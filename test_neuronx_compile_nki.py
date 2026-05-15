"""
EquiformerV3: torch_neuronx compilation with NKI fused-linear kernel.

Compares three execution modes:
  1. CPU baseline        - standard PyTorch forward pass
  2. Neuron XLA baseline - model.to("neuron") JIT, F.linear compiled by XLA
  3. Neuron + NKI fused  - F.linear replaced by nki_op-wrapped matmul;
                           XLA traces the NKI op into the same NEFF so there
                           is ONE dispatch per forward (no per-call PCIe overhead)

NKI kernel: kernels/matmul_tiled.py (nki_matmul_tiled)

Key design:
  - nki_op + wrap_nki registers the NKI kernel as a custom torch op so XLA
    can trace it into the model graph and compile it into a single NEFF.
  - otf_graph=False: graph is pre-computed on CPU to avoid mask_select
    CPU fallback on the Neuron device (uses radius_graph_pbc directly).
  - F.linear patch is active during model.to("neuron") AND all inference
    calls so the traced graph stays consistent.

Remaining CPU fallback ops (present in both XLA and NKI models):
  aten::scatter_reduce, aten::atan2, aten::acos, aten::linalg_cross,
  aten::repeat_interleave, aten::uniform_
  These ops are called once per forward for small tensors (edge geometry).
  Each PCIe round-trip is ~600-700 ms → dominates model latency.
"""
import os, sys, time
os.environ["NEURON_CC_FLAGS"] = "--target trn2 --lnc 2"
sys.path.insert(0, "equiformer_v3/src")
sys.path.insert(0, "kernels")

import torch
import torch.nn.functional as F
import torch_neuronx
from torch_neuronx import nki_op, wrap_nki
from ase.build import bulk
from fairchem.core.common.registry import registry
from fairchem.core.preprocessing import AtomsToGraphs
from fairchem.core.common.utils import radius_graph_pbc
import fairchem.experimental.models.equiformer_v3.equiformer_v3  # noqa: F401
from matmul_tiled import nki_matmul_tiled, TILE_M
from collections import Counter

# ── Build input (pre-computed graph, avoids mask_select fallback) ─────────────
atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.64)
a2g   = AtomsToGraphs(max_neigh=20, radius=6.0,
                      r_energy=False, r_forces=False, r_stress=False)
data = a2g.convert_all([atoms])[0]
num_atoms = len(atoms)
data.batch   = torch.zeros(num_atoms, dtype=torch.long)
data.natoms  = torch.tensor([num_atoms])
edge_index, cell_offsets, neighbors = radius_graph_pbc(
    data, radius=6.0, max_num_neighbors_threshold=20
)
data.edge_index    = edge_index
data.cell_offsets  = cell_offsets
data.neighbors     = neighbors

MODEL_KWARGS = dict(
    use_pbc=True, otf_graph=False, regress_forces=True, direct_prediction=True,
    max_neighbors=20, max_radius=6.0, num_radial_basis=64, num_layers=2,
    num_channels=32, attn_hidden_channels=16, num_heads=4,
    attn_alpha_channels=16, attn_value_channels=8, ffn_hidden_channels=32,
    lmax=2, mmax=2, edge_channels=32,
    attn_grid_resolution_list=[8, 4], ffn_grid_resolution_list=[8, 4],
)

WARMUP = 3
RUNS   = 20

# ── Register NKI matmul as a traceable custom op ──────────────────────────────
# Using nki_op + wrap_nki so XLA can trace the NKI kernel into the model graph
# and compile it into a single NEFF — no per-call PCIe dispatch overhead.

@nki_op("nki_bench::linear", mutates_args={})
def nki_linear_op(input: torch.Tensor, weight: torch.Tensor,
                  bias: torch.Tensor | None = None) -> torch.Tensor:
    """F.linear replacement backed by nki_matmul_tiled, traceable by XLA."""
    M, K  = input.shape
    N, K2 = weight.shape
    assert K == K2

    M_pad = ((M + TILE_M - 1) // TILE_M) * TILE_M
    if M_pad != M:
        pad   = torch.zeros(M_pad - M, K, dtype=input.dtype, device=input.device)
        input = torch.cat([input, pad], dim=0)

    a_T = input.t().contiguous()    # [K, M_pad]
    w_T = weight.t().contiguous()   # [K, N]

    out = wrap_nki(nki_matmul_tiled)(a_T, w_T)   # [M_pad, N] float32

    if M_pad != M:
        out = out[:M, :]

    if bias is not None:
        out = out + bias
    return out

# ── Benchmark helper ──────────────────────────────────────────────────────────
def run_benchmark(model, data_input, label, warmup=WARMUP, runs=RUNS):
    with torch.no_grad():
        for _ in range(warmup):
            model(data_input)["energy"].item()
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            model(data_input)["energy"].item()
            latencies.append((time.perf_counter() - t0) * 1000)
    latencies.sort()
    avg = sum(latencies) / len(latencies)
    print(f"\n--- {label} ({runs} runs) ---")
    print(f"  avg  : {avg:.2f} ms  ({num_atoms / (avg / 1000):.1f} atoms/s)")
    print(f"  min  : {latencies[0]:.2f} ms")
    print(f"  p50  : {latencies[len(latencies) // 2]:.2f} ms")
    print(f"  max  : {latencies[-1]:.2f} ms")
    return avg

def print_op_summary():
    fb  = torch_neuronx.get_fallback_ops()
    neu = torch_neuronx.get_executed_ops()
    print(f"  Neuron ops: {len(neu)},  CPU fallbacks: {len(fb)}")
    if fb:
        for op, cnt in sorted(Counter(fb).items(), key=lambda x: -x[1]):
            print(f"    {op}: {cnt}")
    return len(fb)

# ── 1. CPU baseline ───────────────────────────────────────────────────────────
print("=" * 60)
print("1. CPU baseline")
print("=" * 60)
cpu_model = registry.get_model_class("equiformer_v3")(**MODEL_KWARGS).eval()
print(f"Model params: {cpu_model.num_params:,}")
cpu_avg    = run_benchmark(cpu_model, data, "CPU baseline")
cpu_energy = cpu_model(data)["energy"].item()
print(f"  Energy : {cpu_energy:.6f}")

device = torch.device("neuron")

# ── 2. Neuron XLA baseline ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. Neuron XLA baseline (model.to('neuron'), F.linear via XLA)")
print("=" * 60)
xla_model   = registry.get_model_class("equiformer_v3")(**MODEL_KWARGS).eval().to(device)
neuron_data = data.to(device)

print("First call (JIT compile + inference) ...")
torch_neuronx.clear_op_tracking()
t0 = time.perf_counter()
with torch.no_grad():
    xla_energy = xla_model(neuron_data)["energy"].item()
xla_first = time.perf_counter() - t0
print(f"  Compile + first run : {xla_first:.2f}s  Energy: {xla_energy:.6f}")
xla_nfb = print_op_summary()

xla_avg = run_benchmark(xla_model, neuron_data, "Neuron XLA (F.linear by XLA)")

# ── 3. Neuron + NKI fused via nki_op (traced into NEFF) ──────────────────────
print("\n" + "=" * 60)
print("3. Neuron + NKI (nki_op traced into NEFF, one dispatch per forward)")
print("=" * 60)

_orig_linear = F.linear
F.linear = lambda inp, w, b=None: (
    nki_linear_op(inp, w, b) if inp.dim() == 2 else _orig_linear(inp, w, b)
)

nki_model = registry.get_model_class("equiformer_v3")(**MODEL_KWARGS).eval().to(device)

print("First call (JIT compile with NKI custom op traced in) ...")
torch_neuronx.clear_op_tracking()
t0 = time.perf_counter()
with torch.no_grad():
    nki_energy = nki_model(neuron_data)["energy"].item()
nki_first = time.perf_counter() - t0
print(f"  Compile + first run : {nki_first:.2f}s  Energy: {nki_energy:.6f}")
nki_nfb = print_op_summary()

nki_avg = run_benchmark(nki_model, neuron_data, "Neuron + NKI fused (nki_op in NEFF)")
F.linear = _orig_linear

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Performance Summary")
print("=" * 60)
print(f"  CPU baseline              : {cpu_avg:>10.2f} ms")
print(f"  Neuron XLA                : {xla_avg:>10.2f} ms  compile={xla_first:.0f}s  fallbacks={xla_nfb}")
print(f"  Neuron + NKI (nki_op)     : {nki_avg:>10.2f} ms  compile={nki_first:.0f}s  fallbacks={nki_nfb}")

delta = nki_avg - xla_avg
sign  = "+" if delta >= 0 else ""
print(f"\n  XLA  vs CPU  : {cpu_avg/xla_avg:.3f}x  ({sign if cpu_avg/xla_avg < 1 else ''}{cpu_avg/xla_avg - 1:.1%})")
print(f"  NKI  vs CPU  : {cpu_avg/nki_avg:.3f}x")
print(f"  NKI  vs XLA  : {sign}{delta:.2f} ms  ({sign}{100*delta/xla_avg:.1f}%)")
if nki_avg < xla_avg:
    print(f"  => NKI is {xla_avg/nki_avg:.2f}x FASTER than XLA")
else:
    print(f"  => NKI is {nki_avg/xla_avg:.2f}x SLOWER than XLA")

print(f"\n  Energy: CPU={cpu_energy:.6f}  XLA={xla_energy:.6f}  NKI={nki_energy:.6f}")
energy_ok = abs(xla_energy - nki_energy) / (abs(cpu_energy) + 1e-8) < 0.01
print(f"  NKI vs XLA energy match: {'OK (within 1%)' if energy_ok else 'MISMATCH'}")
print(f"\n  Note: {xla_nfb} CPU fallback ops (scatter_reduce, atan2, etc.) each cost")
print(f"  ~{xla_avg / (xla_nfb + 1):.0f} ms PCIe round-trip — these dominate both model latencies.")
