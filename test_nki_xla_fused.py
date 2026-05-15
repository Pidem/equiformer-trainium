"""
Benchmark: NKI matmul registered as XLA custom op vs baseline XLA F.linear.

The key difference from prior benchmarks:
  - BEFORE: nki_matmul_tiled called directly → separate NEFF per call (~1.2s dispatch)
  - NOW:    nki_op + wrap_nki registers the kernel as a custom torch op,
            so XLA traces it into the model graph and compiles ONE NEFF
            that contains the NKI matmul alongside all other ops.

Two conditions:
  1. XLA baseline — model.to("neuron"), XLA compiles F.linear automatically
  2. NKI fused    — F.linear replaced with nki_op-wrapped matmul, XLA traces
                    the NKI kernel into the same NEFF (one dispatch per forward)
"""
import sys, time
sys.path.insert(0, "equiformer_v3/src")
sys.path.insert(0, "kernels")

import torch
import torch.nn.functional as F
import torch_neuronx
from torch_neuronx import nki_op, wrap_nki
from ase.build import bulk
from fairchem.core.common.registry import registry
from fairchem.core.preprocessing import AtomsToGraphs
import fairchem.experimental.models.equiformer_v3.equiformer_v3  # noqa: F401
from matmul_tiled import nki_matmul_tiled, TILE_M

WARMUP = 5
RUNS   = 20

# ── Register NKI matmul as a traceable custom op ──────────────────────────────

@nki_op("nki_bench::linear", mutates_args={})
def nki_linear_op(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """F.linear replacement backed by nki_matmul_tiled, traceable by XLA."""
    M, K  = input.shape
    N, K2 = weight.shape
    assert K == K2

    M_pad = ((M + TILE_M - 1) // TILE_M) * TILE_M
    if M_pad != M:
        pad = torch.zeros(M_pad - M, K, dtype=input.dtype, device=input.device)
        input = torch.cat([input, pad], dim=0)

    a_T = input.t().contiguous()   # [K, M_pad]
    w_T = weight.t().contiguous()  # [K, N]

    out = wrap_nki(nki_matmul_tiled)(a_T, w_T)  # [M_pad, N]

    if M_pad != M:
        out = out[:M, :]

    if bias is not None:
        out = out + bias
    return out


# ── Input setup ───────────────────────────────────────────────────────────────

atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.64)
a2g = AtomsToGraphs(max_neigh=20, radius=6.0, r_energy=False, r_forces=False, r_stress=False)
data = a2g.convert_all([atoms])[0]
num_atoms = len(atoms)
data.batch  = torch.zeros(num_atoms, dtype=torch.long)
data.natoms = torch.tensor([num_atoms])

model_kwargs = dict(
    use_pbc=True, otf_graph=True, regress_forces=True, direct_prediction=True,
    max_neighbors=20, max_radius=6.0, num_radial_basis=64, num_layers=2,
    num_channels=32, attn_hidden_channels=16, num_heads=4,
    attn_alpha_channels=16, attn_value_channels=8, ffn_hidden_channels=32,
    lmax=2, mmax=2, edge_channels=32,
    attn_grid_resolution_list=[8, 4], ffn_grid_resolution_list=[8, 4],
)


def run_benchmark(model, input_data, label, warmup=WARMUP, runs=RUNS):
    with torch.no_grad():
        for _ in range(warmup):
            model(input_data)["energy"].item()
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            model(input_data)["energy"].item()
            latencies.append((time.perf_counter() - t0) * 1000)
    latencies.sort()
    avg = sum(latencies) / len(latencies)
    print(f"\n--- {label} ({runs} runs) ---")
    print(f"  avg : {avg:.2f} ms   min : {latencies[0]:.2f} ms"
          f"   p50 : {latencies[len(latencies)//2]:.2f} ms"
          f"   max : {latencies[-1]:.2f} ms")
    print(f"  throughput: {num_atoms / (avg / 1000):.1f} atoms/s")
    return avg


device = torch.device("neuron")

# ── 1. XLA baseline ───────────────────────────────────────────────────────────

print("=" * 65)
print("1. XLA baseline (model.to('neuron'), F.linear compiled by XLA)")
print("=" * 65)
xla_model = registry.get_model_class("equiformer_v3")(**model_kwargs).eval()
xla_model  = xla_model.to(device)
xla_data   = data.to(device)

print("First call (JIT compile + inference)...")
torch_neuronx.clear_op_tracking()
with torch.no_grad():
    xla_model(xla_data)["energy"].item()

fallback_ops = torch_neuronx.get_fallback_ops()
neuron_ops   = torch_neuronx.get_executed_ops()
print(f"  Ops on Neuron: {len(neuron_ops)}   CPU fallbacks: {len(fallback_ops)}")
if fallback_ops:
    from collections import Counter
    for op, cnt in sorted(Counter(fallback_ops).items(), key=lambda x: -x[1]):
        print(f"    {op}: {cnt}")

xla_avg = run_benchmark(xla_model, xla_data, "XLA baseline (one fused NEFF)")
del xla_model

# ── 2. NKI fused (nki_op inside XLA graph) ───────────────────────────────────

print("\n" + "=" * 65)
print("2. NKI fused: nki_op registered as custom op, traced into NEFF")
print("=" * 65)

_orig_linear = F.linear
F.linear = lambda inp, w, b=None: (
    nki_linear_op(inp, w, b) if inp.dim() == 2 else _orig_linear(inp, w, b)
)

nki_model = registry.get_model_class("equiformer_v3")(**model_kwargs).eval()
nki_model  = nki_model.to(device)

print("First call (JIT compile with NKI custom op traced in)...")
torch_neuronx.clear_op_tracking()
with torch.no_grad():
    nki_model(xla_data)["energy"].item()

F.linear = _orig_linear  # restore

fallback_ops_nki = torch_neuronx.get_fallback_ops()
neuron_ops_nki   = torch_neuronx.get_executed_ops()
print(f"  Ops on Neuron: {len(neuron_ops_nki)}   CPU fallbacks: {len(fallback_ops_nki)}")
if fallback_ops_nki:
    from collections import Counter
    for op, cnt in sorted(Counter(fallback_ops_nki).items(), key=lambda x: -x[1]):
        print(f"    {op}: {cnt}")

# Patch active during benchmark too
F.linear = lambda inp, w, b=None: (
    nki_linear_op(inp, w, b) if inp.dim() == 2 else _orig_linear(inp, w, b)
)
nki_avg = run_benchmark(nki_model, xla_data, "NKI fused (custom op in NEFF)")
F.linear = _orig_linear

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("Summary")
print("=" * 65)
print(f"  XLA baseline     : {xla_avg:>8.2f} ms   ({num_atoms/(xla_avg/1000):.1f} atoms/s)")
print(f"  NKI fused        : {nki_avg:>8.2f} ms   ({num_atoms/(nki_avg/1000):.1f} atoms/s)")
if nki_avg < xla_avg:
    print(f"\n  NKI fused is {xla_avg/nki_avg:.2f}x FASTER than XLA baseline")
else:
    print(f"\n  NKI fused is {nki_avg/xla_avg:.2f}x SLOWER than XLA baseline"
          f"  (XLA fusion still wins at M={39*1} edges)")
