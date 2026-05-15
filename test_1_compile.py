"""EquiformerV3 on Neuron — XLA baseline vs NKI fused linear.

Two conditions, both using model.to("neuron") eager JIT:
  1. XLA baseline  — model compiled automatically by XLA (one fused NEFF)
  2. NKI fused     — F.linear 2D calls replaced with nki_op-wrapped matmul,
                     traced into the same NEFF via wrap_nki
"""
import sys
sys.path.insert(0, "equiformer_v3/src")
sys.path.insert(0, "kernels")

import time
import torch
import torch.nn.functional as F
from ase.build import bulk
from fairchem.core.common.registry import registry
from fairchem.core.preprocessing import AtomsToGraphs
import torch_neuronx
from torch_neuronx import nki_op, wrap_nki
from matmul_tiled import nki_matmul_tiled, TILE_M

import fairchem.experimental.models.equiformer_v3.equiformer_v3  # noqa: F401


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

    a_T = input.t().contiguous()
    w_T = weight.t().contiguous()
    out = wrap_nki(nki_matmul_tiled)(a_T, w_T)

    if M_pad != M:
        out = out[:M, :]
    if bias is not None:
        out = out + bias
    return out


# ── Build input ───────────────────────────────────────────────────────────────

atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.64)
a2g   = AtomsToGraphs(max_neigh=20, radius=6.0,
                      r_energy=False, r_forces=False, r_stress=False)
data  = a2g.convert_all([atoms])[0]
num_atoms = len(atoms)
data.batch     = torch.zeros(num_atoms, dtype=torch.long)
data.natoms    = torch.tensor([num_atoms], dtype=torch.int32)
# neighbors is required by generate_graph when otf_graph=False
data.neighbors = torch.tensor([data.edge_index.shape[1]], dtype=torch.long)

# Neuron requires contiguous tensors; cast float64 to float32.
# Keep int64 for index tensors (edge_index, cell_offsets, neighbors, batch)
# since indexing ops require long dtype.
INDEX_KEYS = {'edge_index', 'cell_offsets', 'neighbors', 'batch'}
for key in data.keys():
    t = data[key]
    if not isinstance(t, torch.Tensor):
        continue
    if t.dtype == torch.float64:
        t = t.to(torch.float32)
    elif t.dtype == torch.int64 and key not in INDEX_KEYS:
        t = t.to(torch.int32)
    data[key] = t.contiguous()

print(f"Graph: {num_atoms} atoms, {data.edge_index.shape[1]} edges")

model_kwargs = dict(
    use_pbc=True, otf_graph=True, regress_forces=True, direct_prediction=True,
    max_neighbors=20, max_radius=6.0, num_radial_basis=64, num_layers=2,
    num_channels=32, attn_hidden_channels=16, num_heads=4,
    attn_alpha_channels=16, attn_value_channels=8, ffn_hidden_channels=32,
    lmax=2, mmax=2, edge_channels=32,
    attn_grid_resolution_list=[8, 4], ffn_grid_resolution_list=[8, 4],
)

WARMUP = 3
RUNS   = 20
device = torch.device("neuron")


def run_benchmark(model, input_data, label, is_neuron=True):
    if is_neuron:
        print(f"\nFirst Neuron call (JIT compile + inference) — {label} ...")
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(input_data)
            _ = out["energy"].item()
        first_run_time = time.perf_counter() - t0
        print(f"  First call time: {first_run_time:.2f}s")

    print(f"Warm-up ({WARMUP} runs) ...")
    with torch.no_grad():
        for _ in range(WARMUP):
            model(input_data)["energy"].item()

    print(f"Benchmarking ({RUNS} runs) ...")
    latencies = []
    with torch.no_grad():
        for _ in range(RUNS):
            t0 = time.perf_counter()
            model(input_data)["energy"].item()
            latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    avg_ms = sum(latencies) / len(latencies)
    print(f"\n--- {label} ({RUNS} runs) ---")
    print(f"  avg        : {avg_ms:.2f} ms")
    print(f"  min        : {latencies[0]:.2f} ms")
    print(f"  p50        : {latencies[len(latencies)//2]:.2f} ms")
    print(f"  max        : {latencies[-1]:.2f} ms")
    print(f"  throughput : {num_atoms / (avg_ms / 1000):.1f} atoms/s")
    return avg_ms


# Keep old name as alias for neuron calls
def run_neuron_benchmark(model, neuron_data, label):
    return run_benchmark(model, neuron_data, label, is_neuron=True)
    print(f"\nFirst Neuron call (JIT compile + inference) — {label} ...")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(neuron_data)
        _ = out["energy"].item()
    first_run_time = time.perf_counter() - t0
    print(f"  First call time: {first_run_time:.2f}s")

    print(f"Warm-up ({WARMUP} runs) ...")
    with torch.no_grad():
        for _ in range(WARMUP):
            model(neuron_data)["energy"].item()

    print(f"Benchmarking ({RUNS} runs) ...")
    latencies = []
    with torch.no_grad():
        for _ in range(RUNS):
            t0 = time.perf_counter()
            model(neuron_data)["energy"].item()
            latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    avg_ms = sum(latencies) / len(latencies)
    print(f"\n--- {label} ({RUNS} runs, compile excluded) ---")
    print(f"  avg        : {avg_ms:.2f} ms")
    print(f"  min        : {latencies[0]:.2f} ms")
    print(f"  p50        : {latencies[len(latencies)//2]:.2f} ms")
    print(f"  max        : {latencies[-1]:.2f} ms")
    print(f"  throughput : {num_atoms / (avg_ms / 1000):.1f} atoms/s")
    return avg_ms


# ── 1. XLA baseline ────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("1. XLA baseline (model.to('neuron'), F.linear compiled by XLA)")
print("=" * 65)

# torch.compile + neuron requires:
# 1. static shapes (otf_graph=False so edges are fixed at data-prep time)
# 2. all tensors contiguous before each compiled subgraph runs
compile_model_kwargs = {**model_kwargs, 'otf_graph': False}

import torch._dynamo
_neuron_backend = torch._dynamo.backends.registry.lookup_backend('neuron')

def _neuron_contiguous(gm, example_inputs):
    contig = [x.contiguous() if isinstance(x, torch.Tensor) else x for x in example_inputs]
    compiled = _neuron_backend(gm, contig)
    def _run(*args):
        args = tuple(a.contiguous() if isinstance(a, torch.Tensor) else a for a in args)
        return compiled(*args)
    return _run

torch._dynamo.register_backend(name='neuron_contiguous', compiler_fn=_neuron_contiguous)

xla_model   = registry.get_model_class("equiformer_v3")(**compile_model_kwargs).eval()
xla_model   = xla_model.to(device)
xla_model   = torch.compile(xla_model, backend='neuron_contiguous')
neuron_data = data.to(device)

xla_avg = run_neuron_benchmark(xla_model, neuron_data, "XLA baseline")
del xla_model

# ── 2. NKI fused ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("2. NKI fused: nki_op registered as custom op, traced into NEFF")
print("=" * 65)

_orig_linear = F.linear
F.linear = lambda inp, w, b=None: (
    nki_linear_op(inp, w, b) if inp.dim() == 2 else _orig_linear(inp, w, b)
)

nki_model = registry.get_model_class("equiformer_v3")(**compile_model_kwargs).eval()
nki_model = nki_model.to(device)
nki_model = torch.compile(nki_model, backend='neuron_contiguous')

nki_avg = run_neuron_benchmark(nki_model, neuron_data, "NKI fused (custom op in NEFF)")
F.linear = _orig_linear

# ── Summary ───────────────────────────────────────────────────────────────────
# CPU baseline measured in a separate process (_cpu_bench.py) because
# torch_neuronx patches torch.einsum to reject CPU tensors.
cpu_avg = 5.96  # ms, CPU eager from _cpu_bench.py

print(f"\n{'='*65}")
print("Summary")
print("=" * 65)
print(f"  CPU eager    : {cpu_avg:>8.2f} ms   ({num_atoms/(cpu_avg/1000):.1f} atoms/s)  [separate process]")
print(f"  XLA baseline : {xla_avg:>8.2f} ms   ({num_atoms/(xla_avg/1000):.1f} atoms/s)")
print(f"  NKI fused    : {nki_avg:>8.2f} ms   ({num_atoms/(nki_avg/1000):.1f} atoms/s)")
print()
print(f"  Neuron XLA vs CPU : {xla_avg/cpu_avg:.2f}x  ({'faster' if xla_avg < cpu_avg else 'SLOWER — Neuron overhead dominates at this batch size'})")
print(f"  NKI fused vs CPU  : {nki_avg/cpu_avg:.2f}x  ({'faster' if nki_avg < cpu_avg else 'SLOWER — Neuron overhead dominates at this batch size'})")
if nki_avg < xla_avg:
    print(f"  NKI fused vs XLA  : {xla_avg/nki_avg:.2f}x FASTER than XLA baseline")
else:
    print(f"  NKI fused vs XLA  : {nki_avg/xla_avg:.2f}x SLOWER than XLA baseline")
