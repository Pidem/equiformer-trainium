"""
Benchmark: EquiformerV3 on Neuron — baseline XLA vs NKI-patched F.linear.

The patch intercepts torch.nn.functional.linear for 2D inputs only and
routes them through nki_fused_linear (which expects pre-transposed inputs).
3D inputs fall back to the default XLA path.

Key finding: aten::linear is already compiled to Neuron by XLA — there are
no CPU fallbacks for linear. This benchmark measures whether explicit NKI
dispatch has lower latency than XLA auto-compilation for the linear ops.

Two conditions:
  1. XLA baseline   — model.to("neuron"), linear compiled automatically by XLA
  2. NKI-patched    — F.linear 2D calls routed through nki_fused_linear
"""
import os, sys, time
os.environ["NEURON_CC_FLAGS"] = "--target trn2 --lnc 1"
os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = "trn2"
os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
sys.path.insert(0, "equiformer_v3/src")
sys.path.insert(0, "kernels")

import torch
import torch_neuronx
from ase.build import bulk
from fairchem.core.common.registry import registry
from fairchem.core.preprocessing import AtomsToGraphs
import fairchem.experimental.models.equiformer_v3.equiformer_v3  # noqa: F401
from fused_linear import nki_fused_linear

# ── Build input ───────────────────────────────────────────────────────────────
atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.64)
a2g = AtomsToGraphs(max_neigh=20, radius=6.0, r_energy=False, r_forces=False, r_stress=False)
data = a2g.convert_all([atoms])[0]
num_atoms = len(atoms)
data.batch = torch.zeros(num_atoms, dtype=torch.long)
data.natoms = torch.tensor([num_atoms])

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

# ── NKI linear wrapper ────────────────────────────────────────────────────────
import torch.nn.functional as F
_orig_linear = F.linear

_nki_call_count = 0

def _nki_linear(input, weight, bias=None):
    """Route 2D linear calls through nki_fused_linear; fall back otherwise."""
    global _nki_call_count
    if input.dim() == 2:
        _nki_call_count += 1
        # nki_fused_linear expects input.T [K,M] and weight.T [K,N]
        input_t = input.t().contiguous()
        weight_t = weight.t().contiguous()
        return nki_fused_linear(input_t, weight_t, bias)
    return _orig_linear(input, weight, bias)


def run_benchmark(model, neuron_data, label, warmup=WARMUP, runs=RUNS):
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(neuron_data)["energy"].item()
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            _ = model(neuron_data)["energy"].item()
            latencies.append((time.perf_counter() - t0) * 1000)
    latencies.sort()
    avg = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    print(f"\n--- {label} ({runs} runs) ---")
    print(f"  avg : {avg:.2f} ms")
    print(f"  min : {latencies[0]:.2f} ms")
    print(f"  p50 : {p50:.2f} ms")
    print(f"  max : {latencies[-1]:.2f} ms")
    print(f"  throughput: {num_atoms / (avg / 1000):.1f} atoms/s")
    return avg

device = torch.device("neuron")

# ── Baseline: XLA auto-compilation (no NKI patch) ────────────────────────────
print("=== Building baseline model (XLA) ===")
baseline_model = registry.get_model_class("equiformer_v3")(**model_kwargs).eval()
baseline_model = baseline_model.to(device)
neuron_data = data.to(device)

print("First call (JIT compile + inference)...")
torch_neuronx.clear_op_tracking()
with torch.no_grad():
    _ = baseline_model(neuron_data)["energy"].item()

fallback_ops = torch_neuronx.get_fallback_ops()
neuron_ops   = torch_neuronx.get_executed_ops()
print(f"  Ops on Neuron: {len(neuron_ops)},  CPU fallbacks: {len(fallback_ops)}")
if fallback_ops:
    from collections import Counter
    for op, cnt in sorted(Counter(fallback_ops).items(), key=lambda x: -x[1]):
        print(f"    {op}: {cnt}")

baseline_avg = run_benchmark(baseline_model, neuron_data, "Baseline XLA (aten::linear on Neuron)")

# ── NKI-patched: F.linear 2D → nki_fused_linear ──────────────────────────────
print("\n=== Building NKI-patched model ===")
F.linear = _nki_linear
_nki_call_count = 0

nki_model = registry.get_model_class("equiformer_v3")(**model_kwargs).eval()
nki_model = nki_model.to(device)

print("First call (JIT compile + inference)...")
torch_neuronx.clear_op_tracking()
with torch.no_grad():
    _ = nki_model(neuron_data)["energy"].item()
print(f"  NKI linear calls intercepted this run: {_nki_call_count}")

F.linear = _orig_linear  # restore before benchmarking to avoid double-counting

fallback_ops_nki = torch_neuronx.get_fallback_ops()
neuron_ops_nki   = torch_neuronx.get_executed_ops()
print(f"  Ops on Neuron: {len(neuron_ops_nki)},  CPU fallbacks: {len(fallback_ops_nki)}")

# Patch active during inference for benchmark
F.linear = _nki_linear
_nki_call_count = 0
nki_avg = run_benchmark(nki_model, neuron_data, "NKI-patched F.linear (2D→nki_fused_linear)")
print(f"  NKI linear calls per run (approx): {_nki_call_count // RUNS}")
F.linear = _orig_linear

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n=== Summary ===")
print(f"  Baseline avg  : {baseline_avg:.2f} ms")
print(f"  NKI-patch avg : {nki_avg:.2f} ms")
delta = nki_avg - baseline_avg
sign  = "+" if delta >= 0 else ""
print(f"  Delta         : {sign}{delta:.2f} ms  ({sign}{100*delta/baseline_avg:.1f}%)")
if nki_avg < baseline_avg:
    print(f"  Speedup       : {baseline_avg/nki_avg:.2f}x  (NKI faster)")
else:
    print(f"  Overhead      : {nki_avg/baseline_avg:.2f}x  (NKI slower — kernel dispatch cost)")
