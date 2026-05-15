"""EquiformerV3 on Neuron (eager JIT) — benchmark vs CPU baseline.

NOTE: scatter_reduce was removed as an NKI target. It is called once per
forward pass with only 168 bytes of I/O, but incurs ~1.2 s of PCIe
dispatch overhead per call (64 KB kernel binary reloaded each time),
making the model ~4x slower. The op runs fine as a CPU fallback in ~8 µs.

Good NKI targets: fused linear (FFN projections) and fused attention (bmm
+ softmax + bmm). See kernels/fused_linear.py for the NKI implementation.
"""
import sys
import time
sys.path.insert(0, "equiformer_v3/src")
sys.path.insert(0, "kernels")

import torch
import torch_neuronx
from ase.build import bulk
from fairchem.core.common.registry import registry
from fairchem.core.preprocessing import AtomsToGraphs

import fairchem.experimental.models.equiformer_v3.equiformer_v3  # noqa: F401

# ── Build input ───────────────────────────────────────────────────────────────
atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.64)
a2g = AtomsToGraphs(max_neigh=20, radius=6.0, r_energy=False, r_forces=False, r_stress=False)
data = a2g.convert_all([atoms])[0]
num_atoms = len(atoms)
data.batch = torch.zeros(num_atoms, dtype=torch.long)
data.natoms = torch.tensor([num_atoms])

model_kwargs = dict(
    use_pbc=True,
    otf_graph=True,
    regress_forces=True,
    direct_prediction=True,
    max_neighbors=20,
    max_radius=6.0,
    num_radial_basis=64,
    num_layers=2,
    num_channels=32,
    attn_hidden_channels=16,
    num_heads=4,
    attn_alpha_channels=16,
    attn_value_channels=8,
    ffn_hidden_channels=32,
    lmax=2,
    mmax=2,
    edge_channels=32,
    attn_grid_resolution_list=[8, 4],
    ffn_grid_resolution_list=[8, 4],
)

WARMUP = 3
RUNS = 20

# ── CPU baseline (NKI kernel runs on CPU via fallback during CPU runs) ────────
cpu_model = registry.get_model_class("equiformer_v3")(**model_kwargs)
cpu_model.eval()
print(f"Model params: {cpu_model.num_params:,}")

with torch.no_grad():
    for _ in range(WARMUP):
        _ = cpu_model(data)

cpu_latencies = []
with torch.no_grad():
    for _ in range(RUNS):
        t0 = time.perf_counter()
        cpu_out = cpu_model(data)
        _ = cpu_out["energy"].item()
        cpu_latencies.append((time.perf_counter() - t0) * 1000)

cpu_latencies.sort()
cpu_avg = sum(cpu_latencies) / len(cpu_latencies)
print(f"\n--- CPU Baseline ({RUNS} runs) ---")
print(f"  avg        : {cpu_avg:.2f} ms")
print(f"  min        : {cpu_latencies[0]:.2f} ms")
print(f"  p50        : {cpu_latencies[len(cpu_latencies)//2]:.2f} ms")
print(f"  max        : {cpu_latencies[-1]:.2f} ms")
print(f"  throughput : {num_atoms / (cpu_avg / 1000):.1f} atoms/s")
print(f"  Energy     : {cpu_out['energy'].item():.6f}")

# ── Neuron — model.to("neuron") triggers JIT compile to NEFF on first call ───
neuron_model = registry.get_model_class("equiformer_v3")(**model_kwargs)
neuron_model.eval()
device = torch.device("neuron")
neuron_model = neuron_model.to(device)
neuron_data = data.to(device)

print("\nFirst Neuron call (JIT compile + inference) ...")
_nki_call_count = 0
torch_neuronx.clear_op_tracking()
t0 = time.perf_counter()
with torch.no_grad():
    _ = neuron_model(neuron_data)
    _ = _["energy"].item()  # sync
first_run_time = time.perf_counter() - t0

print(f"\nWarm-up ({WARMUP} additional runs) ...")
with torch.no_grad():
    for _ in range(WARMUP):
        _ = neuron_model(neuron_data)
        _ = _["energy"].item()

print(f"Benchmarking ({RUNS} runs) ...")
neuron_latencies = []
with torch.no_grad():
    for _ in range(RUNS):
        t0 = time.perf_counter()
        neuron_out = neuron_model(neuron_data)
        _ = neuron_out["energy"].item()  # sync
        neuron_latencies.append((time.perf_counter() - t0) * 1000)

neuron_latencies.sort()
avg_ms = sum(neuron_latencies) / len(neuron_latencies)
compile_time = max(0.0, first_run_time - avg_ms / 1000)
print(f"  Compile time (est.): {compile_time:.2f}s")

print(f"\n--- Neuron + NKI Benchmark ({RUNS} runs, compile excluded) ---")
print(f"  avg        : {avg_ms:.2f} ms")
print(f"  min        : {neuron_latencies[0]:.2f} ms")
print(f"  p50        : {neuron_latencies[len(neuron_latencies)//2]:.2f} ms")
print(f"  max        : {neuron_latencies[-1]:.2f} ms")
print(f"  throughput : {num_atoms / (avg_ms / 1000):.1f} atoms/s")
print(f"  NKI get_counts calls: {_nki_call_count}")

print(f"\n--- Speedup vs CPU ---")
print(f"  CPU avg        : {cpu_avg:.2f} ms  ({num_atoms / (cpu_avg / 1000):.1f} atoms/s)")
print(f"  Neuron+NKI avg : {avg_ms:.2f} ms  ({num_atoms / (avg_ms / 1000):.1f} atoms/s)")
print(f"  Latency speedup: {cpu_avg/avg_ms:.2f}x")

# ── Op summary ────────────────────────────────────────────────────────────────
fallback_ops = torch_neuronx.get_fallback_ops()
neuron_ops = torch_neuronx.get_executed_ops()
print(f"\n--- Op Execution Summary ---")
print(f"  Ops on Neuron : {len(neuron_ops)}")
print(f"  Ops on CPU    : {len(fallback_ops)}")
if fallback_ops:
    from collections import Counter
    for op, count in sorted(Counter(fallback_ops).items(), key=lambda x: -x[1]):
        print(f"    {op}: {count}")
