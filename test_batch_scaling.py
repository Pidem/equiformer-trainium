"""Batch-size sweep to find where NKI matmul kernels become viable for EquiformerV3.

For a NaCl unit cell (2 atoms, ~39 edges), the SO2 linear layers have M=num_edges
which is too small at B=1 to amortize NKI dispatch overhead (~1s per kernel launch).
This script sweeps batch sizes and reports:
  - num_edges (= M dimension of all edge linear ops)
  - CPU latency
  - Neuron (XLA) latency
  - NKI viability estimate (M >= 128 minimum, M >= 512 for full TE utilization)
"""
import sys
import time
sys.path.insert(0, "equiformer_v3/src")

import torch
import torch_neuronx
from ase.build import bulk
from torch_geometric.data import Batch
from fairchem.core.common.registry import registry
from fairchem.core.preprocessing import AtomsToGraphs
import fairchem.experimental.models.equiformer_v3.equiformer_v3  # noqa: F401

ATOMS = bulk("NaCl", crystalstructure="rocksalt", a=5.64)
A2G = AtomsToGraphs(max_neigh=20, radius=6.0, r_energy=False, r_forces=False, r_stress=False)

MODEL_KWARGS = dict(
    use_pbc=True, otf_graph=True, regress_forces=True, direct_prediction=True,
    max_neighbors=20, max_radius=6.0, num_radial_basis=64, num_layers=2,
    num_channels=32, attn_hidden_channels=16, num_heads=4,
    attn_alpha_channels=16, attn_value_channels=8, ffn_hidden_channels=32,
    lmax=2, mmax=2, edge_channels=32,
    attn_grid_resolution_list=[8, 4], ffn_grid_resolution_list=[8, 4],
)

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
WARMUP = 5
RUNS = 10


def make_batch(batch_size):
    data_list = [A2G.convert_all([ATOMS])[0] for _ in range(batch_size)]
    batch = Batch.from_data_list(data_list)
    batch.batch = batch.batch
    batch.natoms = torch.tensor([2] * batch_size, dtype=torch.long)
    return batch


def nki_viability(num_edges):
    if num_edges < 128:
        return "too small (M<128)"
    elif num_edges < 512:
        return "marginal (128<=M<512)"
    else:
        return "VIABLE (M>=512)"


print("=" * 70)
print("EquiformerV3 Batch Scaling: CPU vs Neuron")
print("=" * 70)

# ── CPU baseline sweep ─────────────────────────────────────────────────────────
print("\n--- CPU Baseline ---")
print(f"{'B':>5}  {'edges':>6}  {'NKI viable?':>22}  {'avg ms':>8}  {'atoms/s':>9}")
print("-" * 60)

cpu_results = {}
cpu_model = registry.get_model_class("equiformer_v3")(**MODEL_KWARGS)
cpu_model.eval()

for bs in BATCH_SIZES:
    batch = make_batch(bs)
    num_edges = batch.edge_index.shape[1]

    # warmup
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = cpu_model(batch)

    latencies = []
    with torch.no_grad():
        for _ in range(RUNS):
            t0 = time.perf_counter()
            out = cpu_model(batch)
            _ = out["energy"].sum().item()
            latencies.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(latencies) / len(latencies)
    atoms_per_s = (bs * 2) / (avg_ms / 1000)
    cpu_results[bs] = (num_edges, avg_ms, 0.0, atoms_per_s)
    viability = nki_viability(num_edges)
    print(f"{bs:>5}  {num_edges:>6}  {viability:>22}  {avg_ms:>8.1f}  {atoms_per_s:>9.1f}")

del cpu_model

# ── Neuron sweep ───────────────────────────────────────────────────────────────
print("\n--- Neuron (XLA JIT) ---")
print(f"{'B':>5}  {'edges':>6}  {'NKI viable?':>22}  {'avg ms':>8}  {'speedup':>8}  {'atoms/s':>9}")
print("-" * 72)

neuron_results = {}
neuron_model = registry.get_model_class("equiformer_v3")(**MODEL_KWARGS)
neuron_model.eval()
device = torch.device("neuron")
neuron_model = neuron_model.to(device)

for bs in BATCH_SIZES:
    batch = make_batch(bs)
    num_edges = batch.edge_index.shape[1]
    neuron_batch = batch.to(device)

    print(f"  B={bs}: compiling + warming up...", flush=True)

    # First call triggers JIT compile; subsequent warmup calls ensure NEFF is cached
    with torch.no_grad():
        for _ in range(WARMUP + 1):
            out = neuron_model(neuron_batch)
            _ = out["energy"].sum().item()

    latencies = []
    with torch.no_grad():
        for _ in range(RUNS):
            t0 = time.perf_counter()
            out = neuron_model(neuron_batch)
            _ = out["energy"].sum().item()
            latencies.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(latencies) / len(latencies)
    atoms_per_s = (bs * 2) / (avg_ms / 1000)
    cpu_avg = cpu_results[bs][1]
    speedup = cpu_avg / avg_ms
    neuron_results[bs] = (num_edges, avg_ms, speedup, atoms_per_s)
    viability = nki_viability(num_edges)
    print(f"{bs:>5}  {num_edges:>6}  {viability:>22}  {avg_ms:>8.1f}  {speedup:>7.2f}x  {atoms_per_s:>9.1f}")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Summary: Neuron speedup vs CPU, and NKI threshold crossings")
print("=" * 70)
print(f"{'B':>5}  {'edges':>6}  {'CPU ms':>8}  {'Neuron ms':>10}  {'speedup':>8}  {'NKI?':>22}")
print("-" * 72)
for bs in BATCH_SIZES:
    num_edges, cpu_ms, _, _ = cpu_results[bs]
    num_edges, n_ms, speedup, _ = neuron_results[bs]
    viability = nki_viability(num_edges)
    marker = " *" if num_edges >= 512 else ("+" if num_edges >= 128 else "")
    print(f"{bs:>5}  {num_edges:>6}  {cpu_ms:>8.1f}  {n_ms:>10.1f}  {speedup:>7.2f}x  {viability}{marker}")

print("\nKey thresholds:")
print("  M >= 128: minimum for NKI nc_matmul (stationary [128,K], moving [K,128])")
print("  M >= 512: PSUM free dim fills (full Tensor Engine utilization)")
print("  *  = full TE utilization possible")
print("  +  = marginal, worth profiling")
