"""CPU-only benchmark — must run in a process without torch_neuronx imported."""
import sys, time
sys.path.insert(0, "equiformer_v3/src")

import torch
from ase.build import bulk
from fairchem.core.common.registry import registry
from fairchem.core.preprocessing import AtomsToGraphs
import fairchem.experimental.models.equiformer_v3.equiformer_v3  # noqa: F401

atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.64)
a2g = AtomsToGraphs(max_neigh=20, radius=6.0, r_energy=False, r_forces=False, r_stress=False)
data = a2g.convert_all([atoms])[0]
num_atoms = len(atoms)
data.batch  = torch.zeros(num_atoms, dtype=torch.long)
data.natoms = torch.tensor([num_atoms], dtype=torch.int32)

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

model = registry.get_model_class("equiformer_v3")(**model_kwargs).eval()

with torch.no_grad():
    for _ in range(WARMUP):
        model(data)["energy"].item()

latencies = []
with torch.no_grad():
    for _ in range(RUNS):
        t0 = time.perf_counter()
        model(data)["energy"].item()
        latencies.append((time.perf_counter() - t0) * 1000)

latencies.sort()
avg = sum(latencies) / len(latencies)
print(f"\n--- CPU eager ({RUNS} runs) ---")
print(f"  avg        : {avg:.2f} ms")
print(f"  min        : {latencies[0]:.2f} ms")
print(f"  p50        : {latencies[len(latencies)//2]:.2f} ms")
print(f"  max        : {latencies[-1]:.2f} ms")
print(f"  throughput : {num_atoms / (avg / 1000):.1f} atoms/s")
print(f"CPU_AVG_MS={avg:.4f}")
