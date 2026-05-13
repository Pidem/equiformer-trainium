"""Benchmark EquiformerV3 inference on Neuron (Trainium) — varying system sizes."""
import sys
sys.path.insert(0, "equiformer_v3/src")

import time
import torch
import numpy as np
from ase import Atoms
from ase.build import bulk
from fairchem.core.common.registry import registry
from fairchem.core.preprocessing import AtomsToGraphs

import fairchem.experimental.models.equiformer_v3.equiformer_v3  # noqa: F401


def make_supercell(n_repeat):
    """Create NaCl supercell with n_repeat^3 unit cells."""
    atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.64)
    atoms = atoms.repeat((n_repeat, n_repeat, n_repeat))
    return atoms


def benchmark(model, data, name, warmup=3, runs=10):
    """Time inference runs and report stats."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(data)

    # Timed runs
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(data)
        # Sync (force completion)
        _ = outputs['energy'].item()
        times.append(time.perf_counter() - start)

    times = np.array(times)
    num_atoms = len(data.pos)
    print(f"  {name} ({num_atoms} atoms): "
          f"mean={times.mean()*1000:.1f}ms, "
          f"std={times.std()*1000:.1f}ms, "
          f"min={times.min()*1000:.1f}ms, "
          f"throughput={num_atoms/times.mean():.0f} atoms/s")
    return times.mean(), num_atoms


# Build structures of increasing size
print("Building structures...")
structures = {
    "NaCl 1x1x1 (2 atoms)": make_supercell(1),
    "NaCl 2x2x2 (16 atoms)": make_supercell(2),
    "NaCl 3x3x3 (54 atoms)": make_supercell(3),
    "NaCl 4x4x4 (128 atoms)": make_supercell(4),
    "NaCl 5x5x5 (250 atoms)": make_supercell(5),
}

for name, atoms in structures.items():
    print(f"  {name}: {len(atoms)} atoms")

# Convert to graphs
a2g = AtomsToGraphs(max_neigh=20, radius=12.0, r_energy=False, r_forces=False, r_stress=False)

# Full EquiformerV3 (OC20 config: N=8, L=6, C=128)
model = registry.get_model_class("equiformer_v3")(
    use_pbc=True,
    otf_graph=True,
    regress_forces=True,
    regress_stress=False,
    direct_prediction=True,
    max_neighbors=20,
    max_radius=12.0,
    num_radial_basis=128,
    max_num_elements=128,
    num_layers=8,
    num_channels=128,
    attn_hidden_channels=64,
    num_heads=8,
    attn_alpha_channels=64,
    attn_value_channels=16,
    ffn_hidden_channels=512,
    norm_type="merge_layer_norm",
    lmax=6,
    mmax=2,
    attn_grid_resolution_list=[20, 8],
    ffn_grid_resolution_list=[20, 20],
    edge_channels=128,
    use_atom_edge_embedding=True,
    use_envelope=True,
    attn_activation="sep-merge_gates2_swiglu",
    use_attn_renorm=True,
    use_add_merge=False,
    use_rad_l_parametrization=True,
    softcap=None,
    ffn_activation="sep-merge_gates2_swiglu",
    use_grid_mlp=True,
    use_gate_force_head=True,
    alpha_drop=0.0,
    attn_mask_rate=0.0,
    attn_weights_drop=0.0,
    value_drop=0.0,
    drop_path_rate=0.0,
    proj_drop=0.0,
    ffn_drop=0.0,
)
model.eval()
print(f"\nModel params: {model.num_params:,}")

# Move to Neuron
device = torch.device("neuron")
model = model.to(device)

# Benchmark each structure size
print(f"\n{'='*60}")
print(f"Benchmark: EquiformerV3 (N=8, L=6, C=128) on Neuron")
print(f"{'='*60}")

results = []
for name, atoms in structures.items():
    data_list = a2g.convert_all([atoms])
    data = data_list[0]
    num_atoms = len(atoms)
    data.batch = torch.zeros(num_atoms, dtype=torch.long)
    data.natoms = torch.tensor([num_atoms])
    data = data.to(device)

    mean_time, n_atoms = benchmark(model, data, name)
    results.append((n_atoms, mean_time))

# Summary
print(f"\n{'='*60}")
print(f"{'Atoms':<10} {'Time (ms)':<12} {'Throughput (atoms/s)':<20}")
print(f"{'-'*60}")
for n_atoms, t in results:
    print(f"{n_atoms:<10} {t*1000:<12.1f} {n_atoms/t:<20.0f}")
