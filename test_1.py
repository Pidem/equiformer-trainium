"""Simple inference with EquiformerV3 on Neuron (Trainium)."""
import sys
sys.path.insert(0, "equiformer_v3/src")

import torch
from ase.build import bulk
from fairchem.core.common.registry import registry
from fairchem.core.preprocessing import AtomsToGraphs

# Import the model so it registers itself
import fairchem.experimental.models.equiformer_v3.equiformer_v3  # noqa: F401

# 1. Create a simple NaCl unit cell
atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.64)

# 2. Convert to PyG graph
a2g = AtomsToGraphs(max_neigh=20, radius=6.0, r_energy=False, r_forces=False, r_stress=False)
data = a2g.convert_all([atoms])[0]
num_atoms = len(atoms)
data.batch = torch.zeros(num_atoms, dtype=torch.long)
data.natoms = torch.tensor([num_atoms])

# 3. Instantiate model
model = registry.get_model_class("equiformer_v3")(
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
model.eval()
print(f"Model params: {model.num_params:,}")

# 4. Move model and data to neuron
device = torch.device("neuron")
model = model.to(device)
data = data.to(device)

# 5. Run inference
with torch.no_grad():
    outputs = model(data)

print(f"Energy: {outputs['energy'].item():.6f}")
print(f"Forces shape: {outputs['forces'].shape}")
print(f"Forces:\n{outputs['forces']}")
