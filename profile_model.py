"""Profile EquiformerV3 inference on Neuron (eager, RUNTIME mode)."""
import sys
sys.path.insert(0, "/root/equiformer_v3/src")

import torch
from torch.profiler import profile, ProfilerActivity
from torch_neuronx.profiling import NeuronConfig, ProfileMode, NeuronProfiler
from ase.build import bulk
from fairchem.core.common.registry import registry
from fairchem.core.preprocessing import AtomsToGraphs
from fairchem.core.common.utils import radius_graph_pbc

import fairchem.experimental.models.equiformer_v3.equiformer_v3  # noqa: F401

# 1. Create structure
atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.64)

# 2. Convert to graph (CPU)
a2g = AtomsToGraphs(max_neigh=20, radius=6.0, r_energy=False, r_forces=False, r_stress=False)
data = a2g.convert_all([atoms])[0]
num_atoms = len(atoms)
data.batch = torch.zeros(num_atoms, dtype=torch.long)
data.natoms = torch.tensor([num_atoms])

# Pre-compute graph on CPU so radius_graph_pbc (masked_select) runs on CPU, not Neuron
edge_index, cell_offsets, neighbors = radius_graph_pbc(data, radius=6.0, max_num_neighbors_threshold=20)
data.edge_index = edge_index
data.cell_offsets = cell_offsets
data.neighbors = neighbors

# 3. Small model (otf_graph=False: use pre-computed graph)
model = registry.get_model_class("equiformer_v3")(
    use_pbc=True,
    otf_graph=False,
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
device = torch.device("neuron")
model = model.to(device)
data = data.to(device)

# 4. Profile (RUNTIME only — eager mode doesn't produce NEFF device traces)
neuron_config = NeuronConfig(
    modes=[ProfileMode.RUNTIME],
    profile_output_dir="./profile_output",
)
exporter = NeuronProfiler(neuron_config)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
    experimental_config=neuron_config,
    on_trace_ready=exporter.export_trace,
    acc_events=True,
) as prof:
    with torch.no_grad():
        outputs = model(data)
    torch.neuron.synchronize()

print(f"Energy: {outputs['energy'].item():.6f}")
print("Traces saved to ./profile_output/")
