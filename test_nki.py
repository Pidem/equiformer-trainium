"""Inference with EquiformerV3 on Neuron — NKI scatter_reduce kernel, with timing."""
import sys
import time
sys.path.insert(0, "equiformer_v3/src")

import torch
import torch_neuronx
from ase.build import bulk
from fairchem.core.common.registry import registry
from fairchem.core.preprocessing import AtomsToGraphs

import fairchem.experimental.models.equiformer_v3.equiformer_v3  # noqa: F401

# ── Patch get_counts with the NKI kernel ──────────────────────────────────────
import fairchem.core.common.utils as _fcc_utils
sys.path.insert(0, "kernels")
from scatter_reduce import nki_scatter_reduce_sum_kernel

_nki_call_count = 0

def _nki_get_counts(x: torch.Tensor, length: int):
    global _nki_call_count
    _nki_call_count += 1
    result = nki_scatter_reduce_sum_kernel(x.int(), int(length))
    # Cast back to the original dtype expected by the caller
    return result.to(x.dtype)

_fcc_utils.get_counts = _nki_get_counts


# ── Build input ───────────────────────────────────────────────────────────────
atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.64)
a2g = AtomsToGraphs(max_neigh=20, radius=6.0, r_energy=False, r_forces=False, r_stress=False)
data = a2g.convert_all([atoms])[0]
num_atoms = len(atoms)
data.batch = torch.zeros(num_atoms, dtype=torch.long)
data.natoms = torch.tensor([num_atoms])

# ── Instantiate model ─────────────────────────────────────────────────────────
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

device = torch.device("neuron")
model = model.to(device)
data = data.to(device)

# ── Run inference with timing ─────────────────────────────────────────────────
torch_neuronx.clear_op_tracking()
t0 = time.perf_counter()
with torch.no_grad():
    outputs = model(data)
t1 = time.perf_counter()

print(f"Energy: {outputs['energy'].item():.6f}")
print(f"Forces shape: {outputs['forces'].shape}")
print(f"Forces:\n{outputs['forces']}")
print(f"\nInference wall time (NKI scatter_reduce): {(t1 - t0)*1000:.1f} ms")
print(f"NKI get_counts calls: {_nki_call_count}")

fallback_ops = torch_neuronx.get_fallback_ops()
neuron_ops   = torch_neuronx.get_executed_ops()

print(f"\n--- Op Execution Summary ---")
print(f"Ops on Neuron : {len(neuron_ops)}")
print(f"Ops on CPU    : {len(fallback_ops)}")

if fallback_ops:
    from collections import Counter
    counts = Counter(fallback_ops)
    print(f"\nCPU fallback ops (name: count):")
    for op, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {op}: {count}")
