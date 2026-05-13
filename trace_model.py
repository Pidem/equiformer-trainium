"""Trace EquiformerV3 to a single NEFF and capture an inspect-mode profile."""
import os
import sys
import types

os.environ["NEURON_RT_NUM_CORES"] = "1"
os.environ["NEURON_CC_FLAGS"] = "--target trn2 --lnc 1"
os.environ["NEURON_RT_INSPECT_ENABLE"] = "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"] = "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"] = "./output_traced"
os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
sys.path.insert(0, "equiformer_v3/src")

import torch
import torch.nn as nn

"""
Mock torch_scatter and torch_cluster (same as test_1.py).
"""
torch_scatter = types.ModuleType("torch_scatter")


def scatter(src, index, dim=-1, out=None, dim_size=None, fill_value=0, reduce="sum"):
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
    size = list(src.size())
    size[dim] = dim_size
    if out is None:
        out = src.new_full(size, fill_value)
    if reduce in ("sum", "add"):
        return out.scatter_add_(dim, index.expand_as(src), src)
    if reduce == "mean":
        count = src.new_zeros(size)
        out.scatter_add_(dim, index.expand_as(src), src)
        count.scatter_add_(dim, index.expand_as(src), src.new_ones(src.shape))
        return out / count.clamp(min=1)
    if reduce == "max":
        return out.scatter_reduce_(dim, index.expand_as(src), src, reduce="amax")
    if reduce == "min":
        return out.scatter_reduce_(dim, index.expand_as(src), src, reduce="amin")
    raise ValueError(f"Unknown reduce: {reduce}")


torch_scatter.scatter = scatter
sys.modules["torch_scatter"] = torch_scatter

torch_scatter_utils = types.ModuleType("torch_scatter.utils")


def broadcast(src, other, dim):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    return src.expand(other.size())


torch_scatter_utils.broadcast = broadcast
sys.modules["torch_scatter.utils"] = torch_scatter_utils

torch_cluster = types.ModuleType("torch_cluster")
torch_cluster.radius_graph = lambda *a, **kw: None
sys.modules["torch_cluster"] = torch_cluster

import torch_neuronx
from ase.build import bulk
from fairchem.core.common.registry import registry
from fairchem.core.preprocessing import AtomsToGraphs
import fairchem.experimental.models.equiformer_v3.equiformer_v3  # noqa: F401


class TensorWrapper(nn.Module):
    """
    Wrap EquiformerV3 so its forward takes plain tensors (traceable),
    builds a SimpleNamespace `data` internally, and returns (energy, forces).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        atomic_numbers,
        pos,
        cell,
        natoms,
        edge_index,
        cell_offsets,
        neighbors,
        batch,
    ):
        data = types.SimpleNamespace(
            atomic_numbers=atomic_numbers,
            pos=pos,
            cell=cell,
            natoms=natoms,
            edge_index=edge_index,
            cell_offsets=cell_offsets,
            neighbors=neighbors,
            batch=batch,
        )
        out = self.model(data)
        return out["energy"], out["forces"]


def build_inputs():
    atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.64)
    a2g = AtomsToGraphs(
        max_neigh=20, radius=6.0, r_energy=False, r_forces=False, r_stress=False
    )
    data = a2g.convert_all([atoms])[0]
    num_atoms = len(atoms)
    return (
        data.atomic_numbers.long(),
        data.pos.float(),
        data.cell.float(),
        torch.tensor([num_atoms], dtype=torch.long),
        data.edge_index.long(),
        data.cell_offsets.float(),
        torch.tensor([data.edge_index.shape[1]], dtype=torch.long),
        torch.zeros(num_atoms, dtype=torch.long),
    )


def main():
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
    print(f"Model params: {model.num_params:,}")

    wrapped = TensorWrapper(model).eval()
    example = build_inputs()

    """
    Sanity-check on CPU.
    """
    with torch.no_grad():
        e_cpu, f_cpu = wrapped(*example)
    print(f"CPU energy: {e_cpu.item():.6f}, forces shape: {tuple(f_cpu.shape)}")

    """
    Compile via torch.compile with the Neuron Dynamo backend.
    Inputs are moved to a Neuron device.
    """
    device = torch.device("neuron:0")
    wrapped_dev = wrapped.to(device)
    example_dev = tuple(t.to(device) for t in example)

    from torch_neuronx.neuron_dynamo_backend import neuron_backend

    print("Input dtypes:")
    for n, t in zip(
        ["atomic_numbers", "pos", "cell", "natoms", "edge_index", "cell_offsets", "neighbors", "batch"],
        example_dev,
    ):
        print(f"  {n}: dtype={t.dtype}, shape={tuple(t.shape)}")

    print("torch.compile with neuron_dynamo_backend...")
    compiled = torch.compile(wrapped_dev, backend=neuron_backend)

    print("Warmup + capture runs...")
    with torch.no_grad():
        for _ in range(3):
            e, f = compiled(*example_dev)
    torch.neuron.synchronize()
    print(f"Neuron energy: {e.item():.6f}, forces shape: {tuple(f.shape)}")
    print("Done. NEFF written under ./output_traced/")


if __name__ == "__main__":
    main()
