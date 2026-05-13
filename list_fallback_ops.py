"""Run EquiformerV3 once on Neuron eager and report which ops fell back to CPU."""
import os
import sys
import types
from collections import Counter

os.environ["NEURON_RT_NUM_CORES"] = "1"
os.environ["NEURON_CC_FLAGS"] = "--target trn2 --lnc 1"
os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
sys.path.insert(0, "equiformer_v3/src")

import torch

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


def main():
    atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.64)
    a2g = AtomsToGraphs(
        max_neigh=20, radius=6.0, r_energy=False, r_forces=False, r_stress=False
    )
    data = a2g.convert_all([atoms])[0]
    num_atoms = len(atoms)
    data.batch = torch.zeros(num_atoms, dtype=torch.long)
    data.natoms = torch.tensor([num_atoms])

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

    device = torch.device("neuron:0")
    model = model.to(device)
    data = data.to(device)

    torch_neuronx.clear_op_tracking()

    with torch.no_grad():
        outputs = model(data)
    torch.neuron.synchronize()

    executed = torch_neuronx.get_executed_ops()
    fallback = torch_neuronx.get_fallback_ops()

    print(f"Energy: {outputs['energy'].item():.6f}")
    print(f"Forces shape: {tuple(outputs['forces'].shape)}")
    print()
    print(f"Executed on Neuron: {len(executed)} op invocations, {len(set(executed))} unique")
    print(f"Fell back to CPU:   {len(fallback)} op invocations, {len(set(fallback))} unique")
    print()

    print("=== Fallback ops (CPU) — counts ===")
    fb_counts = Counter(fallback)
    for op, n in fb_counts.most_common():
        print(f"  {n:>6d}  {op}")

    print()
    print("=== Executed ops (Neuron) — counts ===")
    ex_counts = Counter(executed)
    for op, n in ex_counts.most_common():
        print(f"  {n:>6d}  {op}")


if __name__ == "__main__":
    main()
