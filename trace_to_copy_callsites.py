"""Capture Python frames for every _to_copy / .item() call inside equiformer."""
import os
import sys
import traceback
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


PROJECT_ROOT = "/home/ubuntu/equiformer-trainium/"


def first_user_frame():
    """
    Walk the Python stack outward and return the first frame inside the project,
    skipping our own tracer wrappers.
    """
    stack = traceback.extract_stack()
    for frame in reversed(stack):
        fname = frame.filename
        if fname.endswith("trace_to_copy_callsites.py"):
            continue
        if PROJECT_ROOT in fname and "site-packages" not in fname:
            return f"{fname.replace(PROJECT_ROOT, '')}:{frame.lineno} in {frame.name}"
    return None


def install_tracers():
    """
    Patch a handful of methods that force Neuron->host transfers and record where they're called.
    """
    sites_to_dev = Counter()
    sites_item = Counter()
    sites_int = Counter()
    sites_float = Counter()
    sites_to_copy = Counter()

    orig_to = torch.Tensor.to
    orig_item = torch.Tensor.item
    orig_int = torch.Tensor.__int__
    orig_float = torch.Tensor.__float__
    orig_cpu = torch.Tensor.cpu

    def patched_to(self, *args, **kw):
        site = first_user_frame()
        if site is not None:
            sites_to_dev[site] += 1
        return orig_to(self, *args, **kw)

    def patched_cpu(self, *args, **kw):
        site = first_user_frame()
        if site is not None:
            sites_to_copy[site] += 1
        return orig_cpu(self, *args, **kw)

    def patched_item(self, *a, **k):
        site = first_user_frame()
        if site is not None:
            sites_item[site] += 1
        return orig_item(self, *a, **k)

    def patched_int(self):
        site = first_user_frame()
        if site is not None:
            sites_int[site] += 1
        return orig_int(self)

    def patched_float(self):
        site = first_user_frame()
        if site is not None:
            sites_float[site] += 1
        return orig_float(self)

    torch.Tensor.to = patched_to
    torch.Tensor.cpu = patched_cpu
    torch.Tensor.item = patched_item
    torch.Tensor.__int__ = patched_int
    torch.Tensor.__float__ = patched_float

    return {
        ".to(...)": sites_to_dev,
        ".cpu()": sites_to_copy,
        ".item()": sites_item,
        "int(tensor)": sites_int,
        "float(tensor)": sites_float,
    }


def build():
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
    return model.to(device), data.to(device)


def main():
    model, data = build()

    """
    Warm up before installing the tracers so init/compile doesn't pollute counts.
    """
    with torch.no_grad():
        _ = model(data)
    torch.neuron.synchronize()

    sites = install_tracers()

    with torch.no_grad():
        out = model(data)
    torch.neuron.synchronize()
    print(f"Energy: {out['energy'].item():.6f} (after second forward)")

    for label, ctr in sites.items():
        if not ctr:
            continue
        total = sum(ctr.values())
        print()
        print(f"=== {label} — {total} calls across {len(ctr)} sites ===")
        print(f"{'count':>6s}  site")
        for site, n in ctr.most_common(20):
            print(f"{n:>6d}  {site}")


if __name__ == "__main__":
    main()
