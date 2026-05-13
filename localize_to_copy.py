"""Localize aten::_to_copy call sites in equiformer to specific source lines."""
import os
import re
import sys
import types
from collections import Counter, defaultdict

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
from torch.profiler import profile, ProfilerActivity
from ase.build import bulk
from fairchem.core.common.registry import registry
from fairchem.core.preprocessing import AtomsToGraphs
import fairchem.experimental.models.equiformer_v3.equiformer_v3  # noqa: F401


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


def is_user_frame(frame_str):
    """
    Filter for source lines inside the equiformer model or our own scripts.
    """
    if "equiformer_v3" in frame_str or "fairchem" in frame_str:
        return True
    if frame_str.startswith("/home/ubuntu/equiformer-trainium/") and "site-packages" not in frame_str:
        return True
    return False


def first_user_frame(stack):
    """
    `stack` is a list of frame strings, leaf-first. Return the first user frame.
    """
    for fr in stack:
        if is_user_frame(fr):
            return fr
    return stack[-1] if stack else "<no stack>"


def main():
    model, data = build()

    """
    Warmup so compilation/lowering doesn't pollute the profile.
    """
    for _ in range(2):
        with torch.no_grad():
            _ = model(data)
        torch.neuron.synchronize()

    print("Profiling with stack traces...")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        with_stack=True,
        record_shapes=True,
    ) as prof:
        with torch.no_grad():
            _ = model(data)
        torch.neuron.synchronize()

    """
    Walk every event and aggregate _to_copy and copy_ events by call site.
    """
    by_site_to_copy = defaultdict(lambda: {"count": 0, "us": 0.0, "shapes": Counter()})
    by_site_copy_ = defaultdict(lambda: {"count": 0, "us": 0.0, "shapes": Counter()})

    events = prof.events()
    for ev in events:
        if ev.name not in ("aten::_to_copy", "aten::copy_"):
            continue
        stack = list(ev.stack) if ev.stack else []
        site = first_user_frame(stack)
        bucket = by_site_to_copy if ev.name == "aten::_to_copy" else by_site_copy_
        rec = bucket[site]
        rec["count"] += 1
        rec["us"] += ev.cpu_time
        if ev.input_shapes:
            rec["shapes"][str(ev.input_shapes)] += 1

    def report(label, table):
        rows = sorted(table.items(), key=lambda kv: kv[1]["us"], reverse=True)
        total_us = sum(r["us"] for _, r in rows)
        print()
        print(f"=== {label} — top 15 by total CPU time (total: {total_us/1000:.1f} ms) ===")
        print(f"{'count':>6s}  {'cpu_us':>12s}  site")
        for site, rec in rows[:15]:
            short = re.sub(r".*/(equiformer-trainium/.*)", r"\1", site)
            print(f"{rec['count']:>6d}  {rec['us']:>12.1f}  {short}")
            top_shape = rec["shapes"].most_common(1)
            if top_shape:
                shape_str, n = top_shape[0]
                shape_str = shape_str[:100]
                print(f"        most-common shape ({n}x): {shape_str}")

    report("aten::_to_copy", by_site_to_copy)
    report("aten::copy_", by_site_copy_)

    """
    Dtype check on the input — confirms whether int64 inputs are forcing casts.
    """
    print()
    print("=== input dtypes (post .to(device)) ===")
    for name in [
        "atomic_numbers",
        "pos",
        "cell",
        "natoms",
        "edge_index",
        "cell_offsets",
        "neighbors",
        "batch",
    ]:
        if hasattr(data, name):
            t = getattr(data, name)
            print(f"  {name}: dtype={t.dtype}, shape={tuple(t.shape)}")


if __name__ == "__main__":
    main()
