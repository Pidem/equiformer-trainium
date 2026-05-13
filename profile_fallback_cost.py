"""Rank EquiformerV3 op cost: CPU fallback time and Neuron<->CPU transfer overhead."""
import os
import sys
import types

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


def main():
    model, data = build()
    fallback_set = set()
    for n_warmup in range(2):
        torch_neuronx.clear_op_tracking()
        with torch.no_grad():
            _ = model(data)
        torch.neuron.synchronize()
        if n_warmup == 1:
            fallback_set = set(torch_neuronx.get_fallback_ops())

    print(f"Fallback ops detected: {len(fallback_set)}")
    print()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        record_shapes=False,
    ) as prof:
        with torch.no_grad():
            _ = model(data)
        torch.neuron.synchronize()

    events = prof.key_averages()

    rows = []
    for ev in events:
        rows.append(
            {
                "key": ev.key,
                "cpu_time_us": ev.self_cpu_time_total,
                "cpu_time_total_us": ev.cpu_time_total,
                "device_time_us": ev.self_device_time_total,
                "count": ev.count,
            }
        )

    print("=== Top 25 ops by self_cpu_time_total ===")
    print(f"{'op':<55s}  {'count':>6s}  {'self_cpu(us)':>14s}  {'tot_cpu(us)':>13s}  {'dev(us)':>10s}  {'fallback?':>10s}")
    print("-" * 125)
    for r in sorted(rows, key=lambda x: x["cpu_time_us"], reverse=True)[:25]:
        is_fb = "YES" if r["key"] in fallback_set else ""
        print(
            f"{r['key'][:55]:<55s}  {r['count']:>6d}  {r['cpu_time_us']:>14.1f}  "
            f"{r['cpu_time_total_us']:>13.1f}  {r['device_time_us']:>10.1f}  {is_fb:>10s}"
        )

    print()
    print("=== Fallback ops only, ranked by self_cpu_time_total ===")
    print(f"{'op':<55s}  {'count':>6s}  {'self_cpu(us)':>14s}  {'tot_cpu(us)':>13s}")
    print("-" * 100)
    fb_rows = [r for r in rows if r["key"] in fallback_set]
    for r in sorted(fb_rows, key=lambda x: x["cpu_time_us"], reverse=True):
        print(
            f"{r['key'][:55]:<55s}  {r['count']:>6d}  {r['cpu_time_us']:>14.1f}  "
            f"{r['cpu_time_total_us']:>13.1f}"
        )

    print()
    print("=== Neuron<->CPU transfer rows ===")
    print(f"{'op':<55s}  {'count':>6s}  {'self_cpu(us)':>14s}  {'dev(us)':>10s}")
    print("-" * 100)
    for r in rows:
        if any(t in r["key"] for t in ("cpu_to_neuron", "neuron_to_cpu", "_to_copy", "copy_")):
            print(
                f"{r['key'][:55]:<55s}  {r['count']:>6d}  {r['cpu_time_us']:>14.1f}  "
                f"{r['device_time_us']:>10.1f}"
            )


if __name__ == "__main__":
    main()
