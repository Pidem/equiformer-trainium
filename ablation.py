"""Cumulative ablation: time + numeric check after each optimization.

Each step applies an optimization in-place on the *same* model, so improvements
stack. After every step the forward output is compared against the baseline
to confirm we have not changed the math.
"""
import os
import statistics
import sys
import time
import types

os.environ["NEURON_RT_NUM_CORES"] = "1"
os.environ["NEURON_CC_FLAGS"] = "--target trn2 --lnc 1"
os.environ["NEURON_RT_VISIBLE_CORES"] = "0"
sys.path.insert(0, "equiformer_v3/src")

import torch

torch_scatter = types.ModuleType("torch_scatter")


def _scatter(src, index, dim=-1, out=None, dim_size=None, fill_value=0, reduce="sum"):
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


torch_scatter.scatter = _scatter
sys.modules["torch_scatter"] = torch_scatter

torch_scatter_utils = types.ModuleType("torch_scatter.utils")


def _broadcast(src, other, dim):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    return src.expand(other.size())


torch_scatter_utils.broadcast = _broadcast
sys.modules["torch_scatter.utils"] = torch_scatter_utils

torch_cluster = types.ModuleType("torch_cluster")
torch_cluster.radius_graph = lambda *a, **kw: None
sys.modules["torch_cluster"] = torch_cluster

import torch_neuronx
from ase.build import bulk
from fairchem.core.common.registry import registry
from fairchem.core.preprocessing import AtomsToGraphs
from fairchem.experimental.models.equiformer_v3 import wigner
from fairchem.experimental.models.equiformer_v3 import so3
from fairchem.experimental.models.equiformer_v3 import layer_norm as ln_mod
import fairchem.experimental.models.equiformer_v3.equiformer_v3  # noqa: F401


WARMUP = 3
N_TIMED = 15


def build_data():
    """
    NaCl 2x2x3 supercell -> 24 atoms, deterministic.
    """
    atoms = bulk("NaCl", crystalstructure="rocksalt", a=5.64).repeat((2, 2, 3))
    a2g = AtomsToGraphs(
        max_neigh=20, radius=6.0, r_energy=False, r_forces=False, r_stress=False
    )
    data = a2g.convert_all([atoms])[0]
    n = len(atoms)
    data.batch = torch.zeros(n, dtype=torch.long)
    data.natoms = torch.tensor([n])
    return data, n


def build_model(seed):
    torch.manual_seed(seed)
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
    """
    Scale all parameters down so the model produces O(1) forces.
    Random-init equiformer otherwise gives forces ~1e14 because of
    repeated linear layers with no normalization on the output side.
    Weight scaling does not change math identities between ablation runs;
    it only makes the relative-error check interpretable.
    """
    with torch.no_grad():
        for p in model.parameters():
            p.mul_(0.01)
    model.eval()
    return model


def time_forward(model, data):
    """
    Median wall-clock of N_TIMED forwards, after WARMUP warmups.
    Synchronization on each iteration so timing reflects device latency.
    """
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(data)
        torch.neuron.synchronize()

        times = []
        last_out = None
        for _ in range(N_TIMED):
            torch.neuron.synchronize()
            t0 = time.perf_counter()
            out = model(data)
            torch.neuron.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
            last_out = out

    energy = last_out["energy"].detach().cpu().clone()
    forces = last_out["forces"].detach().cpu().clone()
    return statistics.median(times), min(times), times, energy, forces


def diff_vs(reference, current):
    """
    Returns absolute and relative max error for both energy and forces.
    Relative is normalized by the reference's |.|_max (or 1.0 if smaller),
    so very small reference values don't blow up the relative number.
    """
    e_ref, f_ref = reference
    e_cur, f_cur = current
    e_abs = (e_ref - e_cur).abs().max().item()
    f_abs = (f_ref - f_cur).abs().max().item()
    e_scale = max(e_ref.abs().max().item(), 1.0)
    f_scale = max(f_ref.abs().max().item(), 1.0)
    return e_abs, f_abs, e_abs / e_scale, f_abs / f_scale


"""
=========================================================================
Optimization patches.
Each function mutates the imported source modules / model in-place.
Patches are cumulative — once applied, they stay applied for later runs.
=========================================================================
"""


def patch_wigner_cache():
    """
    Cache (inds, reversed_inds, frequencies) per (l, device, dtype) inside
    wigner._z_rot_mat. Mathematically identical: just hoists the torch.arange
    calls out of the per-call hot path.
    """
    cache: dict = {}

    def _z_rot_mat_cached(angle, l):
        shape, device, dtype = angle.shape, angle.device, angle.dtype
        M = angle.new_zeros((*shape, 2 * l + 1, 2 * l + 1))
        key = (l, str(device), dtype)
        cached = cache.get(key)
        if cached is None:
            inds = torch.arange(0, 2 * l + 1, 1, device=device, dtype=torch.int32)
            reversed_inds = torch.arange(2 * l, -1, -1, device=device, dtype=torch.int32)
            frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
            cache[key] = (inds, reversed_inds, frequencies)
        else:
            inds, reversed_inds, frequencies = cached
        M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
        M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
        return M

    wigner._z_rot_mat = _z_rot_mat_cached


def patch_expand_index_int32(model):
    """
    Convert all `expand_index` buffers (used by SO3Linear and the equivariant
    norm classes) from int64 -> int32 on the Neuron device. The buffer is only
    used as an index in torch.index_select, so int32 is bit-exact.
    """
    for m in model.modules():
        if hasattr(m, "expand_index"):
            buf = m.expand_index
            if buf.dtype == torch.int64:
                m.expand_index = buf.to(torch.int32)


def patch_wigner_nki_kernel():
    """
    Replace wigner._z_rot_mat with the NKI kernel implementation.
    The kernel produces M[b, i, j] for the cos-diagonal/sin-antidiagonal
    pattern using a single fused HBM-to-HBM call instead of the dispatcher
    chain (3x arange, 2x sin/cos, 2x advanced index_put, 1x mul).
    """
    from nki_z_rot_mat import z_rot_mat_nki

    def _z_rot_mat_with_kernel(angle, l):
        """
        Wrapper that calls the NKI kernel for any leading shape of `angle`.
        """
        return z_rot_mat_nki(angle, l)

    wigner._z_rot_mat = _z_rot_mat_with_kernel


def main():
    data_cpu, n_atoms = build_data()
    print(f"Compound: NaCl supercell, {n_atoms} atoms, "
          f"{data_cpu.edge_index.shape[1]} edges (max_neigh=20, r=6.0)")

    model = build_model(seed=0)
    device = torch.device("neuron:0")
    model = model.to(device)
    data = data_cpu.to(device)

    """
    Baseline.
    """
    print()
    hdr = f"{'step':<24s}  {'med_ms':>8s}  {'min_ms':>8s}  {'spdup':>6s}  {'|F|_max':>10s}  {'rel ΔE':>10s}  {'rel ΔF':>10s}"
    print(hdr)
    print("-" * len(hdr))

    median0, min0, _, e0, f0 = time_forward(model, data)
    base_e, base_f = e0, f0
    base_median = median0
    print(
        f"{'baseline':<24s}  {median0:>8.2f}  {min0:>8.2f}  {1.0:>5.2f}x  "
        f"{f0.abs().max().item():>10.3e}  {0.0:>10.2e}  {0.0:>10.2e}"
    )

    def step(label, model_, data_, base_e_, base_f_):
        med, mn, _, e, f = time_forward(model_, data_)
        _, _, rE, rF = diff_vs((base_e_, base_f_), (e, f))
        print(
            f"{label:<24s}  {med:>8.2f}  {mn:>8.2f}  {base_median/med:>5.2f}x  "
            f"{f.abs().max().item():>10.3e}  {rE:>10.2e}  {rF:>10.2e}"
        )

    """
    Step 1: wigner inds/freq cache.
    """
    patch_wigner_cache()
    step("+ wigner_cache", model, data, base_e, base_f)

    """
    Step 2: expand_index buffers as int32.
    """
    patch_expand_index_int32(model)
    step("+ expand_index_int32", model, data, base_e, base_f)

    """
    Step 3: replace wigner._z_rot_mat with the NKI kernel.
    """
    patch_wigner_nki_kernel()
    step("+ wigner_nki_kernel", model, data, base_e, base_f)

    print()
    print("rel ΔE / rel ΔF are |error| / max(|baseline|, 1.0).")
    print("On Neuron, fp32 reorderings typically show rel error ≤ 1e-4.")
    print("rel error > 1e-2 means the math actually changed.")


if __name__ == "__main__":
    main()
