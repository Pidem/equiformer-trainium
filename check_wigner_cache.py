"""Verify the wigner cache patch is mathematically identical on CPU.

This isolates whether the math change comes from the patch itself or from
something the Neuron backend does with the patched code.
"""
import sys
import torch

sys.path.insert(0, "equiformer_v3/src")
from fairchem.experimental.models.equiformer_v3 import wigner

torch.manual_seed(0)

orig_z_rot_mat = wigner._z_rot_mat


def patched(angle, l):
    cache = patched._cache
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


patched._cache = {}

for l in [0, 1, 2]:
    angle = torch.randn(5, 7, dtype=torch.float32)
    a = orig_z_rot_mat(angle, l)
    b = patched(angle, l)
    diff = (a - b).abs().max().item()
    print(f"l={l}, shape={tuple(a.shape)}, max abs diff (CPU vs CPU) = {diff:.3e}, bit-exact={torch.equal(a, b)}")
