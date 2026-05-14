"""NKI kernel for wigner._z_rot_mat.

Output: M[b, i, j] of shape [B, F, F] where F = 2l+1, with
    M[b, i, i]      = cos(freq[i] * angle[b])  (diagonal)
    M[b, i, F-1-i]  = sin(freq[i] * angle[b])  (antidiagonal)
    M[b, i, j]      = 0 elsewhere

Implementation:
    Host pre-builds:
        freq_kk[k]: freq[i] when k = i*F + j, for any j
        diag_kk[k]: 1 when k = i*F + i, else 0
        anti_kk[k]: 1 when k = i*F + (F-1-i), else 0

    Kernel:
        prod[b, k] = angle[b, 0] * freq_kk[k]
        s = sin(prod);  c = cos(prod)
        M_flat[b, k] = c[b, k] * diag_kk[k] + s[b, k] * anti_kk[k]

All elementwise, all uniform [B, K] shapes, no per-row F-broadcast.
"""
from __future__ import annotations

import torch
import nki
import nki.isa as nisa
import nki.language as nl


P_MAX = 128


def _div_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


@nki.jit
def _z_rot_mat_kernel(angle_kk, freq_kk_bcast, diag_kk_bcast, anti_kk_bcast):
    """
    Inputs (HBM, all with partition=P_MAX, free=K):
        angle_kk:        [P_MAX, K] f32 -- angle[b] replicated across K columns
        freq_kk_bcast:   [P_MAX, K] f32
        diag_kk_bcast:   [P_MAX, K] f32
        anti_kk_bcast:   [P_MAX, K] f32

    Output:
        M_flat: [P_MAX, K] f32
    """
    B = angle_kk.shape[0]
    K = angle_kk.shape[1]

    M_flat = nl.ndarray((B, K), dtype=angle_kk.dtype, buffer=nl.shared_hbm)

    angle_sb = nl.ndarray((P_MAX, K), dtype=angle_kk.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=angle_sb[0:B, 0:K], src=angle_kk[0:B, 0:K])

    freq_sb = nl.ndarray((P_MAX, K), dtype=angle_kk.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=freq_sb[0:B, 0:K], src=freq_kk_bcast[0:B, 0:K])

    diag_sb = nl.ndarray((P_MAX, K), dtype=angle_kk.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=diag_sb[0:B, 0:K], src=diag_kk_bcast[0:B, 0:K])

    anti_sb = nl.ndarray((P_MAX, K), dtype=angle_kk.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=anti_sb[0:B, 0:K], src=anti_kk_bcast[0:B, 0:K])

    """
    prod[b, k] = angle_sb[b, k] * freq_sb[b, k] -- both [B, K], no broadcast.
    """
    prod_sb = nl.ndarray((P_MAX, K), dtype=angle_kk.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=prod_sb[0:B, 0:K],
        data1=angle_sb[0:B, 0:K],
        data2=freq_sb[0:B, 0:K],
        op=nl.multiply,
    )

    s_sb = nl.ndarray((P_MAX, K), dtype=angle_kk.dtype, buffer=nl.sbuf)
    c_sb = nl.ndarray((P_MAX, K), dtype=angle_kk.dtype, buffer=nl.sbuf)
    s_sb[0:B, 0:K] = nl.sin(prod_sb[0:B, 0:K])
    c_sb[0:B, 0:K] = nl.cos(prod_sb[0:B, 0:K])

    """
    out = c_sb * diag_sb + s_sb * anti_sb (all [B, K] x [B, K]).
    """
    out_sb = nl.ndarray((P_MAX, K), dtype=angle_kk.dtype, buffer=nl.sbuf)
    tmp_sb = nl.ndarray((P_MAX, K), dtype=angle_kk.dtype, buffer=nl.sbuf)

    nisa.tensor_tensor(
        dst=out_sb[0:B, 0:K],
        data1=c_sb[0:B, 0:K],
        data2=diag_sb[0:B, 0:K],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=tmp_sb[0:B, 0:K],
        data1=s_sb[0:B, 0:K],
        data2=anti_sb[0:B, 0:K],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=out_sb[0:B, 0:K],
        data1=out_sb[0:B, 0:K],
        data2=tmp_sb[0:B, 0:K],
        op=nl.add,
    )

    nisa.dma_copy(dst=M_flat[0:B, 0:K], src=out_sb[0:B, 0:K])

    return M_flat


_constants_cache: dict = {}


def _ensure_constants(l, dtype, device):
    """
    Build [P_MAX, K] broadcast tensors of:
        freq_kk[k]: freq[i] where k = i*F + j (any j)
        diag_kk[k]: 1 where k = i*F + i
        anti_kk[k]: 1 where k = i*F + (F-1-i)
    """
    F = 2 * l + 1
    K = F * F
    key = (l, str(device), dtype)
    cached = _constants_cache.get(key)
    if cached is not None:
        return cached

    freq = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
    freq_kk = freq.unsqueeze(1).expand(F, F).reshape(K).contiguous()

    diag = torch.zeros((F, F), dtype=dtype, device=device)
    anti = torch.zeros((F, F), dtype=dtype, device=device)
    for i in range(F):
        diag[i, i] = 1.0
        anti[i, F - 1 - i] = 1.0
    diag_kk = diag.reshape(K).contiguous()
    anti_kk = anti.reshape(K).contiguous()

    out = (
        freq_kk.unsqueeze(0).expand(P_MAX, K).contiguous(),
        diag_kk.unsqueeze(0).expand(P_MAX, K).contiguous(),
        anti_kk.unsqueeze(0).expand(P_MAX, K).contiguous(),
    )
    _constants_cache[key] = out
    return out


def z_rot_mat_nki(angle: torch.Tensor, l: int) -> torch.Tensor:
    F = 2 * l + 1
    leading = angle.shape
    flat = angle.reshape(-1).contiguous()
    B_real = flat.shape[0]
    dtype = flat.dtype
    device = flat.device

    n_tiles = _div_ceil(B_real, P_MAX)
    B_padded = n_tiles * P_MAX
    if B_padded != B_real:
        pad = torch.zeros(B_padded - B_real, dtype=dtype, device=device)
        flat = torch.cat([flat, pad], dim=0)

    freq_kk_b, diag_kk_b, anti_kk_b = _ensure_constants(l, dtype, device)

    K = F * F
    out_tiles = []
    for t in range(n_tiles):
        chunk = flat[t * P_MAX:(t + 1) * P_MAX].contiguous()
        """
        Pre-broadcast chunk [P_MAX] -> [P_MAX, K] so the kernel only does
        same-shape elementwise ops.
        """
        chunk_kk = chunk.unsqueeze(1).expand(P_MAX, K).contiguous()
        M_chunk = _z_rot_mat_kernel(chunk_kk, freq_kk_b, diag_kk_b, anti_kk_b)
        out_tiles.append(M_chunk)

    M_padded = torch.cat(out_tiles, dim=0) if len(out_tiles) > 1 else out_tiles[0]
    M = M_padded[:B_real]
    return M.reshape(*leading, F, F)


if __name__ == "__main__":
    import os
    import sys

    os.environ.setdefault("NEURON_RT_NUM_CORES", "1")
    os.environ.setdefault("NEURON_CC_FLAGS", "--target trn2 --lnc 1")
    os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")

    sys.path.insert(0, "equiformer_v3/src")
    from fairchem.experimental.models.equiformer_v3 import wigner as _wigner

    torch.manual_seed(0)
    device = torch.device("neuron:0")

    print(f"{'l':>3s}  {'B':>5s}  {'max abs err':>12s}  {'max rel err':>12s}  {'allclose':>8s}")
    for l in [0, 1, 2]:
        for B in [1, 5, 200]:
            angle_cpu = torch.randn(B, dtype=torch.float32)
            ref = _wigner._z_rot_mat(angle_cpu, l)

            angle_nx = angle_cpu.to(device)
            out = z_rot_mat_nki(angle_nx, l)
            torch.neuron.synchronize()
            out_cpu = out.cpu()

            abs_err = (out_cpu - ref).abs().max().item()
            ref_max = max(ref.abs().max().item(), 1.0)
            rel_err = abs_err / ref_max
            ok = torch.allclose(out_cpu, ref, atol=1e-5, rtol=1e-5)
            print(f"{l:>3d}  {B:>5d}  {abs_err:>12.3e}  {rel_err:>12.3e}  {str(ok):>8s}")
