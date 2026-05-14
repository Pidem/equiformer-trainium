"""NKI fused kernel for the full wigner_D chain: Xa @ J @ Xb @ J @ Xc.

The X matrices are structured (cos on diagonal, sin on antidiagonal). J is a
constant L²×L² matrix per l. Decompose the chain into:

    M0 = Xa @ J                          (left-mul by structured -> row-scale)
    M1 = M0 @ Xb                          (right-mul by structured -> elementwise + j-reverse)
    M2 = M1 @ J                           (right-mul by dense -> nc_matmul)
    out = M2 @ Xc                         (right-mul by structured -> elementwise + j-reverse)

For the right-multiplications by the structured X matrices, we avoid an
explicit j-reversal by computing TWO parallel intermediates:
    M  [b, i, j]                         (normal layout)
    M_jrev [b, i, j] = M[b, i, F-1-j]    (j reversed)
using different precomputed constants (J_diag_perm, J_anti_perm, IJperm_T_const).

This means the kernel does exactly two nc_matmul calls (M1 -> M2 and M1 -> M2_jrev)
and a small number of elementwise multiplies and adds.

Inputs are flattened to [B, K] where K = F*F = (2l+1)² and B is padded to P_MAX.
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
def _wigner_D_kernel(
    alpha_kk,
    beta_kk,
    gamma_kk,
    freq_i_kk,
    freq_j_kk,
    J_diag_kk,
    J_anti_kk,
    J_diag_perm_kk,
    J_anti_perm_kk,
    IJ_T_const,
    IJperm_T_const,
):
    """
    Inputs (HBM, all f32):
        alpha_kk, beta_kk, gamma_kk: [P_MAX, K]
            alpha[b], beta[b], gamma[b] each replicated across K.
        freq_i_kk: [P_MAX, K]
            freq[i] indexed by k=i*F+j (each row identical).
        freq_j_kk: [P_MAX, K]
            freq[j] indexed by k=i*F+j (each row identical).
        J_diag_kk:    [P_MAX, K]    J[i, j]            indexed by k=i*F+j
        J_anti_kk:    [P_MAX, K]    J[F-1-i, j]
        J_diag_perm_kk: [P_MAX, K]  J[i, F-1-j]
        J_anti_perm_kk: [P_MAX, K]  J[F-1-i, F-1-j]
        IJ_T_const:    [K, K]   = (I_F ⊗ J).T
        IJperm_T_const: [K, K]  = (I_F ⊗ J_perm).T   where J_perm[k, j] = J[k, F-1-j]

    Output:
        M_flat: [P_MAX, K] f32 -- result of Xa @ J @ Xb @ J @ Xc.
    """
    B = alpha_kk.shape[0]
    K = alpha_kk.shape[1]

    M_flat = nl.ndarray((B, K), dtype=alpha_kk.dtype, buffer=nl.shared_hbm)

    """
    Load all [B, K] inputs into SBUF.
    """
    alpha_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)
    beta_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)
    gamma_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)
    freq_i_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)
    freq_j_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)
    J_diag_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)
    J_anti_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)
    J_diag_perm_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)
    J_anti_perm_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)

    nisa.dma_copy(dst=alpha_sb[0:B, 0:K], src=alpha_kk[0:B, 0:K])
    nisa.dma_copy(dst=beta_sb[0:B, 0:K], src=beta_kk[0:B, 0:K])
    nisa.dma_copy(dst=gamma_sb[0:B, 0:K], src=gamma_kk[0:B, 0:K])
    nisa.dma_copy(dst=freq_i_sb[0:B, 0:K], src=freq_i_kk[0:B, 0:K])
    nisa.dma_copy(dst=freq_j_sb[0:B, 0:K], src=freq_j_kk[0:B, 0:K])
    nisa.dma_copy(dst=J_diag_sb[0:B, 0:K], src=J_diag_kk[0:B, 0:K])
    nisa.dma_copy(dst=J_anti_sb[0:B, 0:K], src=J_anti_kk[0:B, 0:K])
    nisa.dma_copy(dst=J_diag_perm_sb[0:B, 0:K], src=J_diag_perm_kk[0:B, 0:K])
    nisa.dma_copy(dst=J_anti_perm_sb[0:B, 0:K], src=J_anti_perm_kk[0:B, 0:K])

    IJ_T_sb = nl.ndarray((K, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)
    IJperm_T_sb = nl.ndarray((K, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=IJ_T_sb[0:K, 0:K], src=IJ_T_const[0:K, 0:K])
    nisa.dma_copy(dst=IJperm_T_sb[0:K, 0:K], src=IJperm_T_const[0:K, 0:K])

    """
    Trig: c_a/s_a use freq_i (Xa is structured by i index).
    c_b/s_b use freq_j (right-mul by Xb means c_b is indexed by j in the formula).
    Same for c_g/s_g.
    """
    prod_a_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)
    c_a_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)
    s_a_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=prod_a_sb[0:B, 0:K],
        data1=alpha_sb[0:B, 0:K],
        data2=freq_i_sb[0:B, 0:K],
        op=nl.multiply,
    )
    c_a_sb[0:B, 0:K] = nl.cos(prod_a_sb[0:B, 0:K])
    s_a_sb[0:B, 0:K] = nl.sin(prod_a_sb[0:B, 0:K])

    prod_b_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)
    c_b_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)
    s_b_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=prod_b_sb[0:B, 0:K],
        data1=beta_sb[0:B, 0:K],
        data2=freq_j_sb[0:B, 0:K],
        op=nl.multiply,
    )
    c_b_sb[0:B, 0:K] = nl.cos(prod_b_sb[0:B, 0:K])
    s_b_sb[0:B, 0:K] = nl.sin(prod_b_sb[0:B, 0:K])

    prod_g_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)
    c_g_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)
    s_g_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=prod_g_sb[0:B, 0:K],
        data1=gamma_sb[0:B, 0:K],
        data2=freq_j_sb[0:B, 0:K],
        op=nl.multiply,
    )
    c_g_sb[0:B, 0:K] = nl.cos(prod_g_sb[0:B, 0:K])
    s_g_sb[0:B, 0:K] = nl.sin(prod_g_sb[0:B, 0:K])

    """
    Step 1: M0 = Xa @ J. Elementwise:
        M0[b, i, j] = c_a[b, i] * J[i, j] + s_a[b, i] * J[F-1-i, j]
                    = c_a * J_diag + s_a * J_anti                (in flat form)
    Also compute M0_jrev = Xa @ J with j reversed:
        M0_jrev[b, i, j] = M0[b, i, F-1-j]
                         = c_a * J_diag_perm + s_a * J_anti_perm
    """
    M0_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)
    M0_jrev_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)
    tmp_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)

    nisa.tensor_tensor(
        dst=M0_sb[0:B, 0:K],
        data1=c_a_sb[0:B, 0:K],
        data2=J_diag_sb[0:B, 0:K],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=tmp_sb[0:B, 0:K],
        data1=s_a_sb[0:B, 0:K],
        data2=J_anti_sb[0:B, 0:K],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=M0_sb[0:B, 0:K],
        data1=M0_sb[0:B, 0:K],
        data2=tmp_sb[0:B, 0:K],
        op=nl.add,
    )

    nisa.tensor_tensor(
        dst=M0_jrev_sb[0:B, 0:K],
        data1=c_a_sb[0:B, 0:K],
        data2=J_diag_perm_sb[0:B, 0:K],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=tmp_sb[0:B, 0:K],
        data1=s_a_sb[0:B, 0:K],
        data2=J_anti_perm_sb[0:B, 0:K],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=M0_jrev_sb[0:B, 0:K],
        data1=M0_jrev_sb[0:B, 0:K],
        data2=tmp_sb[0:B, 0:K],
        op=nl.add,
    )

    """
    Step 2: M1 = M0 @ Xb (right-mul by structured matrix). Elementwise:
        M1[b, i, j] = M0[b, i, j] * c_b[b, j] + M0[b, i, F-1-j] * s_b[b, j]
                    = M0 * c_b + M0_jrev * s_b                   (in flat form)
    """
    M1_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=M1_sb[0:B, 0:K],
        data1=M0_sb[0:B, 0:K],
        data2=c_b_sb[0:B, 0:K],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=tmp_sb[0:B, 0:K],
        data1=M0_jrev_sb[0:B, 0:K],
        data2=s_b_sb[0:B, 0:K],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=M1_sb[0:B, 0:K],
        data1=M1_sb[0:B, 0:K],
        data2=tmp_sb[0:B, 0:K],
        op=nl.add,
    )

    """
    Step 3: M2 = M1 @ A where A = I_F ⊗ J. Done as an unrolled loop over k:
        M2[b, j_out] = Σ_k M1[b, k] * A[k, j_out]
    For each k we do an elementwise multiply of M1[:, k:k+1] (partition broadcast)
    by A[k, :] (a constant row), accumulating into M2.

    But we can't do per-row elementwise broadcast from a [B, 1] to [B, K] easily
    due to free-dim broadcast restrictions. Workaround: make M1_kk_per_k where
    each row k is M1[b, k] replicated across K columns. That's K such tiles.

    Simpler approach: use IJ_T_sb as a constant [K, K] table, and for each k
    multiply M1[:, k:k+1] (singleton free, no broadcast available) by IJ_T_sb
    row k... still hits the singleton-free issue.

    We circumvent by precomputing M1 in a [B, K, K] layout where M1_bcast[b, k, j_out]
    = M1[b, k]. Then M2[b, j_out] = Σ_k M1_bcast[b, k, j_out] * IJ_T_const[k, j_out].

    Since K is tiny (e.g. 25), we accumulate across k in a Python-unrolled affine_range.
    Each iteration does:
        contrib[b, :] = M1[b, k] * IJ_T_row_k[:]   (need to broadcast M1[b, k] across K)
    We sidestep broadcast by storing each "M1[:, k] replicated K times across free"
    from a precomputed [B, K*K] buffer M1_bcast_sb where columns [k*K:(k+1)*K]
    contain M1[:, k] replicated K times. We build that by F²=25 elementwise copies
    from M1_sb: per-k copy writes M1_bcast_sb[0:B, k*K:(k+1)*K] = broadcast of
    M1_sb[0:B, k:k+1].

    To do that broadcast within elementwise rules, we can use tensor_tensor with a
    [B, K] ones constant (which does same-shape multiply) only if the lhs and rhs
    are the same shape -- so first we need a fan-out.

    The simplest trick: use IJ_T_sb as a moving operand of nc_matmul -- but we hit
    the partition-transpose problem.

    PRACTICAL FIX: do K small per-k broadcasts via affine_range, using
    nl.broadcast_to which is documented to handle this case.
    """
    M2_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)
    M2_jrev_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)
    contrib_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)

    """
    Build a broadcast [B, K] of M1[:, 0] using nl.broadcast_to.
    """
    M1_col_bcast = nl.broadcast_to(M1_sb[0:B, 0:1], shape=(B, K))
    """
    IJ_T_row_k is the k-th row of IJ_T -- it has partition dim 1 (only one row).
    We need it broadcast across partition. Using nl.broadcast_to handles that.
    """
    IJ_row_bcast = nl.broadcast_to(IJ_T_sb[0:1, 0:K], shape=(B, K))
    nisa.tensor_tensor(
        dst=M2_sb[0:B, 0:K],
        data1=M1_col_bcast,
        data2=IJ_row_bcast,
        op=nl.multiply,
    )
    IJperm_row_bcast = nl.broadcast_to(IJperm_T_sb[0:1, 0:K], shape=(B, K))
    nisa.tensor_tensor(
        dst=M2_jrev_sb[0:B, 0:K],
        data1=M1_col_bcast,
        data2=IJperm_row_bcast,
        op=nl.multiply,
    )

    """
    Accumulate k = 1..K-1.
    """
    for k in nl.affine_range(K - 1):
        kk = k + 1
        M1_col_bcast = nl.broadcast_to(M1_sb[0:B, kk:kk + 1], shape=(B, K))
        IJ_row_bcast = nl.broadcast_to(IJ_T_sb[kk:kk + 1, 0:K], shape=(B, K))
        nisa.tensor_tensor(
            dst=contrib_sb[0:B, 0:K],
            data1=M1_col_bcast,
            data2=IJ_row_bcast,
            op=nl.multiply,
        )
        nisa.tensor_tensor(
            dst=M2_sb[0:B, 0:K],
            data1=M2_sb[0:B, 0:K],
            data2=contrib_sb[0:B, 0:K],
            op=nl.add,
        )

        IJperm_row_bcast = nl.broadcast_to(IJperm_T_sb[kk:kk + 1, 0:K], shape=(B, K))
        nisa.tensor_tensor(
            dst=contrib_sb[0:B, 0:K],
            data1=M1_col_bcast,
            data2=IJperm_row_bcast,
            op=nl.multiply,
        )
        nisa.tensor_tensor(
            dst=M2_jrev_sb[0:B, 0:K],
            data1=M2_jrev_sb[0:B, 0:K],
            data2=contrib_sb[0:B, 0:K],
            op=nl.add,
        )

    """
    Step 4: out = M2 @ Xc (right-mul by structured matrix). Elementwise:
        out[b, i, j] = M2[b, i, j] * c_g[b, j] + M2[b, i, F-1-j] * s_g[b, j]
                    = M2 * c_g + M2_jrev * s_g                   (in flat form)
    """
    out_sb = nl.ndarray((P_MAX, K), dtype=alpha_kk.dtype, buffer=nl.sbuf)
    nisa.tensor_tensor(
        dst=out_sb[0:B, 0:K],
        data1=M2_sb[0:B, 0:K],
        data2=c_g_sb[0:B, 0:K],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=tmp_sb[0:B, 0:K],
        data1=M2_jrev_sb[0:B, 0:K],
        data2=s_g_sb[0:B, 0:K],
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


def _ensure_constants(l, J: torch.Tensor, dtype, device):
    """
    Build all the constants the kernel needs, broadcasting [F,F]/[K] data
    across the partition dim where applicable.
    Cache by (l, J memory address, dtype, device) — J is fixed per l in
    equiformer (loaded from Jd.pt) so this caches once per l.
    """
    F = 2 * l + 1
    K = F * F
    key = (l, J.data_ptr(), str(device), dtype)
    cached = _constants_cache.get(key)
    if cached is not None:
        return cached

    """
    freq broadcasts.
    """
    freq = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
    freq_i = freq.unsqueeze(1).expand(F, F).reshape(K)
    freq_j = freq.unsqueeze(0).expand(F, F).reshape(K)
    freq_i_kk = freq_i.unsqueeze(0).expand(P_MAX, K).contiguous()
    freq_j_kk = freq_j.unsqueeze(0).expand(P_MAX, K).contiguous()

    """
    J broadcasts. J has shape [F, F].
    J_diag[i, j]      = J[i, j]
    J_anti[i, j]      = J[F-1-i, j]
    J_diag_perm[i, j] = J[i, F-1-j]
    J_anti_perm[i, j] = J[F-1-i, F-1-j]
    """
    J_d = J.to(dtype=dtype, device=device)
    flip_i = torch.arange(F - 1, -1, -1, dtype=torch.long, device=device)
    flip_j = flip_i
    J_diag = J_d.reshape(K)
    J_anti = J_d[flip_i, :].reshape(K)
    J_diag_perm = J_d[:, flip_j].reshape(K)
    J_anti_perm = J_d[flip_i, :][:, flip_j].reshape(K)

    J_diag_kk = J_diag.unsqueeze(0).expand(P_MAX, K).contiguous()
    J_anti_kk = J_anti.unsqueeze(0).expand(P_MAX, K).contiguous()
    J_diag_perm_kk = J_diag_perm.unsqueeze(0).expand(P_MAX, K).contiguous()
    J_anti_perm_kk = J_anti_perm.unsqueeze(0).expand(P_MAX, K).contiguous()

    """
    For the dense matmul step: A = I_F ⊗ J  shape [F², F²].
    A[i*F+p, k*F+j] = (i==k) * J[p, j].
    """
    eye_F = torch.eye(F, dtype=dtype, device=device)
    A = torch.zeros(K, K, dtype=dtype, device=device)
    for i in range(F):
        A[i * F:(i + 1) * F, i * F:(i + 1) * F] = J_d
    A_T = A.t().contiguous()

    """
    A_perm = I_F ⊗ J_perm where J_perm[k, j] = J[k, F-1-j].
    A_perm[i*F+p, k*F+j] = (i==k) * J[p, F-1-j].
    """
    J_perm = J_d[:, flip_j]
    A_perm = torch.zeros(K, K, dtype=dtype, device=device)
    for i in range(F):
        A_perm[i * F:(i + 1) * F, i * F:(i + 1) * F] = J_perm
    A_perm_T = A_perm.t().contiguous()

    out = (
        freq_i_kk,
        freq_j_kk,
        J_diag_kk,
        J_anti_kk,
        J_diag_perm_kk,
        J_anti_perm_kk,
        A_T,
        A_perm_T,
    )
    _constants_cache[key] = out
    return out


def wigner_D_nki(l: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
    """
    Drop-in replacement for wigner.wigner_D for a single l.
    Returns M of shape [B, F, F] where F = 2l+1.
    """
    F = 2 * l + 1
    K = F * F
    leading = alpha.shape

    a_flat = alpha.reshape(-1).contiguous()
    b_flat = beta.reshape(-1).contiguous()
    g_flat = gamma.reshape(-1).contiguous()
    B_real = a_flat.shape[0]
    dtype = a_flat.dtype
    device = a_flat.device

    n_tiles = _div_ceil(B_real, P_MAX)
    B_padded = n_tiles * P_MAX
    if B_padded != B_real:
        pad = torch.zeros(B_padded - B_real, dtype=dtype, device=device)
        a_flat = torch.cat([a_flat, pad], dim=0)
        b_flat = torch.cat([b_flat, pad], dim=0)
        g_flat = torch.cat([g_flat, pad], dim=0)

    (
        freq_i_kk_b,
        freq_j_kk_b,
        J_diag_kk_b,
        J_anti_kk_b,
        J_diag_perm_kk_b,
        J_anti_perm_kk_b,
        IJ_T,
        IJperm_T,
    ) = _ensure_constants(l, J, dtype, device)

    out_tiles = []
    for t in range(n_tiles):
        a_chunk = a_flat[t * P_MAX:(t + 1) * P_MAX].unsqueeze(1).expand(P_MAX, K).contiguous()
        b_chunk = b_flat[t * P_MAX:(t + 1) * P_MAX].unsqueeze(1).expand(P_MAX, K).contiguous()
        g_chunk = g_flat[t * P_MAX:(t + 1) * P_MAX].unsqueeze(1).expand(P_MAX, K).contiguous()
        M_chunk = _wigner_D_kernel(
            a_chunk,
            b_chunk,
            g_chunk,
            freq_i_kk_b,
            freq_j_kk_b,
            J_diag_kk_b,
            J_anti_kk_b,
            J_diag_perm_kk_b,
            J_anti_perm_kk_b,
            IJ_T,
            IJperm_T,
        )
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
            alpha = torch.randn(B, dtype=torch.float32)
            beta = torch.randn(B, dtype=torch.float32)
            gamma = torch.randn(B, dtype=torch.float32)
            ref = _wigner.wigner_D(l, alpha, beta, gamma)

            J = _wigner._Jd[l].to(dtype=torch.float32)

            alpha_n = alpha.to(device)
            beta_n = beta.to(device)
            gamma_n = gamma.to(device)
            J_n = J.to(device)
            out = wigner_D_nki(l, alpha_n, beta_n, gamma_n, J_n)
            torch.neuron.synchronize()
            out_cpu = out.cpu()

            abs_err = (out_cpu - ref).abs().max().item()
            ref_max = max(ref.abs().max().item(), 1.0)
            rel_err = abs_err / ref_max
            ok = torch.allclose(out_cpu, ref, atol=1e-4, rtol=1e-4)
            print(f"{l:>3d}  {B:>5d}  {abs_err:>12.3e}  {rel_err:>12.3e}  {str(ok):>8s}")
