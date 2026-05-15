"""
NKI tiled matmul kernel for micro-benchmarking nc_matmul vs XLA at small shapes.

Computes:
    out = a_T.T @ w_T

HBM layout (caller pre-transposes; matches the rest of this codebase):
    a_T : [K, M_pad]   (this is the input, transposed)
    w_T : [K, N]       (this is the weight, transposed)
    out : [M_pad, N]   (returned, float32)

Shape contracts (caller is responsible):
    - K   in {32, 64, 192} for the EquiformerV3 edge-linear use case
          (K  > 128 is handled here via K-tiling along the partition dim)
    - N   <= 512                       (one moving-tile, no N-tiling needed)
    - M_pad multiple of 128            (caller pads, no M-remainder needed)

Tile strategy (follows the standard nc_matmul pattern):
    stationary: a_tile [K_tile, TILE_M=128]   partition=K_tile (<=128)
    moving:     b_tile [K_tile, N]            partition=K_tile (<=128)
    psum    :         [TILE_M=128, N]         accumulated across K-tiles
                                              via hardware accumulation
                                              (multiple writes to same PSUM)
"""

import math
import nki
import nki.language as nl
import nki.isa as nisa
import torch

P_MAX  = nl.tile_size.pmax                  # 128 (partition / matmul K-per-tile)
TILE_M = nl.tile_size.gemm_stationary_fmax  # 128 (stationary free dim)
N_MAX  = nl.tile_size.gemm_moving_fmax      # 512 (moving free dim)


def kernel_assert(condition: bool, error_text: str):
    """Assert with NKI-formatted error message."""
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def div_ceil(n: int, d: int) -> int:
    """Ceiling division: smallest integer >= n/d."""
    return (n + d - 1) // d


@nki.jit
def nki_matmul_tiled(a_T, w_T):
    """
    Tiled matmul: out = a_T.T @ w_T.

    Args:
        a_T (nl.ndarray): [K, M_pad] @ HBM, M_pad % 128 == 0
        w_T (nl.ndarray): [K, N]     @ HBM, N <= 512

    Returns:
        nl.ndarray: [M_pad, N] @ HBM, float32

    Notes:
        - K is tiled to <=128 along the partition dimension; multiple
          nc_matmul writes to the same PSUM trigger hardware accumulation.
        - N <= 512 fits in one moving tile, so no N-tiling.
        - M_pad is a multiple of 128, so no M-remainder.
        - Output is always float32 (matches PSUM dtype directly).
    """
    K, M_pad = a_T.shape
    K2, N    = w_T.shape

    kernel_assert(K == K2, f"K mismatch: a_T K={K}, w_T K={K2}")
    kernel_assert(N <= N_MAX, f"N={N} exceeds moving free max {N_MAX}")
    kernel_assert(M_pad % TILE_M == 0,
                  f"M_pad={M_pad} must be multiple of {TILE_M}")

    out_dtype = nl.float32
    output_hbm = nl.ndarray((M_pad, N), dtype=out_dtype, buffer=nl.shared_hbm)

    n_tiles_m = M_pad // TILE_M
    n_tiles_k = div_ceil(K, P_MAX)

    for i_m in nl.affine_range(n_tiles_m):
        m_start = i_m * TILE_M

        # PSUM: uninitialized — first nc_matmul initialises, subsequent
        # writes to the same buffer accumulate in hardware.
        psum = nl.ndarray((TILE_M, N), dtype=nl.float32, buffer=nl.psum)

        for i_k in nl.affine_range(n_tiles_k):
            k_start = i_k * P_MAX
            k_end   = min(K, k_start + P_MAX)
            tk      = k_end - k_start

            # stationary a_tile: [tk, TILE_M] from a_T[K, M_pad]
            a_tile = nl.ndarray((tk, TILE_M), dtype=a_T.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=a_tile,
                src=a_T[k_start:k_end, m_start:m_start + TILE_M],
            )

            # moving b_tile: [tk, N] from w_T[K, N]
            b_tile = nl.ndarray((tk, N), dtype=w_T.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=b_tile,
                src=w_T[k_start:k_end, 0:N],
            )

            # psum[0:TILE_M, 0:N] += a_tile.T @ b_tile
            nisa.nc_matmul(
                dst=psum[0:TILE_M, 0:N],
                stationary=a_tile,
                moving=b_tile,
            )

        # PSUM -> SBUF (required: PSUM cannot be DMA-copied to HBM directly)
        result_sbuf = nl.ndarray((TILE_M, N), dtype=out_dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=result_sbuf, src=psum)

        # SBUF -> HBM
        nisa.dma_copy(
            dst=output_hbm[m_start:m_start + TILE_M, 0:N],
            src=result_sbuf,
        )

    return output_hbm


# === Python wrapper ============================================================

def matmul_tiled(a, w):
    """
    Standard row-major wrapper: out = a @ w.T.

    Args:
        a (torch.Tensor): [M, K]   on XLA device
        w (torch.Tensor): [N, K]   on XLA device   (nn.Linear weight convention)

    Returns:
        torch.Tensor: [M, N] float32 on the same device as `a`.

    Notes:
        - M is padded up to a multiple of 128 before calling the kernel; the
          padded rows are sliced off before returning.
        - K and N must match the kernel contract: K <= 192, N <= 512.
        - Both `a_T` and `w_T` are produced by transposing — the kernel expects
          the contraction dim as the leading (partition) dim.
    """
    M, K  = a.shape
    N, K2 = w.shape
    assert K == K2, f"K mismatch: a has K={K}, w has K={K2}"
    assert N <= N_MAX, f"N={N} exceeds moving free max {N_MAX}"

    # Pad M up to a multiple of 128.
    M_pad = ((M + TILE_M - 1) // TILE_M) * TILE_M
    if M_pad != M:
        pad_rows = M_pad - M
        pad = torch.zeros(
            (pad_rows, K), dtype=a.dtype, device=a.device
        )
        a_padded = torch.cat([a, pad], dim=0)        # [M_pad, K]
    else:
        a_padded = a

    # Transpose to kernel layout: a_T [K, M_pad], w_T [K, N].
    a_T = a_padded.transpose(0, 1).contiguous()      # [K, M_pad]
    w_T = w.transpose(0, 1).contiguous()             # [K, N]

    out_padded = nki_matmul_tiled(a_T, w_T)          # [M_pad, N] float32

    if M_pad != M:
        return out_padded[:M, :]
    return out_padded
