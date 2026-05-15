"""
NKI fused linear kernel: output = input @ weight.T  (+ optional bias)

HBM layout expected by this kernel (differs from nn.Linear convention):
  input_t_hbm:  [K, M]  transposed input  (input.T)
  weight_t_hbm: [K, N]  transposed weight (weight.T; weight is [N,K] in nn.Linear)
  bias_hbm:     [N]     optional, same dtype as input

The caller pre-transposes both tensors. This lets DMA loads stay contiguous
in both K-dimension tiles (P-dim = K).

Tile strategy (follows nki-writing common-patterns exactly):
  stationary: a_tile [K=128, M=128]  from input_t[K, M]
  moving:     b_tile [K=128, N=512]  from weight_t[K, N]
  psum:              [M=128, N=512]  = a_tile.T @ b_tile = input @ weight.T

Returns:
  output: [M, N]  same dtype as input
"""
import math
import nki
import nki.language as nl
import nki.isa as nisa

TILE_K = nl.tile_size.pmax                  # 128
TILE_M = nl.tile_size.gemm_stationary_fmax  # 128
TILE_N = nl.tile_size.gemm_moving_fmax      # 512


def _cdiv(a, b):
    return math.ceil(a / b)


@nki.jit
def nki_fused_linear(input_t_hbm, weight_t_hbm, bias_hbm=None):
    """
    Compute  output = input_t.T @ weight_t  (= input @ weight.T, + bias).

    Args:
        input_t_hbm:  [K, M]  bfloat16/float16/float32  (input transposed)
        weight_t_hbm: [K, N]  same dtype                (weight transposed)
        bias_hbm:     [N]     optional, same dtype

    Returns:
        output: [M, N]  same dtype as input
    """
    K, M = input_t_hbm.shape
    K2, N = weight_t_hbm.shape
    assert K == K2

    out_dtype = input_t_hbm.dtype
    output_hbm = nl.ndarray((M, N), dtype=out_dtype, buffer=nl.shared_hbm)

    n_tiles_m = _cdiv(M, TILE_M)
    n_tiles_k = _cdiv(K, TILE_K)
    n_tiles_n = _cdiv(N, TILE_N)

    for i_m in nl.affine_range(n_tiles_m):
        m_start = i_m * TILE_M
        m_end   = min(M, m_start + TILE_M)
        tm      = m_end - m_start

        for i_n in nl.affine_range(n_tiles_n):
            n_start = i_n * TILE_N
            n_end   = min(N, n_start + TILE_N)
            tn      = n_end - n_start

            # PSUM: uninitialized — first nc_matmul initialises, subsequent accumulate
            psum = nl.ndarray((TILE_M, TILE_N), dtype=nl.float32, buffer=nl.psum)

            for i_k in nl.affine_range(n_tiles_k):
                k_start = i_k * TILE_K
                k_end   = min(K, k_start + TILE_K)
                tk      = k_end - k_start

                # a_tile: stationary [K=tk, M=tm] from input_t[K, M]
                a_tile = nl.ndarray((tk, tm), dtype=out_dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=a_tile,
                              src=input_t_hbm[k_start:k_end, m_start:m_end])

                # b_tile: moving [K=tk, N=tn] from weight_t[K, N]
                b_tile = nl.ndarray((tk, tn), dtype=out_dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=b_tile,
                              src=weight_t_hbm[k_start:k_end, n_start:n_end])

                # psum[0:tm, 0:tn] += a_tile.T @ b_tile = input_tile @ weight_tile.T
                nisa.nc_matmul(dst=psum[0:tm, 0:tn],
                               stationary=a_tile,
                               moving=b_tile)

            # PSUM → SBUF (cast float32 → out_dtype)
            result_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=out_dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=result_sbuf[0:tm, 0:tn],
                             src=psum[0:tm, 0:tn])

            # Bias: broadcast [tn] bias over M rows
            if bias_hbm != None:
                bias_sbuf = nl.ndarray((tm, tn), dtype=out_dtype, buffer=nl.sbuf)
                for i_row in nl.affine_range(tm):
                    nisa.dma_copy(dst=bias_sbuf[i_row:i_row+1, 0:tn],
                                  src=bias_hbm[n_start:n_end])
                nisa.tensor_tensor(dst=result_sbuf[0:tm, 0:tn],
                                   data1=result_sbuf[0:tm, 0:tn],
                                   data2=bias_sbuf[0:tm, 0:tn],
                                   op=nl.add)

            nisa.dma_copy(dst=output_hbm[m_start:m_end, n_start:n_end],
                          src=result_sbuf[0:tm, 0:tn])

    return output_hbm
