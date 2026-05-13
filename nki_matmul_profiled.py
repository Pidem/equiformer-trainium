"""NKI matmul kernel invoked via native PyTorch, wrapped with the Neuron profiler.

Implements a tiled matrix multiply: C[M,N] = A[M,K] @ B[K,N]
using NKI tile-level operations on the TensorEngine.

Usage:
    ssh nki_bootcamp
    source /home/ubuntu/nki_bootcamp_venv/bin/activate
    NEURON_RT_NUM_CORES=1 python3 nki_matmul_profiled.py
"""

import os

PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "matmul_profile_output")
os.environ.setdefault("NEURON_PLATFORM_TARGET_OVERRIDE", "trn2")
os.environ["TORCH_NEURONX_NEFF_CACHE_DIR"] = PROFILE_DIR
os.environ["NEURON_RT_ENABLE_DGE_NOTIFICATIONS"] = "1"
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"


import torch
import torch_neuronx  # noqa: F401

import nki
import nki.language as nl
import nki.isa as nisa
from torch.profiler import profile, ProfilerActivity
from torch_neuronx.profiling import NeuronConfig, ProfileMode, NeuronProfiler

os.makedirs(PROFILE_DIR, exist_ok=True)

# Tile sizes: partition dim (P) must be <= 128 for SBUF tile constraint
# For matmul: stationary has shape (K_par, M_free), moving has shape (K_par, N_free)
# Result is (M_free, N_free) in PSUM
TILE_M = 128
TILE_K = 128
TILE_N = 128


@nki.jit
def nki_matmul(a_input, b_input):
    """Tiled matmul: C[M,N] = A[M,K] @ B[K,N].

    NKI matmul semantics (TensorEngine):
      result[P_lhs_free, P_rhs_free] = lhs[P_contract, P_lhs_free]^T @ rhs[P_contract, P_rhs_free]

    So to compute A[M,K] @ B[K,N]:
      - lhs (stationary, SBUF): shape (TILE_K, TILE_M) — K is partition/contraction
      - rhs (moving, SBUF): shape (TILE_K, TILE_N) — K is partition/contraction
      - result (PSUM): shape (TILE_M, TILE_N)

    We load A transposed so K is the partition (first) dimension.
    B naturally has K as the first dimension.
    """
    M, K = a_input.shape
    K2, N = b_input.shape

    c_output = nl.ndarray((M, N), dtype=a_input.dtype, buffer=nl.shared_hbm)

    for m in nl.affine_range(M // TILE_M):
        for n in nl.affine_range(N // TILE_N):
            # First iteration: no accumulate (avoids bfloat16 Memset to PSUM bug on gen3)
            a_tile = nl.load(
                a_input[m * TILE_M : (m + 1) * TILE_M, 0 : TILE_K]
            )
            b_tile = nl.load(
                b_input[0 : TILE_K, n * TILE_N : (n + 1) * TILE_N]
            )
            result = nl.ndarray((TILE_M, TILE_N), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_matmul(result, a_tile, b_tile, is_transpose=True, accumulate=False)

            # Remaining iterations: accumulate
            for k in nl.affine_range(1, K // TILE_K):
                a_tile = nl.load(
                    a_input[m * TILE_M : (m + 1) * TILE_M,
                            k * TILE_K : (k + 1) * TILE_K]
                )
                b_tile = nl.load(
                    b_input[k * TILE_K : (k + 1) * TILE_K,
                            n * TILE_N : (n + 1) * TILE_N]
                )
                nisa.nc_matmul(result, a_tile, b_tile, is_transpose=True, accumulate=True)

            # Move PSUM to SBUF then DMA to HBM
            result_sbuf = nl.ndarray((TILE_M, TILE_N), dtype=a_input.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=result_sbuf, src=result)
            nisa.dma_copy(
                dst=c_output[m * TILE_M : (m + 1) * TILE_M,
                             n * TILE_N : (n + 1) * TILE_N],
                src=result_sbuf,
            )

    return c_output


def main():
    device = torch.device("neuron")

    M, K, N = 512, 512, 512
    a = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    b = torch.randn(K, N, dtype=torch.bfloat16, device=device)


    # Profile
    neuron_config = NeuronConfig(
        modes=[ProfileMode.DEVICE, ProfileMode.RUNTIME],
        profile_output_dir=PROFILE_DIR,
    )
    exporter = NeuronProfiler(neuron_config)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.PrivateUse1],
        experimental_config=neuron_config,
        on_trace_ready=exporter.export_trace,
    ) as prof:
        c = nki_matmul(a, b)
        torch.neuron.synchronize()
        prof.step()

    print("Profiling done.")
    print(f"\nProfile output: {PROFILE_DIR}/")

    # List output
    for root, dirs, files in os.walk(PROFILE_DIR):
        level = root.replace(PROFILE_DIR, "").count(os.sep)
        indent = "  " * level
        rel = os.path.relpath(root, PROFILE_DIR)
        if rel != ".":
            print(f"{indent}{rel}/")
        for f in sorted(files):
            path = os.path.join(root, f)
            size_kb = os.path.getsize(path) / 1024
            print(f"{indent}  {f}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
