"""
NKI kernel for scatter_reduce (sum): equivalent to
    torch.zeros(length).scatter_reduce(0, index, ones, reduce="sum")

Strategy (SDK 0.1.0 does not support non-unique scatter-add via DMA):
  For each bin b in [0, length):
    1. Compare index == b  → boolean mask [1, N]
    2. Cast mask to float32
    3. tensor_reduce over free-dim (sum) → scalar count [1, 1]
    4. Store count into output[b]

This is O(length) passes over the index tile. Efficient for small atom
counts (length = num_atoms, typically 2–200 in a molecular batch).
"""
import math
import nki
import nki.isa as nisa
import nki.language as nl


TILE = nl.tile_size.pmax  # 128


@nki.jit
def nki_scatter_reduce_sum_kernel(index, length):
    """
    Count occurrences: out[b] = sum(index == b) for b in [0, length).

    Args:
        index:   [N]        int32 indices in [0, length)
        length:  Python int  number of output bins
    Returns:
        out: [length] float32 on shared HBM
    """
    N = index.shape[0]

    out = nl.ndarray((length,), dtype=nl.float32, buffer=nl.shared_hbm)

    # Reshape 1D index to 2D [1, N] — SBUF requires ≥2 dims.
    # Tile the free dim in chunks of TILE to respect pmax=128.
    # We accumulate partial counts in SBUF, one [1, 1] per bin.
    counts = nl.ndarray((1, length), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=counts, value=0.0)

    index2d = index.reshape((1, N))

    for i in nl.sequential_range(math.ceil(N / TILE)):
        start = i * TILE
        end   = min(N, start + TILE)
        chunk = end - start

        # Load index chunk into SBUF: [1, chunk]
        idx_tile = nl.ndarray((1, chunk), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx_tile, src=index2d[0:1, start:end])

        # For each bin b, check equality and accumulate
        for b in nl.sequential_range(length):
            # Compare idx_tile == b  →  [1, chunk] int32 (0 or 1)
            eq_tile = nl.ndarray((1, chunk), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=eq_tile,
                data=idx_tile,
                op0=nl.equal,
                operand0=b,
            )

            # Sum across free dim → [1, 1] int32
            partial = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_reduce(
                dst=partial,
                op=nl.add,
                data=eq_tile,
                axis=(1,),
                keepdims=True,
            )

            # Accumulate into counts[0, b]
            cur = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=cur, src=counts[0:1, b:b+1])
            partial_f = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=partial_f, src=partial)
            nisa.tensor_tensor(
                dst=counts[0:1, b:b+1],
                data1=cur,
                data2=partial_f,
                op=nl.add,
            )

    # Store counts to HBM output
    out2d = out.reshape((1, length))
    nisa.dma_copy(dst=out2d[0:1, 0:length], src=counts)

    return out


if __name__ == "__main__":
    import torch

    torch.manual_seed(0)
    device = torch.device("neuron")

    def ref_get_counts(index, length):
        return torch.zeros(length, dtype=torch.float32).scatter_reduce(
            0, index.long(), torch.ones(len(index)), reduce="sum"
        )

    # ── Test 1: basic counts ──────────────────────────────────────────────────
    length = 8
    index = torch.tensor([0, 2, 2, 3, 0, 7, 3, 3], dtype=torch.int32)
    ref = ref_get_counts(index, length)
    result = nki_scatter_reduce_sum_kernel(index.to(device), length).cpu()

    diff = (result - ref).abs().max().item()
    ok = diff < 1e-5
    print(f"Test 1 basic     : {'PASS' if ok else 'FAIL'}  ref={ref.tolist()}  got={result.tolist()}")
    assert ok

    # ── Test 2: equiformer-style (2 atoms, 20 neighbors each) ─────────────────
    length2 = 2
    index2 = torch.cat([
        torch.zeros(20, dtype=torch.int32),
        torch.ones(20, dtype=torch.int32),
    ])
    ref2 = ref_get_counts(index2, length2)
    result2 = nki_scatter_reduce_sum_kernel(index2.to(device), length2).cpu()

    diff2 = (result2 - ref2).abs().max().item()
    ok2 = diff2 < 1e-5
    print(f"Test 2 equiformer: {'PASS' if ok2 else 'FAIL'}  ref={ref2.tolist()}  got={result2.tolist()}")
    assert ok2

    # ── Test 3: larger random ─────────────────────────────────────────────────
    N3, length3 = 256, 16
    index3 = torch.randint(0, length3, (N3,), dtype=torch.int32)
    ref3 = ref_get_counts(index3, length3)
    result3 = nki_scatter_reduce_sum_kernel(index3.to(device), length3).cpu()

    diff3 = (result3 - ref3).abs().max().item()
    ok3 = diff3 < 1e-5
    print(f"Test 3 random    : {'PASS' if ok3 else 'FAIL'}  max_diff={diff3:.6f}")
    assert ok3

    print("\nAll tests PASSED")
