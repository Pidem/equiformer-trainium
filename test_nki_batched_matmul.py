"""NKI nc_matmul vs XLA linear at EquiformerV3 edge-linear shapes for each batch size.

For a NaCl unit cell (2 atoms, ~39 edges per molecule), batch B gives M ~= 39*B edges.
The SO2 linear layers are MatMul(M, K) @ MatMul(K, N)^T.

Shapes tested match the actual layers in the model:
  - RadialFunction MLP: [M, 64]->[32], [M, 32]->[32], [M, 32]->[192]
  - SO2 linear: [M, 32]->[32], [M, 192]->[32]
  - Attention QKV: [M, 32]->[64], [M, 64]->[32]
"""
import sys
import time
sys.path.insert(0, "equiformer_v3/src")

import torch
import torch_neuronx
from kernels.matmul_tiled import matmul_tiled, TILE_M

EDGES_PER_MOLECULE = 39
WARMUP = 5
RUNS = 50

# (K, N, label) — the actual shapes in EquiformerV3 SO2/attention layers
SHAPES = [
    (64,  32,  "radial_0:  64->32"),
    (32,  32,  "radial_1:  32->32"),
    (32,  192, "radial_2:  32->192"),
    (32,  32,  "so2_lin:   32->32"),
    (192, 32,  "so2_lin:  192->32"),
    (32,  64,  "attn_qkv:  32->64"),
    (64,  32,  "attn_out:  64->32"),
]

BATCH_SIZES = [1, 2, 4, 8, 13, 16, 32]
device = torch.device("neuron")


def edges_for_batch(bs):
    return EDGES_PER_MOLECULE * bs


def bench_xla(M, K, N):
    x = torch.randn(M, K, device=device, dtype=torch.float32)
    w = torch.randn(N, K, device=device, dtype=torch.float32)
    for _ in range(WARMUP):
        torch.nn.functional.linear(x, w).sum().item()
    latencies = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        torch.nn.functional.linear(x, w).sum().item()
        latencies.append((time.perf_counter() - t0) * 1000)
    return sum(latencies) / len(latencies)


def bench_nki(M, K, N):
    x = torch.randn(M, K, device=device, dtype=torch.float32)
    w = torch.randn(N, K, device=device, dtype=torch.float32)
    for _ in range(WARMUP):
        matmul_tiled(x, w).sum().item()
    latencies = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        matmul_tiled(x, w).sum().item()
        latencies.append((time.perf_counter() - t0) * 1000)
    return sum(latencies) / len(latencies)


print("=" * 80)
print("NKI nc_matmul vs XLA F.linear — EquiformerV3 edge-linear shapes")
print(f"Edges/molecule: {EDGES_PER_MOLECULE}  |  Warmup: {WARMUP}  |  Runs: {RUNS}")
print("=" * 80)

for K, N, label in SHAPES:
    print(f"\n{'─'*80}")
    print(f"Layer: {label}  (K={K}, N={N})")
    print(f"  {'B':>4}  {'M':>5}  {'M_pad':>6}  {'XLA µs':>9}  {'NKI µs':>9}  {'speedup':>8}")
    print(f"  {'-'*55}")
    for bs in BATCH_SIZES:
        M = edges_for_batch(bs)
        M_pad = ((M + TILE_M - 1) // TILE_M) * TILE_M

        xla_ms = bench_xla(M, K, N)
        nki_ms = bench_nki(M, K, N)

        xla_us = xla_ms * 1000
        nki_us = nki_ms * 1000
        speedup = xla_ms / nki_ms
        marker = " <<< WIN" if speedup > 1.05 else ("" if speedup > 0.95 else " slower")
        print(f"  {bs:>4}  {M:>5}  {M_pad:>6}  {xla_us:>9.1f}  {nki_us:>9.1f}  {speedup:>7.2f}x{marker}")

print(f"\n{'='*80}")
print("Key thresholds:")
print(f"  M >= 128 (B >= 4):  minimum for nc_matmul (TILE_M = {TILE_M})")
print(f"  M >= 512 (B >= 14): full Tensor Engine utilization")
print("  NKI WIN = >5% faster than XLA")
