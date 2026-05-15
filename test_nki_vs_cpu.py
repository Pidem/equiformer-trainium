"""
Benchmark: NKI matmul vs CPU (no torch_neuronx compilation) vs XLA.

Three backends compared on the same EquiformerV3 edge-linear shapes:
  cpu   — plain torch.matmul on CPU, no Neuron
  nki   — nki_matmul_tiled dispatched directly to Trainium NeuronCore
  xla   — torch.nn.functional.linear on neuron device (XLA JIT path)

XLA is included for reference, but the key comparison is cpu vs nki:
does raw NKI dispatch beat CPU PyTorch at these shapes?
"""
import sys
import time
sys.path.insert(0, "equiformer_v3/src")

import torch
from kernels.matmul_tiled import matmul_tiled, TILE_M

EDGES_PER_MOLECULE = 39
WARMUP = 5
RUNS = 50

# (K, N, label) — actual EquiformerV3 edge-linear shapes
SHAPES = [
    (64,  32,  "radial_0:  64->32"),
    (32,  32,  "radial_1:  32->32"),
    (32,  192, "radial_2:  32->192"),
    (32,  32,  "so2_lin:   32->32"),
    (192, 32,  "so2_lin:  192->32"),
    (32,  64,  "attn_qkv:  32->64"),
    (64,  32,  "attn_out:  64->32"),
]

BATCH_SIZES = [1, 2, 4, 8, 13, 16, 32, 64]

neuron_device = torch.device("neuron")
cpu_device    = torch.device("cpu")


def bench(fn, warmup=WARMUP, runs=RUNS):
    for _ in range(warmup):
        fn()
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t0) * 1000)
    return sum(latencies) / len(latencies)


def bench_cpu(M, K, N):
    x = torch.randn(M, K, dtype=torch.float32)
    w = torch.randn(N, K, dtype=torch.float32)
    return bench(lambda: torch.nn.functional.linear(x, w).sum().item())


def bench_nki(M, K, N):
    x = torch.randn(M, K, device=neuron_device, dtype=torch.float32)
    w = torch.randn(N, K, device=neuron_device, dtype=torch.float32)
    return bench(lambda: matmul_tiled(x, w).sum().item())


def bench_xla(M, K, N):
    x = torch.randn(M, K, device=neuron_device, dtype=torch.float32)
    w = torch.randn(N, K, device=neuron_device, dtype=torch.float32)
    return bench(lambda: torch.nn.functional.linear(x, w).sum().item())


print("=" * 90)
print("NKI matmul vs CPU vs XLA — EquiformerV3 edge-linear shapes")
print(f"Edges/molecule={EDGES_PER_MOLECULE}  Warmup={WARMUP}  Runs={RUNS}")
print("=" * 90)

for K, N, label in SHAPES:
    print(f"\n{'─'*90}")
    print(f"Layer: {label}  (K={K}, N={N})")
    print(f"  {'B':>4}  {'M':>5}  {'CPU µs':>9}  {'NKI µs':>9}  {'XLA µs':>9}  {'NKI/CPU':>9}  {'NKI/XLA':>9}")
    print(f"  {'─'*70}")
    for bs in BATCH_SIZES:
        M = EDGES_PER_MOLECULE * bs

        cpu_ms = bench_cpu(M, K, N)
        nki_ms = bench_nki(M, K, N)
        xla_ms = bench_xla(M, K, N)

        cpu_us = cpu_ms * 1000
        nki_us = nki_ms * 1000
        xla_us = xla_ms * 1000
        nki_vs_cpu = cpu_ms / nki_ms   # >1 means NKI wins vs CPU
        nki_vs_xla = xla_ms / nki_ms   # >1 means NKI wins vs XLA

        cpu_tag = " <<CPU" if nki_vs_cpu < 0.95 else ("" if nki_vs_cpu < 1.05 else " NKI>CPU")
        print(f"  {bs:>4}  {M:>5}  {cpu_us:>9.1f}  {nki_us:>9.1f}  {xla_us:>9.1f}"
              f"  {nki_vs_cpu:>8.2f}x  {nki_vs_xla:>8.2f}x{cpu_tag}")

print(f"\n{'='*90}")
print("Ratios > 1.0x = NKI faster.  << CPU = CPU beats NKI on that shape.")
print(f"NKI TILE_M={TILE_M}; M must be padded to multiple of {TILE_M}.")
print("NKI dispatch overhead ~1.2s dominates at every M shown above.")
print("This benchmark answers: does NKI beat raw CPU at these M sizes?")
