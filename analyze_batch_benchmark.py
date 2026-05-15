"""Parse and summarize output from test_nki_batched_matmul.py.

Run this after the benchmark completes:
    python3 analyze_batch_benchmark.py [output_file]

Default output file path is the background job output from 2026-05-14.
"""
import sys
import re

DEFAULT_OUTPUT = (
    "/tmp/claude-0/-root-NKIBootcamp-equiformer-trainium"
    "/70b4b997-d4f3-45f7-a1e7-8f22d83f14a3/tasks/b33pzuv2o.output"
)

path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT

try:
    with open(path) as f:
        text = f.read()
except FileNotFoundError:
    print(f"Output file not found: {path}")
    print("The benchmark may still be running, or /tmp was cleared on reboot.")
    print("Re-run: python3 test_nki_batched_matmul.py")
    sys.exit(1)

if not text.strip():
    print("Output file is empty — benchmark may still be compiling.")
    sys.exit(1)

print(text)

# Parse result rows: "  B  M  M_pad  XLA µs  NKI µs  speedup"
row_re = re.compile(
    r"^\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x(.*)?$"
)

wins = []   # (layer, B, M, speedup)
losses = []

current_layer = ""
for line in text.splitlines():
    m = re.match(r"Layer: (.+?)\s+\(K=", line)
    if m:
        current_layer = m.group(1).strip()
    r = row_re.match(line)
    if r:
        B, M, M_pad = int(r.group(1)), int(r.group(2)), int(r.group(3))
        xla_us, nki_us, speedup = float(r.group(4)), float(r.group(5)), float(r.group(6))
        tag = r.group(7).strip() if r.group(7) else ""
        if speedup > 1.05:
            wins.append((current_layer, B, M, speedup))
        else:
            losses.append((current_layer, B, M, speedup))

print("\n" + "=" * 70)
print("ANALYSIS SUMMARY")
print("=" * 70)

if not wins and not losses:
    print("No data rows parsed — benchmark may be incomplete.")
    sys.exit(0)

if wins:
    print(f"\nNKI WINS ({len(wins)} cells where NKI > XLA by >5%):")
    for layer, B, M, sp in sorted(wins, key=lambda x: -x[3]):
        print(f"  B={B:>2}  M={M:>5}  speedup={sp:.2f}x  [{layer}]")
    min_win_B = min(w[1] for w in wins)
    print(f"\n  Minimum viable batch size: B={min_win_B} (M={min_win_B*39} edges)")
else:
    print("\nNKI never beat XLA at any shape/batch tested.")
    print("Conclusion: individual NKI layer dispatch overhead dominates at all tested sizes.")
    print("Recommended path: use XLA-fused model (test_1_compile.py) and focus on")
    print("larger batch inference (test_batch_scaling.py) for throughput gains.")

if losses:
    best_loss = max(losses, key=lambda x: x[3])
    print(f"\nClosest near-miss: B={best_loss[1]}, M={best_loss[2]}, speedup={best_loss[3]:.2f}x [{best_loss[0]}]")

print("\nNext steps:")
if wins:
    print("  1. Patch only the winning layer(s) into the full model")
    print("  2. Run test_1_compile.py with the patch to measure end-to-end speedup")
    print("  3. Run test_batch_scaling.py to confirm throughput at that batch size")
else:
    print("  1. Run test_batch_scaling.py — measure XLA model throughput vs CPU at B=1..64")
    print("  2. Find the batch size where Neuron atoms/s peaks (XLA fusion always wins here)")
    print("  3. The NKI story for this model is: fuse inside XLA, not patch individual ops")
