# NKI bootcamp project

setup environment: [2026 NKI Bootcamp using DLAMI](https://quip-amazon.com/2uksAC4PRAzt)
git clone https://github.com/Pidem/equiformer-trainium
pip install -r requirements.txt

### run test_1.py with torch_neuronx print fallback_ops:

 ┌────────────┬────────────────────────────────┬───────────────────────────────────┬────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────┐
 │ Priority │  Op  │  Where  │ Role │ Impact │
 ├────────────┼────────────────────────────────┼───────────────────────────────────┼────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
 │ **1 — High**  │ aten::scatter_reduce.two_out  │ utils.py:787 get_counts()  │ Aggregates neighbor counts during graph construction  │ Called every forward pass during neighbor trimming  │
 ├────────────┼────────────────────────────────┼───────────────────────────────────┼────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
 │ **1 — High**  │ aten::sort.values_stable  │ utils.py:861 │ Sorts neighbors by distance (picks closest N per atom) │ Called every forward pass, shapes the graph topology  │
 ├────────────┼────────────────────────────────┼───────────────────────────────────┼────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
 │ **1 — High**  │ aten::repeat_interleave.Tensor │ utils.py:849 │ Expands index offsets for graph trimming  │ Called every forward pass in neighbor-mask computation  │
 ├────────────┼────────────────────────────────┼───────────────────────────────────┼────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
 │ **1 — High**  │ aten::index_fill_.int_Scalar  │ utils.py:901 │ Writes into neighbor mask │ Called every forward pass │
 ├────────────┼────────────────────────────────┼───────────────────────────────────┼────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
 │ **2 — Medium** │ aten::atan2.out │ so3.py:389 │ Extracts Euler angle γ from rotation matrix per edge  │ Called in every Wigner D-matrix computation (once per layer) │
 ├────────────┼────────────────────────────────┼───────────────────────────────────┼────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
 │ **2 — Medium** │ aten::acos.out  │ likely o3.xyz_to_angles via e3nn │ Converts edge vectors to spherical angles α │ Part of same rotation extraction pipeline as atan2  │
 ├────────────┼────────────────────────────────┼───────────────────────────────────┼────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
 │ **2 — Medium** │ aten::linalg_cross.out  │ likely in o3 rotation utilities  │ Cross product for constructing edge rotation matrices │ Part of the same rotation frame setup │
 ├────────────┼────────────────────────────────┼───────────────────────────────────┼────────────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────┤
 │ **3 — Low** │ aten::uniform_  │ so3.py:584 / input_block.py:55-56 │ Weight initialization (nn.init.uniform_)  │ **Only runs once at model init, not during inference**  │
 └────────────┴────────────────────────────────┴───────────────────────────────────┴────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────┘


### using nki skills to analyze feasibility:

 **Strategic Verdict**
 
 ┌───────────────────┬─────────────┬────────────┬─────────────────────────────────────────────┐
 │ Op  │ Feasibility │  Effort  │  Recommendation │
 ├───────────────────┼─────────────┼────────────┼─────────────────────────────────────────────┤
 │ scatter_reduce │ ✅ Yes │ Low │ Write it │
 ├───────────────────┼─────────────┼────────────┼─────────────────────────────────────────────┤
 │ index_fill_  │ ✅ Yes │ Medium  │ Write it │
 ├───────────────────┼─────────────┼────────────┼─────────────────────────────────────────────┤
 │ sort │ ⚠️ Partial │ High  │ Skip — use max8 top-K only if threshold ≤ 8 │
 ├───────────────────┼─────────────┼────────────┼─────────────────────────────────────────────┤
 │ repeat_interleave │ ❌ No  │ Infeasible │ Skip entirely  │
 └───────────────────┴─────────────┴────────────┴─────────────────────────────────────────────┘

 **Summary**
 
 ┌──────────────────────────┬──────────────────────────────┬────────────────────────────────┐
 │ │  Without NKI │ With NKI │
 ├──────────────────────────┼──────────────────────────────┼────────────────────────────────┤
 │ scatter_reduce execution │ CPU fallback, ~0 ms overhead │ 24 × 130 ms dispatch = 3125 ms │
 ├──────────────────────────┼──────────────────────────────┼────────────────────────────────┤
 │ On-device kernel time │ — │ 24 × 10.7 µs = 0.26 ms  │
 ├──────────────────────────┼──────────────────────────────┼────────────────────────────────┤
 │ Model latency (avg) │ 816 ms  │ 3125 ms │
 └──────────────────────────┴──────────────────────────────┴────────────────────────────────┘
 
 **The NKI kernel itself is not the problem** — 10.7 µs is perfectly reasonable. The problem is that this op is the wrong candidate for NKI: it's tiny (168 bytes of I/O), called 24 times per forward, and was
 already handled efficiently as a CPU fallback. NKI kernels pay off when they replace large compute-bound ops that are bottlenecking on-device execution, not when they replace small CPU fallbacks.
 

 ---
 **Root cause: kernel binary reload dominates every call**
 
 **What the profile shows**


 ┌─────────────────┬──────────────────────────────────┬──────────────────┐
 │  Source │ Metric │ Value  │
 ├─────────────────┼──────────────────────────────────┼──────────────────┤
 │ Profile  │ On-device exec time │ **10.7 µs** │
 ├─────────────────┼──────────────────────────────────┼──────────────────┤
 │ Profile  │ Actual data transfer (168 B I/O) │ **0.5 µs**  │
 ├─────────────────┼──────────────────────────────────┼──────────────────┤
 │ Profile  │ static_dma_size loaded per call │ **65,852 B (64 KB)** │
 ├─────────────────┼──────────────────────────────────┼──────────────────┤
 │ Profile  │ dma_queue_count │ 33 packets  │
 ├─────────────────┼──────────────────────────────────┼──────────────────┤
 │ Micro-benchmark │ NKI round-trip wall time  │ **1,206 ms**  │
 ├─────────────────┼──────────────────────────────────┼──────────────────┤
 │ Micro-benchmark │ CPU scatter_reduce  │ **0.008 ms**  │
 ├─────────────────┼──────────────────────────────────┼──────────────────┤
 │ Micro-benchmark │ Overhead ratio  │ **149,607×**  │
 └─────────────────┴──────────────────────────────────┴──────────────────┘
 
 **The actual bottleneck: 64 KB kernel binary reloaded every call**
 
 On-device compute:  10.7 µs  (0.0009% of wall time)
 Actual I/O (168 B):  0.5 µs
 Kernel binary reload: 65,852 B DMA'd to NeuronCore on every call
 ─────────────────────────────────────────────────────────────────
 Total wall time per call: 1,206 ms (99.999% is dispatch overhead)
 
 The static_dma_size = 65,852 B in the profile is the compiled kernel binary being DMA'd to the NeuronCore on every invocation. The kernel never amortizes this because get_counts is called once per model
 forward — there's no batching of calls on-device.


 **The numbers fully explain the benchmark degradation**
 
 ┌────────────────────────────────────────┬─────────────────────┐
 │ Scenario │ Per-forward latency │
 ├────────────────────────────────────────┼─────────────────────┤
 │ CPU scatter_reduce (0.008 ms × 1 call) │ ~0.008 ms  │
 ├────────────────────────────────────────┼─────────────────────┤
 │ NKI round-trip (1,206 ms × 1 call)  │ ~1,206 ms  │
 ├────────────────────────────────────────┼─────────────────────┤
 │ Measured model slowdown (3,125 − 816) │ ~2,309 ms  │
 └────────────────────────────────────────┴─────────────────────┘
 
 The remaining ~1,100 ms gap is the NKI call interrupting the Neuron execution stream — forcing a host-device sync that stalls the rest of the eager JIT ops that were pipelined.
 
 **Why scatter_reduce was faster as a CPU fallback**
 
 In test_1_compile (without NKI), aten::scatter_reduce appeared in the CPU fallback list. It ran entirely on CPU in ~0.008 ms with zero device dispatch. By replacing it with a NKI kernel, we inserted a
 1.2-second PCIe round-trip in the middle of every forward pass.


 **Fix: NKI kernels need to live inside the device execution loop**
 
 NKI pays off only when the kernel is called in a tight device-side loop over large tiles — e.g., inside a fused attention or matmul kernel where the device never returns to host between calls. The right
 target would be replacing one of the **76 on-device ops** that are already on Neuron, not a CPU fallback op that runs in 8 µs.


### Why the NKI scatter_reduce kernel makes performance worse
  

The core problem: dispatch overhead >> compute work

  The scatter_reduce operation in this model handles a trivially small tensor: for a NaCl unit cell (2 atoms × 20 neighbors = 40 edges), the total I/O is only 168 bytes (40 × int32 indices + 2 × float32
  outputs).

  Every time the NKI kernel is called, the Neuron runtime must:
  1. Ship the 64 KB NEFF binary over PCIe to the device
  2. Set up descriptors, allocate buffers, execute, sync
  
  That dispatch overhead is ~1.2 seconds per call, while the actual computation completes in nanoseconds. The CPU fallback does the same work in ~8 µs — roughly 150,000× faster.

  Since scatter_reduce is called once per forward pass, the model becomes ~4× slower overall.

  The kernel algorithm also has structural inefficiency

  Looking at kernels/scatter_reduce.py:47-87, the algorithm does O(length × N/TILE) sequential passes:

  for i in sequential_range(ceil(N/128)):       # 1 tile for N=40
      for b in sequential_range(length):        # 2 iterations
          compare → reduce → accumulate

  The nested sequential_range loops prevent any pipelining. For tiny length=2, this is just 2 scalar operations dressed up as a kernel — there is no meaningful parallelism for the Neuron engines to exploit.

  Why NKI helps for some ops but not this one

  NKI is beneficial when:
  - The tensor is large enough that kernel execution time >> PCIe dispatch overhead
  - The operation is compute-bound (matmuls, softmax over many elements)
  
  scatter_reduce here is the opposite: it's a tiny, memory-bound, once-per-pass op. The overhead is fixed at ~1.2s regardless of problem size.

  What the file already documents

  test_nki_compile.py:4-10 already captures this finding:

  ▎ scatter_reduce was removed as an NKI target. It is called once per forward pass with only 168 bytes of I/O, but incurs ~1.2 s of PCIe dispatch overhead per call (64 KB kernel binary reloaded each time), 
  ▎ making the model ~4x slower. The op runs fine as a CPU fallback in ~8 µs.
  ▎
  ▎ Good NKI targets: fused linear (FFN projections) and fused attention (bmm + softmax + bmm).

  Bottom line: NKI kernels are not free to call — the Neuron dispatch cost is substantial. Only use NKI for operations where the data is large enough (typically tens of KB to MB) that the computation time
  amortizes the fixed ~1s dispatch overhead.




### **CPU Fallback Ops: NKI Kernel Candidates**

 
 **The scatter_reduce lesson applied**


 The scatter_reduce NKI kernel failed because:
 1. **Tiny I/O** (168 bytes) — ~1.2s PCIe dispatch overhead dwarfs actual compute
 2. **Called once per forward pass** — no amortization opportunity 
 3. **~8µs on CPU** — the CPU fallback was 150,000× cheaper than the NKI dispatch
 
 Applying the same criteria — total_data_size >> dispatch_cost AND called frequently enough AND compute-heavy on CPU — here is how the real inference fallbacks rank:


 ---
 **Tier 1: Strong NKI Candidates**
 
 ┌────────────────────┬───────────┬─────────┬─────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │  Op  │ Count/run │ Avg │ Total  │  Why it's a good target │
 │ │  │ (ms)  │  (s)  │  │
 ├────────────────────┼───────────┼─────────┼─────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ **aten::index_select** │ 3.0/run  │ 485ms  │ 34.4s  │ Large gather on edge/node features; operates on [N_edges, channels] tensors. High compute, no scatter complexity, NKI has local_gather / │
 │ │  │  │  │ nc_n_gather primitives.  │
 ├────────────────────┼───────────┼─────────┼─────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ **aten::linear**  │ 3.2/run  │ 94ms │ 7.3s │ FFN projections and attention linears. This is y = xW^T + b — textbook matmul, the NKI sweet spot. Already noted in test_nki_compile.py as the │
 │ │  │  │  │ primary candidate. High I/O, high arithmetic intensity.  │
 ├────────────────────┼───────────┼─────────┼─────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ **aten::embedding** │ 0.6/run  │ 532ms  │ 8.0s │ Large lookup of edge/node feature embeddings into [N_edges, embed_dim]. Essentially a gather over HBM — good fit for DMA-backed NKI gather.  │
 └────────────────────┴───────────┴─────────┴─────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘


 ---
 **Tier 2: Viable with Care**
 
 ┌──────────────────┬───────────┬────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Op │ Count/run │ Avg  │  Why / Caveats  │
 │ │  │ (ms) │  │
 ├──────────────────┼───────────┼────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ **aten::bmm** │ 4.5/run  │ 7.6ms │ Batched matmul in attention ([B, H, S, S]). Low avg latency (~7ms each) means dispatch is still a risk, but multiple calls per layer could justify a fused │
 │ │  │ │ attention kernel (bmm + softmax + bmm). Already flagged in test_nki_compile.py.  │
 ├──────────────────┼───────────┼────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ **aten::index_put_** │ ~1/run │ 939ms │ Scatter-write back to node/edge tensors — the inverse of index_select. NKI's dma_copy with dst_rmw_op=nl.add supports non-unique scatter-add. But each call is │
 │ │  │ │ infrequent (appears to be graph construction, not layer computation).  │
 └──────────────────┴───────────┴────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘


 ---
 **Tier 3: NOT good targets (same failure mode as scatter_reduce)**
 
 ┌───────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Op  │  Why to avoid  │
 ├───────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ aten::ge, aten::is_nonzero │ ~8 calls/run but trivially small tensors; pure control flow/mask operations. CPU does them in microseconds, NKI dispatch would dominate. │
 ├───────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ aten::item, aten::_local_scalar_dense  │ Scalar extraction — by definition 1 element. Irreducibly small I/O. │
 ├───────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ aten::arange, aten::sub, aten::add, aten::mul │ High call count but very fast per call (µs–ms). Many are pure index arithmetic on small integer tensors.  │
 ├───────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ aten::atan2, aten::acos, aten::linalg_cross  │ Only 2–4 calls total across 24 runs — these are graph construction ops, not inference ops.  │
 ├───────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ aten::scatter_reduce │ Already tried. 3 calls / 24 runs = once per forward pass, tiny I/O. Same failure as the NKI attempt.  │
 ├───────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ aten::_to_copy / aten::to  │ 36–44 calls/run, but these are host↔device transfers. NKI can't eliminate PCIe copies.  │
 └───────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘


 ---
 **Priority recommendation**
 
 The test_nki_compile.py comment already identifies the right targets. The profiling data confirms them quantitatively:


 1. **aten::linear (FFN projections)** — best NKI target. Classic matmul, large tensors, high arithmetic intensity. The kernels/fused_linear.py file already exists — this is the right path.
 2. **Fused attention (bmm + softmax + bmm)** — second best. The 4.5/run × 7.6ms = ~34ms/pass from bmm alone, plus the softmax overhead that isn't captured here. Fusing reduces kernel dispatch count from 3 to 1
 and enables on-chip accumulation.
 3. **aten::index_select** — highest single-op time (34.4s), but it's a gather with dynamic indices, which is harder in NKI. The per-call size (484ms avg on CPU) suggests large tensors (~`[N_edges=40,
 channels=32]` or larger), so dispatch overhead would be amortized. This is worth profiling before committing to the implementation.



### **NKI Library / Samples Kernels for EquiformerV3**

 
 **Op 1: aten::linear — 7.3s CPU time, 3.2 calls/run**


 **Directly usable: mlp** from nki-library (core/mlp/mlp.py)


 The highest-priority match. It implements the full FFN block (gate_proj + up_proj + down_proj with optional SwiGLU/SiLU activation), takes input [B, S, H], and auto-dispatches between a CTE variant (large S)
 and TKG variant (small S, B×S ≤ 96). Supports bf16/fp16/fp32 natively.


 Caveat: if EquiformerV3's FFN is a plain 2-matrix block (up + down, no gate), set skip_gate_proj=True. If it's the gated variant, it's a direct drop-in.


 **Also useful: output_projection_cte** (core/output_projection/) — handles a single linear layer out = attn @ weight + bias, matching any standalone aten::linear. Best for S ≥ 512. The QKV projection variant
 qkv_cte covers the case where Q, K, V weights are fused.


 **For arbitrary shapes: nki_matmul_fully_optimized_** from nki-samples tutorials — a standalone tiled matmul, bias must be added separately, requires shape multiples (K×1024, M×2048, N×1024) so may need parameter
 tuning for EquiformerV3's small channels=32.


 ---
 **Op 2: Fused attention (bmm + softmax + bmm) — ~34ms/pass across 4.5 calls/run**
 
 **Directly usable: attention_cte** from nki-library (core/attention/attention_cte.py)


 Production-grade, computes exactly softmax(scale * Q @ K^T) @ V. Handles seqlen up to 131072, d ≤ 128, supports causal masks, GQA, and context parallelism. This is the primary recommendation.


 **For small graphs: attention_tkg** — same computation but optimized for the token-generation regime (B×S ≤ 96). Since EquiformerV3 with a NaCl unit cell has only 2 atoms × 20 neighbors = 40 edges, the sequence
 length is tiny — **attention_tkg is likely the correct variant here**.


 **For custom adaptation: attn_fwd_v8a** from nki-samples tutorials — the most optimized of 8 progressive tutorial kernels. Clean NKI code with d_head=128 constraint (which matches attn_alpha_channels=16 in the
 model config — this is a **mismatch** to investigate before using this kernel).


 ---
 **Op 3: aten::index_select / aten::embedding — 34.4s + 8.0s CPU time**
 
 **No direct match exists in either repository.**


 The closest is indexed_flatten from nki-library (core/subkernels/indexed_flatten.py) — it gathers *contiguous blocks* of rows from a 2D tensor using block offsets (designed for MoE token routing where tokens
 are pre-sorted by expert). It is **not** a random-access gather by arbitrary integer indices, which is what index_select needs.


 This means a custom NKI kernel would need to be written using nl.load with nl.mgrid indexing or DMA scatter/gather ISA primitives. However, at 34.4s total CPU time, this is the highest-value gap — the 484ms
 average per call on CPU suggests large-enough tensors to amortize NKI dispatch overhead.


 ---
 **Summary**
 
 ┌────────────────────┬────────────┬───────────────────────────────┬────────────────────────────────────────────────────┐
 │  Op  │ CPU Time │  Best Available Kernel  │  Usability │
 ├────────────────────┼────────────┼───────────────────────────────┼────────────────────────────────────────────────────┤
 │ aten::linear (FFN) │ 7.3s  │ mlp (nki-library)  │ **Drop-in** — matches FFN block pattern │
 ├────────────────────┼────────────┼───────────────────────────────┼────────────────────────────────────────────────────┤
 │ Fused attention │ ~34ms/pass │ attention_tkg (nki-library)  │ **Drop-in** — TKG variant matches tiny graph seqlen │
 ├────────────────────┼────────────┼───────────────────────────────┼────────────────────────────────────────────────────┤
 │ aten::index_select │ 34.4s │ indexed_flatten (nki-library) │ **Partial** — block gather only, not arbitrary indices │
 ├────────────────────┼────────────┼───────────────────────────────┼────────────────────────────────────────────────────┤
 │ aten::embedding │ 8.0s  │ No match │ **Must write custom** │
 └────────────────────┴────────────┴───────────────────────────────┴────────────────────────────────────────────────────┘


 **Recommended action:** Start with attention_tkg and mlp from nki-library — both are drop-in and address the second-largest bottleneck class. The index_select gap (largest single op by total time) has no existing
 kernel and requires a custom implementation using nisa.dma_copy with dge_mode=swdge or nc_n_gather.
* * *
 ---
 **Benchmark Results implementing mlp kernel from nki library**
 
 ┌────────────────────┬─────────────┬──────────────┐
 │ │ Avg latency │ Throughput │
 ├────────────────────┼─────────────┼──────────────┤
 │ **Baseline XLA**  │ 4,402 ms │ 0.5 atoms/s │
 ├────────────────────┼─────────────┼──────────────┤
 │ **NKI-patched linear** │ 19,857 ms  │ 0.1 atoms/s │
 ├────────────────────┼─────────────┼──────────────┤
 │ **Delta** │ +15,455 ms │ **4.51× slower** │
 └────────────────────┴─────────────┴──────────────┘
 
 **NKI explicit dispatch is ~4.5× slower** than XLA auto-compilation for this model.
 
 ---
 **Why**
 
 This is the same problem as scatter_reduce, just at larger scale. The 20 intercepted F.linear calls each incur the full NKI kernel binary dispatch overhead (~64 KB reload over PCIe per call). The shapes are
 tiny — M=40, K in {16–192} — nowhere near the [128×512] minimum tile size needed to amortize Tensor Engine startup cost.
 
 XLA wins here because it **traces the entire model graph once**, compiles a single NEFF that fuses all linear ops together, and executes it in one shot. The NKI kernel launches 20 separate NEFFs per forward pass.
 
 ---
 **What would actually help**
 
 ┌───────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ Target  │  Why it's a good NKI candidate  │
 ├───────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ **Fused attention** (bmm + softmax + bmm) │ One large kernel binary for the full QK·V computation; avoids 3 separate HBM round-trips │
 ├───────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ **Larger batch / multi-graph** inference │ Once M ≥ 128, the Tensor Engine saturates and per-NEFF overhead amortizes  │
 ├───────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
 │ **RadialFunction fusion**  │ The [40, 128]→[32]→[32]→[192] MLP is called 4× — fusing all 3 linears + SiLU into one NKI kernel would save 2 HBM round-trips │
 └───────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
 
 The nki_fused_linear kernel is now correct and could be useful in a proper fusion context — but patching F.linear one-at-a-time is the wrong granularity for this model.



 **The pattern is now clear:** Both scatter_reduce and fused_linear failed for the same fundamental reason — patching individual ops inserts separate NEFF dispatches, each reloading a 64KB kernel binary over PCIe.
 The model's tiny shapes (M=40) mean compute is nanoseconds but dispatch is ~1s per call.


 **The only viable next step is fused attention** (bmm + softmax + bmm), because:
 1. It replaces 3 separate dispatches with 1, collapsing the overhead by 3×
 2. attention_tkg from nki-library is designed for exactly this regime (B×S ≤ 96, which fits 40 edges perfectly)
 3. It enables on-chip accumulation — Q, K, V never need to round-trip through HBM between the two matmuls
 


 The aten::bmm calls in the profile are **not** a QKV attention — they're Wigner D-matrix rotations (so3.py:463,471) with shapes [40, 9, 9] × [40, 9, 32]. The EquiformerV3 "attention" is graph-based (GraphSoftmax
 over edges), not a sequence-level QK^T V attention at all. There is no bmm + softmax + bmm pattern in the forward pass — the bmm calls are Wigner rotations and the softmax is a custom scatter-based graph
 softmax over neighbors.
 
 **attention_tkg is the wrong kernel for this model.** It implements transformer token-generation attention (Q/K/V with KV cache, RoPE, block tables) — completely different from what EquiformerV3 does.


 Here's what's actually happening with the 4.5 bmm calls/run at 7.6ms avg:


 ┌──────────────────────────┬──────────────────────────────┬──────────────────────────────────┐
 │ Shape  │ Operation  │ Source │
 ├──────────────────────────┼──────────────────────────────┼──────────────────────────────────┤
 │ [40, 9, 9] × [40, 9, 32] │ Wigner rotation (rotate)  │ so3.py:463  │
 ├──────────────────────────┼──────────────────────────────┼──────────────────────────────────┤
 │ [40, 9, 9] × [40, 9, 32] │ Wigner inverse rotation │ so3.py:471  │
 ├──────────────────────────┼──────────────────────────────┼──────────────────────────────────┤
 │ Others  │ Outer products, cell updates │ heads/rank2.py, equiformer_v3.py │
 └──────────────────────────┴──────────────────────────────┴──────────────────────────────────┘


 These are batched small matmuls (B=40, M=9, K=9, N=32). Each is only 23KB of data — too small to amortize NKI dispatch overhead just like the linear layers were.


 **The right NKI target for this model is the SO2 convolution** (SO2MLinear.forward), which calls F.linear with shapes [40, 2, in_features] where in_features grows with lmax and num_channels — those are the large
 linear ops that dominate. But even those face the same small-M dispatch problem.


 **Honest assessment:** For a NaCl unit cell (2 atoms, 40 edges), this model is too small for any individual NKI kernel to pay off. The only realistic win would be:
 1. **Larger batches** — batch multiple molecules so M ≥ 128 before calling NKI
 2. **Full graph fusion** — compile the entire forward pass as one XLA NEFF (what test_nki_compile.py already does), which is already faster than CPU
 

 yes explore batching multple molecules to hit the threshold where NKI would help
● The benchmark is still compiling (each unique shape compiles a new NEFF which takes time). I'll wait for the notification. While waiting, let me explain the setup and what to expect:

  The benchmark is running test_nki_batched_matmul.py. Here's the experiment structure:
  
  What it tests:
  - 7 layer shapes × 7 batch sizes (B=1 to B=32)
  - Each cell: bench_xla() vs bench_nki() — 5 warmup + 50 timed runs
  - NKI kernel: kernels/matmul_tiled.py — uses nc_matmul with K-tiling for K=192
  
  What crossover to expect:
  - B=1–3: M=39–117, below TILE_M=128 → NKI pads to 128, adding overhead → expect XLA wins
  - B=4: M=156, M_pad=256 → NKI has 2 tiles, XLA has no padding → first fair fight
  - B=16+: M=624+, M_pad same → NKI should saturate Tensor Engine, XLA stays scalar
  
  The critical question is whether the per-matmul_tiled() dispatch overhead is still ~1s (PCIe NEFF reload) or if it amortizes. If each kernel call still costs ~1s of PCIe overhead independently of M, NKI still
   loses for individual layer calls. The only escape is batching multiple layers into one kernel OR using the XLA compiled model where all layers are fused in one NEFF.
    

● The results are definitive. NKI nc_matmul is ~1,200,000 µs on every call regardless of shape — vs XLA's 350–1250 µs. That's a **~3000x slowdown**, and it's flat across all batch sizes, which is the smoking gun:
 **the cost is entirely dispatch/PCIe overhead, not compute**.


 Key observations:
 - NKI time is nearly constant (~1.2s) across all 7 layers × 7 batch sizes — compute time is essentially zero; you're paying pure kernel-launch overhead
 - XLA times range 350–1250 µs and scale with work (as expected for fused ops)
 - This is the same failure mode as the scatter_reduce and fused_linear patches — individually dispatched NKI ops can never win when XLA fuses the whole graph into a single NEFF
 
 **Conclusion:** NKI is not viable as a drop-in op replacement for this model. The right approach is what the script recommends — measure the XLA model's throughput ceiling with test_batch_scaling.py and treat NKI
 as something you'd use inside XLA fusion (custom fused graph), not as patched individual layers.


 Let me update the project memory to reflect this.


 Recalled **2** memories, wrote **2** memories (ctrl+o to expand)


● **Results: NKI lost at every shape and batch size tested.**

 The NKI nc_matmul times were ~1.2 seconds flat across all 49 test cases (7 layers × 7 batch sizes). XLA ranged from 350–1250 µs. The constant NKI time is the giveaway — you're paying PCIe dispatch overhead on
 every call, not compute time.

 This is now three strikes for the individual-op-patching strategy:
 1. nki_scatter_reduce — 1.2s overhead/call
 2. nki_fused_linear (mlp from nki-library) — 4.5× slower
 3. nc_matmul batched — 3000× slower, flat across all sizes
 
 **The architecture conclusion is firm:** XLA fuses the whole model graph into one NEFF. Any individual NKI op patch splits that into separate NEFFs and pays per-call PCIe reload for each one (~64KB kernel binary
 × ~20 calls/forward pass).

 The recommended next step is test_batch_scaling.py — measure where the XLA model's throughput peaks as you increase batch size. That's the number to beat (or accept as the ceiling).


