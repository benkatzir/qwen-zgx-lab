# Measured attention and recurrent-state bottlenecks

The tested kernels do not provide a demonstrated path to **45–50 independent
128K streams at 40 output tokens/s each**. At concurrency 45, the best tested
native NVFP4 attention page size yields an optimistic **25.70 tokens/s/stream
with eight accepted positions per pass**, or **27.72 with sixteen**, after
adding only the model's FP32 recurrent-state updates. Model weights, input and
output projections, MoE, convolution, output normalization/gating, drafting,
sampling, vision and scheduling are still excluded.

These are **projections from separately measured kernel blocks, not measured
end-to-end model throughput or a proof against every possible implementation**.
They show why an attention-only figure above 40 does not satisfy the request.
The real model benchmark and the >=97% BF16-relative task-quality evaluation
remain separate requirements.

All measurements below come from the user's NVIDIA GB10, compute capability
12.1, with PyTorch 2.13.0+cu130, FlashInfer 0.6.16.post3 and vLLM 0.28.0. Attention
used 131072 historical tokens per independent request, plus the verification
positions, BF16 queries, 16 query heads, 2 KV heads and dimension 256. The
recurrent benchmark used 32 value heads, 16 key heads, 128×128 matrices and
native FP32 state, including each speculative token's checkpoint write.

| Path and shape | Ten attention calls | Thirty FP32 GDN updates | Sum of measured medians | Optimistic accepted tokens/s/stream |
|---|---:|---:|---:|---:|
| NVFP4 XQA, C45, q8, page128 |190.033ms|121.224ms|311.257ms|**25.70**|
| NVFP4 XQA, C45, q16, page128 |341.777ms|235.379ms|577.157ms|**27.72**|
| NVFP4 XQA, C50, q8, page16 |276.323ms|133.835ms|410.158ms|19.50|
| NVFP4 XQA, C50, q16, page16 |470.507ms|261.365ms|731.872ms|21.86|
| FP8 XQA, C45, q4, page16 |256.129ms|65.015ms|321.144ms|12.46|
| FP8 XQA, C50, q4, page16 |303.998ms|72.471ms|376.470ms|10.63|
| Stock TQ4, C50, q4, page128 |5065.183ms|72.471ms|5137.655ms|0.78|

`q` is target verification length: q4 corresponds to at most three accepted
draft tokens plus one target token; q8/q16 similarly allow at most eight/sixteen
outputs per pass. The projection is `q × 1000 / (attention_ms + GDN_ms)` and
assumes **every** position becomes an accepted output. To produce 40 per stream,
the entire pass must finish within 100 ms at q4, 200 ms at q8, or 400 ms at q16.
Measured attention plus GDN alone already exceeds those budgets in the table.
Lower draft acceptance and the omitted model work worsen the projection.

Page size 128 was the best tested NVFP4 attention layout for C45 q8/q16. Page size 64 was
close: adding the same GDN measurements gives 25.55 and 27.64 tokens/s/stream.
C50 results above use page size 16; they must not be presented as the best attainable
C50 page-size tuning. The initial NVFP4 sweep passed all 12 shapes, followed by
four additional successful C45 page-size shapes.

Correctness checks were independent of the timed attention kernels: manual
NVFP4 unpacking followed by FP32 causal GQA attention, tested on the first,
middle and last requests where available. Checks covered eager execution,
graph replay, changed persistent queries/sequence lengths and restored full
context. All passed; the largest relative RMSE across these NVFP4 checks was
0.003019 against the **same quantized** KV values. All eight FP32 GDN shapes
passed an independent Torch recurrence checking every first-request checkpoint,
plus an eager-versus-graph comparison from identical starting state. Maximum
GDN state error was 2.385e-7. These validate tested kernel semantics; they do
**not** establish task accuracy, or >=97% retention versus unquantized BF16.

The GDN source exported from the installed image is byte-identical to the
v0.28.0 release. Its ordinary decode uses the packed Triton recurrence; this
model's value/key-head ratio 2 fails the CUDA MTP path's ratio 8 condition, so
speculation uses the sigmoid-gating Triton recurrence. The benchmark follows
those paths and writes a full state checkpoint for every verification token.
TurboQuant's q4 test likewise follows the stock per-request continuation loop,
whose queries independently load/unpack the compressed cache; it does not
replace that loop with a more favorable flattened batch.

Each microbenchmark reuses one layer's buffer across the required layer count
to stay below 12 GiB; every request still has distinct KV/state slots. Timings
are CUDA-event medians after warmup, using fixed-shape graph replay. Reusing
buffers and omitting intervening model work can change cache, power and launch
behavior. Adding the two measured medians models serialized kernel work at
matching batch/query shapes; it is not a measurement of the full integrated
model. Nominal payload bandwidth in the JSON is not a hardware-counter DRAM
measurement. The [orchestration script](micro_remaining.sh) runs each GPU
container in the foreground and waits before starting the next; the
[full-model launch script](run_full45.sh) waits for that process to exit and
its completion marker before starting the server. The operator confirmed no
other serving or GPU work overlapped. A separate source-file read used a
GPU-free container. Thus the GDN, tuned NVFP4 and checked FP8 timings above
were collected sequentially, not in competition for the GPU.

For context only, switching recurrent checkpoints to BF16 halves their byte
size but changes model precision and has **not** been quality-certified here.
An idealized memory-only calculation at NVIDIA's advertised 273 GB/s gives
46.67 ms for C45 q8 or 88.15 ms for C45 q16, using
`45 × 30 × (1+q) × 32 × 128 × 128 × 2` bytes. Combining those unmeasured ideal
state times with measured page size 128 NVFP4 attention gives 33.80 or 37.22 tokens/s,
still before all other model work. This calculation assumes the state payload
traverses unified memory and ignores cache effects and all compute/overhead;
it is a bandwidth-model scenario, **not a measured BF16-state result or a
universal performance bound**. [NVIDIA DGX Spark specifications](https://www.nvidia.com/en-us/products/workstations/dgx-spark/).

Raw evidence:

- [NVFP4 page128 results](results/attention-nvfp4-page128.json), [page64 results](results/attention-nvfp4-page64.json), [initial twelve-shape NVFP4 sweep](results/attention-128k-nvfp4.json).
- [FP32 GDN results](results/gdn-fp32.json), [FP8 XQA with graph checks](results/attention-xqa-checked.json), [stock TQ4 results](results/attention-128k-tq4.json).
- Harnesses: [NVFP4 attention](bench_attention_nvfp4.py), [GDN](bench_gdn.py), [stock TQ](bench_attention_tq.py), [FP8 attention](bench_attention.py).
- Source audit: [vLLM0.28 Qwen GDN dispatch](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py), [TQ continuation path](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/attention/backends/turboquant_attn.py), [FlashInfer XQA](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16/flashinfer/xqa.py).
