# Qwen GDN state-update benchmark

`bench_gdn.py` calls the installed public vLLM FLA functions directly. The
exported installed Qwen layer source is byte-identical to the v0.28.0 tag
(SHA256 `da3c9bb565a740e9dc32c4c019b45b57015ebfa958f9900910b81acaa32937b9`).
For this model's 16 key heads and 32 value heads, ordinary q=1 decode uses
`fused_recurrent_gated_delta_rule_packed_decode` (Triton). Speculative q>1
uses `fused_sigmoid_gating_delta_rule_update` (Triton). The separate fused
CUDA MTP route requires value/key-head ratio eight; ratio two cannot select
it. A server log saying `GDN decode kernel: cuda` does not override this
per-call shape gate.

The benchmark generates BF16 Q/K/V after convolution and BF16 gating inputs,
with FP32 A_log and native FP32 recurrent state. It uses valid positive state
indices and chooses the last previously accepted checkpoint as initial state.
Every speculative query writes its own complete state checkpoint. States
remain FP32 inside the recurrence, even when the optional BF16 checkpoint
format is selected. The independent reference reproduces that behavior and
the q=1 packed path's BF16 sigmoid rounding. Kernel output and every first-
request checkpoint are checked. A captured block is also compared with eager
execution from identical initial state.

One request/layer state contains `32 × 128 × 128 × 4 = 2097152` bytes (2 MiB).
The FP32 batch-50 minimum state payload over thirty layers is
`50 × (1 + q) × 2097152 × 30` bytes: one initial read and q checkpoint writes.
For q=1/4/8/16 that is 6.291/15.729/28.312/53.477 decimal GB per target pass.
At advertised 273 GB/s alone, these amounts take at least approximately
23.0/57.6/103.7/195.9 ms, before input reads, redundant loads, arithmetic or
other layers. Real state-kernel timing, not these bandwidth estimates, is
what the harness measures. Its effective payload bandwidth is not a DRAM
counter measurement.

Only one layer's state pool is allocated and reused for thirty serial calls;
at batch50 q16 it is approximately 1.56 GiB plus scratch. Each request has
distinct slots. The twelve-GiB Torch allocator cap and an explicit free-memory
check bound allocations. This omits full-model capacity, real learned input
distributions, convolution, projections, output RMS norm/gate, MoE, full
attention, drafting, sampling and vision. BF16-state runs change precision and
cannot inherit the FP32 accuracy claim. Adding separately measured full-
attention and state-kernel times is an optimistic projection for this serial
execution path, not a substitute for the complete model's simultaneous-stream
test or an architecture-independent impossibility proof.

Primary source:

- [Qwen GDN dispatch, v0.28.0](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py): non-spec core 1632–1684, fused MTP gate 1798–1812.
- [Packed recurrent implementation](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/third_party/flash_linear_attention/ops/fused_recurrent.py): packed path, beta rounding and in-place state update.
- [Speculative fused sigmoid recurrence](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py): accepted-state selection and per-token checkpoint stores.
- [Official packed-kernel test](https://github.com/vllm-project/vllm/blob/v0.28.0/tests/kernels/test_fused_recurrent_packed_decode.py): representative grouped value heads and comparison to explicit gating/reference recurrence.
