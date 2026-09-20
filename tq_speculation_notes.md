# Measured FP8 verification bound and TurboQuant source audit

The measured batch-50, 131072-token-history XQA CUDA graph times for ten
full-attention calls were 291.61 ms for q=1, 300.08 ms for q=4, and 295.13 ms
for q=8. Dividing query count by elapsed time gives optimistic attention-only
rates of **3.43, 13.33, and 27.11 tokens/s per stream**. With MTP3, at most
three accepted drafts plus one target token fit the q=4 pass; MTP7 similarly
has at most eight outputs. They cannot reach 40 per stream through this
measured FP8 kernel at these shapes even with 100% acceptance and zero cost
for drafting, the other 30 GDN layers, projections/MoE, vision, sampling and
scheduling. This is evidence about this dense FP8 path, not every possible
compression or attention algorithm. Original harness reference checks were
before graph capture; the updated harness also checks output after graph replay.

Stock vLLM 0.28 **does not give TurboQuant speculation the same explicit
multiquery reuse as XQA**. `TurboQuantMetadataBuilder` sets
`supports_spec_as_decode=False`, and `max_query_len > 1` selects prefill.
For continuation q <= 128, the implementation loops over requests, expands
one request's block-table row q times, supplies incremental sequence lengths
for causality, and calls the ordinary TQ decode kernel. That kernel launches
over `(synthetic query, query head, KV split)`, loading and unpacking K/V
separately in those programs. Hardware caches may service repeated accesses,
so this establishes repeated logical reads and absent explicit query-block
reuse, **not** a guaranteed q-fold increase in actual DRAM traffic. The
standalone `bench_attention_tq.py` follows this per-request loop exactly,
including query rotation and fused unpack/dequantization. Its manually
captured fixed-shape graph is an optimistic kernel experiment, not a claim
that stock serving captures that same continuation path.

The strongest existing native-kernel candidate beyond that TQ path is
**FlashInfer XQA with NVFP4 KV and BF16 Q/output**, with native FA2 continuation
prefill. FlashInfer 0.6.16 contains the NVFP4 XQA API and accepts SM12x; the
packed KV data uses D/2 bytes plus D/16 scale bytes per vector. However,
vLLM 0.28 explicitly gates its NVFP4 FlashInfer integration to SM100, so this
is integration work, not a working stock CLI recipe on GB10. Correct wiring
needs scale layout, separate/compatible data-scale strides, Q/output dtype,
prefill dispatch and graph correctness checks, not merely removal of an
architecture guard. The public consumer-Blackwell work below provides a
concrete starting point. Calibrated scales and exact-model 128K multimodal
quality evaluation are necessary; **no located evidence certifies >=97%
BF16 task-score retention for Qwen3.6-35B-A3B under this recipe**. NVFP4 weight
accuracy results do not establish NVFP4 KV accuracy.

A smaller stock experiment is `--attention-backend TRITON_ATTN
--kv-cache-dtype int4_per_token_head`: its v0.28 packed kernel uses rotated
INT4 K/V and tile reuse across GQA heads and query positions. But with this
model's GQA ratio eight, its default BLOCK_M=16 gives BLOCK_Q=2, so q=4/q=8
still span multiple query tiles/cache scans. It is not a demonstrated route
to the requested throughput or quality. Raising tile size, or adding a
multiquery TQ kernel that preserves the same stored representation, could
reduce reads without introducing additional quantization; both require
new GPU correctness/performance work. None of these untested candidates
should be advertised as retaining 97% until the paired task evaluation passes.

Sources:

- [vLLM 0.28 TQ backend](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/attention/backends/turboquant_attn.py): metadata 253/322, continuation 853–930, CUDA AoS path.
- [vLLM 0.28 TQ decode kernel](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/attention/ops/triton_turboquant_decode.py): separate query/head/split programs and fused quantized reads.
- [FlashInfer 0.6.16 XQA](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16/flashinfer/xqa.py): uint8 packed NVFP4 with scale tensors, SM12x capability gate.
- [vLLM 0.28 FlashInfer](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/attention/backends/flashinfer.py#L517): NVFP4 SM100 restriction.
- [Consumer-Blackwell NVFP4 integration PR 46329](https://github.com/vllm-project/vllm/pull/46329): ongoing implementation and calibration/layout/capture work, primarily Gemma validation; not exact-target evidence.
- [Qwen hybrid prototype report 49011](https://github.com/vllm-project/vllm/issues/49011): author-reported RTX5090 Qwen3.6-27B prototype, not our GB10 or model; useful engineering details, no 97% evaluation.
- [vLLM 0.28 packed INT4 kernel](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/attention/ops/int4_per_token_head.py): rotated cache and BLOCK_Q default near lines 721–727.
