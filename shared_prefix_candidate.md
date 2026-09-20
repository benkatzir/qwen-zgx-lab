# Conditional shared-prefix candidate — untested

Source audit: vLLM 0.28.0, Qwen3.6-35B-A3B multimodal hybrid GDN, FLASHINFER on GB10, MTP3. No GPU experiment was run for this candidate.

Exact token-prefix sharing is a valid cache optimization that reuses already computed model state without adding a new quantization or approximation. It can reduce memory and prefill work when all requests share the same document/system prompt. It does not establish capacity or throughput for arbitrary independent 128K histories, and it does not repair quality loss from other parts of the serving stack.

## Finding: supported prefix cache, disabled cascade

vLLM's `FlashInferMetadataBuilder.use_cascade_attention()` unconditionally returns `False`; the source comment says cascade is disabled because it does not work. It separately rejects different query/cache dtypes, which would exclude the current BF16-query/FP8-KV combination anyway. Although cascade wrapper construction remains in the file, normal runtime selection never chooses it. No POD execution path was found in this backend.

Consequently enabling prefix caching does **not** activate supported shared-prefix cascade computation on this stack. Requests still execute their own attention queries. Sharing physical KV pages may improve hardware-cache locality, but that is a measurement question, not a promised multiplier.

[Disabled cascade selection](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/attention/backends/flashinfer.py#L1722), [inactive cascade planning implementation](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/attention/backends/flashinfer.py#L1473).

## Hybrid/MTP compatibility

Qwen3.5 model code is the exact architecture used by this Qwen3.6 checkpoint. Both target and MTP classes reject `mamba_cache_mode=all` and direct users to `align`. Enabling prefix caching selects align by default; align requires chunked prefill. The v0.28 V1 runner has a specific MTP/EAGLE hybrid align path that copies the accepted recurrent state after verification.

Shared prefix states can initialize each request, but divergent suffixes still need separate GDN states and speculative checkpoints. Align also needs one more active state slot per request/layer than cache mode none, before allocator effects.

- [Target cache-mode restriction](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/models/qwen3_5.py#L322)
- [MTP restriction](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/models/qwen3_5_mtp.py#L225)
- [Prefix-cache/align configuration](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/models/config.py#L603)
- [Hybrid speculative accepted-state handling](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/worker/gpu_model_runner.py#L1614)

## Minimal conditional launch change

Keep the validated target checkpoint, FP32 GDN state, FP8 KV, FLASHINFER, MTP3, and vision encoder. Replace `--no-enable-prefix-caching` with:

```bash
--enable-prefix-caching \
--mamba-cache-mode align \
--enable-chunked-prefill
```

Keep block-size selection automatic initially. Do not patch cascade on or switch to `mamba_cache_mode=all`.

If tested later:

1. Warm one request with the exact common 131,072-token prefix and generate one token. Wait for completion so the prefix is available before starting the batch.
2. Start 45 requests containing those exact same prefix token IDs plus distinct short suffixes. Do not use different per-request cache salts. Token-ID completion prompts make the common prefix unambiguous; equivalent rendered text is not sufficient if tokenization/templates differ.
3. Record actual cached token counts/hit metrics, simultaneous decoding concurrency, per-stream timing, and peak memory. Cache alignment can leave a small common tail to recompute; report actual hits rather than assuming every prefix token was reused.
4. Label any result **45 requests sharing one 128K document**. It is not an independent-history result or evidence for arbitrary workloads.

Priority is low: root already measured the MTP3 short-2K C50 run at minimum 17.42 and mean 19.41 tokens/s per stream. Prefix reuse alone therefore has no compelling existing evidence for achieving 40 tokens/s at C45–50. This candidate remains untested and should not delay the actual independent-128K concurrency measurements.
