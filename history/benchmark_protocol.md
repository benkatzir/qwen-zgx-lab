# Acceptance protocol: Qwen3.6-35B-A3B on one GB10

Status: research and a proposed test contract, not measured results. Prepared 2026-09-18. No access to the workstation was used for this audit.

## What counts as success

- One physical GB10 with 128 GB unified memory. Record exact device, usable memory, driver, CUDA, power/temperature, container digest, server commit, model revision and every launch argument.
- Evaluate both 45 and 50 independent, simultaneously decoding requests. The stricter final target is 50. A server accepting 50 queued requests is not 50 active decodes.
- Each request contains at least 131,072 model tokens before decoding, including chat-template and image/video tokens. Do not satisfy the requirement by configuring `max_model_len=131072` and benchmarking a 1K prompt. Keep the configured model limit at least prompt plus output, preferably the native 262,144.
- Preserve supported image/video inputs, vision weights, processor, native reasoning mode, tools/chat template and production generation settings. Do not use a language-model-only flag. Multimodal capability and mixed multimodal load require separate evidence.
- Every stream sustains at least 40 accepted output tokens/second while all target streams are actively decoding. At 50 streams this also implies at least 2,000 aggregate accepted output tokens/second. Neither aggregate throughput alone nor a single-stream result proves this.
- Treat the explicit 97% minimum as the binding quality threshold. Define it as task-score retention against the unquantized official BF16 checkpoint with BF16 KV. This is not 97% exact token agreement and not 97% absolute task accuracy. A BF16 score of 80 means the candidate floor is 77.6 under a relative-score definition. The user's parenthetical “90+ preferred” conflicts with the explicit 97% minimum and should not be used to silently weaken it.

## Capacity and bandwidth audit

The official config has 40 layers, with full attention every fourth layer (10 layers), two KV heads, and head dimension 256. The other 30 layers have recurrent GatedDeltaNet states. Full-attention KV cache for one 128Ki-token sequence is:

`131072 * 10 * 2(K,V) * 2 KV heads * 256 dimensions * bytes_per_element`.

| KV bits | KV GiB/stream | KV GiB/50 streams | Minimum cache read GB/s at 50 x 40 tok/s | Ideal tok/s/stream at 50 streams and 273 GB/s |
|---|---:|---:|---:|---:|
| 16 | 2.500 | 125.000 | 5368.7 | 2.03 |
| 8 | 1.250 | 62.500 | 2684.4 | 4.07 |
| 4 | 0.625 | 31.250 | 1342.2 | 8.14 |
| 3 | 0.469 | 23.438 | 1006.6 | 10.85 |
| 2 | 0.313 | 15.625 | 671.1 | 16.27 |
| 1 | 0.156 | 7.813 | 335.5 | 32.54 |

These are optimistic *single-token dense decode* bounds, with perfectly packed payloads, no metadata, no model-weight reads, no recurrent state traffic, no output KV writes, no CPU bandwidth competition and 100% advertised memory bandwidth. They are not universal impossibility proofs: speculative verification can reuse KV for several output positions and sparse attention can avoid reads. At 45 streams the bandwidth requirement is 90% of the table. Weights, recurrent state, vision memory, graph pools, activation workspaces and cache allocation padding must be added to capacity. The recurrent matrix alone is about 60 MiB/stream if all 30 layers store 32 x 128 x 128 FP32 values; actual runtime allocations can be larger.

At 50 streams the required cache-reuse factor is at least 9.83 with FP8, 4.92 with ideal 4-bit, or 2.46 with ideal 2-bit. This assumes zero other traffic, so a deployable solution needs materially more reuse or sparsity. A k=3 speculative draft can produce at most four accepted tokens per verification step, even with perfect acceptance. At full attention and FP8 cache, that ideal is still insufficient for 45-50 streams at 40 tok/s. Rejected draft tokens never count toward throughput.

Full-attention arithmetic alone is approximately `4 * 131072 * 10 * 16 query_heads * 256 = 21.475 GFLOPs` per output token. At 50 x 40 this is 42.95 TFLOP/s, excluding projection/MLP/recurrent work, quantization and softmax. Compare with the measured precision-specific attention kernel, not the advertised sparse FP4 peak. Speculation reduces repeated memory reads; it does not remove the target arithmetic for every verified output position.

## Performance stages

1. Run a smoke test at short context, including text, image OCR and a short video. Record the actual selected kernels. Reject unsupported fallback or silently disabled modalities.
2. Sweep active concurrency 1, 2, 4, 8, 16, 32, 45, 50 at actual prompt lengths 4K, 32K, 64K, 128Ki. Use independent prompt bodies and an early request-specific nonce. Disable shared-prefix optimization in the independent test, or prove prefix overlap is negligible. Report any separate shared-prefix experiment as a different workload.
3. For each case record actual prompt/completion counts, TTFT, accepted output tokens, per-stream decode time, inter-token latency, active decoding count, queued count, preemptions, swapping, KV occupancy, memory, thermal/power and speculation acceptance. Timestamp with a monotonic clock. Token counts should come from server token IDs or usage; streamed chunks may contain multiple tokens and are not token counts.
4. The critical test needs at least a 60-second common interval in which all 45/50 requests are actively decoding at full prompt depth. Prefill all independent contexts before the measurement when the serving stack permits, or prime and retain each unique prefix then submit the measured continuations. Verify that this warm test really has retained all contexts and never label its latency as cold-start latency. A separate simultaneous cold-start burst measures prefill and admission performance.
5. Use sufficiently long outputs (initially 8192 tokens; increase if needed) so early-finishing requests cannot give late survivors artificially high speeds. For the performance stress case only, allow a fixed output length/ignore-EOS setting and disclose it. Accuracy tests must honor normal stopping. A performance workload that is merely repeated predictable text can inflate speculative acceptance; include representative code, prose, reasoning and visual-answer workloads.
6. Compute a per-request rate on the common interval, not merely total tokens divided by run wall time. Also report conventional decode rate `(output_tokens-1)/(last_token_time-first_token_time)` and end-to-end rate including TTFT, clearly labeled. Strict pass requires the minimum common-interval rate to be at least 40 tok/s, no failures and no silent truncation. Report p50/p95/p99 inter-token latency so long stalls are visible. “p95 stream rate >=40” is weaker than “each >=40”.
7. Repeat the final target at least three times with independent prompt sets after warmup, and perform a sustained run with replacement requests to expose thermals and scheduling/preemption. A short best run is not a sustained service result. Separately report whether ongoing 128K prefills reduce the decode service level.

## Quality stages

Use the official BF16 base as the reference, one request at a time if necessary. Preserve the exact checkpoint revision, processor, image resolution/frame sampling, chat template, reasoning setting, sampling settings, maximum answer length and evaluation code. Do not compare a candidate with thinking disabled to a differently configured published base score.

Suggested candidate matrix, proceeding only when the previous stage survives:

1. Official BF16 weights + BF16 KV, no speculative decoding (reference).
2. Official FP8 weights + BF16 KV, then FP8 KV (separate weight and cache effects).
3. Quality-focused NVFP4 weights + BF16 KV, then FP8 KV.
4. Add exact speculative decoding to the strongest qualified candidate; verify the rejection/sampling implementation and count only accepted tokens.
5. Experimental 4/2-bit cache or sparse attention only with independent calibration and a fresh held-out quality suite. Test its interaction with weight quantization and speculation; independent quality scores do not multiply into a guarantee.

Quality coverage should include:

- Text/reasoning: GPQA Diamond, MATH-500 or other held-out math; coding with actual unit tests such as LiveCodeBench; instruction following and the user's intended tool-calling workload.
- Full-context: RULER's several task families (multi-needle, tracing, aggregation), multi-round context retrieval and representative 128K documents/codebases with multi-hop questions. Needles placed at many depths; many distinct prompts and seeds. A single exact passkey is a functional probe, not evidence of 97% general quality.
- Vision: OCR/document/chart and visual reasoning benchmarks using supported image input. Test text-only, image-only and mixed long-text-plus-image cases. Include video if preservation of video support is required.
- Long generation: substantial actual decoding to exercise cache quantization. A one-shot teacher-forced perplexity pass may not test the quantized incremental KV cache at all.

Predeclare the benchmark mix and aggregation. Report every domain and task, not just an average that can hide a coding or vision regression. The conservative contract is >=97% retention in every critical domain as well as the aggregate. If only an aggregate is required, make that explicit. Keep calibration and tuning sets disjoint from final evaluation.

For evidence beyond a point estimate, run paired comparisons and compute a one-sided 95% lower confidence bound on `candidate_score - 0.97 * BF16_score`, resampling original evaluation items and clustering repeated seeds by item. Pass only if the lower bound is nonnegative. If multiple domains must individually pass, use simultaneous intervals or a multiplicity correction. Small 30-100-question smoke tests cannot certify a 3% relative margin; sample size depends on baseline score, candidate loss and paired disagreement rate. If the candidate only equals the 97% boundary in truth, finite samples cannot reliably establish a positive lower bound. Use generous quality margin and report uncertainty.

## Research findings and limits

- NVIDIA's NVFP4 card is the stronger starting quality candidate: its eight published text, coding, long-context and multimodal metrics are all at least 99.16% of its BF16 reference scores. These were measured on GB300; they do not prove this exact GB10 runtime, KV scheme, or 128K concurrent workload. Its supplied Spark example uses FP8 KV and MTP3 but only four maximum sequences. Unsloth Fast separately reports three near-BF16 text metrics and an SM121-specific serving path; the large “Original Qwen3.6 BF16 Reference Benchmarks” table on its page is not a quantized-model evaluation.
- The Red Hat NVFP4 model card reports strong recovery on several tasks, but LiveCodeBench V6 is 74.67 versus base 77.33 (96.55% retention). BFCL overall is 97.01%. This is evidence for trying that quant, not an all-task 97% certificate, and does not certify FP8/low-bit KV or 128K multimodal workloads.
- OSCAR is a promising INT2 cache candidate. Its authors' repo includes an experimental Qwen3.5 hybrid branch and a separate VL branch. Published Qwen3.5-35B-A3B GPQA results are 82.32 vs 80.30. No reviewed evidence here proves Qwen3.6-35B-A3B on GB10 with multimodality, 128K, and high concurrency. Its reported effective cache precision is about 2.28 bits including residual windows/metadata, not the optimistic 2.0-bit row above. Rotations should be calibrated for the exact target checkpoint.
- TurboQuant illustrates why bit counts are not enough. vLLM's May study found accuracy degradation for aggressive 3-bit settings on older Qwen models and slower end-to-end serving despite capacity gains. The May statement that hybrid models were unsupported is stale: current vLLM source includes hybrid handling. Availability must be checked against the actually installed version and SM121 kernels.
- Query-aware sparse attention (e.g. Quest) can reduce bandwidth beyond scalar KV quantization while retaining all stored cache pages, but approximate attention is workload dependent. The authors' benchmarks on other architectures do not establish the requested quality guarantee or drop-in hybrid/multimodal support.
- Exact speculative sampling preserves the target model distribution in theory. It does not turn a quantized or sparsified target back into the BF16 reference, and it provides workload-dependent accepted lengths. Unverified draft-only generation, approximate acceptance, context summarization or retrieval truncation are changed-quality methods.

## Primary sources

- Model configuration: https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json
- Model card and modality/context usage: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- NVIDIA hardware specification: https://www.nvidia.com/en-us/products/workstations/dgx-spark/
- Red Hat NVFP4 recipe and quality: https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-NVFP4
- NVIDIA NVFP4 recipe and quality: https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4
- Unsloth NVFP4 Fast recipe and quality: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4-Fast
- OSCAR paper: https://arxiv.org/abs/2605.17757
- OSCAR implementation/status: https://github.com/FutureMLS-Lab/OSCAR
- vLLM TurboQuant evaluation: https://vllm.ai/blog/2026-05-11-turboquant
- Current vLLM TurboQuant source: https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/turboquant/
- KIVI paper: https://arxiv.org/abs/2402.02750
- Quest paper and implementation: https://arxiv.org/abs/2406.10774 and https://github.com/mit-han-lab/Quest
- Exact speculative decoding: https://arxiv.org/abs/2211.17192
- RULER: https://github.com/NVIDIA/RULER
