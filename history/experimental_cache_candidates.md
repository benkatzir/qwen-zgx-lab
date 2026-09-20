# Experimental low-bit cache candidates for one GB10

Source audit: 2026-09-18. These are **not hardware-verified results**. The ZGX became unreachable during setup; no claim below establishes the user's complete target on that machine.

Target: 45–50 simultaneously generating independent, occupied 131,072-token contexts, at least 40 generated tokens/s for every stream, image/video support, and at least 97% of BF16/base task accuracy. Configured maximum context, short prompts, shared prefixes, aggregate throughput, and quantization-only quality scores do not establish that target.

## Conclusion and experiment order

1. Stock vLLM 0.28.0 + Unsloth NVFP4-Fast + TurboQuant 4-bit is the most bounded low-bit-cache experiment. It preserves the model architecture and needs no OSCAR fork. Try no speculation, then MTP 3, then MTP 8 only if measured acceptance and latency justify it.
2. Stock TurboQuant 3-bit is a useful capacity/bandwidth experiment, but is a substantially riskier accuracy candidate. Source comments report considerably worse perplexity than the 4-bit preset on other models. There is no Qwen3.6 long-context/multimodal 97% result.
3. OSCAR hybrid INT2 offers better cache compression and promising adjacent-model quality, but **its audited hybrid implementation explicitly excludes speculative decoding**. Combining OSCAR flags with MTP or DFlash does not produce the intended hybrid INT2 pool. It also needs model-specific calibration and SM121 validation. This is an engineering project, not an already available recipe for the full target.

No audited candidate presently proves the complete target. The available evidence also does not justify claiming a mathematical impossibility for every future custom kernel.

## Exact architecture and rough cache sizes

The pinned Fast checkpoint, revision `1c3f884bc99aac2524f6d49bcbac8c88401afd66`, declares `Qwen3_5MoeForConditionalGeneration` / `qwen3_5_moe`, despite its Qwen3.6 marketing name. It has 40 layers, 30 linear-attention layers, and 10 full-attention layers at indices 3,7,11,15,19,23,27,31,35,39. Full attention has 16 query heads, 2 KV heads, head dimension 256. It has one MTP hidden layer and an intact BF16 vision tower. Consequently Qwen3.5 hybrid model code is relevant to this exact checkpoint; a missing `qwen3_6` Python class is not itself a blocker.

[Pinned checkpoint config](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4-Fast/blob/1c3f884bc99aac2524f6d49bcbac8c88401afd66/config.json)

Approximate full-attention history allocation for 50 independent 131,072-token contexts, excluding weights, Mamba states/checkpoints, workspace, allocation padding, and other overhead:

| Cache | Bytes per KV-head pair per position | Total history, decimal GB |
|---|---:|---:|
| BF16 | 1,024 | 134.22 |
| FP8 | 512 | 67.11 |
| TurboQuant k8v4 | 388 | 50.86 |
| TurboQuant 4bit_nc | 262 | 34.34 |
| TurboQuant 3bit_nc | 198 | 25.95 |
| OSCAR INT2, group 128, FP32 scale/zero | 160 | 20.97, plus BF16 windows |

Formula: `50 * 131072 * 10 full-attention layers * 2 KV heads * slot_bytes`. OSCAR has 128 packed data bytes plus 32 scale/zero bytes per head pair, rather than a metadata-free 128 bytes. These numbers describe history capacity, not throughput or total process memory.

## OSCAR hybrid source audit

Audited repository: `FutureMLS-Lab/OSCAR`, branch `zhongzhu/hybrid-model`, commit **`19f85e13059de3da60686af4cbdd778b5671d9ff`**. Read-only clone is in `work/oscar-hybrid-source/`. The repository includes a vendored `sglang-research` tree; installing ordinary current SGLang is not equivalent to installing this code.

### Speculation is explicitly gated out

`model_runner_kv_cache_mixin.py:623–635` builds the hybrid mixed INT2 pool only if `self.server_args.speculative_algorithm is None`. The server's `_unified_mixed_kv_active()` duplicates that restriction. This excludes both NEXTN/MTP and DFLASH. Their source modules being present in the repository does not remove this gate. Removing only the condition is unsafe: allocator lifecycle, speculative writes, accepted-token commit/rollback, BF16 recent-window aging, and Mamba state rollback must agree.

- [Pinned pool construction](https://github.com/FutureMLS-Lab/OSCAR/blob/19f85e13059de3da60686af4cbdd778b5671d9ff/sglang-research/python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py#L623)
- [Pinned server argument gates](https://github.com/FutureMLS-Lab/OSCAR/blob/19f85e13059de3da60686af4cbdd778b5671d9ff/sglang-research/python/sglang/srt/server_args.py#L2640)
- [Upstream integration PR 32129](https://github.com/sgl-project/sglang/pull/32129) was open/unmerged when inspected and describes speculative decoding as unsupported for the mixed HP windows path.

### Multimodal architecture is present, but still needs an image/video test

The branch's `Qwen3_5MoeForConditionalGeneration` inherits `Qwen3VLForConditionalGeneration`. OSCAR absorption is called from the conditional-generation loading path too. Thus the branch does not inherently require discarding the vision tower. However, the supplied hybrid evaluation is text GPQA; it does not establish image/video quality with INT2 KV.

Its calibration launcher notes a vision-routing/page-size issue with FA3 and uses page size 1 for BF16 dumping. Mixed INT2 requires page size 8. This is an additional reason to explicitly test actual image/video requests rather than infer multimodality solely from successful text loading. Use Triton prefill/decode on GB10; do not copy the H100 FA3-prefill recipe unchanged.

[Conditional model and absorption code](https://github.com/FutureMLS-Lab/OSCAR/blob/19f85e13059de3da60686af4cbdd778b5671d9ff/sglang-research/python/sglang/srt/models/qwen3_5.py#L1630)

### Unsloth Fast cannot use the supplied absorption setting unchanged

`_maybe_absorb_oscar_v_rotation_qwen35()` explicitly supports only dense BF16/FP16/FP32 `qkv_proj` weights and raises on FP8. The current Fast checkpoint's dense attention projections are FP8. Set `SGLANG_OSCAR_ABSORB_V_ROTATION=0` for a first attempt; otherwise implement and validate FP8 folding/requantization or select a checkpoint with dense attention. Disabling absorption keeps runtime rotation work and may cost speed. The provided model-specific eval script force-exports absorption=1, so it cannot simply be used unchanged for Fast.

[Exact absorption restriction](https://github.com/FutureMLS-Lab/OSCAR/blob/19f85e13059de3da60686af4cbdd778b5671d9ff/sglang-research/python/sglang/srt/models/qwen3_5.py#L898)

### Calibration and quality evidence

The public RotationZoo includes Qwen3.5-35B-A3B, not Qwen3.6-35B-A3B. Identical dimensions do not make learned rotations interchangeable. Dump actual Qwen3.6 post-RoPE Q/K/V at the ten full-attention layers and compute its rotations, preferably using representative text and vision inputs with a separate evaluation holdout. The supplied dump script permits changing `MODEL` and TP size, and the compute script supplies head dimension 256 and the correct ten layer IDs. It defaults to four GPUs; change that explicitly for a single GB10.

The author reports GPQA BF16 80.30% versus OSCAR 82.32% for **Qwen3.5**-35B-A3B. This is promising but is one adjacent-model test, not evidence of 97% retention for Qwen3.6, 128K retrieval, multimodality, or the combined NVFP4+INT2 stack.

- [OSCAR README and reported evaluations](https://github.com/FutureMLS-Lab/OSCAR)
- [RotationZoo](https://huggingface.co/Zhongzhu/OSCAR-RotationZoo)
- [Hybrid calibration dump](https://github.com/FutureMLS-Lab/OSCAR/blob/19f85e13059de3da60686af4cbdd778b5671d9ff/rotation/qwen3.5-35B-A3B/save_qkv_qwen35_35b.sh)
- [Layer-specific rotation computation](https://github.com/FutureMLS-Lab/OSCAR/blob/19f85e13059de3da60686af4cbdd778b5671d9ff/rotation/qwen3.5-35B-A3B/compute_rotation.sh)

### Isolated candidate command, after building and calibrating

This is a **source-informed, untested skeleton**, not a ready verified installer. Pin the audited commit in a separate environment. Check ARM64 CUDA/Triton/FlashInfer/SM121 compatibility and compressed-tensors NVFP4 loading before allocating a large cache. Do not run repository scripts blindly: some contain machine-specific environment activation and cleanup commands.

```bash
env \
 SGLANG_ENABLE_MIXED_KV_WINDOWS=1 \
 SGLANG_LLOYD_MAX=1 \
 SGLANG_OSCAR_K_ROTATION_PATH=/calibration/qwen36/k_rotation.pt \
 SGLANG_OSCAR_V_ROTATION_PATH=/calibration/qwen36/v_rotation.pt \
 SGLANG_OSCAR_K_CLIP_RATIO=0.96 \
 SGLANG_OSCAR_V_CLIP_RATIO=0.92 \
 SGLANG_OSCAR_ABSORB_V_ROTATION=0 \
 SGLANG_MIXED_KV_PREFIX_TOKENS=64 \
 SGLANG_MIXED_KV_RECENT_TOKENS=256 \
 SGLANG_MIXED_KV_HP_DTYPE=bfloat16 \
 SGLANG_MIXED_KV_SCALE_DTYPE=float32 \
 python -m sglang.launch_server \
 --model-path /models/pinned-qwen36-nvfp4-fast \
 --tp-size 1 --kv-cache-dtype int2 --kv-cache-quant-group-size 128 \
 --prefill-attention-backend triton --decode-attention-backend triton \
 --mamba-scheduler-strategy extra_buffer --page-size 8 \
 --max-running-requests 64 --context-length 132096 \
 --chunked-prefill-size 4096 --mem-fraction-static 0.85 \
 --host 127.0.0.1 --port 30000
```

Do not add a speculative algorithm. Do not disable radix cache with `extra_buffer`; the model-specific source comments say that combination raises. Use unique prefixes to avoid prefix-cache inflation of the independent-context benchmark. The pool reserves extra BF16 prefix capacity based on request slots, so bounding max-running-requests matters for real memory usage.

## Stock vLLM 0.28.0 TurboQuant and deep MTP

The audited backend supports head dimension 256 and four presets. It uses a combined packed K/V slot, not the conventional two-buffer cache shape. The default hybrid model treatment does not skip the first/last two attention layers, unlike its aggressive presets on dense models.

- [v0.28.0 TurboQuant backend](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/attention/backends/turboquant_attn.py)
- [Quantization preset configuration](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/layers/quantization/turboquant/config.py)
- [MTP configuration](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/config/speculative.py)

Experiment changes to the baseline recipe in `serving_recipe_candidates.md`:

```bash
--attention-backend TURBOQUANT --kv-cache-dtype turboquant_4bit_nc
# or, for the more aggressive accuracy experiment:
--attention-backend TURBOQUANT --kv-cache-dtype turboquant_3bit_nc
# separately benchmark no speculation, then:
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
# then, only as an experiment:
--speculative-config '{"method":"mtp","num_speculative_tokens":8,"moe_backend":"triton"}'
```

Important enabling conditions and limits:

- Qwen3.6 has one MTP layer. The vLLM config permits eight drafts through repeated reuse of that layer; this does not mean there are eight independent trained prediction heads. The source warns that repeated forward passes can lower acceptance.
- TurboQuant metadata sets `supports_spec_as_decode=False`. Verification with multiple query tokens goes through the prefill path.
- Continuations with query length at most 128 use the quantized decode kernel with synthetic per-query sequence lengths, so a 9-query verification block from MTP8 has a conceivable supported path. The source loops over requests and treats each verification query as a decode request. It does **not** establish an optimized multi-query verification kernel that reads the long KV history once and amortizes it across all accepted tokens.
- Large continuation chunks dequantize the cached history into FP16 scratch and use Flash Attention. Long-context prefill time and workspace must be measured independently from steady-state decode.
- CUDA graph behavior, hybrid Mamba rollback, acceptance, head-256 Triton compilation on SM121, actual multimodal requests, and quality need hardware tests. Source-level admission is not an end-to-end proof.
- Preset comments report generic perplexity deltas of +2.71% (4bit_nc), +10.63% (k3v4_nc), and +20.59% (3bit_nc). These are not Qwen3.6 accuracy percentages and cannot be converted into a 97%-retention guarantee. They do justify testing 4-bit before 3-bit.

Even ignoring weights and all other work, ordinary one-token-at-a-time attention at 2,000 total output tokens/s would consume roughly 1.04 TB/s reading the 3-bit histories or 0.84 TB/s reading the OSCAR histories once per generated token. Those are diagnostic calculations, not a universal impossibility proof: accepted multi-token verification could amortize reads if implemented efficiently. They explain why cache capacity alone does not satisfy the request and why deep speculation needs measured accepted tokens per verifier call and actual throughput, not just a draft-depth flag.

## Concrete acceptance test

First verify one text and one image/video request with each stack, then test occupied 128K contexts at C=1,8,16,32,45,50 using unique inputs. Record per-stream generated token timing after all streams are active, minimum/p10/median tokens/s, aggregate throughput, resident requests, preemptions, TTFT, accepted tokens per MTP verification, cache occupancy, and memory. Quality must compare the final combined stack against BF16 on the same held-out text, image/video, and long-context tasks. A small smoke test can reject a candidate; it cannot certify the 97% minimum.

## Follow-up audit: MTP8/12 and recurrent-state memory

The Fast checkpoint explicitly sets `text_config.mamba_ssm_dtype=float32`. vLLM's Qwen3.5 configuration updater honors that default and warns on an explicit override. Its 30 GDN layers have recurrent state shape `(32,128,128)` per state slot. The BF16 convolution state has 8,192 channels and history length `3 + num_speculative_tokens`. With prefix caching disabled, each request needs `1 + num_speculative_tokens` state slots per GDN layer. Align mode needs one additional slot. These are source-level allocations; padding can raise actual usage.

| Draft depth | GDN memory at C50, FP32 state | GDN memory at C50, 16-bit state |
|---:|---:|---:|
| 0 | 3.00 GiB | 1.53 GiB |
| 3 | 12.27 GiB | 6.41 GiB |
| 7 | 25.27 GiB | 13.55 GiB |
| 8 | 28.63 GiB | 15.45 GiB |
| 12 | 42.55 GiB | 23.51 GiB |

These totals exclude full-attention KV, draft-model KV, weights, allocator padding, and workspace. Increasing MTP8 to MTP12 adds approximately **13.92 GiB** of FP32 GDN state at C50 before those overheads. Each extra speculative slot also creates recurrent state writes; deeper speculation is not free even with perfect draft acceptance.

Sources:

- [State shape calculation](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/layers/mamba/mamba_utils.py#L247)
- [Mamba memory accounting](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/kv_cache_interface.py#L668)
- [Qwen-specific dtype handling](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/models/config.py#L781)

### FP16 is allowed, but is not an accuracy-preserving guarantee

The Triton GDN verification kernel accumulates FP32 and writes the full recurrent state after each candidate token, casting to the selected cache dtype. FP16 or BF16 therefore changes the state used in later recurrence. Neither is a lossless alternative to the checkpoint's FP32 state. The exact 32-value-head/16-key-head kernel test covers FP32 and BF16 at draft depths 1 and 3; it does not test FP16 or certify long-context task accuracy. BF16 is the better-supported experimental 16-bit option, while its lower mantissa precision still requires quality testing.

The fused CUDA implementation accepts BF16 and FP32 recurrent state, but excludes FP16. Its speculative fast path additionally requires a value/key head ratio of **8**, while this exact 35B model has ratio **2**. Thus its maximum verification width of 8 does **not** create a special MTP7 fast path for this model: it is already excluded at every speculative depth. Do not select MTP7 based only on that width constant.

- [Actual GDN recurrence and state writes](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py#L102)
- [Exact-shape numerical tests](https://github.com/vllm-project/vllm/blob/v0.28.0/tests/kernels/test_fused_sigmoid_gating_delta_rule.py#L105)
- [Full fused MTP eligibility condition](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L1796)

### Supported controls that retain the configured FP32 recurrence

- Use `--no-enable-prefix-caching` and confirm `mamba_cache_mode=none`. Relative to align, this saves one state slot per request/layer, approximately 3.18 GiB at MTP8/C50. The benchmark uses independent prefixes, so this does not remove a needed sharing benefit.
- Limit `--max-num-seqs` to the actual 50-request target to avoid unnecessary graph/profiling buffers. This does not reduce the state required for those 50 active requests.
- Reduce configured speculative depth when marginal acceptance does not pay for its state and draft work. Dynamic depth schedules retain allocations for the configured maximum and are not a substitute for lowering that maximum.
- Avoid excessive CUDA graph capture shapes, but preserve shapes actually used. C50/MTP8 verification has 450 query tokens; a capture ceiling of 64 cannot cover that shape. TurboQuant's verification path may remain piecewise because it treats speculation as prefill, so inspect runtime capture behavior rather than forcing a graph mode.
- `--use-replayssm` is not applicable: v0.28 validates it for Nemotron-H and rejects speculative decoding. Stochastic FP16 rounding also does not make reduced-precision GDN state lossless; the Triton rounding option requires data-center SM100, not SM121.

No audited stock flag retains FP32 recurrence while storing only one checkpoint for arbitrary-depth GDN speculative verification. A new accepted-prefix replay implementation could trade recomputation for intermediate-state storage, but requires kernel/allocator changes and correctness tests; it is not the existing ReplaySSM feature.

Best next bounded configuration remains **FP32 GDN + TurboQuant4bit + MTP3, followed by MTP8** if acceptance and measured memory justify it. Test MTP12 only after MTP8 demonstrates substantial useful acceptance; combining 3-bit KV and 16-bit GDN just to make deeper speculation fit introduces two unvalidated quality changes. None of these controls alone establishes 40 tokens/s per stream at C50.

The state-write requirement also weakens the performance case for this conservative stack. One verified token writes approximately 62.9 MB of FP32 GDN state across the 30 layers; 2,000 accepted tokens/s therefore entails at least approximately 125.8 GB/s of state writes, before rejected candidates and other traffic. For TQ4/MTP8, an optimistic accounting that reads target history just once per nine-token verification and one MTP attention-layer history per draft gives approximately 274.7 GB/s of KV reads at perfect acceptance, before those state writes and weight traffic. Cache reuse and actual implementation determine DRAM traffic, so this is a diagnostic rather than an unconditional hardware lower bound, but it makes 40 tokens/s at C50 a weak expectation for the conservative recipe.

The aggressive experimental corner is TQ3 plus BF16 GDN plus deep MTP; it reduces both KV and state traffic but has no established 97% quality retention and still depends on unusually high acceptance and efficient verification. A successful short prompt or code-copy benchmark would not establish the general independent-128K requirement. Meaningful lossless progress beyond the stock knobs likely requires a GDN accepted-prefix replay implementation and a verification attention kernel that amortizes KV reads across draft queries, or a different well-matched speculative method.
