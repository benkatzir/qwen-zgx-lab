# Parallel block drafting and native FP4 KV: bounded source audit

2026-09-18. Source-informed experimental candidates; not end-to-end verified on the ZGX. Root's existing FP8 XQA microbenchmark found C50/Q8 target attention alone takes about 295 ms for ten layers, an optimistic 27.1 tokens/s per stream before GDN, weights, drafting, or rejection. Therefore replacing the drafter alone cannot make that measured FP8/Q8 path meet 40 tokens/s.

## Concrete trained drafts

### DFlash: suitable public block-16 checkpoint

Use `z-lab/Qwen3.6-35B-A3B-DFlash`, pinned revision **`f181eece646affea2c38b2765f1aaa01a9734ccd`**. Current config: 6 layers, five SWA-4096 and one full attention, 8 KV heads of dimension 128, BF16, trained block size 16; approximately 0.386B parameters/0.77 GB of BF16 weights. Target feature taps are 1,6,11,16,22,27,32,37. Old posts describing an eight-layer all-full-attention draft do not describe this pinned checkpoint.

The authors retrained at 40K sequence length. Their B200/BF16/SGLang text benchmarks give mean accepted lengths of 4.33–5.74 at block 8 and 5.12–7.63 at block 16. They recommend block 8 for higher concurrency and 16 for single-stream speed. Neither test is a GB10, filled-128K, or multimodal quality result.

[Model/config and benchmark source](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash), [pinned config](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash/blob/f181eece646affea2c38b2765f1aaa01a9734ccd/config.json).

### DSpark: preferable draft cache geometry

Use `RedHatAI/Qwen3.6-35B-A3B-speculator.dspark`, revision **`53814b238c3a6ce5f332066a6bedb9d179777ede`**. Its five layers are all SWA-2048, with 2 KV heads of dimension 256, trained block size 8, a 32K draft vocabulary and a small sequential Markov head after the parallel backbone. Approximately 0.95B BF16 parameters. H100 validation reports mean acceptance length 3.39 for tool calls, 3.92 for RAG, 4.58 for HumanEval and 5.03 for mathematical reasoning. Training sequences were 16K. There is no native block-16 capability established by this checkpoint.

Its card deploys eight speculative tokens; the official vLLM recipe lists seven. Both are reasonable bounded tests, with the checkpoint's `sample_from_anchor=true` affecting draft query accounting. Do not set 15 merely to emulate DFlash16.

[Model/config source](https://huggingface.co/RedHatAI/Qwen3.6-35B-A3B-speculator.dspark), [official vLLM recipe](https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.6-35B-A3B.yaml).

### EAGLE candidate is not ready for stock serving

The public `zenith1232/qwen36-eagle3-drafter-v4` uses a custom one-layer MoE draft head. Its author explicitly says vanilla vLLM cannot load it without a custom MoE patch. This is not a more reproducible next candidate than supported DFlash/DSpark. Standard autoregressive EAGLE3 also does not automatically acquire parallel block drafting by setting a flag; that requires a correspondingly trained draft.

[Author's serving restriction](https://huggingface.co/zenith1232/qwen36-eagle3-drafter-v4).

## vLLM 0.28 compatibility and commands

Use V2 explicitly. Mixed SWA/full DFlash refuses V1; DSpark selects V2 automatically unless an environment override prevents it. V2 includes hybrid GDN state rollback and both parallel speculators. Do not install the stale PR40898 suggested by an old DFlash card: that PR is closed/unmerged, while current v0.28 contains its own mixed-attention support.

The following are replacements for the existing MTP speculative config, retaining the target's current launch flags and vision encoder. First use the working target FP8 attention configuration for correctness; lower-bit target cache is a separate experiment.

```bash
export VLLM_USE_V2_MODEL_RUNNER=1

# DFlash block8 (7 drafts + bonus); block16 uses num_speculative_tokens=15.
--speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-35B-A3B-DFlash","revision":"f181eece646affea2c38b2765f1aaa01a9734ccd","num_speculative_tokens":7,"attention_backend":"FLASHINFER","kv_cache_dtype":"auto"}'

# Official recipe's DSpark depth; separately compare depth8 from the model card.
--speculative-config '{"method":"dspark","model":"RedHatAI/Qwen3.6-35B-A3B-speculator.dspark","revision":"53814b238c3a6ce5f332066a6bedb9d179777ede","num_speculative_tokens":7,"attention_backend":"FLASHINFER","kv_cache_dtype":"auto"}'
```

These are **untested command candidates**, not a certification that every backend/model combination works. Draft `auto` keeps BF16 KV for the initial correctness run and avoids inheriting TurboQuant or target NVFP4. The draft is dense, so do not force the target's NVFP4 MoE backend onto it. Retain standard rejection sampling; never use synthetic acceptance for a quality/performance claim.

The current FlashInfer backend advertises SM12x XQA support for speculative and non-causal queries. This is appropriate for DFlash's non-causal full-attention layer. SGLang's model-card example instead targets B200 with FA4/TRTLLM backends; it is not a directly portable GB10 recipe.

Relevant source:

- [DFlash mixed-attention V2 requirement](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/models/qwen3_dflash.py#L133)
- [V2 selection](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/config/vllm.py#L615)
- [FlashInfer SM12x speculative/non-causal support](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/attention/backends/flashinfer.py#L982)

### Vision is retained, but test it

V2's DFlash constructor initially sets `supports_mm_inputs=False`. This does not disable the target vision tower. The common draft loader determines whether external image embeddings can be passed to the draft; otherwise it warns and uses text-only draft inputs, while the target still runs its multimodal encoder. DFlash also receives target hidden-state features after the multimodal prefill. Thus the required test is an actual image/video request and correct target verification, not removing vision to make loading work.

[Draft embedding behavior](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/worker/gpu/spec_decode/speculator.py#L176), [target multimodal input handling](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/worker/gpu/model_runner.py#L1551).

### Memory and target quality

Parallel drafting eliminates repeated sequential draft-model attention passes. It does **not** eliminate target GDN speculative checkpoints. With BF16 recurrent state, C50 uses approximately 13.55 GiB at depth7, 15.45 GiB at depth8, and 30.03 GiB at depth15, before padding, full-attention KV, weights and draft cache. BF16 recurrent state itself remains a lossy override of this target's FP32 setting.

Ideal BF16 draft history geometry at C50/128K is approximately 25 GiB for DFlash's single full-attention layer plus about 3.91 GiB across its five 4096-token SWA windows. DSpark's five 2048-token SWA windows total approximately 0.98 GiB. Actual allocation depends on vLLM hybrid grouping/padding and eviction; inspect reported cache specs rather than assuming these ideal totals.

Correct standard speculative verification preserves the distribution of the **served target**. It cannot restore quality already lost by NVFP4 target weights, low-bit KV, BF16 GDN states, incorrect scales, or broken kernels. Acceptance affects speed; a less accurate drafter alone does not imply less accurate verified output.

## Official native NVFP4 XQA exists, but stock vLLM does not wire it

FlashInfer **v0.6.16.post3 already provides** `xqa_batch_decode_with_kv_cache(..., kv_cache_sf=...)` and `xqa(..., k_sf_cache=..., v_sf_cache=...)`. The NVFP4 kernel explicitly allows SM12x and handles speculative query blocks. Packed KV is uint8 (two E2M1 values per byte), per-16 scales use linear E4M3 bytes, and K/V global scales must be applied correctly. Head dimension 256 is supported; the official v0.6.18 test specifically exercises head256 NVFP4 including a Q4 case.

Stock vLLM0.28 gates NVFP4 KV to SM100 and routes it through trtllm-gen with FP8 Q/output. Its SM121 XQA path needs different Q/output dtypes, scale/layout plumbing and prefill routing. Removing only the device gate is insufficient.

- [Installed-version public wrapper](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post3/flashinfer/decode.py#L3524)
- [Installed-version XQA implementation](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post3/flashinfer/xqa.py)
- [Official NVFP4 XQA test](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/tests/attention/test_xqa_batch_decode.py#L865)
- [Stock vLLM device gate](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/attention/backends/flashinfer.py#L517)

Three integration PRs remain open/unmerged: [54772](https://github.com/vllm-project/vllm/pull/54772), [46329](https://github.com/vllm-project/vllm/pull/46329), and [56550](https://github.com/vllm-project/vllm/pull/56550). They are not equivalent patches. The latter includes HND layout and linear V-scale writes; earlier prototypes describe wrong strided scale reads and unsafe full CUDA graphs. [Issue49011](https://github.com/vllm-project/vllm/issues/49011) reports a Qwen3.6-27B/head256 prototype on an RTX5090, with MTP4, a 49K needle check, and PIECEWISE graphs. This is adjacent-model integration evidence, not 97% task accuracy or the requested C50 result.

## Isolated test artifact

`work/bench_attention_nvfp4.py` tests the official existing kernel without modifying vLLM, upgrading libraries, loading weights, or allocating ten full caches. It uses matched interleaved data/scale buffers, independently dequantizes FP4 nibbles and FP8 scales, and checks FP32 attention on first/middle/last request streams. CUDA graph output is checked before and after mutating persistent queries and sequence lengths, then restoring the full context. Bad/unsupported graphs receive no graph throughput result; eager timing remains separate.

Local validation: Python syntax and `--help` only. Root runs GPU tests. A successful kernel test would establish a candidate for serving integration, not long-context model accuracy. The next decisive measurement is native NVFP4 XQA C45/C50 at Q8/Q16, including correctness; if attention alone still exceeds the token-time budget, parallel drafting cannot repair that configuration.

### Final query-precision check: FP8 Q/output is unsupported here

The exact installed-version source, FlashInfer **v0.6.16.post3**, accepts only `torch.float16` and `torch.bfloat16` query inputs for ordinary GQA XQA. `gen_xqa_module()` raises on FP8 query dtype. The public `xqa()` function permits FP8 output only when KV dtype is also FP8; with NVFP4's uint8 packed KV, output must match the BF16/FP16 query dtype. Providing different scale factors cannot bypass these compile-time type restrictions. The separate MLA XQA generator accepts FP8 queries, but this Qwen35B architecture uses GQA, not MLA. No FP8-query/output option was added to the benchmark.

[Exact query dtype gate](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post3/flashinfer/jit/xqa.py#L64), [output dtype restriction](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post3/flashinfer/xqa.py#L371).

Root subsequently measured native NVFP4 at C45/page128: ten attention layers take approximately 190.03 ms at Q8 and 341.78 ms at Q16. Its separate 30-layer FP32 GDN measurements are approximately 121.22 ms and 235.38 ms respectively. Their sum gives optimistic serial component ceilings of about 25.7 and 27.7 tokens/s per stream, assuming every verified token is accepted, before MoE, other projections, draft work and scheduling. These are component measurements supplied by root, not a full-model serving result or proof about a future fused implementation.
