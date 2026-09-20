# Qwen3.6-35B-A3B hybrid cache allocation: vLLM 0.28.0

Read-only source audit, 2026-09-18. Sources below are pinned to upstream tag `v0.28.0`, not moving `main`. These are capacity predictions, not measured workstation results or throughput claims. Assumptions: one GPU/TP=1; 10 full-attention and 30 GDN layers; BF16 activation/conv state; FP32 GDN recurrent state; FP8 full-attention cache; no speculation unless explicitly discussed; 50 requests; maximum length 139,264 (131,072 prompt + 8,192 output).

## Main result

Stock v0.28 should need about **69.98 GiB of its cache pool for 50 completely filled 139,264-token requests**, including one null block, under the default FlashInfer alignment. This contains full-attention KV and recurrent states. Add the actual loaded weights, activation peak, CUDA graph pool, vision encoder peak, multimodal caches, framework allocations and host memory separately. The startup `Available KV cache memory` and group-aware `Maximum concurrency for 139,264 tokens per request` lines are the practical cross-checks.

At 131,072 tokens before output growth, the corresponding requirement is about **65.98 GiB**. A run can therefore admit 50 input contexts and still run out while all outputs grow if only the starting occupancy was budgeted. There is no fourfold full-context KV penalty from having 30 GDN layers: their states are constant-size in `none` mode and use shared-pool blocks separately from full attention.

## How the prediction follows from source

1. The [GDN shape calculator](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/layers/mamba/mamba_utils.py#L246) gives convolution shape `8192 x (3 + num_speculative_tokens)` and recurrent shape `32 x 128 x 128` at TP=1. Without speculation: BF16 convolution = 49,152 bytes; FP32 recurrent state = 2,097,152 bytes. Total GDN state page = **2,146,304 bytes = 2.046875 MiB per layer/request**. Thirty layers use 61.40625 MiB/request before padding.
2. The [Qwen config hook](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/models/config.py#L781) reads the checkpoint's `mamba_ssm_dtype` when the cache override is `auto`. Qwen's value is `float32`. The [Mamba config hook](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/models/config.py#L590) forces `mamba_cache_mode=none` when prefix caching is disabled and defaults the Mamba logical block length to `max_model_len`. This does **not** mean a GDN state is stored at every token.
3. The [platform alignment code](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/platforms/interface.py#L767) grows the full-attention manager block until one attention page is at least a GDN page. FP8 attention here uses `2 K/V x 2 heads x 256 = 1024 bytes/token/layer`. With alignment 16, a manager block is **2096 tokens**, and its 2,146,304-byte page exactly matches GDN: no physical page padding. The manager block is divided into supported physical kernel blocks; it does not require a FlashInfer kernel with literal page size 2096.
4. [FlashInfer's SM121 path](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/attention/backends/flashinfer.py#L419) advertises kernel sizes 16/32/64; the extra large-page support requires capability family 100. The inherited [preferred-size selection](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/attention/backend.py#L207) retains default 16. This prediction changes if the installed container patches that logic, another backend is selected, a user block size is specified, or cache dtypes differ.
5. [Group creation](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/core/kv_cache_utils.py#L1102) splits 10 attention + 30 GDN into four groups of 10 layers. There are no dummy padding layers in this 1:3 configuration. The allocator maintains ten shared pools; each logical pool block spans one 2.046875-MiB page in each pool, i.e. 20.46875 MiB across layers. A full request consumes `ceil(139264/2096) + 3 = 67 + 3 = 70` logical blocks. Fifty requests plus the null block require 3501 blocks, or 69.981536865 GiB. At 131072, each request needs 63+3=66 blocks.

## Few useful flags, in order

Keep the initial run's capacity contract explicit:

```bash
--max-num-seqs 50 \
--max-model-len 139264 \
--kv-cache-dtype fp8 \
--no-enable-prefix-caching \
--mamba-cache-mode none \
--mamba-ssm-cache-dtype float32 \
--enable-chunked-prefill
```

Do not set `--mamba-block-size` with prefix caching off; v0.28 [validates and rejects](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/config/vllm.py#L2543) a non-default value in that combination. Do not reduce recurrent precision to BF16 as a harmless allocator adjustment: that changes model numerics and requires quality validation.

If actual cache capacity is below 50:

- **Reduce activation reservation first:** use `--max-num-batched-tokens 4096` if currently 8192 or higher. With chunked prefill, this limits work per scheduler iteration without shortening the accepted context. Keep `max_num_seqs=50`. The [profile run](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/worker/gpu_model_runner.py#L6558) uses that token budget; the [scheduler](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/config/scheduler.py#L237) also derives encoder budgets from it. Smaller chunks may cost prefill speed; measure.
- **Capture only required decode graph sizes:** for this no-speculation test, try `--compilation-config '{"cudagraph_capture_sizes":[1,2,4,8,16,32,45,50],"max_cudagraph_capture_size":50}'`. Default graph maximum can reach twice `max_num_seqs` (100 here), with extra captured sizes. This [explicit list](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/config/vllm.py#L1923) can reduce graph workspace/reservation without disabling graphs. Revise capture sizes if speculation is later enabled: token batch size can then be `sequences x (k+1)`.
- **Optional very small capacity gain:** `--block-size 32` or `--block-size 64` produces manager block 2112 and state page 2.0625 MiB. At this specific maximum length it needs 66 full-attention blocks + 3 state blocks per request, so 50 requests plus null use **69.508667 GiB**, about 0.47 GiB less. At the initial 128K depth it is slightly less efficient (66.487427 GiB). This is rounding arithmetic, not a major optimization; test kernel speed before selecting it.
- **Control repeated multimodal host caching:** `--mm-processor-cache-gb 0` preserves image/video processing and weights while disabling reuse of preprocessing results. The default capacity is 4 GiB multiplied by API plus engine process counts ([definition](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/config/multimodal.py#L152)); this is potential host cache occupancy, not necessarily preallocated RAM. On unified memory it can matter. Zero can slow repeated images; a small nonzero value is an alternative.

Only after observing actual activation, graph, host and vision headroom should the owner of the test raise `--gpu-memory-utilization` from 0.87 or specify `--kv-cache-memory-bytes`. The [explicit byte flag](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/engine/arg_utils.py#L1210) overrides the utilization-based cache budget; it is not additive. The [worker](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/worker/gpu_worker.py#L489) still runs a model profile but skips automatic budget sizing. Do not use `--num-gpu-blocks-override` to fabricate capacity beyond available memory.

## Effects that can otherwise mislead

- `max_num_seqs=50` is a scheduler maximum, not a promise of 50 fully resident contexts. [Actual capacity](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/core/kv_cache_utils.py#L917) is available pool blocks divided by blocks needed per request across all groups. Max sequences affects graph sizes, input buffers and profiling batch shape; the main pool size is computed from memory. Lowering it may release overhead but also lowers active concurrency.
- Prefix caching switches GDN from `none` (one state block) to `align` (two). [MambaSpec](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/kv_cache_interface.py#L667) adds speculative blocks to either mode. Disabling prefix caching is appropriate for independent contexts and saves roughly another 3 GiB versus align at 50, excluding retained prefix entries.
- With MTP k=3, [the layer spec](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/layers/mamba/abstract.py#L63) reserves three extra state blocks per GDN layer, and the conv state grows to six positions. The **target layers alone** then require approximately 78.74 GiB at 50 x 139264 with alignment16. MTP's own layer, cache grouping/padding and workspaces add more; do not reuse the no-MTP 69.98-GiB estimate.
- `--skip-mm-profiling` preserves modality execution but omits vision-encoder/embedding-cache peaks from startup sizing ([definition](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/config/multimodal.py#L217)). A text-only capacity success using it does not certify multimodal memory safety. For the final multimodal run, remove it or measure the intended image/video peak and reserve equivalent headroom. Do not disable vision, set modality counts to zero, or shrink pixels/frames silently to pass.
- Keep the hybrid cache manager active; disabling it is not a documented capacity improvement for this GDN configuration. Also retain the default full-input reservation behavior: turning it off can admit requests that later thrash or preempt rather than creating memory.

Confirm resolved block size, recurrent dtype/mode, number of speculative blocks, cache groups and actual cache bytes in logs before treating these predictions as the workstation's allocation. A 50x cache-capacity line supplies no evidence for 40 tokens/sec per stream.
