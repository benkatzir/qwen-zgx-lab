# Attention microbenchmark: appropriate GB10 kernel

The wrapper's `backend="auto"` result is not evidence of the fastest vLLM
attention path on GB10. On SM121, vLLM 0.28 uses FlashInfer's dedicated **XQA**
decode API when its compatibility and availability checks pass. Its decode query
remains BF16 with FP8 KV. The `trtllm-gen` kernel is the SM100 path; forcing that
backend through a generic wrapper does not reproduce the SM121 serving path.

Use either of the equivalent entry points, with the updated files together:

```sh
python bench_attention_xqa.py --batches 1 --query-lengths 1 4 --context 256 --repeats 2
python bench_attention_xqa.py --output attention_xqa_results.json
# Alternatively, only one file is needed:
python bench_attention.py --kernel xqa --output attention_xqa_results.json
```

The XQA variant directly calls
`flashinfer.decode.xqa_batch_decode_with_kv_cache`, uses page size 16 by default,
and packs the speculative causal mask with the same uint32-to-uint16 layout as
vLLM 0.28. BF16 queries, FP8 E4M3 KV, 16 query heads, 2 KV heads, dimension 256,
and a full visible historical prefix match the target model's full-attention
geometry. Query lengths 4 and 8 model target verification blocks with perfect
acceptance only in the projected throughput field; they do not measure a draft
model or acceptance. A separate FP32 attention calculation checks stream 0 for
each shape before timing, including the causal draft mask.

The lower-level FlashInfer XQA function supports both NHD and HND. vLLM requires
HND on SM100, but does not impose this requirement on SM121; NHD here is supported.
The public implementation allows the requested head dimension and GQA ratio to
reach JIT generation. Actual compilation and correctness on the installed build
must pass before treating any result as measured evidence.

Timing includes ten serial attention calls, excludes initialization, JIT,
planning, reference checking and warmup, and reports eager and captured CUDA
graph timings. The graph is warmed once after capture. One layer's cache is
reused for the ten calls to keep the batch-50 run near 6.3 GiB plus scratch and
reference tensors, beneath a 12 GiB Torch allocator limit. Each request owns
distinct pages, and even the batch-one 128K layer cache exceeds GPU L2. Reusing
this buffer omits the full model's capacity requirements and all intervening
projections, recurrent layers, MoE work, vision, scheduling, drafting and token
acceptance. It may leave the GPU in a different cache/power state from serving.

`effective_kv_payload_GBps` is nominal valid KV payload divided by measured time,
**not** a hardware-counter measurement of DRAM reads. q > 1 divides that payload
once per verification pass: actual implementations may reread it. Rates are
optimistic attention-only projections, never end-to-end throughput claims.

At batch 50, 131072 historical tokens, ten full-attention layers and FP8 KV,
reading every cache once moves about 67.1 GB per target pass. At the official
273 GB/s bandwidth, this alone allows approximately 4.07 target passes/s.
Perfectly reused q=8 blocks would therefore allow only 32.5 accepted tokens/s
per stream before all other work. About 9.83 accepted positions per pass are
needed for 40 tokens/s. This bound is specific to dense attention with FP8 KV;
it is not a proof against all compression, sparse attention or speculation
schemes. Deep speculation on this hybrid model also increases recurrent-state
allocation, so the no-speculation cache budget cannot simply be retained.

Primary implementation references (pinned releases):

- [vLLM 0.28 FlashInfer backend](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/attention/backends/flashinfer.py): `_get_flashinfer_trtllm_api_decode_kernel`, dedicated XQA call, query dtype, and `_make_xqa_draft_block_mask`.
- [vLLM 0.28 FlashInfer compatibility checks](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/utils/flashinfer.py): SM12x decode support and API selection.
- [FlashInfer 0.6.16 decode API](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16/flashinfer/decode.py): `xqa_batch_decode_with_kv_cache` and direct TRTLLM API backend dispatch.
- [FlashInfer 0.6.16 XQA implementation](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16/flashinfer/xqa.py): supported capabilities, layouts, mask and JIT parameters.
- [NVIDIA DGX Spark specifications](https://www.nvidia.com/en-us/products/workstations/dgx-spark/): advertised 273 GB/s unified-memory bandwidth.

The installed FlashInfer is reported as 0.6.16.post3. These references establish
the release-family API and selection logic, not that every post-release binary
or generated kernel is byte-identical. The harness records installed versions
and runs an independent correctness check to address that distinction.
