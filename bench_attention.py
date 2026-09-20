#!/usr/bin/env python3
"""Bounded-memory FlashInfer attention microbenchmark for Qwen3.6-35B-A3B.

Run inside the installed GPU container; this script does not install anything:
  python /work/bench_attention.py --batches 1 8 50 --query-lengths 1 4 8 \
      --context 131072 --output /work/attention_results.json

Only attention is timed: ten serial invocations reuse ONE layer's distinct-per-
request FP8 cache. At batch 50 this is ~6.26 GiB, rather than ten caches (~62.5).
Queries are BF16, GQA is 16 query heads / 2 KV heads / head dimension 256.
`context` counts historical tokens; each call appends query_length KV positions.
No weights, model downloads, serving process changes or remote calls are made.

effective_kv_payload_GBps is valid KV payload / measured GPU time, NOT profiler-
measured DRAM traffic. Repeated reads, metadata and scratch traffic are excluded.
All-tokens-accepted token rates are attention-only optimistic projections, NOT
model throughput or an actual speculative acceptance measurement.

Public APIs consulted (2026-09-18):
 https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/decode.py
 https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/prefill.py
"""

import argparse
import gc
import inspect
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path


GIB = 1024 ** 3
Q_HEADS = 16
KV_HEADS = 2
HEAD_DIM = 256


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8, 50])
    p.add_argument("--query-lengths", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--context", type=int, default=131072,
                   help="Historical tokens per request, before the query block.")
    p.add_argument("--layers", type=int, default=10)
    p.add_argument("--page-size", type=int, default=None,
                   help="Default 16 for XQA (vLLM SM121 path), otherwise 128.")
    p.add_argument("--backend", default="auto")
    p.add_argument("--kernel", choices=["auto", "decode", "prefill", "xqa"], default="auto",
                   help="auto: wrappers; xqa: direct SM121 decode/verification API used by vLLM.")
    p.add_argument("--decode-use-tensor-cores", type=int, choices=[0, 1], default=1)
    p.add_argument("--timing", choices=["both", "eager", "graph"], default="both")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--max-measurement-seconds", type=float, default=30)
    p.add_argument("--workspace-mib", type=int, default=128)
    p.add_argument("--memory-cap-gib", type=float, default=12,
                   help="Torch CUDA allocator cap; native libraries may allocate separately.")
    p.add_argument("--skip-reference-check", action="store_true")
    p.add_argument("--seed", type=int, default=73021)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.page_size is None:
        args.page_size = 16 if args.kernel == "xqa" else 128
    if args.kernel == "xqa" and args.backend not in ("auto", "xqa"):
        p.error("The direct XQA variant requires --backend auto or xqa.")
    if args.kernel == "xqa" and args.page_size not in (16, 32, 64):
        p.error("Use page size 16, 32 or 64 for the vLLM SM121 XQA path.")
    if any(x <= 0 for x in args.batches + args.query_lengths):
        p.error("Batch and query lengths must be positive.")
    if min(args.context, args.layers, args.page_size, args.repeats,
           args.workspace_mib, args.memory_cap_gib) <= 0 or args.warmup < 1:
        p.error("Sizes/repetitions must be positive, and warmup >=1.")
    if args.kernel == "decode" and any(q != 1 for q in args.query_lengths):
        p.error("Use auto/prefill for multiple queries; decode mode is limited to q=1.")
    return args


def emit(record):
    print(json.dumps(record, sort_keys=True), flush=True)


def percentile(values, p):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def signature(obj):
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "unavailable"


def init_fp8_cache(torch, cache):
    # Initialize in <=32-MiB FP32 chunks; never materialize a full BF16/FP32 cache.
    per_page = math.prod(cache.shape[1:])
    pages_per_chunk = max(1, (32 * 1024 ** 2) // (per_page * 4))
    for start in range(0, cache.shape[0], pages_per_chunk):
        n = min(pages_per_chunk, cache.shape[0] - start)
        tmp = torch.randn((n, *cache.shape[1:]), dtype=torch.float32, device="cuda")
        cache[start:start + n].copy_(tmp.to(cache.dtype))
        del tmp
    torch.cuda.synchronize()


def check_reference(torch, q, cache, out, pages_per_request, kv_len, q_len):
    """Validate stream 0 against FP32 attention over the same quantized K/V.

    Uses grouped einsum without repeating K/V eight times. At 128K it needs
    about 0.7 GiB transient memory. It is outside every timed interval.
    """
    k = cache[:pages_per_request, 0].reshape(-1, KV_HEADS, HEAD_DIM)[:kv_len].float()
    v = cache[:pages_per_request, 1].reshape(-1, KV_HEADS, HEAD_DIM)[:kv_len].float()
    groups = Q_HEADS // KV_HEADS
    query = q[:q_len].float().reshape(q_len, KV_HEADS, groups, HEAD_DIM)
    query = query.permute(1, 2, 0, 3)
    scores = torch.einsum("hgtd,nhd->hgtn", query, k) / math.sqrt(HEAD_DIM)
    positions = torch.arange(kv_len, device="cuda")
    end_positions = kv_len - q_len + torch.arange(q_len, device="cuda")
    causal_mask = positions[None, :] > end_positions[:, None]
    scores.masked_fill_(causal_mask[None, None, :, :], float("-inf"))
    probabilities = scores.softmax(dim=-1)
    expected = torch.einsum("hgtn,nhd->thgd", probabilities, v).reshape(q_len, Q_HEADS, HEAD_DIM)
    actual = out[:q_len].float()
    delta = actual - expected
    maximum = delta.abs().max().item()
    relative_rmse = ((delta.square().mean().sqrt() /
                      expected.square().mean().sqrt().clamp_min(1e-10)).item())
    finite = bool(torch.isfinite(actual).all().item())
    passed = finite and maximum < 5e-3 and relative_rmse < 0.05
    result = {"passed": passed, "max_abs_error": maximum,
              "relative_rmse": relative_rmse,
              "scope": "stream 0, same FP8 KV values, independent FP32 attention"}
    if not passed:
        raise RuntimeError("Reference attention mismatch: " + json.dumps(result))
    return result


def measure(torch, fn, args):
    torch.cuda.synchronize()
    samples = []
    wall_start = time.monotonic()
    for _ in range(args.repeats):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
        if time.monotonic() - wall_start > args.max_measurement_seconds:
            break
    return samples


def timing_summary(samples, payload_bytes, batch, q_len, layers):
    median_ms = statistics.median(samples)
    seconds = median_ms / 1000
    return {
        "block_gpu_ms_samples": samples,
        "block_gpu_ms_median": median_ms,
        "block_gpu_ms_p95": percentile(samples, 95),
        "one_attention_layer_ms_median": median_ms / layers,
        "effective_kv_payload_GBps": payload_bytes * layers / seconds / 1e9,
        "attention_only_all_accepted_tokens_per_second_per_stream": q_len / seconds,
        "attention_only_all_accepted_tokens_per_second_aggregate": batch * q_len / seconds,
        "required_acceptance_fraction_for_40_tps_ignoring_other_work": 40 * seconds / q_len,
    }


def run_case(torch, flashinfer, args, batch, q_len):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    kv_len = args.context + q_len
    pages_per_request = math.ceil(kv_len / args.page_size)
    pages = pages_per_request * batch
    allocated_cache_bytes = pages * 2 * args.page_size * KV_HEADS * HEAD_DIM
    payload_bytes = batch * kv_len * 2 * KV_HEADS * HEAD_DIM
    workspace_bytes = args.workspace_mib * 1024 ** 2
    # Reserve for reference FP32 tensors, graph internals, query/output and native scratch.
    estimated_bytes = allocated_cache_bytes + workspace_bytes + 2 * GIB
    free, _ = torch.cuda.mem_get_info()
    if estimated_bytes > args.memory_cap_gib * GIB:
        raise RuntimeError(f"Refusing allocation: estimated {estimated_bytes / GIB:.2f} GiB "
                           f"exceeds {args.memory_cap_gib:.2f} GiB cap.")
    if estimated_bytes > free * 0.8:
        raise RuntimeError(f"Insufficient free GPU memory: need ~{estimated_bytes/GIB:.2f} GiB, "
                           f"available {free/GIB:.2f} GiB (20% headroom required).")
    emit({"event": "case_start", "batch": batch, "query_length": q_len,
          "historical_tokens": args.context, "kv_length": kv_len,
          "cache_allocation_GiB": allocated_cache_bytes / GIB,
          "estimated_total_GiB": estimated_bytes / GIB})
    torch.manual_seed(args.seed + batch * 100 + q_len)
    cache = torch.empty((pages, 2, args.page_size, KV_HEADS, HEAD_DIM),
                        dtype=torch.float8_e4m3fn, device="cuda")
    init_fp8_cache(torch, cache)
    q = torch.randn((batch * q_len, Q_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(q)
    workspace = torch.zeros(workspace_bytes, device="cuda", dtype=torch.uint8)
    indptr = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * pages_per_request
    indices = torch.arange(pages, dtype=torch.int32, device="cuda")
    last_page = torch.full((batch,), (kv_len - 1) % args.page_size + 1,
                           dtype=torch.int32, device="cuda")
    qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * q_len
    graph_requested = args.timing in ("graph", "both")
    kernel = args.kernel
    if kernel == "auto":
        kernel = "decode" if q_len == 1 else "prefill"
    if kernel == "xqa":
        from flashinfer.decode import xqa_batch_decode_with_kv_cache
        wrapper = None
        block_tables = indices.view(batch, pages_per_request)
        seq_lens = torch.full((batch,), kv_len, dtype=torch.uint32, device="cuda")
        draft_mask = None
        if q_len > 1:
            # Same packed causal draft-block mask as vLLM0.28 flashinfer.py.
            # Context prefix remains fully visible; these bits mask only the
            # q_len candidate positions appended at the end of each sequence.
            packed_words = (q_len + 31) // 32
            row = torch.arange(q_len, device="cuda").unsqueeze(1)
            col = torch.arange(packed_words * 32, device="cuda").unsqueeze(0)
            boolean = col <= row
            bits = 1 << torch.arange(32, device="cuda", dtype=torch.int64)
            packed = ((boolean.reshape(q_len, packed_words, 32).to(torch.int64) * bits)
                      .sum(dim=-1).to(torch.uint32))
            per_request_mask = packed.view(torch.uint16).reshape(q_len, packed_words * 2)
            draft_mask = per_request_mask.unsqueeze(0).expand(batch, -1, -1).contiguous()
    elif kernel == "decode":
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, kv_layout="NHD", use_cuda_graph=graph_requested,
            use_tensor_cores=bool(args.decode_use_tensor_cores), backend=args.backend,
            paged_kv_indptr_buffer=indptr if graph_requested else None,
            paged_kv_indices_buffer=indices if graph_requested else None,
            paged_kv_last_page_len_buffer=last_page if graph_requested else None)
        wrapper.plan(indptr, indices, last_page, Q_HEADS, KV_HEADS, HEAD_DIM, args.page_size,
                     pos_encoding_mode="NONE", q_data_type=torch.bfloat16,
                     kv_data_type=torch.float8_e4m3fn, sm_scale=1 / math.sqrt(HEAD_DIM))
    else:
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, kv_layout="NHD", use_cuda_graph=graph_requested, backend=args.backend,
            qo_indptr_buf=qo_indptr if graph_requested else None,
            paged_kv_indptr_buf=indptr if graph_requested else None,
            paged_kv_indices_buf=indices if graph_requested else None,
            paged_kv_last_page_len_buf=last_page if graph_requested else None)
        wrapper.plan(qo_indptr, indptr, indices, last_page, Q_HEADS, KV_HEADS,
                     HEAD_DIM, args.page_size, causal=True, pos_encoding_mode="NONE",
                     q_data_type=torch.bfloat16, kv_data_type=torch.float8_e4m3fn,
                     sm_scale=1 / math.sqrt(HEAD_DIM))

    def layer_block():
        # Same-stream execution; ten calls model only the full-attention count.
        # No attempt is made to model the 30 recurrent layers or any projections.
        for _ in range(args.layers):
            if kernel == "xqa":
                xqa_batch_decode_with_kv_cache(
                    query=q, kv_cache=cache, workspace_buffer=workspace,
                    block_tables=block_tables, seq_lens=seq_lens,
                    max_seq_len=kv_len, bmm1_scale=1 / math.sqrt(HEAD_DIM),
                    bmm2_scale=1.0, window_left=-1, out=out,
                    kv_layout="NHD", q_len_per_req=q_len, mask=draft_mask)
            else:
                wrapper.run(q, cache, out=out)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(args.warmup):
            layer_block()
    warmup_stream.synchronize()
    if not bool(torch.isfinite(out).all().item()):
        raise RuntimeError("Attention output contains non-finite values.")
    reference = None
    if not args.skip_reference_check:
        reference = check_reference(torch, q, cache, out, pages_per_request, kv_len, q_len)
        torch.cuda.synchronize()
    result = {
        "event": "case_result", "batch": batch, "query_length": q_len,
        "historical_tokens": args.context, "kv_length": kv_len,
        "full_attention_layer_calls": args.layers,
        "kernel": kernel, "backend_requested": args.backend,
        "backend_resolved": "xqa" if kernel == "xqa" else str(getattr(wrapper, "_backend", "unreported")),
        "public_api": ("flashinfer.decode.xqa_batch_decode_with_kv_cache" if kernel == "xqa"
                       else type(wrapper).__name__),
        "draft_mask": "packed causal uint16" if kernel == "xqa" and q_len > 1 else None,
        "query_dtype": str(q.dtype), "kv_dtype": str(cache.dtype), "layout": "NHD",
        "cache_allocation_GiB": allocated_cache_bytes / GIB,
        "one_layer_valid_kv_payload_bytes": payload_bytes,
        "full_10_layer_distinct_cache_GiB_projected": payload_bytes * args.layers / GIB,
        "reference_check": reference,
    }
    if args.timing in ("both", "eager"):
        samples = measure(torch, layer_block, args)
        result["eager"] = timing_summary(samples, payload_bytes, batch, q_len, args.layers)
    if graph_requested:
        graph = torch.cuda.CUDAGraph()
        # Planning/JIT/warmup above must finish before capture. All inputs and outputs persist.
        torch.cuda.synchronize()
        try:
            with torch.cuda.graph(graph, stream=warmup_stream):
                layer_block()
            graph.replay()
            torch.cuda.synchronize()
            if not args.skip_reference_check:
                result["graph_reference_check"] = check_reference(
                    torch, q, cache, out, pages_per_request, kv_len, q_len)
            samples = measure(torch, graph.replay, args)
            result["graph"] = timing_summary(samples, payload_bytes, batch, q_len, args.layers)
        except Exception as exc:
            result["graph_error"] = f"{type(exc).__name__}: {exc}"
            if args.timing == "graph":
                raise
        del graph
    torch.cuda.synchronize()
    result["torch_peak_allocated_GiB"] = torch.cuda.max_memory_allocated() / GIB
    result["torch_peak_reserved_GiB"] = torch.cuda.max_memory_reserved() / GIB
    emit(result)
    return result


def main():
    args = parse_args()
    import torch
    import flashinfer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required. Run inside the target GPU container.")
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.memory_cap_gib * GIB / props.total_memory))
    torch.backends.cuda.matmul.allow_tf32 = False
    metadata = {
        "event": "metadata", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": platform.node(), "python": sys.version, "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "flashinfer": getattr(flashinfer, "__version__", "unknown"),
        "gpu": props.name, "compute_capability": [props.major, props.minor],
        "device_total_GiB": props.total_memory / GIB,
        "l2_cache_size_bytes_if_exposed": getattr(props, "L2_cache_size", None),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "flashinfer_decode_init_signature": signature(flashinfer.BatchDecodeWithPagedKVCacheWrapper.__init__),
        "flashinfer_decode_plan_signature": signature(flashinfer.BatchDecodeWithPagedKVCacheWrapper.plan),
        "flashinfer_prefill_init_signature": signature(flashinfer.BatchPrefillWithPagedKVCacheWrapper.__init__),
        "flashinfer_prefill_plan_signature": signature(flashinfer.BatchPrefillWithPagedKVCacheWrapper.plan),
        "limitations": [
            "Measured attention kernel only; excludes weights, projections, recurrent layers, vision, drafting, scheduling and prefill.",
            "One layer cache reused ten times; request caches are distinct. No full-model memory/capacity claim.",
            "GPU-event elapsed time and effective payload bandwidth, not profiler DRAM counters.",
            "q>1 rates assume all positions yield accepted tokens; actual speculative acceptance is not measured.",
            "No positional encoding inside kernel: matches attention after externally applied RoPE.",
            "Torch memory fraction cap does not govern allocations made outside the Torch allocator.",
        ],
    }
    emit(metadata)
    results, errors = [], []
    for batch in args.batches:
        for q_len in args.query_lengths:
            try:
                with torch.inference_mode():
                    results.append(run_case(torch, flashinfer, args, batch, q_len))
            except Exception as exc:
                error = {"event": "case_error", "batch": batch, "query_length": q_len,
                         "type": type(exc).__name__, "error": str(exc)}
                errors.append(error)
                emit(error)
                traceback.print_exc(file=sys.stderr)
                # A failed GPU operation can poison the context; stop rather than report bogus cases.
                if isinstance(exc, torch.cuda.OutOfMemoryError) or "CUDA" in str(exc):
                    break
            finally:
                gc.collect()
                torch.cuda.empty_cache()
        if errors and (errors[-1]["type"] == "OutOfMemoryError" or "CUDA" in errors[-1]["error"]):
            break
    report = {"metadata": metadata, "results": results, "errors": errors}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    emit({"event": "complete", "successful_cases": len(results), "failed_cases": len(errors)})
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
