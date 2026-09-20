#!/usr/bin/env python3
"""Isolated native FlashInfer SM12x NVFP4 XQA attention benchmark.

No installation, upgrades, server changes, or weights. Requires bench_attention.py
alongside this file, solely for timing/report helpers. Defaults: actual 131072-token
history, independent request caches, C=1/45/50, Q=1/4/8/16, ten serial layer calls
reusing one layer cache. At C50 packed data+scales occupy about 3.52 GiB.

Uses the public API already present in FlashInfer v0.6.16.post3. Layout follows
the official test_xqa_batch_decode_nvfp4_kv fixture: BOTH packed data and linear
FP8 block scales use matched [page,2,token,head,dim] interleaving. This avoids
the historical separate-scale-page-stride bug. K/V global scales differ and
are non-unit. The reference manually decodes FP4 nibbles and FP8 scale bytes,
then computes FP32 attention without any FlashInfer reference kernel.

Example: python bench_attention_nvfp4.py --batches 1 --query-lengths 1 4 \
  --context 256 --output nvfp4_smoke.json

Sources:
 https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post3/flashinfer/decode.py
 https://github.com/flashinfer-ai/flashinfer/blob/v0.6.16.post3/flashinfer/xqa.py
 https://github.com/flashinfer-ai/flashinfer/blob/v0.6.18/tests/attention/test_xqa_batch_decode.py
"""

import argparse
import gc
import inspect
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path

from bench_attention import GIB, Q_HEADS, KV_HEADS, HEAD_DIM
from bench_attention import emit, measure, signature, timing_summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batches", nargs="+", type=int, default=[1, 45, 50])
    p.add_argument("--query-lengths", nargs="+", type=int, default=[1, 4, 8, 16])
    p.add_argument("--context", type=int, default=131072)
    p.add_argument("--layers", type=int, default=10)
    p.add_argument("--page-size", type=int, choices=[16, 32, 64, 128], default=16)
    p.add_argument("--timing", choices=["both", "eager", "graph"], default="both")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--max-measurement-seconds", type=float, default=30)
    p.add_argument("--workspace-mib", type=int, default=128)
    p.add_argument("--memory-cap-gib", type=float, default=12)
    p.add_argument("--reference-streams", type=int, default=3)
    p.add_argument("--reference-max-relative-rmse", type=float, default=0.20,
                   help="Kernel-vs-dequantized-reference guard; not model accuracy.")
    p.add_argument("--reference-max-abs-error", type=float, default=0.10,
                   help="Official NVFP4 test uses atol/rtol=0.1; we also require relative RMSE.")
    p.add_argument("--seed", type=int, default=73021)
    p.add_argument("--output", type=Path)
    a = p.parse_args()
    if any(v <= 0 for v in a.batches + a.query_lengths) or min(a.context, a.layers, a.warmup,
            a.repeats, a.reference_streams, a.memory_cap_gib) <= 0:
        p.error("All sizes/repetition counts must be positive.")
    if a.memory_cap_gib > 12 or a.workspace_mib < 16:
        p.error("Memory cap must be <=12 GiB; workspace must be >=16 MiB.")
    return a


def causal_mask(torch, batch, qlen):
    if qlen == 1:
        return None
    words = (qlen + 31) // 32
    row = torch.arange(qlen, device="cuda").unsqueeze(1)
    col = torch.arange(words * 32, device="cuda").unsqueeze(0)
    bits = 1 << torch.arange(32, device="cuda", dtype=torch.int64)
    packed = (((col <= row).reshape(qlen, words, 32).to(torch.int64) * bits)
              .sum(-1).to(torch.uint32))
    return packed.view(torch.uint16).reshape(1, qlen, words * 2).expand(batch, -1, -1).contiguous()


def initialize_cache(torch, data, scales):
    from flashinfer import SfLayout
    from flashinfer.fp4_quantization import nvfp4_quantize
    globals_ = [2.0 / (448.0 * 6.0), 3.0 / (448.0 * 6.0)]
    npage, _, page, heads, _ = data.shape
    # Only one <=16 MiB BF16 source chunk is materialized at a time.
    chunk_pages = max(1, (16 * 1024**2) // (page * heads * HEAD_DIM * 2))
    for kind, global_scale in enumerate(globals_):
        inv = torch.tensor([1.0 / global_scale], device="cuda", dtype=torch.float32)
        clip = 2.0 if kind == 0 else 3.0
        for start in range(0, npage, chunk_pages):
            count = min(chunk_pages, npage - start)
            src = torch.randn((count * page * heads, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
            src.mul_(clip / 4).clamp_(-clip, clip)
            packed, sf = nvfp4_quantize(src, inv, sfLayout=SfLayout.layout_linear,
                                       do_shuffle=False, sf_vec_size=16, enable_pdl=False)
            if sf.numel() != count * page * heads * (HEAD_DIM // 16):
                raise RuntimeError(f"Unexpected linear scale layout: {tuple(sf.shape)}")
            data[start:start + count, kind].copy_(packed.view(count, page, heads, HEAD_DIM // 2))
            scales[start:start + count, kind].copy_(sf.view(torch.uint8).view(count, page, heads, HEAD_DIM // 16))
            del src, packed, sf
    torch.cuda.synchronize()
    return globals_


def independent_dequantize(torch, data, scales, global_scale):
    """Manual E2M1 low/high nibble and E4M3FN per-16 scale decoding."""
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.,
                        -0., -.5, -1., -1.5, -2., -3., -4., -6.], device="cuda")
    packed = data.contiguous().reshape(-1, KV_HEADS, HEAD_DIM // 2)
    values = torch.empty((*packed.shape[:-1], HEAD_DIM), dtype=torch.float32, device="cuda")
    values[..., 0::2] = lut[(packed & 15).long()]
    values[..., 1::2] = lut[(packed >> 4).long()]
    sf = scales.contiguous().reshape(-1, KV_HEADS, HEAD_DIM // 16).to(torch.int32)
    exponent = (sf >> 3) & 15
    mantissa = sf & 7
    decoded_sf = torch.where(exponent == 0, mantissa.float() / 512,
                             (1 + mantissa.float() / 8) * torch.pow(2., exponent.float() - 7))
    decoded_sf *= torch.where((sf & 128) != 0, -1., 1.)
    if bool(((exponent == 15) & (mantissa == 7)).any().item()):
        raise RuntimeError("Non-finite FP8 block scale in quantized cache")
    values.view(*values.shape[:-1], HEAD_DIM // 16, 16).mul_(decoded_sf.unsqueeze(-1) * global_scale)
    return values


def check_reference(torch, args, q, out, data, sf, globals_, pages_per_req, kv_len, batch, qlen):
    count = min(args.reference_streams, batch)
    streams = sorted(set(round(i * (batch - 1) / max(1, count - 1)) for i in range(count)))
    checks = []
    for stream in streams:
        sl = slice(stream * pages_per_req, (stream + 1) * pages_per_req)
        k = independent_dequantize(torch, data[sl, 0], sf[sl, 0], globals_[0])[:kv_len]
        v = independent_dequantize(torch, data[sl, 1], sf[sl, 1], globals_[1])[:kv_len]
        query = q[stream * qlen:(stream + 1) * qlen].float()
        query = query.reshape(qlen, KV_HEADS, Q_HEADS // KV_HEADS, HEAD_DIM).permute(1, 2, 0, 3)
        scores = torch.einsum("hgtd,nhd->hgtn", query, k) / math.sqrt(HEAD_DIM)
        positions = torch.arange(kv_len, device="cuda")
        ends = kv_len - qlen + torch.arange(qlen, device="cuda")
        scores.masked_fill_(positions[None, None, None, :] > ends[None, None, :, None], float("-inf"))
        probs = scores.softmax(-1)
        expected = torch.einsum("hgtn,nhd->thgd", probs, v).reshape(qlen, Q_HEADS, HEAD_DIM)
        actual = out[stream * qlen:(stream + 1) * qlen].float()
        diff = actual - expected
        max_abs = diff.abs().max().item()
        rel = (diff.square().mean().sqrt() / expected.square().mean().sqrt().clamp_min(1e-12)).item()
        finite = bool(torch.isfinite(actual).all().item())
        passed = finite and max_abs <= args.reference_max_abs_error and rel <= args.reference_max_relative_rmse
        checks.append({"stream": stream, "passed": passed, "max_abs_error": max_abs,
                       "relative_rmse": rel, "reference_rms": expected.square().mean().sqrt().item()})
        del k, v, query, scores, probs, expected, actual, diff
    result = {"passed": all(c["passed"] for c in checks), "checks": checks,
              "scope": "Independent manual NVFP4 dequantization, FP32 GQA causal attention"}
    if not result["passed"]:
        raise RuntimeError("Reference mismatch: " + json.dumps(result))
    return result


def run_case(torch, args, batch, qlen):
    from flashinfer.decode import xqa_batch_decode_with_kv_cache as xqa
    kv_len = args.context + qlen
    pages_per_req = math.ceil(kv_len / args.page_size)
    pages = batch * pages_per_req
    packed_bytes = pages * 2 * args.page_size * KV_HEADS * (HEAD_DIM // 2 + HEAD_DIM // 16)
    # Two dequantized stream tensors, scores/probs, quantization chunks, graph/JIT headroom.
    estimate = packed_bytes + args.workspace_mib * 1024**2 + 2.0 * GIB
    free, _ = torch.cuda.mem_get_info()
    if estimate > args.memory_cap_gib * GIB or estimate > free * .8:
        raise RuntimeError(f"Estimated {estimate/GIB:.2f}GiB exceeds cap or free-memory headroom")
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(args.seed + batch * 100 + qlen)
    emit({"event": "case_start", "batch": batch, "query_length": qlen,
          "historical_tokens": args.context, "cache_GiB": packed_bytes/GIB})
    data = torch.empty((pages, 2, args.page_size, KV_HEADS, HEAD_DIM // 2), device="cuda", dtype=torch.uint8)
    sf = torch.empty((pages, 2, args.page_size, KV_HEADS, HEAD_DIM // 16), device="cuda", dtype=torch.uint8)
    globals_ = initialize_cache(torch, data, sf)
    q = torch.randn((batch * qlen, Q_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16)
    out = torch.empty_like(q)
    workspace = torch.zeros(args.workspace_mib * 1024**2, device="cuda", dtype=torch.uint8)
    tables = torch.arange(pages, device="cuda", dtype=torch.int32).view(batch, pages_per_req)
    lengths = torch.full((batch,), kv_len, dtype=torch.uint32, device="cuda")
    mask = causal_mask(torch, batch, qlen)

    def block():
        for _ in range(args.layers):
            xqa(query=q, kv_cache=data, workspace_buffer=workspace,
                block_tables=tables, seq_lens=lengths, max_seq_len=kv_len,
                bmm1_scale=globals_[0] / math.sqrt(HEAD_DIM), bmm2_scale=globals_[1],
                window_left=-1, out=out, kv_layout="NHD", q_len_per_req=qlen,
                mask=mask, kv_cache_sf=sf, enable_pdl=False)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(args.warmup):
            block()
    stream.synchronize()
    ref = check_reference(torch, args, q, out, data, sf, globals_, pages_per_req, kv_len, batch, qlen)
    valid_bytes = batch * kv_len * 2 * KV_HEADS * (HEAD_DIM // 2 + HEAD_DIM // 16)
    result = {"event": "case_result", "batch": batch, "query_length": qlen,
              "historical_tokens": args.context, "kv_length": kv_len,
              "full_attention_layer_calls": args.layers, "cache_allocation_GiB": packed_bytes/GIB,
              "query_dtype": "bfloat16", "kv_dtype": "NVFP4", "layout": "NHD",
              "data_shape": list(data.shape), "scale_shape": list(sf.shape),
              "k_global_scale": globals_[0], "v_global_scale": globals_[1],
              "reference_check": ref, "one_layer_valid_kv_payload_bytes": valid_bytes}
    # Always obtain eager timing as a fallback even when graph-only is requested.
    result["eager"] = timing_summary(measure(torch, block, args), valid_bytes, batch, qlen, args.layers)
    if args.timing != "eager":
        graph = torch.cuda.CUDAGraph()
        try:
            torch.cuda.synchronize()
            with torch.cuda.graph(graph, stream=stream):
                block()
            graph.replay()
            torch.cuda.synchronize()
            result["graph_reference_check"] = check_reference(
                torch, args, q, out, data, sf, globals_, pages_per_req, kv_len, batch, qlen)
            # Changing persistent query and lengths catches graph replay using stale values.
            q.normal_()
            shorter = max(qlen, kv_len - min(args.page_size, args.context))
            lengths.fill_(shorter)
            graph.replay()
            torch.cuda.synchronize()
            result["graph_mutated_input_reference_check"] = check_reference(
                torch, args, q, out, data, sf, globals_, pages_per_req, shorter, batch, qlen)
            lengths.fill_(kv_len)
            graph.replay()
            torch.cuda.synchronize()
            result["graph_restored_context_reference_check"] = check_reference(
                torch, args, q, out, data, sf, globals_, pages_per_req, kv_len, batch, qlen)
            result["graph"] = timing_summary(measure(torch, graph.replay, args), valid_bytes, batch, qlen, args.layers)
        except Exception as exc:
            result["graph_error"] = f"{type(exc).__name__}: {exc}"
            result["usable_timing"] = "eager_only"
            # A mismatched graph must never receive a throughput result.
            result.pop("graph", None)
            if "CUDA" in str(exc) or isinstance(exc, torch.cuda.OutOfMemoryError):
                raise
        del graph
    torch.cuda.synchronize()
    result["torch_peak_allocated_GiB"] = torch.cuda.max_memory_allocated()/GIB
    result["torch_peak_reserved_GiB"] = torch.cuda.max_memory_reserved()/GIB
    emit(result)
    return result


def main():
    args = parse_args()
    import torch
    import flashinfer
    from flashinfer.decode import xqa_batch_decode_with_kv_cache
    if not torch.cuda.is_available():
        raise RuntimeError("Run inside the existing CUDA GPU container")
    props = torch.cuda.get_device_properties(0)
    if props.major != 12:
        raise RuntimeError("This NVFP4 XQA benchmark requires SM120/SM121")
    if "kv_cache_sf" not in inspect.signature(xqa_batch_decode_with_kv_cache).parameters:
        raise RuntimeError("Installed FlashInfer lacks NVFP4 XQA scale API; no upgrade attempted")
    torch.cuda.set_per_process_memory_fraction(min(1., args.memory_cap_gib*GIB/props.total_memory))
    torch.backends.cuda.matmul.allow_tf32 = False
    metadata = {"event": "metadata", "hostname": platform.node(), "python": sys.version,
                "torch": torch.__version__, "flashinfer": getattr(flashinfer, "__version__", "unknown"),
                "cuda": torch.version.cuda, "gpu": props.name, "compute_capability": [props.major, props.minor],
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "xqa_signature": signature(xqa_batch_decode_with_kv_cache),
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "limitations": ["Kernel-only synthetic attention, not model accuracy or model throughput.",
                    "One-layer distinct-request cache reused ten times; cache reuse may favor timings.",
                    "All-accepted rates assume perfect speculation and omit weights/GDN/draft work.",
                    "Reference checks quantized KV semantics, not accuracy against original BF16 KV.",
                    "Torch allocator cap does not control CUDA allocations outside PyTorch."]}
    emit(metadata)
    report = {"metadata": metadata, "results": [], "errors": []}
    stop = False
    for batch in args.batches:
        for qlen in args.query_lengths:
            try:
                with torch.inference_mode():
                    report["results"].append(run_case(torch, args, batch, qlen))
            except Exception as exc:
                record = {"event": "case_error", "batch": batch, "query_length": qlen,
                          "type": type(exc).__name__, "error": str(exc)}
                report["errors"].append(record)
                emit(record)
                traceback.print_exc(file=sys.stderr)
                stop = "CUDA" in str(exc) or isinstance(exc, torch.cuda.OutOfMemoryError)
            finally:
                gc.collect()
                torch.cuda.empty_cache()
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(report, indent=2)+"\n")
            if stop:
                break
        if stop:
            break
    emit({"event": "complete", "successful_cases": len(report["results"]), "failed_cases": len(report["errors"])})
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
