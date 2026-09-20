#!/usr/bin/env python3
"""Bounded vLLM 0.28 TurboQuant attention microbenchmark; no model required.

Smoke: python bench_attention_tq.py --batches 1 --query-lengths 1 4 --context 256 --repeats 2
Full:  python bench_attention_tq.py --output attention_tq_results.json

q=1 uses one batch-wide decode call. q>1 faithfully follows stock CUDA TQ's
small-continuation path: a Python loop over requests, each with q synthetic
decode requests, expanded block table, and incremental causal sequence lengths.
This includes query rotation, fused quantized reads and softmax reduction.
It excludes newly generated KV stores, weights, GDN, projections and drafting.
One layer's distinct-per-request cache is reused for ten layer calls.
"""

import argparse
import gc
import inspect
import json
import math
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path

GIB = 1024 ** 3
HQ, HK, D = 16, 2, 256


def emit(x):
    print(json.dumps(x, sort_keys=True), flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--batches', type=int, nargs='+', default=[1, 50])
    p.add_argument('--query-lengths', type=int, nargs='+', default=[1, 4])
    p.add_argument('--context', type=int, default=131072)
    p.add_argument('--layers', type=int, default=10)
    p.add_argument('--page-size', type=int, default=128)
    p.add_argument('--preset', choices=['turboquant_4bit_nc', 'turboquant_3bit_nc'],
                   default='turboquant_4bit_nc')
    p.add_argument('--timing', choices=['eager', 'graph', 'both'], default='both')
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--max-measurement-seconds', type=float, default=30)
    p.add_argument('--memory-cap-gib', type=float, default=12)
    p.add_argument('--seed', type=int, default=73021)
    p.add_argument('--skip-reference-check', action='store_true')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if min(a.batches + a.query_lengths + [a.context, a.layers, a.page_size,
                                        a.warmup, a.repeats, a.memory_cap_gib]) <= 0:
        p.error('All sizes and repetitions must be positive.')
    if max(a.query_lengths) > 128:
        p.error('Only the stock small-continuation q<=128 path is modeled.')
    if a.page_size % 16:
        p.error('Use a page size divisible by 16.')
    return a


def hadamard(torch):
    h = torch.ones((1, 1), dtype=torch.float32)
    while h.shape[0] < D:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return (h / math.sqrt(D)).to('cuda')


def init_cache(torch, store, cfg, cache, h, midpoints, batch, kv_len,
               pages_per_request, page_size, retain_original):
    # Only stream 0's original K/V are retained for descriptive quantization error.
    originals = (torch.empty((kv_len, HK, D), device='cuda', dtype=torch.bfloat16),
                 torch.empty((kv_len, HK, D), device='cuda', dtype=torch.bfloat16)) \
        if retain_original else None
    for b in range(batch):
        for start in range(0, kv_len, 4096):
            n = min(4096, kv_len - start)
            k = torch.randn((n, HK, D), device='cuda', dtype=torch.bfloat16)
            v = torch.randn_like(k)
            slots = torch.arange(n, device='cuda', dtype=torch.int64)
            slots += b * pages_per_request * page_size + start
            store(k, v, cache, slots, h, midpoints,
                  mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
                  value_quant_bits=cfg.effective_value_quant_bits, key_fp8=False)
            if b == 0 and originals is not None:
                originals[0][start:start+n].copy_(k)
                originals[1][start:start+n].copy_(v)
            del k, v, slots
    torch.cuda.synchronize()
    return originals


def unpack_reference(torch, cfg, slots, centroids):
    """Independent Torch unpack, retaining FP32 math used inside decode.

    Returns keys in the rotated basis and ordinary values. It does not call
    the kernel under test or its FP16-only full-dequant helper.
    """
    dims = torch.arange(D, device='cuda', dtype=torch.int64)
    kb = dims * cfg.key_mse_bits
    ki, ks = kb // 8, kb % 8
    raw = slots[..., ki].to(torch.int32) | (slots[..., ki + 1].to(torch.int32) << 8)
    idx = (raw >> ks) & ((1 << cfg.key_mse_bits) - 1)
    kr = centroids[idx]
    if cfg.norm_correction:
        kr = kr / (kr.square().sum(-1, keepdim=True) + 1e-16).sqrt()
    norm_offset = math.ceil(D * cfg.key_mse_bits / 8)
    norms = slots[..., norm_offset:norm_offset+2].contiguous().view(torch.float16).float()
    kr = kr * norms
    vb = dims * cfg.effective_value_quant_bits
    vi, vs = vb // 8, vb % 8
    offset = cfg.key_packed_size
    rawv = slots[..., offset+vi].to(torch.int32)
    if cfg.effective_value_quant_bits == 3:
        rawv |= slots[..., offset+vi+1].to(torch.int32) << 8
    vidx = (rawv >> vs) & ((1 << cfg.effective_value_quant_bits) - 1)
    scale_offset = offset + math.ceil(D * cfg.effective_value_quant_bits / 8)
    scale_zero = slots[..., scale_offset:scale_offset+4].contiguous().view(torch.float16).float()
    vr = vidx.float() * scale_zero[..., :1] + scale_zero[..., 1:2]
    return kr, vr


def attention_reference(torch, q, k, v):
    q_len, kv_len = q.shape[0], k.shape[0]
    query = q.reshape(q_len, HK, HQ // HK, D).permute(1, 2, 0, 3)
    scores = torch.einsum('hgtd,nhd->hgtn', query, k) / math.sqrt(D)
    positions = torch.arange(kv_len, device='cuda')
    ends = kv_len - q_len + torch.arange(q_len, device='cuda')
    scores.masked_fill_(positions[None, None, None, :] > ends[None, None, :, None],
                        float('-inf'))
    return torch.einsum('hgtn,nhd->thgd', scores.softmax(-1), v).reshape(q_len, HQ, D)


def errors(torch, actual, expected):
    delta = actual.float() - expected.float()
    return {'max_abs_error': delta.abs().max().item(),
            'rmse': delta.square().mean().sqrt().item(),
            'relative_rmse': (delta.square().mean().sqrt() /
                              expected.float().square().mean().sqrt().clamp_min(1e-12)).item(),
            'finite': bool(torch.isfinite(actual).all().item())}


def build_references(torch, cfg, cache, centroids, h, q, originals, ppr, kv_len, q_len):
    k = torch.empty((kv_len, HK, D), device='cuda', dtype=torch.float32)
    v = torch.empty_like(k)
    # Physical pages of stream 0 are contiguous; padded final slots are excluded.
    raw = cache[:ppr].reshape(-1, HK, cfg.slot_size_aligned)
    for start in range(0, kv_len, 4096):
        end = min(kv_len, start + 4096)
        kr, vr = unpack_reference(torch, cfg, raw[start:end], centroids)
        k[start:end].copy_(kr)
        v[start:end].copy_(vr)
    qr = q[:q_len].float() @ h
    exact_quantized = attention_reference(torch, qr, k, v)
    original = attention_reference(torch, q[:q_len].float(),
                                   originals[0].float(), originals[1].float())
    return exact_quantized, original


def validate(torch, actual, expected):
    result = errors(torch, actual, expected)
    # Kernel correctness only, allowing BF16 output rounding/reduction differences.
    # This threshold is NOT a downstream model-accuracy or quantization-quality gate.
    torch.testing.assert_close(actual.float(), expected.float(), rtol=0.02, atol=0.002)
    if not result['finite']:
        raise RuntimeError('Kernel output is not finite.')
    result['passed_kernel_numeric_check'] = True
    result['numeric_tolerance'] = {'rtol': 0.02, 'atol': 0.002}
    return result


def measure(torch, fn, args):
    samples, wall_start = [], time.monotonic()
    torch.cuda.synchronize()
    for _ in range(args.repeats):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
        if time.monotonic() - wall_start > args.max_measurement_seconds:
            break
    return samples


def summarize(samples, batch, q_len, layers, payload):
    ms = statistics.median(samples)
    return {'ten_layer_gpu_ms_samples': samples, 'layer_calls': layers,
            'block_gpu_ms_median': ms, 'one_layer_gpu_ms_median': ms/layers,
            'attention_only_all_accepted_tps_per_stream': q_len * 1000/ms,
            'attention_only_all_accepted_tps_aggregate': batch*q_len*1000/ms,
            'nominal_one_pass_kv_payload_GBps': payload * layers/(ms/1000)/1e9,
            'nominal_payload_note': 'Counts each cache once per call block; NOT actual DRAM traffic, and queries/heads issue separate logical loads.'}


def run_case(torch, store, decode, get_centroids, Config, args, batch, q_len):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    cfg = Config.from_cache_dtype(args.preset, D)
    kv_len = args.context + q_len
    ppr = math.ceil(kv_len/args.page_size)
    pages = batch * ppr
    cache_bytes = pages * args.page_size * HK * cfg.slot_size_aligned
    estimated = cache_bytes + 3 * GIB
    free, _ = torch.cuda.mem_get_info()
    if estimated > args.memory_cap_gib*GIB or estimated > free*0.8:
        raise RuntimeError(f'Refusing allocation: estimated {estimated/GIB:.2f}GiB, '
                           f'cap {args.memory_cap_gib:.2f}GiB, available {free/GIB:.2f}GiB.')
    emit({'event': 'case_start', 'batch': batch, 'query_length': q_len,
          'context': args.context, 'cache_GiB': cache_bytes/GIB, 'estimated_GiB': estimated/GIB})
    torch.manual_seed(args.seed + batch*100 + q_len)
    cache = torch.empty((pages, args.page_size, HK, cfg.slot_size_aligned),
                        dtype=torch.uint8, device='cuda')
    h = hadamard(torch)
    centroids = get_centroids(D, cfg.centroid_bits).to(device='cuda', dtype=torch.float32)
    sorted_c = centroids.sort().values
    midpoints = (sorted_c[:-1] + sorted_c[1:])/2
    originals = init_cache(torch, store, cfg, cache, h, midpoints, batch, kv_len,
                           ppr, args.page_size, not args.skip_reference_check)
    q = torch.randn((batch*q_len, HQ, D), device='cuda', dtype=torch.bfloat16)
    out = torch.empty_like(q)
    bt = torch.arange(pages, device='cuda', dtype=torch.int32).view(batch, ppr)
    lens = torch.full((batch,), kv_len, device='cuda', dtype=torch.int32)
    synth_lens = torch.arange(args.context+1, kv_len+1, device='cuda', dtype=torch.int32)
    # Mirrors _decode_attention's WorkspaceManager buffers for q1.
    mid = torch.empty((batch, HQ, 32, D+1), device='cuda', dtype=torch.float32)
    lse = torch.empty((batch, HQ), device='cuda', dtype=torch.float32)
    kwargs = dict(kv_cache=cache, Pi=h, centroids=centroids, scale=1/math.sqrt(D),
                  mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
                  value_quant_bits=cfg.effective_value_quant_bits, key_fp8=False,
                  norm_correction=cfg.norm_correction, PiT=h)

    def layer_block():
        for _ in range(args.layers):
            if q_len == 1:
                decode(query=q, block_table=bt, seq_lens=lens,
                       mid_o_buf=mid, output_buf=out, lse_buf=lse,
                       max_num_kv_splits=32, **kwargs)
            else:
                # Faithful stock _prefill_attention continuation <=128 branch.
                # Stock does not pass shared buffers/max splits on this branch.
                for b in range(batch):
                    start, end = b*q_len, (b+1)*q_len
                    result = decode(query=q[start:end],
                                    block_table=bt[b:b+1].expand(q_len, -1),
                                    seq_lens=synth_lens, **kwargs)
                    out[start:end].copy_(result)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(args.warmup):
            layer_block()
    stream.synchronize()
    if not bool(torch.isfinite(out).all().item()):
        raise RuntimeError('Non-finite attention output.')
    result = {'event': 'case_result', 'batch': batch, 'query_length': q_len,
              'historical_tokens': args.context, 'kv_length': kv_len,
              'preset': args.preset, 'slot_bytes_per_kv_head': cfg.slot_size_aligned,
              'cache_GiB': cache_bytes/GIB,
              'stock_path': 'batch_decode' if q_len == 1 else 'per_request_synthetic_decode_continuation',
              'finite_all_streams': True}
    expected = None
    if not args.skip_reference_check:
        expected, original = build_references(torch, cfg, cache, centroids, h, q,
                                              originals, ppr, kv_len, q_len)
        result['eager_exact_dequant_reference'] = validate(torch, out[:q_len], expected)
        result['quantization_only_vs_original_BF16_KV_attention'] = errors(torch, expected, original)
        result['kernel_plus_quantization_vs_original_BF16_KV_attention'] = errors(torch, out[:q_len], original)
        result['reference_scope'] = 'Stream0; synthetic seeded Gaussian QKV, not model accuracy. Quantization error is descriptive only.'
        del original, originals
        originals = None
    payload = batch * kv_len * HK * cfg.slot_size_aligned
    if args.timing in ('eager', 'both'):
        result['eager'] = summarize(measure(torch, layer_block, args), batch, q_len, args.layers, payload)
    if args.timing in ('graph', 'both'):
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        try:
            with torch.cuda.graph(graph, stream=stream):
                layer_block()
            graph.replay()
            torch.cuda.synchronize()
            if expected is not None:
                result['graph_exact_dequant_reference'] = validate(torch, out[:q_len], expected)
            result['graph'] = summarize(measure(torch, graph.replay, args), batch, q_len, args.layers, payload)
        except Exception as exc:
            result['graph_error'] = f'{type(exc).__name__}: {exc}'
            if args.timing == 'graph':
                raise
        del graph
    result['torch_peak_allocated_GiB'] = torch.cuda.max_memory_allocated()/GIB
    result['torch_peak_reserved_GiB'] = torch.cuda.max_memory_reserved()/GIB
    emit(result)
    return result


def main():
    args = parse_args()
    import torch
    import vllm
    from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
    from vllm.model_executor.layers.quantization.turboquant.centroids import get_centroids
    from vllm.v1.attention.ops.triton_turboquant_store import triton_turboquant_store
    from vllm.v1.attention.ops.triton_turboquant_decode import triton_turboquant_decode_attention
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(min(1., args.memory_cap_gib*GIB/props.total_memory))
    torch.backends.cuda.matmul.allow_tf32 = False
    metadata = {'event': 'metadata', 'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'hostname': platform.node(), 'python': sys.version,
                'torch': torch.__version__, 'vllm': vllm.__version__,
                'torch_cuda': torch.version.cuda, 'gpu': props.name,
                'compute_capability': [props.major, props.minor],
                'args': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                'decode_signature': str(inspect.signature(triton_turboquant_decode_attention)),
                'limitations': [
                    'One layer cache reused ten times; not a full-model memory/capacity test.',
                    'Excludes KV updates, recurrent GDN layers, model projections/MoE, drafting, vision, sampling and scheduling.',
                    'q>1 faithfully loops requests like stock v0.28 CUDA TQ; do not equate with one flattened B*q decode call.',
                    'Manual fixed-shape CUDA graph is an optimistic kernel experiment, not evidence stock serving captures the same q>1 path.',
                    'Random-tensor quantization error is not downstream task accuracy or a97% quality certificate.',
                    'Nominal payload bandwidth is not measured DRAM traffic.',
                    'Torch allocation cap does not govern external library allocations.'
                ]}
    emit(metadata)
    results, failures = [], []
    fatal = False
    for b in args.batches:
        for q_len in args.query_lengths:
            try:
                with torch.inference_mode():
                    results.append(run_case(torch, triton_turboquant_store,
                        triton_turboquant_decode_attention, get_centroids,
                        TurboQuantConfig, args, b, q_len))
            except Exception as exc:
                item = {'event':'case_error', 'batch':b, 'query_length':q_len,
                        'type':type(exc).__name__, 'error':str(exc)}
                failures.append(item)
                emit(item)
                traceback.print_exc(file=sys.stderr)
                fatal = isinstance(exc, torch.cuda.OutOfMemoryError) or 'CUDA' in str(exc)
            finally:
                gc.collect()
                torch.cuda.empty_cache()
            if fatal:
                break
        if fatal:
            break
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'metadata':metadata, 'results':results,
                                          'errors':failures}, indent=2)+'\n')
    emit({'event':'complete', 'successful_cases':len(results), 'failed_cases':len(failures)})
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
