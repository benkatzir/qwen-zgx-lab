#!/usr/bin/env python3
"""State-update microbenchmark for the installed vLLM0.28 Qwen3.5/3.6 GDN.

Smoke: python bench_gdn.py --batches 1 --query-lengths 1 4 --layers 2 --repeats 2
Full:  python bench_gdn.py --output gdn_results.json

Actual pinned/installed model dispatch: q1 uses the packed Triton recurrent
kernel; q>1 uses fused_sigmoid_gating_delta_rule_update (Triton). The CUDA MTP
fast path requires V-heads/K-heads=8; this model has32/16=2 and cannot select it.
Inputs are seeded BF16 post-convolution QKV/gates. FP32 recurrent state is the
default; --state-dtype bfloat16 is a model-precision change, not a free speedup.
One layer's state pool is reused30times. Each request has distinct state slots
for every verification token; speculative kernels write every checkpoint.
No SSH, model loading, convolution, projection, norm or GPU installation.
"""

import argparse
import gc
import hashlib
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
H, HV, K, V = 16, 32, 128, 128
QKV_DIM = 2*H*K + HV*V


def emit(value):
    print(json.dumps(value, sort_keys=True), flush=True)


def args_parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--batches', type=int, nargs='+', default=[45, 50])
    p.add_argument('--query-lengths', type=int, nargs='+', default=[1, 4, 8, 16])
    p.add_argument('--layers', type=int, default=30)
    p.add_argument('--state-dtype', choices=['float32','bfloat16'], default='float32')
    p.add_argument('--timing', choices=['eager','graph','both'], default='both')
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--repeats', type=int, default=7)
    p.add_argument('--max-measurement-seconds', type=float, default=30)
    p.add_argument('--memory-cap-gib', type=float, default=12)
    p.add_argument('--seed', type=int, default=73021)
    p.add_argument('--skip-reference-check', action='store_true')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if min(args.batches+args.query_lengths+[args.layers,args.warmup,args.repeats,
                                           args.memory_cap_gib]) <= 0:
        p.error('Sizes, warmups and repetitions must be positive.')
    return args


def provenance(fn):
    path = inspect.getsourcefile(fn)
    return {'callable': f'{fn.__module__}.{fn.__name__}',
            'signature': str(inspect.signature(fn)), 'source_path': path,
            'source_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if path and Path(path).exists() else None}


def metrics(torch, actual, expected):
    a, e = actual.float(), expected.float()
    delta = a-e
    return {'max_abs_error': delta.abs().max().item(),
            'relative_rmse': (delta.square().mean().sqrt()/
                              e.square().mean().sqrt().clamp_min(1e-12)).item(),
            'finite': bool(torch.isfinite(a).all().item())}


def reference(torch, mixed, a, b, A_log, dt_bias, initial, q_len, state_dtype):
    """Independent FP32 recurrence for request0, matching public kernel math.

    State stays FP32 inside a q-token pass, even when checkpoints store BF16.
    Packed q1 rounds sigmoid(beta) to the BF16 gate dtype; spec path does not.
    """
    q, key, val = mixed[:q_len].float().split([H*K,H*K,HV*V], dim=-1)
    q, key = q.reshape(q_len,H,K), key.reshape(q_len,H,K)
    val = val.reshape(q_len,HV,V)
    q = q / (q.square().sum(-1,keepdim=True)+1e-6).sqrt()
    key = key / (key.square().sum(-1,keepdim=True)+1e-6).sqrt()
    q = q.repeat_interleave(HV//H,dim=1) / math.sqrt(K)
    key = key.repeat_interleave(HV//H,dim=1)
    state = initial.float()
    outputs, checkpoints = [], []
    for t in range(q_len):
        g = -A_log.float().exp() * torch.nn.functional.softplus(a[t].float()+dt_bias.float())
        beta = b[t].float().sigmoid()
        if q_len == 1:
            beta = beta.to(b.dtype).float()
        state = state * g.exp()[:,None,None]
        delta = (val[t] - (state*key[t,:,None,:]).sum(-1))*beta[:,None]
        state = state + delta[:,:,None]*key[t,:,None,:]
        outputs.append((state*q[t,:,None,:]).sum(-1))
        checkpoints.append(state.to(state_dtype))
    return torch.stack(outputs).to(mixed.dtype), torch.stack(checkpoints)


def measure(torch, fn, args):
    torch.cuda.synchronize()
    times, wall = [], time.monotonic()
    for _ in range(args.repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
        if time.monotonic()-wall > args.max_measurement_seconds:
            break
    return times


def summarize(times, batch, q_len, layers, state_bytes):
    ms = statistics.median(times)
    # Per layer: one initial state read + q intermediate/final checkpoint writes.
    logical_bytes = batch*(1+q_len)*state_bytes*layers
    return {'block_gpu_ms_samples':times, 'block_gpu_ms_median':ms,
            'one_layer_gpu_ms_median':ms/layers,
            'state_only_all_accepted_tps_per_stream':q_len*1000/ms,
            'state_only_all_accepted_tps_aggregate':batch*q_len*1000/ms,
            'nominal_state_read_write_payload_GBps':logical_bytes/(ms/1000)/1e9,
            'payload_note':'One state read plus q checkpoint writes per layer; not hardware-counter DRAM traffic.'}


def run_case(torch, packed, spec, args, batch, q_len):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    state_dtype = getattr(torch,args.state_dtype)
    state_bytes = HV*V*K*torch.empty((),dtype=state_dtype).element_size()
    slots = 1+batch*q_len  # slot0 is vLLM's null block, never a real request.
    pool_bytes = slots*state_bytes
    estimate = pool_bytes + 2*GIB
    free, _ = torch.cuda.mem_get_info()
    if estimate > args.memory_cap_gib*GIB or estimate > free*0.8:
        raise RuntimeError(f'Refusing allocation: estimate {estimate/GIB:.2f}GiB, '
                           f'cap {args.memory_cap_gib:.2f}GiB, free {free/GIB:.2f}GiB.')
    emit({'event':'case_start','batch':batch,'query_length':q_len,
          'state_dtype':args.state_dtype,'state_pool_GiB':pool_bytes/GIB})
    torch.manual_seed(args.seed+batch*100+q_len)
    state = torch.empty((slots,HV,V,K),device='cuda',dtype=state_dtype).normal_(0,0.05)
    state[0].zero_()
    tokens = batch*q_len
    mixed = torch.randn((tokens,QKV_DIM),device='cuda',dtype=torch.bfloat16)
    a = torch.randn((tokens,HV),device='cuda',dtype=torch.bfloat16)
    b = torch.randn_like(a)
    A_log = torch.randn((HV,),device='cuda',dtype=torch.float32)*0.25-1.0
    dt_bias = torch.randn((HV,),device='cuda',dtype=torch.bfloat16)*0.25
    indices = torch.arange(1,slots,device='cuda',dtype=torch.int32).view(batch,q_len)
    cu_lens = torch.arange(batch+1,device='cuda',dtype=torch.int32)*q_len
    # Assume all previous draft positions accepted: start from the last slot.
    # Current q updates still write every slot0..q-1, as stock MTP does.
    accepted = torch.full((batch,),q_len,device='cuda',dtype=torch.int32)
    q_raw,k_raw,v_raw = mixed.split([H*K,H*K,HV*V],dim=-1)
    q = q_raw.view(1,tokens,H,K)
    key = k_raw.view(1,tokens,H,K)
    val = v_raw.view(1,tokens,HV,V)
    last_out = torch.empty((batch,1,HV,V),device='cuda',dtype=torch.bfloat16) if q_len==1 else None

    def one_layer():
        nonlocal last_out
        if q_len == 1:
            packed(mixed_qkv=mixed,a=a,b=b,A_log=A_log,dt_bias=dt_bias,
                   scale=K**-0.5,initial_state=state,out=last_out,
                   ssm_state_indices=indices[:,0],use_qk_l2norm_in_kernel=True)
        else:
            last_out,_ = spec(A_log=A_log,a=a,b=b,dt_bias=dt_bias,
                q=q,k=key,v=val,initial_state=state,inplace_final_state=True,
                cu_seqlens=cu_lens,ssm_state_indices=indices,
                num_accepted_tokens=accepted,use_qk_l2norm_in_kernel=True)

    def layer_block():
        for _ in range(args.layers):
            one_layer()

    result = {'event':'case_result','batch':batch,'query_length':q_len,
              'state_dtype':args.state_dtype,'state_pool_GiB':pool_bytes/GIB,
              'per_request_one_layer_state_bytes':state_bytes,
              'layer_calls':args.layers,'previous_accepted_positions':q_len,
              'kernel':'packed_recurrent_triton' if q_len==1 else 'sigmoid_gating_recurrent_triton',
              'new_checkpoint_writes_per_request_per_layer':q_len}
    if not args.skip_reference_check:
        initial = state[q_len].clone()
        expected_out,expected_states = reference(torch,mixed,a,b,A_log,dt_bias,
                                                 initial,q_len,state_dtype)
        one_layer()
        torch.cuda.synchronize()
        actual_out = last_out.reshape(tokens,HV,V)[:q_len]
        actual_states = state[1:1+q_len]
        result['independent_reference'] = {
            'scope':'First request, one layer/pass, all checkpoint states and output positions.',
            'output':metrics(torch,actual_out,expected_out),
            'checkpoints':metrics(torch,actual_states,expected_states),
            'numeric_tolerance':{'rtol':0.02,'atol':0.002},
            'note':'Kernel-math check only, not task accuracy or permission to reduce state precision.'}
        torch.testing.assert_close(actual_out,expected_out,rtol=0.02,atol=0.002)
        torch.testing.assert_close(actual_states,expected_states,rtol=0.02,atol=0.002)
        result['independent_reference']['passed'] = True
        del initial,expected_out,expected_states
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(args.warmup):
            layer_block()
    stream.synchronize()
    if not bool(torch.isfinite(last_out).all().item()):
        raise RuntimeError('Non-finite GDN output.')
    if args.timing in ('eager','both'):
        result['eager'] = summarize(measure(torch,layer_block,args),batch,q_len,args.layers,state_bytes)
    if args.timing in ('graph','both'):
        # Validate the captured30-call block against eager from identical first-
        # request state. Requests are independent; copying only these slots keeps
        # memory bounded. The one-pass independent recurrence was checked above.
        saved = state[1:1+q_len].clone()
        layer_block()
        torch.cuda.synchronize()
        expected_out = last_out.reshape(tokens,HV,V)[:q_len].clone()
        expected_states = state[1:1+q_len].clone()
        state[1:1+q_len].copy_(saved)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        try:
            with torch.cuda.graph(graph,stream=stream):
                layer_block()
            state[1:1+q_len].copy_(saved)
            graph.replay()
            torch.cuda.synchronize()
            actual_out = last_out.reshape(tokens,HV,V)[:q_len]
            actual_states = state[1:1+q_len]
            torch.testing.assert_close(actual_out,expected_out,rtol=0.02,atol=0.002)
            torch.testing.assert_close(actual_states,expected_states,rtol=0.02,atol=0.002)
            result['graph_vs_eager_same_start'] = {
                'passed':True,'output':metrics(torch,actual_out,expected_out),
                'checkpoints':metrics(torch,actual_states,expected_states)}
            result['graph'] = summarize(measure(torch,graph.replay,args),batch,q_len,args.layers,state_bytes)
        except Exception as exc:
            result['graph_error'] = f'{type(exc).__name__}: {exc}'
            if args.timing == 'graph':
                raise
        del graph,saved,expected_out,expected_states
    result['finite_output'] = bool(torch.isfinite(last_out).all().item())
    result['torch_peak_allocated_GiB'] = torch.cuda.max_memory_allocated()/GIB
    result['torch_peak_reserved_GiB'] = torch.cuda.max_memory_reserved()/GIB
    emit(result)
    return result


def main():
    args = args_parse()
    import torch
    import vllm
    from vllm.third_party.flash_linear_attention.ops import (
        fused_recurrent_gated_delta_rule_packed_decode as packed,
        fused_sigmoid_gating_delta_rule_update as spec)
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(min(1.,args.memory_cap_gib*GIB/props.total_memory))
    torch.backends.cuda.matmul.allow_tf32=False
    metadata = {'event':'metadata','utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                'hostname':platform.node(),'python':sys.version,'torch':torch.__version__,
                'vllm':vllm.__version__,'gpu':props.name,
                'compute_capability':[props.major,props.minor],
                'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                'packed_kernel':provenance(packed),'spec_kernel':provenance(spec),
                'limitations':[
                    'Only GDN post-convolution state update/gating and its wrapper copies/allocations; excludes convolution, output norm/gate, projections, MoE, full attention, drafting, sampling and vision.',
                    'State buffers for one layer are reused30times; not full-model capacity. Distinct state/checkpoint slots per request.',
                    'q1 and q>1 paths match v0.28.0 installed Qwen GDN dispatch at value/key-head ratio2. CUDA fused MTP requires ratio8 and is not selected.',
                    'Repeated inputs/state updates are a synthetic kernel workload, not real-model decoding or acceptance.',
                    'FP32 is native recurrent state; BF16 option changes model precision and requires separate quality evaluation.',
                    'Nominal state payload bandwidth is not hardware-counter DRAM traffic.',
                    'Manual fixed-shape graph timings do not establish stock end-to-end graph capture/performance.',
                    'Torch allocator limit does not limit native allocations outside its allocator.'
                ]}
    emit(metadata)
    results,failures=[],[]
    fatal=False
    for batch in args.batches:
        for q_len in args.query_lengths:
            try:
                with torch.inference_mode():
                    results.append(run_case(torch,packed,spec,args,batch,q_len))
            except Exception as exc:
                failure={'event':'case_error','batch':batch,'query_length':q_len,
                         'type':type(exc).__name__,'error':str(exc)}
                failures.append(failure);emit(failure)
                traceback.print_exc(file=sys.stderr)
                fatal=isinstance(exc,torch.cuda.OutOfMemoryError) or 'CUDA' in str(exc)
            finally:
                gc.collect();torch.cuda.empty_cache()
            if fatal:break
        if fatal:break
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps({'metadata':metadata,'results':results,'errors':failures},indent=2)+'\n')
    emit({'event':'complete','successful_cases':len(results),'failed_cases':len(failures)})
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
