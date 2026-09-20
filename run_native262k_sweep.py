"""Sequential, resumable native-window sweep on the existing qwen-lab server."""
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

ROOT = Path('/home/ben/qwen-lab')
RESULTS = ROOT / 'results'
STATE = RESULTS / 'native262k-sweep-state.json'
COUNTS = [1, 2, 4, 8, 16]


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(state):
    state['updated_utc'] = utc()
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(state, indent=2) + '\n')
    tmp.replace(STATE)


def server_idle():
    with urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=10) as r:
        models = json.load(r)
    assert models['data'][0]['max_model_len'] == 262144, models
    with urllib.request.urlopen('http://127.0.0.1:8000/metrics', timeout=10) as r:
        metrics = r.read().decode()
    for metric in ('num_requests_running', 'num_requests_waiting'):
        values = [float(line.rsplit(' ', 1)[1]) for line in metrics.splitlines()
                  if line.startswith('vllm:' + metric + '{')]
        assert values and sum(values) == 0, (metric, values)
    return models


def review(path, concurrency):
    d = json.loads(path.read_text())
    assert d['fatal_error'] is None and d['summary'], d.get('fatal_error')
    assert d['config']['concurrency'] == concurrency
    assert d['config']['prompt_tokens'] == 260096
    assert d['config']['output_tokens'] == 2048
    assert d['config']['respect_eos'] and d['config']['chat_prompt']
    assert not d['config']['warm_prefixes'] and not d['config']['disable_thinking']
    assert len(d['requests']) == concurrency
    for r in d['requests']:
        assert not r['error'] and r['done_received'], r.get('error')
        assert r['prompt_tokens_verified'] and r['event_tokens_verified']
    rates = [r['all_stream_overlap_tps'] for r in d['requests']]
    assert all(x is not None for x in rates), 'No common decode window'
    return {'concurrency': concurrency, 'result': str(path),
            'input_tokens': 260096, 'output_cap': 2048,
            'all_reached_native_window': all(r['output_tokens_verified'] for r in d['requests']),
            'minimum_tps': min(rates), 'maximum_tps': max(rates),
            'mean_tps': sum(rates) / concurrency, 'aggregate_tps': sum(rates),
            'overlap_seconds': d['summary']['all_stream_overlap_seconds'],
            'ttft_min_seconds': min(r['ttft_seconds'] for r in d['requests']),
            'ttft_max_seconds': max(r['ttft_seconds'] for r in d['requests']),
            'checks': d['summary']['checks'],
            'metrics': d['summary']['metrics']}


def main():
    lock = open(RESULTS / 'native262k-sweep.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'status': 'running', 'started_utc': utc(), 'wrapper_pid': os.getpid(),
             'concurrency_sweep': COUNTS, 'input_tokens': 260096,
             'output_cap': 2048, 'server_context_limit': 262144, 'completed': []}
    state['models_at_start'] = server_idle()
    telemetry_log = open(ROOT / 'logs/native262k-telemetry.log', 'a')
    telemetry = subprocess.Popen([sys.executable, str(ROOT / 'scripts/telemetry.py'),
        '--out', str(RESULTS / 'native262k-telemetry.jsonl'), '--seconds', '14400',
        '--interval', '10'], stdout=telemetry_log, stderr=subprocess.STDOUT)
    state['telemetry_pid'] = telemetry.pid
    save(state)
    try:
        for concurrency in COUNTS:
            stem = f'native262k-c{concurrency}-260096in-2048out'
            out = RESULTS / (stem + '.json')
            if out.exists():
                row = review(out, concurrency)
            else:
                server_idle()
                cmd = [sys.executable, str(ROOT / 'scripts/bench_streams.py'),
                       '--model', 'qwen-lab', '--concurrency', str(concurrency),
                       '--prompt-tokens', '260096', '--output-tokens', '2048',
                       '--respect-eos', '--target-tps', '18',
                       '--min-overlap-seconds', '10', '--timeout', '14400',
                       '--out', str(out)]
                with open(ROOT / 'logs' / (stem + '.log'), 'w') as log:
                    child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                    state.update(current_concurrency=concurrency, client_pid=child.pid,
                                 current_started_utc=utc(), current_command=cmd)
                    save(state)
                    print(json.dumps({'started': concurrency, 'pid': child.pid, 'utc': utc()}), flush=True)
                    code = child.wait()
                    state['client_exit_code'] = code
                if code not in (0, 2):
                    raise RuntimeError(f'Benchmark process exited {code}')
                row = review(out, concurrency)
            state['completed'].append(row)
            state['client_pid'] = None
            save(state)
            print(json.dumps({'completed': row}), flush=True)
        state['status'] = 'complete'
        state['current_concurrency'] = None
        state['finished_utc'] = utc()
        save(state)
    except Exception as exc:
        state.update(status='needs_review', error=f'{type(exc).__name__}: {exc}')
        save(state)
        raise
    finally:
        telemetry.terminate()
        telemetry.wait(timeout=10)
        telemetry_log.close()
        state['telemetry_pid'] = None
        save(state)


if __name__ == '__main__':
    main()
