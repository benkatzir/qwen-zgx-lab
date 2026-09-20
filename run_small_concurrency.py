#!/usr/bin/env python3
"""Sequential, exploratory 128K chat tests after the full45 run finishes."""
import json
import subprocess
from pathlib import Path
from urllib.request import urlopen

ROOT = Path('/home/ben/qwen-lab')
full = ROOT / 'results/fp8-mtp3-c45-chat-128k.json'
if not full.exists():
    raise SystemExit('Full45 result missing; refusing concurrent GPU work')
with urlopen('http://127.0.0.1:8000/metrics', timeout=10) as response:
    metrics = response.read().decode()
for metric in ('vllm:num_requests_running{', 'vllm:num_requests_waiting{'):
    lines = [line for line in metrics.splitlines() if line.startswith(metric)]
    if not lines or any(float(line.rsplit(' ', 1)[1]) != 0 for line in lines):
        raise SystemExit(f'Server is not confirmed idle: {metric}')

def run(concurrency):
    out = ROOT / f'results/fp8-mtp3-c{concurrency}-chat-128k-exploratory.json'
    log = ROOT / f'logs/fp8-mtp3-c{concurrency}-chat-128k-exploratory.log'
    command = [str(ROOT / 'venv/bin/python'), str(ROOT / 'scripts/bench_streams.py'),
               '--model', 'qwen-lab', '--concurrency', str(concurrency),
               '--prompt-tokens', '131072', '--output-tokens', '1024',
               '--respect-eos', '--min-overlap-seconds', '10', '--timeout', '7200',
               '--out', str(out)]
    print('Starting exploratory concurrency', concurrency, flush=True)
    with log.open('w') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    result = json.loads(out.read_text())
    summary = result.get('summary')
    if result.get('fatal_error') or not summary:
        raise SystemExit(f'Benchmark failed; inspect {out}')
    print(json.dumps({'concurrency': concurrency, 'summary': summary}), flush=True)
    return summary

first = run(4)
rate = first.get('minimum_stream_overlap_tps')
if rate is None:
    raise SystemExit('No valid C4 common rate; manual review required')
second = run(8 if rate >= 40 else 2)
if rate < 40 and (second.get('minimum_stream_overlap_tps') or 0) < 40:
    run(1)
(ROOT / 'results/small-concurrency-complete.txt').write_text('Completed sequential exploratory tests.\n')
