"""Collect completed cases only; never changes the running model or benchmark."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / 'outputs/qwen-zgx-lab'
SOCKET = ROOT / 'work/zgx-ssh-native262k.sock'
REMOTE = '/home/ben/qwen-lab'

parser = argparse.ArgumentParser()
parser.add_argument('--allow-partial', action='store_true')
args = parser.parse_args()

script = '''
import datetime, json, tarfile, urllib.request
from pathlib import Path
root = Path('/home/ben/qwen-lab')
state = json.loads((root/'results/native262k-sweep-state.json').read_text())
if not ALLOW_PARTIAL:
    assert state['status'] == 'complete' and state['telemetry_pid'] is None, state['status']
paths = [root/'results/native262k-sweep-state.json', root/'logs/native262k-sweep.log',
         root/'results/native262k-telemetry.jsonl', root/'logs/native262k-telemetry.log']
for case in state['completed']:
    stem = 'native262k-c{}-260096in-2048out'.format(case['concurrency'])
    paths.extend(root/sub/(stem+suffix) for sub,suffix in
        [('results','.json'), ('results','.trace.jsonl'), ('logs','.log')])
if not ALLOW_PARTIAL:
    with urllib.request.urlopen('http://127.0.0.1:8000/v1/models') as response:
        models = json.load(response)
    with urllib.request.urlopen('http://127.0.0.1:8000/metrics') as response:
        metrics = response.read().decode()
    with urllib.request.urlopen('http://127.0.0.1:8000/health') as response:
        health = response.status
    values = {}
    for name in ('num_requests_running','num_requests_waiting','num_preemptions_total',
                 'prompt_tokens_total','generation_tokens_total'):
        values[name] = sum(float(line.rsplit(' ',1)[1]) for line in metrics.splitlines()
                          if line.startswith('vllm:'+name+'{'))
    assert models['data'][0]['max_model_len'] == 262144 and health == 200
    assert values['num_requests_running'] == values['num_requests_waiting'] == 0, values
    final = root/'results/native262k-final-health.json'
    final.write_text(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'health_http_status':health, 'models':models, 'scheduler':values}, indent=2)+'\\n')
    paths.append(final)
archive = root/'results/native262k-export.tar.gz'
with tarfile.open(archive, 'w:gz') as tar:
    for path in paths:
        tar.add(path, arcname=str(path.relative_to(root)))
print(json.dumps({'completed':[r['concurrency'] for r in state['completed']], 'status':state['status'],
                  'archive':str(archive), 'bytes':archive.stat().st_size}))
'''.replace('ALLOW_PARTIAL', repr(args.allow_partial))
ssh = ['ssh', '-S', str(SOCKET), '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'ben@zgx-140a']
subprocess.run(ssh + ['python3 -'], input=script, text=True, check=True)
archive = (ROOT/'work/native262k-completed-evidence.tar.gz' if args.allow_partial
           else BUNDLE/'native262k-evidence.tar.gz')
subprocess.run(['scp','-o',f'ControlPath={SOCKET}',
    f'ben@zgx-140a:{REMOTE}/results/native262k-export.tar.gz', str(archive)], check=True)
with tarfile.open(archive, 'r:gz') as tar:
    for member in tar.getmembers():
        if member.isfile() and member.name.startswith('results/') and member.name.endswith('.json'):
            name = Path(member.name).name
            (BUNDLE/'results'/name).write_bytes(tar.extractfile(member).read())
comparison = BUNDLE/'results/native262k-comparison.json'
cmd = [sys.executable, str(ROOT/'work/review_native262k_sweep.py'), str(archive), '--out', str(comparison)]
if args.allow_partial:
    cmd.append('--allow-partial')
subprocess.run(cmd, check=True)
d = json.loads(comparison.read_text())
print(json.dumps({'audit_pass':d['audit_pass'], 'complete_sweep':d['complete_sweep'],
    'runs':[{k:r[k] for k in ('concurrency','min_tps','mean_tps','max_tps','aggregate_tps',
                              'common_seconds','ttft_max_seconds','filled_native_budget_all_streams')}
            for r in d['runs']]}, indent=2))
