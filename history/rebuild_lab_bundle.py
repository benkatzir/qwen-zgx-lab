"""Validate and package the completed native-context results, never a partial sweep."""
import hashlib
import json
from pathlib import Path
import re
import zipfile

workspace = Path(__file__).resolve().parent.parent
bundle = workspace / 'outputs/qwen-zgx-lab'
review = json.loads((bundle/'results/native262k-comparison.json').read_text())
state = json.loads((bundle/'results/native262k-sweep-state.json').read_text())
assert review['audit_pass'] and review['complete_sweep']
assert all(r['filled_native_budget_all_streams'] for r in review['runs'])
assert state['status'] == 'complete' and state['telemetry_pid'] is None
readme = (bundle/'README.md').read_text()
assert '**In progress:**' not in readme and '**Pending:**' not in readme
assert not (bundle/'RESUME.md').read_text().startswith('# Active')
for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', readme):
    if not target.startswith(('http:', 'https:', '#')):
        assert (bundle/target.split('#')[0]).exists(), target
files = sorted(p for p in bundle.rglob('*') if p.is_file() and p.name != 'SHA256SUMS')
hashes = {str(p.relative_to(bundle)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
manifest = bundle/'SHA256SUMS'
manifest.write_text(''.join(f'{digest}  {name}\n' for name,digest in hashes.items()))
archive = workspace/'outputs/qwen-zgx-lab.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in files + [manifest]:
        z.write(p, arcname=str(p.relative_to(bundle.parent)))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    assert z.read('qwen-zgx-lab/SHA256SUMS') == manifest.read_bytes()
    for name,digest in hashes.items():
        assert hashlib.sha256(z.read('qwen-zgx-lab/'+name)).hexdigest() == digest, name
print(json.dumps({'zip':str(archive), 'bytes':archive.stat().st_size,
                  'manifest_files_verified':len(hashes), 'zip_crc_and_sha256':'pass'}, indent=2))
