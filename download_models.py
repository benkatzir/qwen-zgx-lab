import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from concurrent.futures import ThreadPoolExecutor

ROOT = Path('/home/ben/qwen-lab')
MODELS = [
    ('nvidia/Qwen3.6-35B-A3B-NVFP4', '1355db6a052410cfd62085d94b58866fd0f2c3c5'),
    ('unsloth/Qwen3.6-35B-A3B-NVFP4-Fast', '1c3f884bc99aac2524f6d49bcbac8c88401afd66'),
]

def download(model):
    repo, revision = model
    info = HfApi().model_info(repo, revision=revision)
    name = repo.split('/')[0]
    (ROOT / f'{name}-revision.json').write_text(json.dumps({'repo': repo, 'revision': info.sha}, indent=2))
    print(f'Downloading {repo} revision {info.sha}', flush=True)
    path = snapshot_download(repo, revision=info.sha, local_dir=ROOT / 'models' / name, max_workers=8)
    print(f'COMPLETE {repo} {path}', flush=True)

with ThreadPoolExecutor(max_workers=2) as pool:
    list(pool.map(download, MODELS))
