#!/usr/bin/env bash
set -euo pipefail
root=/home/ben/qwen-lab
variant=${1:-unsloth}
speculation=${2:-0}
backend=${3:-auto}
name=qwen-lab-server
image=vllm/vllm-openai:v0.28.0
kv_dtype=${KV_DTYPE:-fp8}
attention_backend=${ATTENTION_BACKEND:-FLASHINFER}
gpu_util=${GPU_UTIL:-0.87}
graph_max=${GRAPH_MAX:-64}
prefill_cap=${PREFILL_CAP:-0}
max_model_len=${MAX_MODEL_LEN:-262144}
# Authentication. Every route except GET/HEAD /health requires "Authorization: Bearer <key>".
# The key lives in a root-only file OUTSIDE $root (which is mounted into the container read-write)
# and is bind-mounted read-only as a secret; it is never an environment variable or an argument.
# The launcher refuses to start without it. Create one with:
#   sudo install -d -m 700 /home/ben/qwen-lab-auth
#   sudo sh -c 'umask 077; openssl rand -hex 32 > /home/ben/qwen-lab-auth/api-key'
auth_dir=${AUTH_DIR:-/home/ben/qwen-lab-auth}
api_key_file=${API_KEY_FILE:-$auth_dir/api-key}
test -f "$auth_dir/private_auth.py" || { echo "Missing $auth_dir/private_auth.py (copy auth/private_auth.py from the repo)" >&2; exit 2; }
test -f "$api_key_file" || { echo "Missing API key file: $api_key_file" >&2; exit 2; }
python3 - "$api_key_file" <<'PY'
import pathlib, stat, sys
p = pathlib.Path(sys.argv[1]); st = p.stat(); key = p.read_bytes().strip()
assert stat.S_ISREG(st.st_mode) and not st.st_mode & 0o077, "Key file permissions must be 0600 or stricter"
assert 32 <= len(key) <= 512 and key.isascii() and all(c > 32 for c in key), "Key must be 32..512 non-whitespace ASCII bytes"
PY
extra=()
if [[ "$prefill_cap" != 0 ]]; then
  extra+=(--long-prefill-token-threshold "$prefill_cap")
fi
if [[ "$backend" != auto ]]; then extra+=(--moe-backend "$backend"); fi
if [[ "$speculation" != 0 ]]; then
  extra+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$speculation,\"moe_backend\":\"triton\"}")
fi
docker run -d --name "$name" --device nvidia.com/gpu=all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -e CUTE_DSL_ARCH=sm_121a -e VLLM_USE_DEEP_GEMM=0 -e HF_HOME=/lab/hf \
  -e VLLM_SERVER_DEV_MODE=1 \
  -e PYTHONPATH=/opt/private-auth -e PRIVATE_API_KEY_FILE=/run/secrets/api-key \
  -v "$auth_dir/private_auth.py:/opt/private-auth/private_auth.py:ro" \
  -v "$api_key_file:/run/secrets/api-key:ro" \
  -v "$root:/lab" -v "$root/cache:/root/.cache" "$image" \
  /lab/models/"$variant" --served-model-name qwen-lab \
  --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 \
  --max-model-len "$max_model_len" --max-num-seqs 50 \
  --max-num-batched-tokens 8192 --gpu-memory-utilization "$gpu_util" \
  --max-cudagraph-capture-size "$graph_max" \
  --kv-cache-dtype "$kv_dtype" --attention-backend "$attention_backend" \
  --enable-chunked-prefill --async-scheduling --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image":2,"video":1}' --skip-mm-profiling \
  --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \
  "${extra[@]}" \
  --middleware private_auth.BearerMiddleware --disable-fastapi-docs
