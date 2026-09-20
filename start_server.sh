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
  "${extra[@]}"
