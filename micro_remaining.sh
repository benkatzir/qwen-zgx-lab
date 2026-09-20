#!/usr/bin/env bash
set -u
root=/home/ben/qwen-lab
run_probe() {
  tag=$1
  shift
  docker run --rm --device nvidia.com/gpu=all --ipc=host --entrypoint python3 \
    -e CUTE_DSL_ARCH=sm_121a -v "$root:/lab" -v "$root/cache:/root/.cache" \
    vllm/vllm-openai:v0.28.0 "$@" > "$root/logs/$tag.log" 2>&1
}
if run_probe gdn-smoke /lab/scripts/bench_gdn.py --batches 1 --query-lengths 1 4 --layers 2 --repeats 2 --output /lab/results/gdn-smoke.json; then
  run_probe gdn-fp32 /lab/scripts/bench_gdn.py --output /lab/results/gdn-fp32.json
fi
run_probe attention-nvfp4-page128 /lab/scripts/bench_attention_nvfp4.py --batches 45 --query-lengths 8 16 --page-size 128 --output /lab/results/attention-nvfp4-page128.json
run_probe attention-nvfp4-page64 /lab/scripts/bench_attention_nvfp4.py --batches 45 --query-lengths 8 16 --page-size 64 --output /lab/results/attention-nvfp4-page64.json
run_probe attention-xqa-checked /lab/scripts/bench_attention_xqa.py --batches 45 50 --query-lengths 4 --output /lab/results/attention-xqa-checked.json
date > "$root/results/micro-remaining-complete.txt"
