#!/usr/bin/env bash
set -u
root=/home/ben/qwen-lab
while kill -0 30281 2>/dev/null; do sleep 5; done
test -f "$root/results/micro-remaining-complete.txt" || exit 1
sync
echo 3 > /proc/sys/vm/drop_caches
PREFILL_CAP=128 GRAPH_MAX=192 bash "$root/scripts/start_server.sh" unsloth 3 auto || exit 1
docker logs -f qwen-lab-server > "$root/logs/unsloth-fp8-mtp3-full45.log" 2>&1 &
docker inspect qwen-lab-server --format '{{json .Config.Cmd}}' > "$root/results/unsloth-fp8-mtp3-full45-args.json"
ready=0
for attempt in $(seq 1 300); do
  if curl -fsS --max-time 2 http://127.0.0.1:8000/health >/dev/null; then ready=1; break; fi
  sleep 2
done
if [[ $ready != 1 ]]; then exit 1; fi
sudo -u ben "$root/venv/bin/python" "$root/scripts/smoke_multimodal.py" --model qwen-lab --out "$root/results/modality-fp8-mtp3.json" > "$root/logs/modality-fp8-mtp3.log" 2>&1
sudo -u ben "$root/venv/bin/python" "$root/scripts/bench_streams.py" --model qwen-lab --concurrency 45 --prompt-tokens 131072 --output-tokens 512 --respect-eos --min-overlap-seconds 60 --timeout 14400 --out "$root/results/fp8-mtp3-c45-chat-128k.json" > "$root/logs/fp8-mtp3-c45-chat-128k.log" 2>&1
date > "$root/results/full45-complete.txt"
