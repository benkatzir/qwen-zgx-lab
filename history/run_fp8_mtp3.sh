#!/usr/bin/env bash
set -u
root=/home/ben/qwen-lab
for attempt in $(seq 1 180); do
  if curl -fsS --max-time 2 http://127.0.0.1:8000/health >/dev/null; then break; fi
  sleep 2
done
curl -fsS --max-time 3 http://127.0.0.1:8000/health >/dev/null || exit 1
"$root/venv/bin/python" "$root/scripts/bench_streams.py" --model qwen-lab --concurrency 50 --prompt-tokens 2048 --output-tokens 512 --out "$root/results/fp8-mtp3-c50-2k.json" > "$root/logs/fp8-mtp3-c50-2k.log" 2>&1
"$root/venv/bin/python" "$root/scripts/bench_streams.py" --model qwen-lab --concurrency 1 --prompt-tokens 131072 --output-tokens 8192 --min-overlap-seconds 60 --out "$root/results/fp8-mtp3-c1-128k.json" > "$root/logs/fp8-mtp3-c1-128k.log" 2>&1
"$root/venv/bin/python" "$root/scripts/bench_streams.py" --model qwen-lab --concurrency 4 --prompt-tokens 131072 --output-tokens 8192 --min-overlap-seconds 60 --out "$root/results/fp8-mtp3-c4-128k.json" > "$root/logs/fp8-mtp3-c4-128k.log" 2>&1
