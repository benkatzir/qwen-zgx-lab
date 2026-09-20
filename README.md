# Qwen3.6-35B-A3B on ZGX-140A

- **262K-window result:** four streams sustain at least 18 tok/s each; eight and sixteen fall below 18. Counts 5–7 remain untested.
- **Measured at 128K input:** two streams exceed 40 tok/s each; eight exceed the revised 18 tok/s target. Sixteen fail 18 tok/s. Concurrency 9–15 remains untested.
- **45–50 streams at 40 tok/s each: not achieved.** At least 97% accuracy retention versus BF16 remains unverified.

## Exact current recipe

- Model: `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast`, revision `1c3f884bc99aac2524f6d49bcbac8c88401afd66`.
- Image: `vllm/vllm-openai:v0.28.0`; recorded multiarch digest `sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14`.
- GB10/SM121, TP1; FlashInfer, FP8 KV, native FP32 GDN state; MTP3 with Triton drafting and automatic target MoE backend.
- Native **262,144 total input + output tokens**; 50 maximum sequences; 8,192 batched tokens; GPU utilization 0.87; graph capture 192; long-prefill cap 128.
- Chunked prefill and async scheduling enabled; prefix caching disabled. Vision retained: two images / one video per prompt. Multimodal profiling skipped; development admission endpoints enabled.
- Full flags: [start_server.sh](start_server.sh). Node weights: `/home/ben/qwen-lab/models/unsloth`; scripts: `/home/ben/qwen-lab/scripts`; results: `/home/ben/qwen-lab/results`.

## Start and connect

1. On the ZGX, restart the existing container with the bundled launcher:

```bash
sudo docker stop qwen-lab-server
sudo docker rm qwen-lab-server
sudo env MAX_MODEL_LEN=262144 GPU_UTIL=0.87 PREFILL_CAP=128 GRAPH_MAX=192 \
  KV_DTYPE=fp8 ATTENTION_BACKEND=FLASHINFER \
  bash /home/ben/qwen-lab/scripts/start_server.sh unsloth 3 auto
```

2. Wait for startup, then verify:

```bash
sudo docker logs -f qwen-lab-server
# In another terminal:
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
```

3. From your computer, replace `ZGX_HOST` with its SSH address:

```bash
ssh -N -L 18000:127.0.0.1:8000 ben@ZGX_HOST
```

- OpenAI-compatible base URL: `http://127.0.0.1:18000/v1`; model: `qwen-lab`.
- Stop: `sudo docker stop qwen-lab-server` on the ZGX.

## Measured serving results

Historical MTP3 results. Rates count accepted output tokens while all streams decode simultaneously.

| Streams | Occupied input each | Output each | Minimum tok/s | Aggregate tok/s | Common decode | Outcome |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 128K | 2,048 cap | 73.866 | 73.866 | 27.712 s | Short exploratory run |
| 1 | 8K | 2,048 cap | 90.457 | 90.457 | 22.629 s | Short-context comparison |
| **2** | **128K** | **Natural EOS** | **53.693** | **107.955** | **65.073 s** | **Pass 40 tok/s each** |
| 3 | 128K | Natural EOS | 38.777 | 123.425 | 78.525 s | Minimum below 40 |
| 4 | 128K | 1,024 cap | 35.685 | 156.442 | 21.970 s | Below 40; exploratory |
| **8** | **128K** | **2,048 cap** | **20.180** | **182.125** | **67.838 s** | **Pass revised 18 tok/s each** |
| 16 | 128K | 2,048 cap | 12.663 | 212.862 | 143.802 s | All below 18 |
| 45 | 128K | 512 cap | 6.478 | 304.427 | 69.462 s | Contexts fit; speed fails |
| 50 | 2K | 512 cap | 17.419 | 970.381 | 22.963 s | Short-context comparison |

- **Workload:** synthetic, independent chat histories; greedy (`temperature=0`), with thinking tokens included. 128K means 131,072 occupied input tokens. Capped outputs are not necessarily completed answers.
- **Context cap:** the nine historical rows above used 139,264. The native-window results below use the current 262,144 limit.
- **Natural EOS:** the two-stream run completed 4,031/3,532 tokens. Its fixed 8,192-output-quota check fails, while speed and 60-second overlap pass. Repeated planning may raise MTP acceptance.
- **Quality:** text/image/video smoke probes passed; a small nonthinking screen passed 12/13 cases. No paired BF16 accuracy certification or simultaneous vision-load test.
- Evidence: [results/](results/), `*-evidence.tar.gz`, [quality_gate.md](quality_gate.md).

## Native 262K benchmark

- **Input/output:** 260,096 input + 2,048 output = **262,144 tokens** each; starts 99.22% full and reaches the native limit. Greedy generation, native thinking and natural EOS enabled; independent histories, no prefix warming.

| Streams | Per-stream decode tok/s | Combined tok/s | Max time to first token | Common decode |
|---:|---:|---:|---:|---:|
| 1 | 57.54 | 57.54 | 4.97 min | 35.58 s |
| 2 | 39.85–40.86 | 80.71 | 9.49 min | 50.09 s |
| 4 | 25.59–26.79 | 104.59 | 13.38 min | 76.27 s |
| 8 | 14.00–15.89 | 122.28 | 24.26 min | 128.81 s |
| 16 | 8.19–9.03 | 137.93 | 43.15 min | 226.70 s |

- **Verified:** all 31 requests reached the exact native limit and stopped at the output cap; raw token-ID audit passed. No observed queueing/preemption.
- **Duration:** the harness uses a 10-second minimum; sustained-pass labels require 60 seconds. One- and two-stream runs are exploratory.
- Evidence: [raw traces](native262k-evidence.tar.gz), [independent audit](results/native262k-comparison.json).

- Reproduce when the server is idle (completed result files are reused):

```bash
/home/ben/qwen-lab/venv/bin/python /home/ben/qwen-lab/scripts/run_native262k_sweep.py
```

- One fresh test; choose the stream count and a new output filename:

```bash
/home/ben/qwen-lab/venv/bin/python /home/ben/qwen-lab/scripts/bench_streams.py \
  --model qwen-lab --concurrency 8 --prompt-tokens 260096 --output-tokens 2048 \
  --target-tps 18 --respect-eos --min-overlap-seconds 10 --timeout 14400 \
  --out /home/ben/qwen-lab/results/recheck-c8-native262k.json
```

## Repository contents

- Root: exact recipe, serving/benchmark scripts, measured notes, and evidence archives.
- [results/](results/): all final measurements and independent audits; `*-evidence.tar.gz`: original traces, logs, and telemetry.
- [history/](history/): additional research notes, test harness checks, setup helpers, and the chronological lab journal.
- [RESUME.md](RESUME.md): last verified workstation checkpoint (September 18, 2026); live status may have changed since then.
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md): upstream references and licenses. Model weights, credentials, virtual environments, and duplicate caches are excluded.
- [Full snapshot](qwen-zgx-lab-snapshot.zip): every project file with its directory structure. Verify extracted contents with `shasum -a 256 -c SHA256SUMS`.
