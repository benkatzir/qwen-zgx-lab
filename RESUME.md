# Completed checkpoint — 2026-09-18

- Requested concise README rewrite and native-window concurrency sweep are complete.
- Benchmarks, telemetry and progress monitor have exited. vLLM remains running and idle on `ben@zgx-140a`, loopback port 8000; model `qwen-lab`.
- Server PID `39359`, container `qwen-lab-server`; native context limit **262,144 total input + output tokens**. No serving configuration changes were made during this sweep.
- Exact recipe and commands are in `README.md` and `start_server.sh`. The bundled launcher, benchmark harness and sweep script match their remote SHA256 hashes.
- Remote lab: `/home/ben/qwen-lab`; scripts in `scripts/`, source results/traces in `results/`, logs in `logs/`. Models remain installed.

## Native-window measurements

- Sequential concurrency: 1, 2, 4, 8, 16. Every request used 260,096 actual input tokens and generated 2,048 outputs, reaching 262,144 total.
- Native thinking, greedy generation, natural EOS allowed; synthetic independent chat histories and no prefix warming.
- All 31 requests finished at the output cap. Independent raw SSE/token-ID audit passed for every case, with no observed queueing or preemption.

| Streams | Per-stream decode tok/s | Combined tok/s | Max time to first token | Common decode |
|---:|---:|---:|---:|---:|
| 1 | 57.54 | 57.54 | 4.97 min | 35.58 s |
| 2 | 39.85–40.86 | 80.71 | 9.49 min | 50.09 s |
| 4 | 25.59–26.79 | 104.59 | 13.38 min | 76.27 s |
| 8 | 14.00–15.89 | 122.28 | 24.26 min | 128.81 s |
| 16 | 8.19–9.03 | 137.93 | 43.15 min | 226.70 s |

- **Highest tested concurrency meeting 18 tok/s on every stream for at least 60 seconds: 4.** Counts 5–7 are untested; this is not a proven hardware maximum.
- One- and two-stream cases are shorter than 60 seconds and remain exploratory. Eight and sixteen streams are below 18 tok/s on every stream.
- `native262k-evidence.tar.gz` preserves all five reports, raw SSE traces, logs, telemetry, wrapper state and final health. `results/native262k-comparison.json` contains the independent audit; `review_native262k_sweep.py` reproduces it.
- `results/native262k-runtime-audit.json` confirms identical running argument vectors and native model context; it records the limitation on independently reading container environment variables without elevated access.
- Final server counters equal exactly the sweep totals plus the earlier 17-input/3-output smoke request: 8,062,993 prompt tokens and 63,491 output tokens. Server health HTTP 200; running/waiting/preemption counters are zero.

## Limits and future use

- Earlier 128K measurements are preserved in the README and evidence archives; those used a 139,264 context cap.
- Original 45–50 streams × 40 tok/s target remains unmet. Multimodal smoke tests passed separately; concurrent visual load and 97% BF16-relative accuracy retention remain unverified.
- No benchmark is queued. Do not resume old PIDs. `run_native262k_sweep.py` reuses completed result files; the README also provides a command for a fresh single-case benchmark.
- SSH master `work/zgx-ssh-native262k.sock` may expire. No credentials are stored in the artifacts.
