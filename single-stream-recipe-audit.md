# NVIDIA / 200 TPS comparison audit

Read-only primary-source audit, 2026-09-18. No remote changes or GPU runs.

## What the official sources establish

The [current NVIDIA agent-ready playbook](https://build.nvidia.com/spark/vllm/agent-ready-models) recommends `nvidia/Qwen3.6-35B-A3B-NVFP4` and links directly to the [vLLM GB10 recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B?features=tool_calling,reasoning&hardware=dgx_spark_gb10). That recipe reports a 2026-08-31 measurement: **97.7 decode tokens/s**, concurrency **1**, actual **8K input / 256 output** tokens on SPEED-Bench, aiperf 0.12.0, vLLM 0.28.0 official Docker image, one GB10 at TP1. Launch uses FlashInfer, Marlin MoE and three-token MTP on Triton. This is a measured 8K prompt, not a full 128K history.

The [NVIDIA June 1 Computex article](https://developer.nvidia.com/blog/run-local-ai-agents-with-faster-models-and-multi-node-clustering-on-nvidia-dgx-spark/) reports **2.6× relative throughput improvement**. Its [original chart](https://developer-blogs.nvidia.com/wp-content/uploads/2026/06/image3.webp), visually inspected and saved locally as `work/serving_recipe_sources/nvidia-computex-qwen-throughput.webp`, identifies these conditions in the footnote:

- Before: Unsloth Qwen3.6-35B NVFP4, vLLM 0.20.
- After: NVIDIA NVFP4, May 27 vLLM nightly, MTP3.
- Harness/workload: AIPerf and SPEED-Bench Coding Dataset.

The chart does not give absolute TPS, actual context length or concurrency. The 2.6× ratio therefore cannot be directly applied to a vLLM 0.28 / actual 128K / no-MTP baseline.

## 200+ numbers: provenance remains unresolved

I did **not** locate a NVIDIA-authored, single-stream 200 TPS claim with a complete workload in the current NVIDIA playbook, its linked recipe, model card or Computex article. This is a bounded search result, not proof that no such claim exists. Search results associate prominent 200+ social claims with Atlas Inference/Spark Arena; I did not recover an original 200+ benchmark with its full conditions, so that provenance should not be presented as established.

A [first-person benchmark posted to NVIDIA's community forum](https://forums.developer.nvidia.com/t/benchmark-report-qwen3-6-35b-a3b-nvfp4-on-nvidia-dgx-spark-jetson-thor-blackwell-6000-pro/371810) does show **249.47 aggregate output TPS** for 16 submitted requests with 1,000 input / 1,000 output tokens. The server permits four active sequences, configured maximum context 65,536, NVIDIA NVFP4, FlashInfer, Marlin and MTP3; mean TPOT is 15.14 ms. Its decode-heavy 1,000/8,000 case reports 268.21 aggregate output TPS. These are author measurements on a NVIDIA-hosted community forum, not NVIDIA-authored single-stream/full-context results.

The [Atlas source README](https://github.com/Avarok-Cybersecurity/atlas#-performance) inspected today does not provide a 200 TPS Qwen3.6 single-stream receipt. Its single-stream table describes a tiny capital-of-France prompt and at most 30 output tokens; its current concurrency table measures a different model, Qwen3.8-27B, at input 128/output 1,024. Neither is evidence for Qwen3.6 at 128K. Do not transpose those figures.

## Fair comparison to the local 51.3 TPS result

Per root's run description, the local result is Unsloth NVFP4-Fast, vLLM 0.28, **actual 131,072-token input**, without MTP. The official reproducible measurement differs in occupied history (8K), checkpoint/quantization path (NVIDIA versus Unsloth Fast), MoE backend, and speculation (MTP3 versus none). NVIDIA's own [quantization playbook](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/nvfp4-quantization/README.md) distinguishes its W4A16 interactive recipe from W4A4 experts for higher concurrency.

The 51.3 result is consequently a baseline for that particular full-context configuration, **not a measured maximum for the workstation**. A valid next comparison would reproduce the linked NVIDIA recipe at its actual 8K/256/C1 conditions, then hold the engine/checkpoint/speculation fixed and raise occupied input to 128K. No new benchmark has been run as part of this audit.

## Follow-up: concrete Atlas recipe recovered

The faster-engine lead does correspond to a real first-party recipe; the current vLLM recipe is not exhaustive and does not disprove faster results. The [Atlas-maintained recipe registry](https://github.com/Avarok-Cybersecurity/atlas-recipes) and pinned historical files provide reproducible launch settings:

| Version | Checkpoint and settings | What is actually documented |
| --- | --- | --- |
| [2026-05-27 recipe](https://github.com/Avarok-Cybersecurity/atlas-recipes/blob/431c05ecd950395798e26bddeafa5a2d0971e7a8/recipes/qwen3.6/qwen3.6-35b-a3b-nvfp4.yaml) | RedHatAI/Qwen3.6-35B-A3B-NVFP4; avarok/atlas-gb10:latest; FP8 KV; FP8 MTP head; speculation enabled; SLAi scheduling; prefix caching; configured maximum 131,072 | Real historical recipe, but no input-length, output-length, throughput or concurrency measurement. Description mentions NVFP4 KV/head while actual defaults say FP8; actual defaults are the reproducible settings. |
| [Current recipe, pinned July 18 revision](https://github.com/Avarok-Cybersecurity/atlas-recipes/blob/c99948adb6782d823b36e7dd86dee8d3d35e357f/recipes/qwen3.6/qwen3.6-35b-a3b-nvfp4.yaml) | NVIDIA NVFP4; :dev image; calibrated FP8 KV with boundary layers promoted; BF16 MTP head; num_drafts=1; configured max262144 | July12 C1 measurement, ten repetitions, 400 output tokens, greedy: median116.5 TPS with num_drafts1 versus89.7 without speculation and82.2 with num_drafts2. Actual prompt length is not specified. The later262K cache fit is explicitly described as derived, not a separate live run. |

The original X post and full benchmark behind the syndicated Atlas **200+ TPS** claim were not recovered in this bounded audit. Its exact model revision, prompt occupancy, output count, measurement formula and runtime revision therefore remain unresolved. Do not call that claim false, treat 116.5 as a speed ceiling, or claim its launch reproduces200. The concrete next candidate is the Atlas recipe, with an image digest and checkpoint revision captured before testing.

Root meanwhile measured the current vLLM MTP3 configuration at actual8K input /2,048 output /C1: **90.457 decode TPS** (22.629 seconds). This newly measured result supersedes treating the earlier51.3 no-spec/full-context baseline as the only single-stream data point; it remains a different workload from128K.
