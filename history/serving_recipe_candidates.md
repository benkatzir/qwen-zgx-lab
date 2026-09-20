# Qwen3.6-35B-A3B / single GB10 serving candidates

Research date: 2026-09-18. These are research candidates, NOT proven target performance on the user's machine. No SSH operations performed by this research agent.

## First candidate: Unsloth NVFP4-Fast, stock vLLM 0.28.0

Pinned checkpoint revision verified using Hugging Face API: `1c3f884bc99aac2524f6d49bcbac8c88401afd66`.

The current config/index has W4A4 routed/shared experts in all 40 layers, FP8 attention and dense projections, BF16 vision tower, unquantized MTP head, and shipped K/V calibration scales for all ten full-attention layers. Older recipes mention an FP8 expert tail; that does not appear in this exact revision's config/index.

Run inside the official ARM64 CUDA container already being pulled by parent:

```bash
CUTE_DSL_ARCH=sm_121a VLLM_USE_DEEP_GEMM=0 \
vllm serve unsloth/Qwen3.6-35B-A3B-NVFP4-Fast \
  --revision 1c3f884bc99aac2524f6d49bcbac8c88401afd66 \
  --served-model-name qwen36-test \
  --dtype bfloat16 \
  --attention-backend FLASHINFER \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 132096 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --max-cudagraph-capture-size 64 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --async-scheduling \
  --limit-mm-per-prompt '{"image":1,"video":1}' \
  --reasoning-parser qwen3
```

This reserves a 131072-token prompt plus 1024 generation positions, keeps image/video capability, and avoids benchmark gains from shared-prefix reuse. No quantization or MoE backend override for initial control; inspect actual chosen kernel in startup logs. Adjust memory only from observed capacity/headroom. Start without speculation, then add:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
```

Try MTP 1/2/3 only as warranted by measured acceptance and throughput. A high C throughput gain is not guaranteed; an additional full-attention MTP layer increases cache work.

## Second candidate: native B12X MoE

Same baseline plus `--moe-backend flashinfer_b12x`; retain draft `moe_backend:triton` with MTP. This is opt-in, not a required patch. v0.28.0's NVFP4 oracle deliberately excludes B12X from automatic selection because of an upstream SM121 MMA guard. Its FP8 MoE mapper still does not recognize B12X. The pinned Unsloth revision's routed experts are all NVFP4, so no blanket FP8 mapper patch should be applied unless a real stack trace establishes a relevant FP8 MoE layer. Semantics tests precede load testing.

Sources:
- https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py
- https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/layers/fused_moe/oracle/fp8.py
- https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py

## Third candidate: stock TurboQuant cache

v0.28.0 includes TURBOQUANT attention and named cache formats, head_dim256 tests, and explicit hybrid full-attention detection. Swap initial baseline's attention/cache flags for:

```bash
--attention-backend TURBOQUANT --kv-cache-dtype turboquant_4bit_nc
```

Start without MTP because TQ metadata marks speculative verification as a prefill path. At head_dim256 the per-head KV slot is 262 bytes, compared with 512 FP8 bytes (1.95x smaller); `turboquant_k8v4` is 388 bytes (1.32x smaller) and `turboquant_3bit_nc` 198 bytes (2.59x smaller). These exclude Mamba state, padding, workspaces and draft KV. The source includes alarming quality deltas on other models for aggressive three-bit presets; none establishes 97% retention for this model, so measure quality rather than treating compression ratios as a solution.

Sources:
- https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/attention/backends/turboquant_attn.py
- https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/layers/quantization/turboquant/config.py
- https://github.com/vllm-project/vllm/blob/v0.28.0/tests/quantization/test_turboquant.py

## Published evidence relevant to expectations

- NVIDIA checkpoint preserves text/image/video. Eight vendor evals retain >=99.16% of BF16 score; worst ratio is telecom 94.7/95.5. Evaluated on GB300, not ZGX. Official Spark W4A16 recipe uses Marlin, FP8 KV, MTP3, and max-seqs4. https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4
- Unsloth Fast vendor evals: MMLU-Pro 85.58 vs BF16 85.75, GPQA87.75 vs86.36, AIME2025 91.67 vs92.50. Retains vision and MTP. Their claimed 1.79x throughput comparison is B200 C128, not GB10. https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4-Fast
- Current vLLM recipe: single GB10 97.7 decode tps at C1, 8K input/256 output, v0.28.0, measured 2026-08-31. https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B
- r0b0tlab: stock-like audited W4A4 B12X+MTP2, 2048 input/512 output, C1=80.6, C8=268.8, C16=285.6, C32=344.2 aggregate tps. C32 mean TPOT64.21ms. Their experimental NVFP4 KV booted but failed simple semantics; rejected. https://github.com/r0b0tlab/qwen36-35b-a3b-nvfp4-fast-sm121-vllm
- Hiceron: auto/CUTLASS+MTP3 926 aggregate at C64 (~15/request), short prompts; B12X did not beat auto in their measured profile. https://github.com/hiceron/spark-nvfp4-lab/blob/main/DEPLOY.md
- HowToSpark author reports Unsloth Fast+MTP3 single-stream75.5 tps at actual131K input, TTFT46s;250K input62.9tps, TTFT140s. v0.24.0 tested faster than0.26.0. Community median128K=68tps. Not concurrency proof. https://howtospark.com/recipes/qwen3-6-35b-a3b-nvfp4-fast
- SparkBench's NVIDIA MTP3 recipe actual fill4K=86.3,50K=78.8,100K=31.5tps; single-node GB10. https://sparkbench.dev/models/nvidia_qwen3.6-35b-a3b/
- NVIDIA own quantization playbook differentiates W4A16 interactive and W4A4 experts high-concurrency recipes. https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/nvfp4-quantization/README.md

No reviewed primary source demonstrates all 45-50 independent filled128K requests, >=40 generated tps each, retained modalities, and >=97% measured BF16 accuracy on a single GB10. Cache capacity/configured context alone is not such proof.
