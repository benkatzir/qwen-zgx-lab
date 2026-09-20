# Recurrent-state journal: credible optimization, insufficient target evidence

**Recommendation: do not integrate this solely to pursue the current C45/C50,
128K, 40 tokens/s target.** It is a credible research optimization, but the
measured attention budget and available draft acceptance make it an unlikely
way to satisfy the complete requirement. No GPU work or new downloads were
performed for this assessment. This is not a proof against future software.

At C45, native NVFP4 XQA/page128 takes **341.777 ms for ten attention layers at
q16**. Even eliminating every other operation permits only **46.81 tokens/s
with all sixteen positions accepted**. Reaching 40 would require at least
**13.671 accepted output tokens per verification pass before charging any GDN,
MoE, projections, draft execution, convolution, sampling or scheduling**. With
perfect acceptance, all those operations together have only **58.223 ms** left
in the 400 ms pass budget. Stock FP32 GDN alone measured 235.379 ms. At q8,
attention takes 190.033 ms, leaving only 9.967 ms of the 200 ms budget even with
perfect acceptance. These are component measurements, not integrated throughput.

The trained block-16 DFlash checkpoint's authors report mean accepted lengths
**5.118–7.629**, explicitly defined as `completion_tokens / spec_verify_ct`.
Applying those numbers as a **scenario**, not a prediction for this machine,
to the measured 341.777 ms attention time gives only **14.97–22.32 tokens/s**
before any other model work. Their measurements used B200, BF16, SGLang,
concurrency 1 and ordinary benchmark prompts; they did **not** measure this
user's filled-128K, multimodal, quantized-target workload. Nevertheless, they
provide no evidence for the near-perfect block-16 acceptance needed here.
The trained DFlash block size is 16; arbitrarily setting 32/64 is not an
established high-acceptance recipe. Existing q8→q16 measurements also show
attention time increasing from 190.033 to 341.777 ms and GDN from 121.224 to
235.379 ms. Increasing the block length alone cannot be credited with free KV
reuse. Wider queries remain unmeasured; those two points do not prove their
performance.

## The actual optimization

Stock vLLM's speculative GDN kernel holds the FP32 state in registers during
the token loop but writes a complete checkpoint after every token. The loop is
equivalent to `S *= alpha; d = beta * (v - S @ k); S += outer(d, k)`, followed
by the output reduction. An alternative can retain the initial state, journal
the already computed **FP32 alpha, normalized k, and d**, emit the normal
verification outputs, and save just a final candidate state. After acceptance
is known, select that final state if appropriate, or replay the accepted
prefix's rank-one updates from the retained initial state and commit once.
Replay needs neither another model forward nor the `S @ k`/output reductions.
Keeping projected inputs and recomputing the recurrence is a simpler but more
expensive variant. Sparse checkpoints plus short replay are another tradeoff.

For this geometry the minimal journal is 24,704 bytes per token/layer
(`(16*128 + 32*128 + 32)*4`), versus 2,097,152 bytes for one FP32 state:
about **85× less checkpoint payload**. Across C45, q16 and thirty GDN layers,
that is approximately 509 MiB of journals, plus the retained initial/final
states. These are logical sizes, not measured allocation or bandwidth.
The rewrite must preserve FP32 values and the original operation/FMA order;
merely storing BF16 deltas would add a precision change. It also needs correct
convolution rollback, mixed acceptance lengths, graph replay and pool lifetime
handling. Stock vLLM and the inspected SGLang implementation both save full
intermediate states; neither supplies this journal scheme. It is consequently
new kernel/runtime work, not a launch flag. Avoiding state writes does not
remove the state arithmetic, replay cost or model projections.

If pursued for general serving improvements, the first bounded experiment is
a standalone journal/replay kernel: compare all outputs and committed states
against the current FP32 recurrence for every accepted-prefix length, including
mixed lengths and graph replay, then time **verification plus commit/replay**
at C45 q16. It should precede any runtime integration. For this user's target,
however, that experiment alone cannot resolve the attention/acceptance deficit.
A credible route would additionally need substantially faster multi-query
attention or a demonstrably much better drafter, correct SM121 NVFP4 serving
integration, and a paired long-context/text/image/video quality evaluation
showing ≥97% retention versus BF16. Exact speculative verification preserves
the served target's distribution; it does not certify low-bit target/KV quality.

Evidence: [measured components](measured_bottlenecks.md),
[draft/source audit](parallel_draft_candidates.md),
[authors' DFlash benchmark and acceptance definition](serving_recipe_sources/parallel_draft/dflash-readme.md),
[vLLM FP32 recurrence and per-token stores](vllm028/vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py),
[SGLang per-token state stores](oscar-hybrid-source/sglang-research/python/sglang/srt/layers/attention/fla/fused_recurrent.py).
