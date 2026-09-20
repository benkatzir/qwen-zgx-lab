# Admitting 50 independent 128K prompts in stock vLLM 0.28

## Recommended cold-prefill test

Use the real serving model and its existing no-prefix-cache configuration, with:

```text
--max-num-seqs 50
--max-num-batched-tokens 8192
--enable-chunked-prefill
--long-prefill-token-threshold 128
--max-model-len 139264
--no-enable-prefix-caching
```

The key additional setting is `--long-prefill-token-threshold 128`. In v0.28,
the scheduler caps each running **and** waiting request's new tokens by this
value, then continues to the next request. Fifty 128-token chunks require 6400
tokens, within the 8192-token iteration budget. Thus fifty equal 131072-token
prompts, if admitted together and with sufficient actual cache capacity, can
progress through 1024 chunks approximately together. No-prefix Mamba mode
`none` does not impose the `align` mode's Mamba cache-boundary split. Decode
requests use one token (or a short verification block), so the 128-token cap
does not constrain their ordinary decode throughput.

**TurboQuant exception:** the 128-token recommendation above is for the
FlashInfer FP8 path. v0.28 TurboQuant routes continuation chunks of 128 tokens
or fewer through its per-request synthetic decode path, independently scanning
the compressed cache for each query. Fair-prefilling 50 requests this way can
be extremely expensive. A cap of **160** fits 50 requests into the same budget
(50 × 160 = 8000) while selecting TurboQuant's larger-continuation path, which
dequantizes cached K/V and invokes FlashAttention. This still adds significant
dequantization traffic and is a source-supported mitigation, not a measured
prefill performance claim. See the [pinned TQ continuation branch](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/attention/backends/turboquant_attn.py#L853).

This is a real prefill tuning choice, not a guarantee of admission: the KV pool
must still fit all requests, the scheduler must not preempt them, and the
multimodal encoder budget can introduce skew. Keep full-ISL admission checks.
There is no need to disable modalities. For the first capacity/performance run,
use text-only payloads with vision support retained, and label that limitation;
follow with mixed-modality prompts and actual encoder peak-memory measurements.

For an initial submission barrier, enable `VLLM_SERVER_DEV_MODE=1` at server
launch, then call:

```text
POST /pause?mode=keep&clear_cache=false
GET  /is_paused
# Start all 50 independent streaming completion requests.
POST /resume
```

All requests must have equal actual tokenized input length, independent early
nonces and independent content, and an 8192-token output allowance with
`ignore_eos=true` for this **load test only**. An explicit array of valid input
token IDs in the completions API can eliminate uncertainty about prompt token
counts; the text should still be meaningful and the final task consistent.
Use normal stopping for separate quality evaluation.

`keep` freezes **all** scheduling; it cannot pause decode while prefill continues.
The useful operation is pause once before submission, followed by resume once.
`abort` loses requests; `wait` drains existing requests. Explicit
`clear_cache=false` avoids accidental recomputation. The HTTP route's docstring
says this parameter is ignored in `keep`, but the pinned route actually forwards
it and engine core conditionally resets caches, so do not rely on that sentence.

The pause barrier should be released after the client has submitted all request
bodies and the server has accepted them. HTTP streaming headers establish HTTP
acceptance, not necessarily exact engine-queue arrival. A small arrival skew is
normally absorbed by a 1024-step fair prefill. Do not assert that paused Prometheus
queue gauges necessarily refresh: engine stats publication can depend on steps
or DP load-balancing mode. Capture request IDs and server admission logs if a
strict queue-arrival audit is required. An offline engine whose requests are all
added before the first `step` is another supported way to get an exact initial
barrier, but would require replacing the active serving process for that run.

## Measurement gate

Do not start measuring decode based on `num_requests_running == 50` alone:
running includes partial prefills. Require first output from **every** request,
then start a common at-least-60-second measurement interval. Confirm all 50
remain active, none finishes in that interval, waiting is zero, and no preemption
or input recomputation occurs. Count each stream's actual accepted output tokens
inside that same interval. Report the minimum per-stream rate plus percentiles;
aggregate throughput is supplementary. Report cold-prefill wall time and TTFT
separately, including any artificial initial submission pause.

If the existing server has no per-request prefill cap, it can prefer a small
number of large prefills; increasing output length until the last stream starts
does not itself guarantee the intended capacity, and may exceed the configured
context. The cap is a stock configuration fix, but requires a launch with that
setting; do not pretend `/pause` alone synchronizes completion of prefills.

## Why not prewarm prefix cache first?

Prewarming fifty distinct prefixes and then issuing their continuations can
legitimately measure **warm-prefix decode**, provided all fifty independent full
caches are resident and per-stream context remains at least 128K. It must be
labeled as such and cannot supply cold-prefill or general-arrival evidence.
It also changes this hybrid model's cache behavior (`none` to a supported prefix
mode), adds recurrent-state retention/copies, may evict earlier warmed prefixes,
and changes memory capacity. Do not infer residency from fifty successful warmup
requests or a server-wide average hit rate. The cold fair-prefill method above
avoids those confounders and does not require prefix caching.

Old advice to set `max_num_partial_prefills` or `max_long_partial_prefills` does
not apply to this pinned v0.28 configuration: those fields are absent from
`SchedulerConfig` and `EngineArgs`. The per-request cap is
`long_prefill_token_threshold`.

Primary source:

- [v0.28 scheduler](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/core/sched/scheduler.py): running cap around lines 558–566; waiting cap 952–966; Mamba split selection 316; pause token budget 500; waiting-admission gate 748; request enqueue 2301.
- [v0.28 scheduler configuration](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/config/scheduler.py): exact supported flags and defaults.
- [v0.28 pause/resume HTTP routes](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/entrypoints/serve/dev/rlhf/api_router.py): query parameters and forwarding.
- [v0.28 AsyncLLM pause](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/engine/async_llm.py): `pause_generation` and `resume_generation`.
- [v0.28 engine core](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/engine/core.py): pause states, cache clearing and stats publication.
