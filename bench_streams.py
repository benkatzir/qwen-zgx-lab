#!/usr/bin/env python3
"""Exact-length independent-context vLLM SSE load test, not an accuracy test.

pip install aiohttp
python bench_streams.py --base-url http://127.0.0.1:8000 --model MODEL \
  --concurrency 50 --prompt-tokens 131072 --output-tokens 1024 --out run.json

Input is synthetic archival prose in the server's actual chat template by
default, with a unique first user-content block per request. Native thinking
behavior is preserved unless --disable-thinking is given. Requests use exact-
length token-ID prompts. Prefix hashes and lengths are saved. A 131072-token prompt requires
max_model_len >= 131072 + output_tokens. This deliberately measures occupied
context, not merely a configured context limit. No throughput claim implies
reasoning/vision accuracy or retention of all input information.

Raw .trace.jsonl contains complete SSE payloads and selected Prometheus samples.
Rates use server token IDs or per-event continuous usage, never chunk counts.
All timing is client monotonic time, and includes transport effects.
"""

import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import aiohttp


METRIC_PATTERNS = (
    "num_requests_running", "num_requests_waiting", "num_requests_swapped",
    "num_preemptions", "kv_cache_usage", "gpu_cache_usage", "cpu_cache_usage",
    "prefix_cache", "request_queue_time_seconds", "request_prefill_time_seconds",
    "request_decode_time_seconds", "prompt_tokens_total", "generation_tokens_total",
    "iteration_tokens_total", "time_to_first_token_seconds", "inter_token_latency",
    "spec_decode", "spec_decoding", "request_success_total",
)


def now():
    return time.perf_counter()


class Trace:
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8", buffering=1)
        self.origin = now()

    def write(self, kind, **data):
        self.file.write(json.dumps({"kind": kind, "t": now() - self.origin, **data},
                                   ensure_ascii=False, separators=(",", ":")) + "\n")


def url(base, path):
    root = base.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root + path


async def json_request(session, method, endpoint, **kwargs):
    async with session.request(method, endpoint, **kwargs) as response:
        body = await response.text()
        if response.status >= 400:
            raise RuntimeError(f"HTTP {response.status} from {endpoint}: {body[:1500]}")
        return json.loads(body)


def make_text(index, target_chars, seed):
    rng = random.Random(seed + index * 1000003)
    # A long unique header prevents identical first KV cache blocks even if a
    # tokenizer splits the stream number in a surprising way.
    identity = " ".join(f"{rng.getrandbits(40):010x}" for _ in range(48))
    pieces = [f"Archive identity {identity}.\n",
              "Read this fictional research archive. After the archive, write an extended "
              "plain-language account of how careful observations support reliable decisions.\n"]
    length = sum(map(len, pieces))
    subjects = ["river", "forest", "harbor", "orchard", "hill", "village", "garden"]
    actions = ["measured water flow", "compared written records", "checked the instruments",
               "interviewed the field team", "reviewed the daily observations"]
    n = 0
    while length < target_chars:
        n += 1
        part = (f"Record {n} for archive {index}: The team near the {rng.choice(subjects)} "
                f"{rng.choice(actions)} on day {rng.randrange(1, 366)}. "
                f"They recorded {rng.randrange(10, 999)} observations and kept the original "
                "notes so that another researcher could examine the same evidence. "
                "Unexpected results were checked before the report was revised.\n")
        pieces.append(part)
        length += len(part)
    return "".join(pieces)


async def prompt_envelope(session, args):
    """Obtain the real chat template and tokenizer-confirmed Qwen turn markers."""
    marker_ids = {}
    for marker in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
        result = await json_request(session, "POST", url(args.base_url, "/tokenize"),
                                    json={"model": args.model, "prompt": marker,
                                          "add_special_tokens": False})
        if len(result["tokens"]) == 1:
            marker_ids[marker] = result["tokens"][0]
    if not getattr(args, "chat_prompt", True):
        return [], [], marker_ids
    marker = "BENCHBODYMARKERf9238c71a64b5e02END"
    body = await json_request(session, "POST", url(args.base_url, "/tokenize"),
                              json={"model": args.model, "prompt": marker,
                                    "add_special_tokens": False})
    template_request = {
        "model": args.model, "messages": [
            {"role": "system", "content": "Answer the user's request directly and carefully."},
            {"role": "user", "content": marker}],
        "add_generation_prompt": True}
    if getattr(args, "disable_thinking", False):
        template_request["chat_template_kwargs"] = {"enable_thinking": False}
    rendered = await json_request(session, "POST", url(args.base_url, "/tokenize"), json=template_request)
    ids, body_ids = rendered["tokens"], body["tokens"]
    starts = [index for index in range(len(ids) - len(body_ids) + 1)
              if ids[index:index + len(body_ids)] == body_ids]
    if len(starts) != 1:
        raise RuntimeError("Cannot isolate the user body in server chat template; refusing raw-prompt fallback")
    start = starts[0]
    return ids[:start], ids[start + len(body_ids):], marker_ids


async def build_prompts(session, args, trace):
    chat_prefix, chat_suffix, turn_token_ids = await prompt_envelope(session, args)
    suffix_text = ("\nEnd of archive. Write a detailed, continuous explanation of careful "
                   "research practice. Discuss observations, comparisons, uncertainty, "
                   "replication, and clear communication, using examples from the archive.\n")
    suffix_info = await json_request(session, "POST", url(args.base_url, "/tokenize"),
                                    json={"model": args.model, "prompt": suffix_text,
                                          "add_special_tokens": False})
    suffix = suffix_info["tokens"]
    body_budget = args.prompt_tokens - len(chat_prefix) - len(chat_suffix)
    if body_budget <= 0:
        raise ValueError("prompt-tokens must exceed the actual chat-template envelope length")
    if body_budget <= len(suffix) + 128:
        # Still useful for short runtime smoke tests.
        suffix = []
    prompts, metadata = [], []
    for index in range(args.concurrency):
        target_chars = max(args.prompt_tokens * 6, 1000)
        while True:
            text = make_text(index, target_chars, args.seed)
            info = await json_request(session, "POST", url(args.base_url, "/tokenize"),
                                      json={"model": args.model, "prompt": text,
                                            "add_special_tokens": False})
            token_ids = info["tokens"]
            needed = body_budget - len(suffix)
            if len(token_ids) >= needed:
                token_ids = chat_prefix + token_ids[:needed] + suffix + chat_suffix
                break
            target_chars *= 2
        max_len = info.get("max_model_len")
        if max_len and args.prompt_tokens + args.output_tokens > max_len:
            raise ValueError(f"Server max_model_len={max_len}; request needs "
                             f"{args.prompt_tokens + args.output_tokens} tokens")
        first = token_ids[:min(256, args.prompt_tokens)]
        meta = {"request": index, "input_tokens": len(token_ids),
                "prefix_16_sha256": hashlib.sha256(json.dumps(token_ids[:16]).encode()).hexdigest(),
                "prefix_256_sha256": hashlib.sha256(json.dumps(first).encode()).hexdigest(),
                "prompt_sha256": hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
                "server_max_model_len": max_len, "synthetic_input": True,
                "chat_prompt": getattr(args, "chat_prompt", True),
                "thinking_disabled": getattr(args, "disable_thinking", False),
                "shared_chat_prefix_tokens": len(chat_prefix),
                "assistant_suffix_tokens": len(chat_suffix), "turn_token_ids": turn_token_ids}
        prompts.append(token_ids)
        metadata.append(meta)
        trace.write("prompt", **meta)
        print(f"Prepared request {index + 1}/{args.concurrency}: {len(token_ids)} tokens", flush=True)
    if len({item["prefix_256_sha256"] for item in metadata}) != len(metadata):
        raise ValueError("Prompt prefixes are not unique")
    return prompts, metadata


async def warm_prefixes(session, args, trace, prompts):
    """Populate each independent prefix once; exclude this time from load rates."""
    if not args.warm_prefixes:
        return None
    start = now()
    responses = []
    before = await scrape_metrics(session, args, trace, "before_warming")
    for index, prompt in enumerate(prompts):
        began = now()
        payload = {"model": args.model, "prompt": prompt, "max_tokens": 1,
                   "temperature": 0, "ignore_eos": not getattr(args, "respect_eos", False), "stream": False,
                   "add_special_tokens": False,
                   "seed": args.seed + index}
        response = await json_request(session, "POST", url(args.base_url, "/v1/completions"),
                                      json=payload)
        usage = response.get("usage") or {}
        record = {"request": index, "seconds": now() - began, "usage": usage,
                  "input_tokens_verified": usage.get("prompt_tokens") == args.prompt_tokens}
        responses.append(record)
        trace.write("prefix_warmed", **record)
        if not record["input_tokens_verified"]:
            raise RuntimeError(f"Warming request {index} did not verify exact prompt count")
        await scrape_metrics(session, args, trace, "during_warming")
        print(f"Warmed independent prefix {index + 1}/{args.concurrency}", flush=True)
    after = await scrape_metrics(session, args, trace, "after_warming")
    return {"seconds": now() - start, "requests": responses,
            "metrics_before": before, "metrics_after": after,
            "note": "Same unique prompts reused in measured requests. Prefix caching must be "
                    "enabled by the server; warming does not prove cache retention. Inspect "
                    "prefix-cache hits and cached_tokens. Warming time is excluded from measured TPS."}


def parse_metrics(body):
    samples = []
    for line in body.splitlines():
        if line.startswith("#") or not any(p in line for p in METRIC_PATTERNS):
            continue
        match = re.match(r"([^\s{]+)(\{.*\})?\s+([^\s]+)(?:\s+.*)?$", line)
        if match:
            try:
                value = float(match.group(3))
                if math.isfinite(value):
                    samples.append({"name": match.group(1), "labels": match.group(2) or "",
                                    "value": value})
            except ValueError:
                pass
    return samples


async def scrape_metrics(session, args, trace, phase):
    sample = {"started_at": now(), "at": None, "phase": phase, "samples": [], "error": None}
    try:
        async with session.get(url(args.base_url, "/metrics"),
                               timeout=aiohttp.ClientTimeout(total=10)) as response:
            body = await response.text()
            if response.status != 200:
                raise RuntimeError(f"metrics HTTP {response.status}: {body[:200]}")
            sample["samples"] = parse_metrics(body)
    except Exception as exc:
        sample["error"] = str(exc)
    sample["at"] = now()
    trace.write("metrics", **sample)
    return sample


async def poll_metrics(session, args, trace, done, samples):
    while not done.is_set():
        samples.append(await scrape_metrics(session, args, trace, "during"))
        try:
            await asyncio.wait_for(done.wait(), timeout=args.metrics_interval)
        except asyncio.TimeoutError:
            pass


async def sse_events(response):
    # vLLM's first return_token_ids event includes all prompt_token_ids, which
    # can exceed aiohttp's default readline limit by >10x at a 128K context.
    # Parse raw chunks instead; decode only complete lines so split UTF-8 is safe.
    data = []
    pending = bytearray()
    async for chunk in response.content.iter_any():
        pending.extend(chunk)
        while True:
            end = pending.find(b"\n")
            if end < 0:
                break
            line = bytes(pending[:end]).rstrip(b"\r")
            del pending[:end + 1]
            if not line:
                if data:
                    yield "\n".join(data)
                    data = []
            elif line.startswith(b"data:"):
                data.append(line[5:].lstrip(b" ").decode("utf-8"))
    if pending.startswith(b"data:"):
        data.append(bytes(pending[5:]).lstrip(b" ").decode("utf-8"))
    if data:
        yield "\n".join(data)


def output_diagnostics(token_ids, text, marker_ids, events):
    """Descriptive evidence only: these measurements are not a quality gate."""
    counts = Counter(token_ids)
    special = {}
    for marker, token_id in marker_ids.items():
        positions = [index for index, value in enumerate(token_ids) if value == token_id]
        special[marker] = {"token_id": token_id, "count": len(positions),
                           "first_output_token_index": positions[0] if positions else None}
    grams, first_duplicate = Counter(), None
    for index in range(max(0, len(token_ids) - 15)):
        gram = tuple(token_ids[index:index + 16])
        if grams[gram] and first_duplicate is None:
            first_duplicate = index
        grams[gram] += 1
    windows = sum(grams.values())
    longest, run, previous = 0, 0, None
    for token in token_ids:
        run = run + 1 if token == previous else 1
        longest, previous = max(longest, run), token
    early_rates = {}
    if events:
        for target in (128, 512, 1024):
            end = next((event for event in events if event["cumulative"] >= target), None)
            if end and end["at"] > events[0]["at"]:
                early_rates[str(target)] = {
                    "observed_tokens": end["cumulative"],
                    "seconds": end["at"] - events[0]["at"],
                    "decode_tps": (end["cumulative"] - events[0]["tokens"]) / (end["at"] - events[0]["at"])}
    return {"not_a_quality_score": True, "token_diagnostics_available": bool(token_ids),
            "token_ids_observed": len(token_ids),
            "special_turn_markers": special,
            "new_turn_start_count": special.get("<|im_start|>", {}).get("count", 0),
            "repetition": {"ngram_tokens": 16, "total_windows": windows,
                           "distinct_windows": len(grams),
                           "duplicate_window_fraction": (windows - len(grams)) / windows if windows else None,
                           "most_common_window_occurrences": max(grams.values()) if grams else 0,
                           "first_duplicate_window_start_token": first_duplicate,
                           "longest_identical_token_run": longest,
                           "most_common_token_count": counts.most_common(1)[0][1] if counts else 0},
            "visible_role_label_matches": len(re.findall(r"(?im)^\s*(?:user|assistant)\s*:?\s*$", text)),
            "escaped_role_label_matches": len(re.findall(r"\\n(?:user|assistant)\\n", text)),
            "visible_output_characters": len(text), "early_decode_rates": early_rates,
            "interpretation": "Turn markers, repetition, and early/full rate differences require review. An end marker at natural stopping is normal. No quality pass is inferred."}


async def stream_request(session, args, trace, gate, index, prompt, marker_ids=None):
    result = {"request": index, "start": None, "end": None, "first_token": None,
              "last_token": None, "usage": None, "error": None, "finish_reason": None,
              "http_status": None, "done_received": False, "token_events": [],
              "content_chunks": 0, "stream_token_count": 0, "token_count_basis": [],
              "text_preview": ""}
    payload = {"model": args.model, "prompt": prompt, "max_tokens": args.output_tokens,
               "temperature": 0, "stream": True, "ignore_eos": not getattr(args, "respect_eos", False),
               "add_special_tokens": False,
               "stream_options": {"include_usage": True, "continuous_usage_stats": True},
               "return_token_ids": True, "seed": args.seed + index}
    await gate.wait()
    result["start"] = now()
    trace.write("request_start", request=index, input_tokens=len(prompt),
                output_tokens=args.output_tokens)
    cumulative_usage = 0
    exact_so_far = 0
    generated_ids, text_parts = [], []
    try:
        async with session.post(url(args.base_url, "/v1/completions"), json=payload) as response:
            result["http_status"] = response.status
            if response.status != 200:
                raise RuntimeError(f"HTTP {response.status}: {(await response.text())[:2000]}")
            async for raw in sse_events(response):
                at = now()
                if raw == "[DONE]":
                    result["done_received"] = True
                    trace.write("sse_done", request=index, at=at)
                    break
                event = json.loads(raw)
                trace.write("sse", request=index, at=at, event=event)
                if "error" in event:
                    raise RuntimeError(json.dumps(event["error"]))
                choices = event.get("choices") or []
                usage = event.get("usage")
                if usage:
                    result["usage"] = usage
                if not choices:
                    continue  # Final aggregate usage is not a token arrival.
                choice = choices[0]
                if choice.get("index", 0) != 0 or len(choices) != 1:
                    raise RuntimeError("Expected exactly one completion choice")
                text = choice.get("text") or choice.get("delta", {}).get("content") or ""
                ids = choice.get("token_ids")
                if ids is None:
                    ids = choice.get("delta", {}).get("token_ids")
                token_delta = None
                basis = None
                if isinstance(ids, list):
                    token_delta, basis = len(ids), "token_ids"
                    generated_ids.extend(ids)
                elif usage and usage.get("completion_tokens") is not None:
                    token_delta = usage["completion_tokens"] - cumulative_usage
                    basis = "continuous_usage"
                if usage and usage.get("completion_tokens") is not None:
                    cumulative_usage = usage["completion_tokens"]
                elif token_delta is not None:
                    cumulative_usage += token_delta
                if token_delta is not None and token_delta < 0:
                    raise RuntimeError("Streaming cumulative token count moved backwards")
                if text:
                    text_parts.append(text)
                    result["content_chunks"] += 1
                    result["text_preview"] = (result["text_preview"] + text)[:2000]
                if token_delta is not None and token_delta > 0:
                    exact_so_far += token_delta
                    result["token_events"].append({"at": at, "tokens": token_delta,
                                                   "cumulative": exact_so_far, "basis": basis})
                    if basis not in result["token_count_basis"]:
                        result["token_count_basis"].append(basis)
                    result["first_token"] = result["first_token"] or at
                    result["last_token"] = at
                elif token_delta is None and text:
                    # Preserve timings, but never infer one token per chunk.
                    result["first_token"] = result["first_token"] or at
                    result["last_token"] = at
                if choice.get("finish_reason"):
                    result["finish_reason"] = choice["finish_reason"]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["end"] = now()
    result["stream_token_count"] = exact_so_far
    result["e2e_seconds"] = result["end"] - result["start"]
    result["ttft_seconds"] = (result["first_token"] - result["start"]
                               if result["first_token"] is not None else None)
    events = result["token_events"]
    usage = result["usage"] or {}
    result["prompt_tokens_verified"] = usage.get("prompt_tokens") == args.prompt_tokens
    result["output_tokens_verified"] = usage.get("completion_tokens") == args.output_tokens
    result["event_tokens_verified"] = bool(events) and exact_so_far == usage.get("completion_tokens")
    generated_text = "".join(text_parts)
    result["text_tail"] = generated_text[-2000:]
    result["output_diagnostics"] = output_diagnostics(generated_ids, generated_text, marker_ids or {},
                                                       events if result["event_tokens_verified"] else [])
    result["natural_stop_before_requested_length"] = (getattr(args, "respect_eos", False)
        and result["finish_reason"] == "stop" and not result["output_tokens_verified"])
    result["decode_tps"] = None
    if len(events) > 1 and result["event_tokens_verified"]:
        # Exclude the entire first event: it may contain multiple draft tokens.
        duration = events[-1]["at"] - events[0]["at"]
        result["decode_tps"] = ((exact_so_far - events[0]["tokens"]) / duration
                                  if duration > 0 else None)
    trace.write("request_end", **{k: v for k, v in result.items() if k != "token_events"})
    print(f"Request {index}: output={usage.get('completion_tokens')} "
          f"decode_tps={result['decode_tps']} error={result['error']}", flush=True)
    return result


def metric_sum(sample, part):
    # 0.28 also exposes num_requests_waiting_by_reason; summing a substring
    # would count each waiting request twice. Exclude *_created counter metadata.
    accepted_names = {part, part + "_total", part + "_perc"}
    values = [s["value"] for s in sample["samples"]
              if s["name"].rsplit(":", 1)[-1] in accepted_names]
    return sum(values) if values else None


async def resume_barrier(session, args, trace, barrier):
    """Resume even after an ambiguous pause response; retry transient failures."""
    errors = []
    for attempt in range(3):
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            await json_request(session, "POST", url(args.base_url, "/resume"), timeout=timeout)
            status = await json_request(session, "GET", url(args.base_url, "/is_paused"), timeout=timeout)
            if status.get("is_paused") is not False:
                raise RuntimeError(f"Resume was not confirmed: {status}")
            barrier.update(resume_required=False, resumed_at=now(), resume_confirmed=True,
                           resume_errors=errors)
            barrier["seconds"] = barrier["resumed_at"] - barrier["started_at"]
            trace.write("admission_resumed", **barrier)
            return
        except Exception as exc:
            errors.append(f"attempt {attempt + 1}: {type(exc).__name__}: {exc}")
            trace.write("admission_resume_error", error=errors[-1])
    barrier["resume_errors"] = errors
    raise RuntimeError("Could not confirm server resumed: " + "; ".join(errors))


async def wait_for_admission(session, args, trace, tasks, metrics, barrier):
    while True:
        completed = [task.result() for task in tasks if task.done() and not task.cancelled()]
        if completed:
            raise RuntimeError("Requests completed while admission was paused: " +
                               json.dumps([{k: r[k] for k in ("request", "error", "http_status")}
                                           for r in completed]))
        sample = await scrape_metrics(session, args, trace, "admission_paused")
        metrics.append(sample)
        waiting = metric_sum(sample, "num_requests_waiting")
        barrier["last_observed_waiting"] = waiting
        if waiting is not None and waiting >= args.concurrency:
            barrier.update(all_enqueued_at=now(), all_enqueued_confirmed=True)
            trace.write("admission_all_enqueued", **barrier)
            return
        await asyncio.sleep(min(args.metrics_interval, 0.25))


async def execute_requests(session, args, trace, prompts, metrics, report):
    barrier = {"enabled": getattr(args, "admission_barrier", False),
               "resume_required": False, "resume_confirmed": False,
               "all_enqueued_confirmed": False}
    report["admission_barrier"] = barrier
    gate, done = asyncio.Event(), asyncio.Event()
    tasks, poller = [], None
    try:
        if barrier["enabled"]:
            # A nonempty server would make waiting >= C ambiguous and pausing it
            # would affect requests outside this benchmark.
            running = metric_sum(metrics[-1], "num_requests_running")
            waiting = metric_sum(metrics[-1], "num_requests_waiting")
            if running != 0 or waiting != 0:
                raise RuntimeError(f"Admission barrier requires an idle server; running={running}, waiting={waiting}")
            timeout = aiohttp.ClientTimeout(total=10)
            status = await json_request(session, "GET", url(args.base_url, "/is_paused"), timeout=timeout)
            if status.get("is_paused") is not False:
                raise RuntimeError(f"Server already paused or pause state unknown: {status}")
            barrier.update(started_at=now(), resume_required=True)
            trace.write("admission_pause_requested", **barrier)
            # Set resume_required before the request: the server could pause and
            # the response could be lost, so cleanup must still attempt resume.
            await json_request(session, "POST", url(args.base_url, "/pause"),
                               params={"mode": "keep", "clear_cache": "false"}, timeout=timeout)
            status = await json_request(session, "GET", url(args.base_url, "/is_paused"), timeout=timeout)
            if status.get("is_paused") is not True:
                raise RuntimeError(f"Pause was not confirmed: {status}")
            barrier["pause_confirmed_at"] = now()
        poller = asyncio.create_task(poll_metrics(session, args, trace, done, metrics))
        markers = report["prompts"][0].get("turn_token_ids", {})
        tasks = [asyncio.create_task(stream_request(session, args, trace, gate, i, prompt, markers))
                 for i, prompt in enumerate(prompts)]
        gate.set()
        if barrier["enabled"]:
            try:
                await asyncio.wait_for(wait_for_admission(session, args, trace, tasks, metrics, barrier),
                                       timeout=getattr(args, "admission_timeout", 30))
            except asyncio.TimeoutError as exc:
                raise RuntimeError(f"Admission barrier timed out before {args.concurrency} requests were queued; "
                                   f"last waiting={barrier.get('last_observed_waiting')}") from exc
            await resume_barrier(session, args, trace, barrier)
        results = await asyncio.gather(*tasks)
        if barrier["enabled"]:
            for result in results:
                result["ttft_after_resume_seconds"] = (max(0, result["first_token"] - barrier["resumed_at"])
                    if result["first_token"] is not None else None)
            barrier["timing_note"] = ("Intentional admission pause remains in request TTFT and e2e; "
                                      "post-resume TTFT is separate. Decode overlap uses generated-token events only.")
        return results
    finally:
        resume_error = None
        if barrier["resume_required"]:
            try:
                await asyncio.shield(resume_barrier(session, args, trace, barrier))
            except Exception as exc:
                resume_error = exc
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        done.set()
        if poller:
            await poller
        if resume_error:
            raise resume_error


def summarize(args, results, metrics):
    all_intervals = all(r["first_token"] is not None and r["last_token"] is not None
                        for r in results)
    overlap_start = max(r["first_token"] for r in results) if all_intervals else None
    overlap_end = min(r["last_token"] for r in results) if all_intervals else None
    overlap = max(0, overlap_end - overlap_start) if all_intervals else 0
    for result in results:
        result["all_stream_overlap_tokens"] = None
        result["all_stream_overlap_tps"] = None
        if overlap > 0 and result["event_tokens_verified"]:
            count = sum(e["tokens"] for e in result["token_events"]
                        if overlap_start < e["at"] <= overlap_end)
            result["all_stream_overlap_tokens"] = count
            result["all_stream_overlap_tps"] = count / overlap
    def values(part):
        return [v for sample in metrics if (v := metric_sum(sample, part)) is not None]
    running, waiting = values("num_requests_running"), values("num_requests_waiting")
    preemptions = values("num_preemptions")
    cache = values("kv_cache_usage") or values("gpu_cache_usage")
    # Exclude boundary scrapes whose network interval spans either boundary.
    overlap_metrics = [sample for sample in metrics if overlap > 0 and
                       overlap_start <= sample["started_at"] <= sample["at"] <= overlap_end]
    overlap_running = [v for sample in overlap_metrics
                       if (v := metric_sum(sample, "num_requests_running")) is not None]
    overlap_waiting = [v for sample in overlap_metrics
                       if (v := metric_sum(sample, "num_requests_waiting")) is not None]
    overlap_rates = [r["all_stream_overlap_tps"] for r in results]
    full_success = all(not r["error"] and r["done_received"] and
                       r["prompt_tokens_verified"] and r["output_tokens_verified"] and
                       r["event_tokens_verified"] for r in results)
    metrics_complete = bool(running and waiting and preemptions) and all(
        not sample["error"] for sample in metrics)
    preemption_delta = preemptions[-1] - preemptions[0] if len(preemptions) > 1 else None
    checks = {
        "all_requests_completed_with_exact_token_counts": full_success,
        "all_stream_decode_overlap_at_least_min_seconds": overlap >= args.min_overlap_seconds,
        "every_stream_at_target_tps_during_overlap": bool(overlap_rates) and all(
            rate is not None and rate >= args.target_tps for rate in overlap_rates),
        "scheduler_metrics_available": metrics_complete,
        "all_streams_observed_running_during_overlap": bool(overlap_running) and
            min(overlap_running) >= args.concurrency,
        "no_observed_queueing_during_overlap": bool(overlap_waiting) and max(overlap_waiting) == 0,
        "no_observed_preemptions": preemption_delta == 0,
    }
    totals = sum((r["usage"] or {}).get("completion_tokens", 0) for r in results)
    duration = max(r["end"] for r in results) - min(r["start"] for r in results)
    diagnostics = [r["output_diagnostics"] for r in results]
    repeated_fractions = [d["repetition"]["duplicate_window_fraction"] for d in diagnostics
                          if d["repetition"]["duplicate_window_fraction"] is not None]
    return {
        "strict_throughput_pass": all(checks.values()), "checks": checks,
        "not_validated": ["base-model accuracy retention", "multimodal quality",
                          "representative real-world prompt distribution"],
        "all_stream_overlap_seconds": overlap, "all_stream_overlap_start": overlap_start,
        "all_stream_overlap_end": overlap_end,
        "minimum_stream_overlap_tps": min(overlap_rates) if all(
            r is not None for r in overlap_rates) else None,
        "aggregate_overlap_output_tps": sum(overlap_rates) if all(
            r is not None for r in overlap_rates) else None,
        "aggregate_e2e_output_tps": totals / duration if duration > 0 else None,
        "wall_seconds": duration, "total_output_tokens": totals,
        "output_characteristics": {
            "forced_output_length": not getattr(args, "respect_eos", False),
            "streams_with_token_diagnostics": sum(d["token_diagnostics_available"] for d in diagnostics),
            "streams_with_generated_turn_start": sum(d["new_turn_start_count"] > 0 for d in diagnostics),
            "maximum_duplicate_16gram_fraction": max(repeated_fractions) if repeated_fractions else None,
            "streams_stopping_naturally_before_output_target": sum(r["natural_stop_before_requested_length"] for r in results),
            "quality_verified": False},
        "metrics": {"max_running": max(running) if running else None,
                    "max_waiting": max(waiting) if waiting else None,
                    "max_waiting_during_overlap": max(overlap_waiting) if overlap_waiting else None,
                    "min_running_during_overlap": min(overlap_running) if overlap_running else None,
                    "preemptions_delta": preemption_delta,
                    "max_cache_usage_fraction": max(cache) if cache else None,
                    "scrape_count": len(metrics),
                    "scrape_errors": [s["error"] for s in metrics if s["error"]]},
        "measurement_notes": [
            "Synthetic text prompts; exact token-ID inputs; unique initial 256-token hashes; shared chat envelope recorded.",
            ("Natural EOS stopping enabled; early stopping fails the requested fixed-output-length gate while measured rates remain available."
             if getattr(args, "respect_eos", False) else
             "ignore_eos=True forces the requested output length and may generate repetitive extra turns; this is not useful-answer throughput."),
            "Per-stream decode TPS excludes all tokens in the first received token event.",
            "Overlap rates count exact tokens arriving in (latest first token, earliest last token].",
            "SSE token-event times are receive times, not exact server generation times.",
            "Scheduler metrics are sampled and may miss short queueing or occupancy changes.",
            "Admission queueing before common decoding is descriptive; queueing during overlap fails.",
            "Strict pass is a throughput result only, and does not establish accuracy or multimodality.",
        ],
    }


async def image_smoke(session, args):
    """Exercise image input; preserve the raw answer for human verification."""
    if not args.image_smoke:
        return None
    image_path = Path(args.image_smoke)
    extension = image_path.suffix.lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp"}.get(extension)
    if not mime:
        raise ValueError("--image-smoke expects PNG, JPEG, or WebP")
    encoded = base64.b64encode(image_path.read_bytes()).decode()
    payload = {"model": args.model, "messages": [{"role": "user", "content": [
        {"type": "text", "text": args.image_question},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
    ]}], "max_tokens": 256, "temperature": 0, "stream": False}
    result = await json_request(session, "POST", url(args.base_url, "/v1/chat/completions"),
                                json=payload)
    return {"image": str(image_path), "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "question": args.image_question, "response": result,
            "interpretation": "API image-input smoke test only; inspect answer correctness manually."}


async def run(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    trace_path = out.with_suffix(".trace.jsonl")
    trace = Trace(trace_path)
    started = datetime.now(timezone.utc).isoformat()
    report = {"started_utc": started, "config": vars(args), "trace": str(trace_path),
              "fatal_error": None, "summary": None, "requests": []}
    # Never serialize an API secret into the config or trace.
    headers = {}
    if os.environ.get(args.api_key_env):
        headers["Authorization"] = "Bearer " + os.environ[args.api_key_env]
    timeout = aiohttp.ClientTimeout(total=args.timeout, sock_read=args.timeout)
    connector = aiohttp.TCPConnector(limit=max(args.concurrency + 4, 16))
    try:
        async with aiohttp.ClientSession(timeout=timeout, connector=connector,
                                          headers=headers) as session:
            report["server_models"] = await json_request(session, "GET", url(args.base_url, "/v1/models"))
            prompts, report["prompts"] = await build_prompts(session, args, trace)
            report["prefix_warming"] = await warm_prefixes(session, args, trace, prompts)
            metrics = [await scrape_metrics(session, args, trace, "before")]
            results = await execute_requests(session, args, trace, prompts, metrics, report)
            metrics.append(await scrape_metrics(session, args, trace, "after"))
            report["summary"] = summarize(args, results, metrics)
            # All individual event timestamps remain in the trace.
            report["requests"] = [{k: v for k, v in r.items() if k != "token_events"}
                                  for r in results]
            report["image_smoke"] = await image_smoke(session, args)
    except Exception as exc:
        report["fatal_error"] = f"{type(exc).__name__}: {exc}"
        trace.write("fatal_error", error=report["fatal_error"])
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        trace.file.close()
    print(json.dumps({"out": str(out), "fatal_error": report["fatal_error"],
                      "summary": report["summary"]}, indent=2), flush=True)
    return 0 if report["summary"] and report["summary"]["strict_throughput_pass"] else 2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, default=50)
    parser.add_argument("--prompt-tokens", type=int, default=131072)
    parser.add_argument("--output-tokens", type=int, default=1024)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=float, default=7200,
                        help="Per HTTP request total and read timeout, in seconds")
    parser.add_argument("--metrics-interval", type=float, default=1)
    parser.add_argument("--target-tps", type=float, default=40)
    parser.add_argument("--min-overlap-seconds", type=float, default=5)
    parser.add_argument("--seed", type=int, default=17491)
    parser.add_argument("--chat-prompt", action=argparse.BooleanOptionalAction, default=True,
                        help="Use the server's actual chat template (default); --no-chat-prompt is raw synthetic completion")
    parser.add_argument("--respect-eos", action="store_true",
                        help="Allow natural stopping; short completions retain rates but fail the fixed-length gate")
    parser.add_argument("--disable-thinking", action="store_true",
                        help="Explicitly set enable_thinking=False in the chat template; native behavior is the default")
    parser.add_argument("--warm-prefixes", action="store_true",
                        help="Prefill each unique prompt with one output token before the timed run")
    parser.add_argument("--admission-barrier", action="store_true",
                        help="Use vLLM dev pause/resume endpoints to enqueue all requests before scheduling")
    parser.add_argument("--admission-timeout", type=float, default=30,
                        help="Maximum seconds waiting for all paused requests to appear in scheduler metrics")
    parser.add_argument("--api-key-env", default="VLLM_API_KEY")
    parser.add_argument("--image-smoke", help="Optional local image for separate image-input smoke test")
    parser.add_argument("--image-question", default="Describe the visible objects and their colors precisely.")
    args = parser.parse_args()
    if min(args.concurrency, args.prompt_tokens, args.output_tokens) <= 0:
        parser.error("concurrency, prompt-tokens, and output-tokens must be positive")
    if min(args.timeout, args.metrics_interval, args.min_overlap_seconds, args.admission_timeout) <= 0:
        parser.error("timeout, metrics-interval, min-overlap-seconds, and admission-timeout must be positive")
    return args


if __name__ == "__main__":
    sys.exit(asyncio.run(run(parse_args())))
