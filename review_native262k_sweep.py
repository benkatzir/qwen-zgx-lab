#!/usr/bin/env python3
"""Independently audit native-window SSE evidence; standard library only.

Usage: python3 review_native262k_sweep.py evidence.tar.gz --out comparison.json
The archive is read without extraction. Hardware failing a target is not an
audit error. Missing, inconsistent, or malformed evidence produces exit code 1.
Use --allow-partial while a sweep is running. Missing expected cases are then
listed without failing the audit; omit it for the final five-case audit.
"""

import argparse
import hashlib
import json
import math
import sys
import tarfile
from collections import Counter
from pathlib import Path, PurePosixPath


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def metric(sample, names):
    values = [v["value"] for v in sample.get("samples", [])
              if v.get("name", "").rsplit(":", 1)[-1] in names and number(v.get("value"))]
    return sum(values) if values else None


def audit_pair(report, lines, concurrency, input_tokens=260096, output_tokens=2048,
               context_tokens=262144, target_tps=18.0, min_overlap=60.0):
    """Return compact raw-derived results; do not import the benchmark harness."""
    errors, warnings = [], []

    def expect(condition, message):
        if not condition:
            errors.append(message)

    def compare(label, actual, claimed):
        same = (math.isclose(actual, claimed, rel_tol=1e-9, abs_tol=1e-7)
                if number(actual) and number(claimed) else actual == claimed)
        expect(same, f"{label}: raw={actual!r}, report={claimed!r}")

    states = {i: {"events": [], "ids": [], "texts": [], "prompt_count": None,
                  "prompt_hash": None, "prefix_hash": None, "usage": None,
                  "prompt_usage_counts": set(), "last_usage_count": 0,
                  "prompt": None, "start_event": None, "end": None,
                  "done_count": 0, "finish": None, "sse_errors": [], "last_sse_at": None}
              for i in range(concurrency)}
    metrics = []
    for line_no, line in enumerate(lines, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        kind = row.get("kind")
        if kind == "metrics":
            metrics.append(row)
            continue
        if kind not in {"prompt", "request_start", "request_end", "sse", "sse_done"}:
            continue
        index = row.get("request")
        if type(index) is not int or index not in states:
            errors.append(f"line {line_no}: unexpected request index {index!r}")
            continue
        state, label = states[index], f"request {index}"
        if kind in {"prompt", "request_start", "request_end"}:
            field = {"prompt": "prompt", "request_start": "start_event", "request_end": "end"}[kind]
            expect(state[field] is None, f"{label}: duplicate {kind}")
            state[field] = row
            continue
        if kind == "sse_done":
            state["done_count"] += 1
            continue
        at = row.get("at")
        expect(number(at), f"{label}: nonnumeric SSE timestamp")
        if not number(at):
            continue
        expect(state["last_sse_at"] is None or at >= state["last_sse_at"],
               f"{label}: SSE time moved backwards")
        state["last_sse_at"] = at
        event = row.get("event") or {}
        if event.get("error") is not None:
            state["sse_errors"].append(event["error"])
        usage = event.get("usage")
        if usage:
            state["usage"] = usage
            state["prompt_usage_counts"].add(usage.get("prompt_tokens"))
            count = usage.get("completion_tokens")
            expect(type(count) is int and count >= state["last_usage_count"],
                   f"{label}: missing/decreasing cumulative usage")
            if type(count) is int:
                state["last_usage_count"] = count
        choices = event.get("choices") or []
        if not choices:
            continue  # A final usage event is not a generated-token event.
        expect(len(choices) == 1 and choices[0].get("index", 0) == 0,
               f"{label}: unexpected completion choices")
        choice = choices[0]
        prompt_ids = choice.get("prompt_token_ids")
        if prompt_ids is not None:
            valid = isinstance(prompt_ids, list) and all(type(v) is int and v >= 0 for v in prompt_ids)
            expect(valid, f"{label}: malformed raw prompt token IDs")
            if valid:
                digest = hashlib.sha256(json.dumps(prompt_ids).encode()).hexdigest()
                expect(state["prompt_hash"] in (None, digest), f"{label}: inconsistent repeated prompt IDs")
                state["prompt_hash"], state["prompt_count"] = digest, len(prompt_ids)
                state["prefix_hash"] = hashlib.sha256(json.dumps(prompt_ids[:256]).encode()).hexdigest()
        ids = choice.get("token_ids")
        if ids is None:
            ids = (choice.get("delta") or {}).get("token_ids")
        text = choice.get("text") or (choice.get("delta") or {}).get("content") or ""
        if text:
            state["texts"].append(text)
        if ids is None:
            # Never turn SSE chunks or visible strings into inferred token counts.
            expect(not text, f"{label}: output text without exact token IDs")
        else:
            valid = isinstance(ids, list) and all(type(v) is int and v >= 0 for v in ids)
            expect(valid, f"{label}: malformed output token IDs")
            if valid and ids:
                state["ids"].extend(ids)
                state["events"].append((at, len(ids)))
        if choice.get("finish_reason"):
            expect(state["finish"] in (None, choice["finish_reason"]), f"{label}: conflicting finish reasons")
            state["finish"] = choice["finish_reason"]

    config = report.get("config") or {}
    for key, value in {"concurrency": concurrency, "prompt_tokens": input_tokens,
                       "output_tokens": output_tokens}.items():
        compare(f"config.{key}", value, config.get(key))
    expect(report.get("fatal_error") is None, f"fatal error: {report.get('fatal_error')}")
    requests = report.get("requests") or []
    claimed_requests = {r.get("request"): r for r in requests}
    expect(len(requests) == concurrency and set(claimed_requests) == set(states),
           "reported requests are missing, duplicate, or unexpected")
    claimed_prompts = {r.get("request"): r for r in report.get("prompts") or []}
    expect(len(claimed_prompts) == concurrency, "reported prompt metadata count differs from concurrency")
    complete_intervals = all(state["events"] for state in states.values())
    start = max(s["events"][0][0] for s in states.values()) if complete_intervals else None
    end = min(s["events"][-1][0] for s in states.values()) if complete_intervals else None
    duration = max(0.0, end - start) if complete_intervals else 0.0
    streams = []
    for index, state in states.items():
        label = f"request {index}"
        events, ids = state["events"], state["ids"]
        first, last = (events[0][0], events[-1][0]) if events else (None, None)
        raw_end, usage = state["end"] or {}, state["usage"] or {}
        claimed = claimed_requests.get(index, {})
        expect(bool(raw_end), f"{label}: no request_end record")
        expect(state["start_event"] is not None, f"{label}: no request_start record")
        expect(state["prompt"] is not None, f"{label}: no prompt record")
        expect(state["prompt_count"] is not None, f"{label}: no raw prompt token IDs")
        expect(bool(events), f"{label}: no exact output-token events")
        expect(bool(usage), f"{label}: no SSE usage")
        expect(state["done_count"] == 1, f"{label}: expected one SSE DONE, found {state['done_count']}")
        expect(state["finish"] in {"length", "stop"}, f"{label}: incomplete/unknown finish reason {state['finish']!r}")
        expect(not state["sse_errors"] and not raw_end.get("error"), f"{label}: request/SSE error")
        expect(raw_end.get("http_status") == 200, f"{label}: HTTP status was not 200")
        expect(state["prompt_usage_counts"] == {state["prompt_count"]}, f"{label}: usage and raw input IDs differ")
        expect(usage.get("completion_tokens") == len(ids), f"{label}: final usage and raw output IDs differ")
        expect(usage.get("total_tokens") == (state["prompt_count"] or 0) + len(ids),
               f"{label}: final total token usage differs")
        for source, meta in (("trace", state["prompt"] or {}),
                             ("report", claimed_prompts.get(index, {}))):
            compare(f"{label} {source} prompt count", state["prompt_count"], meta.get("input_tokens"))
            compare(f"{label} {source} prompt hash", state["prompt_hash"], meta.get("prompt_sha256"))
            compare(f"{label} {source} prefix hash", state["prefix_hash"], meta.get("prefix_256_sha256"))
        if state["start_event"]:
            compare(f"{label} requested input count", state["prompt_count"], state["start_event"].get("input_tokens"))
            compare(f"{label} requested output limit", output_tokens, state["start_event"].get("output_tokens"))
        began, ended = raw_end.get("start"), raw_end.get("end")
        timing_valid = all(number(v) for v in (began, first, last, ended)) and began <= first <= last <= ended
        expect(timing_valid, f"{label}: missing/unordered request timings")
        ttft = first - began if timing_valid else None
        decode = (len(ids) - events[0][1]) / (last - first) if events and last > first else None
        overlap_tokens = sum(n for at, n in events if start < at <= end) if duration > 0 else None
        rate = overlap_tokens / duration if duration > 0 else None
        natural_early = bool(config.get("respect_eos") and state["finish"] == "stop" and len(ids) < output_tokens)
        full = state["prompt_count"] == input_tokens and len(ids) == output_tokens
        completed = bool(timing_valid and state["done_count"] == 1 and state["finish"] in {"length", "stop"}
                         and not raw_end.get("error") and not state["sse_errors"] and raw_end.get("http_status") == 200)
        derived = {"first_token": first, "last_token": last, "usage": usage,
                   "stream_token_count": len(ids), "finish_reason": state["finish"],
                   "done_received": state["done_count"] == 1, "ttft_seconds": ttft,
                   "decode_tps": decode, "prompt_tokens_verified": state["prompt_count"] == input_tokens,
                   "output_tokens_verified": len(ids) == output_tokens,
                   "event_tokens_verified": bool(events) and len(ids) == usage.get("completion_tokens"),
                   "natural_stop_before_requested_length": natural_early}
        for key, value in derived.items():
            compare(f"{label} {key}", value, claimed.get(key))
            compare(f"{label} trace request_end.{key}", value, raw_end.get(key))
        for key, value in {"start": began, "end": ended, "all_stream_overlap_tokens": overlap_tokens,
                           "all_stream_overlap_tps": rate}.items():
            compare(f"{label} {key}", value, claimed.get(key))
        grams = Counter(tuple(ids[i:i + 16]) for i in range(max(0, len(ids) - 15)))
        windows = max(0, len(ids) - 15)
        repetition = (windows - len(grams)) / windows if windows else None
        markers = (state["prompt"] or {}).get("turn_token_ids", {})
        turn_count = ids.count(markers["<|im_start|>"]) if "<|im_start|>" in markers else None
        diagnostics = claimed.get("output_diagnostics") or {}
        compare(f"{label} diagnostic token count", len(ids), diagnostics.get("token_ids_observed"))
        compare(f"{label} duplicate 16-grams", repetition,
                (diagnostics.get("repetition") or {}).get("duplicate_window_fraction"))
        if turn_count is not None:
            compare(f"{label} generated turn starts", turn_count, diagnostics.get("new_turn_start_count"))
        streams.append({"request": index, "input_token_ids": state["prompt_count"],
                        "output_token_ids": len(ids), "usage_output_tokens": usage.get("completion_tokens"),
                        "finish_reason": state["finish"], "completed": completed,
                        "filled_native_budget": full and input_tokens + output_tokens == context_tokens,
                        "natural_early_eos": natural_early, "first_token_at": first, "last_token_at": last,
                        "ttft_seconds": ttft, "decode_tps": decode,
                        "common_tokens": overlap_tokens, "common_tps": rate,
                        "generated_turn_starts": turn_count, "duplicate_16gram_fraction": repetition})

    expect(len({s["prefix_hash"] for s in states.values()}) == concurrency,
           "raw initial 256-token prefixes are not unique")
    metrics.sort(key=lambda row: row.get("at", float("inf")))
    within = [m for m in metrics if duration > 0 and number(m.get("started_at")) and number(m.get("at"))
              and start <= m["started_at"] <= m["at"] <= end]

    def values(rows, names):
        return [v for row in rows if (v := metric(row, names)) is not None]

    running = values(metrics, {"num_requests_running"})
    waiting = values(metrics, {"num_requests_waiting"})
    preemptions = values(metrics, {"num_preemptions", "num_preemptions_total"})
    cache = values(metrics, {"kv_cache_usage", "kv_cache_usage_perc"}) or values(metrics, {"gpu_cache_usage", "gpu_cache_usage_perc"})
    overlap_running = values(within, {"num_requests_running"})
    overlap_waiting = values(within, {"num_requests_waiting"})
    scrape_errors = [m["error"] for m in metrics if m.get("error")]
    delta = preemptions[-1] - preemptions[0] if len(preemptions) > 1 else None
    resets = sum(b < a for a, b in zip(preemptions, preemptions[1:]))
    raw_metrics = {"max_running": max(running) if running else None,
                   "max_waiting": max(waiting) if waiting else None,
                   "max_waiting_during_overlap": max(overlap_waiting) if overlap_waiting else None,
                   "min_running_during_overlap": min(overlap_running) if overlap_running else None,
                   "preemptions_delta": delta, "max_cache_usage_fraction": max(cache) if cache else None,
                   "scrape_count": len(metrics), "scrape_errors": scrape_errors}
    summary = report.get("summary") or {}
    for key, value in raw_metrics.items():
        compare(f"metrics.{key}", value, (summary.get("metrics") or {}).get(key))
    raw_metrics.update(common_scrape_count=len(within), preemption_counter_resets=resets,
                       common_running_sample_count=len(overlap_running), common_waiting_sample_count=len(overlap_waiting))
    rates = [s["common_tps"] for s in streams if s["common_tps"] is not None]
    all_rates = len(rates) == concurrency
    minimum, maximum = (min(rates), max(rates)) if all_rates else (None, None)
    aggregate = sum(rates) if all_rates else None
    ttfts = [s["ttft_seconds"] for s in streams if s["ttft_seconds"] is not None]
    full = all(s["filled_native_budget"] and s["completed"] for s in streams)
    metrics_available = bool(running and waiting and preemptions) and not scrape_errors
    scheduler_ok = bool(metrics_available and overlap_running and min(overlap_running) >= concurrency
                        and overlap_waiting and max(overlap_waiting) == 0 and delta == 0 and resets == 0)
    checks = {"all_requests_completed_with_exact_token_counts": all(s["completed"] and
                  s["input_token_ids"] == input_tokens and s["output_token_ids"] == output_tokens and
                  s["usage_output_tokens"] == output_tokens for s in streams),
              "all_stream_decode_overlap_at_least_min_seconds": duration >= config.get("min_overlap_seconds", min_overlap),
              "every_stream_at_target_tps_during_overlap": all_rates and minimum >= config.get("target_tps", target_tps),
              "scheduler_metrics_available": metrics_available,
              "all_streams_observed_running_during_overlap": bool(overlap_running) and min(overlap_running) >= concurrency,
              "no_observed_queueing_during_overlap": bool(overlap_waiting) and max(overlap_waiting) == 0,
              "no_observed_preemptions": delta == 0}
    for key, value in checks.items():
        compare(f"checks.{key}", value, (summary.get("checks") or {}).get(key))
    compare("strict_throughput_pass at report's thresholds", all(checks.values()), summary.get("strict_throughput_pass"))
    for key, value in {"all_stream_overlap_start": start, "all_stream_overlap_end": end,
                       "all_stream_overlap_seconds": duration, "minimum_stream_overlap_tps": minimum,
                       "aggregate_overlap_output_tps": aggregate,
                       "total_output_tokens": sum(s["output_token_ids"] for s in streams)}.items():
        compare(key, value, summary.get(key))
    native_chat = config.get("chat_prompt") is True and config.get("respect_eos") is True and config.get("disable_thinking") is False
    if not native_chat:
        warnings.append("Scenario is not confirmed as chat prompts with natural EOS and native thinking.")
    if config.get("warm_prefixes"):
        warnings.append("Prefixes were warmed; TTFT is not a cold-prefill measurement.")
    if config.get("admission_barrier"):
        warnings.append("Admission pause is included in raw TTFT; no pause adjustment is inferred.")
    if duration < min_overlap:
        warnings.append("Short common interval: rates are exploratory, not a sustained-duration pass.")
    if any(s["natural_early_eos"] for s in streams):
        warnings.append("Natural early EOS: at least one stream never filled the native context budget.")
    if not full:
        warnings.append("Not every completed stream has the exact full native input-plus-output budget.")
    if resets:
        warnings.append("Preemption counter reset observed; zero end-to-end delta cannot certify no preemptions.")
    measured_pass = bool(not errors and all(s["completed"] for s in streams) and all_rates
                         and minimum >= target_tps and duration >= min_overlap and scheduler_ok)
    return {"concurrency": concurrency, "audit_pass": not errors, "audit_errors": errors,
            "warnings": warnings, "filled_native_budget_all_streams": full,
            "natural_early_eos_streams": sum(s["natural_early_eos"] for s in streams),
            "finish_counts": dict(Counter(str(s["finish_reason"]) for s in streams)),
            "common_seconds": duration, "common_start": start, "common_end": end,
            "min_tps": minimum, "mean_tps": aggregate / concurrency if all_rates else None,
            "max_tps": maximum, "aggregate_tps": aggregate,
            "ttft_min_seconds": min(ttfts) if len(ttfts) == concurrency else None,
            "ttft_max_seconds": max(ttfts) if len(ttfts) == concurrency else None,
            "measured_target_and_duration_pass": measured_pass,
            "full_native_target_and_duration_pass": measured_pass and full and native_chat,
            "native_thinking_chat_respect_eos": native_chat, "metrics": raw_metrics,
            "output_diagnostics": {"quality_verified": False,
                "streams_with_generated_turn_starts": sum((s["generated_turn_starts"] or 0) > 0 for s in streams),
                "maximum_duplicate_16gram_fraction": max((s["duplicate_16gram_fraction"] for s in streams
                    if s["duplicate_16gram_fraction"] is not None), default=None)},
            "streams": streams}


def audit_archive(path, concurrencies, target_tps=18.0, min_overlap=60.0, allow_partial=False):
    runs, errors, missing = [], [], []
    with tarfile.open(path, "r:gz") as archive:
        members = {}
        for member in archive.getmembers():
            if member.isfile():
                members.setdefault(PurePosixPath(member.name).name, []).append(member)
        for concurrency in sorted(set(concurrencies)):
            stem = f"native262k-c{concurrency}-260096in-2048out"
            pair = [members.get(stem + suffix, []) for suffix in (".json", ".trace.jsonl")]
            if not all(len(items) == 1 for items in pair):
                message = f"{stem}: expected one report and one trace; found {list(map(len, pair))}"
                if any(len(items) > 1 for items in pair):
                    errors.append(message)
                else:
                    missing.append(concurrency)
                    if not allow_partial:
                        errors.append(message)
                continue
            report_member, trace_member = pair[0][0], pair[1][0]
            try:
                with archive.extractfile(report_member) as source:
                    report = json.load(source)
                with archive.extractfile(trace_member) as source:
                    result = audit_pair(report, source, concurrency, target_tps=target_tps, min_overlap=min_overlap)
                result.update(report_member=report_member.name, trace_member=trace_member.name)
                runs.append(result)
            except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
                errors.append(f"{stem}: cannot audit malformed evidence: {type(exc).__name__}: {exc}")
    passing = [r["concurrency"] for r in runs if r["full_native_target_and_duration_pass"]]
    return {"archive": str(Path(path).resolve()), "audit_pass": bool(runs) and not errors and all(r["audit_pass"] for r in runs),
            "audit_errors": errors, "expected_concurrencies": sorted(set(concurrencies)),
            "missing_expected_concurrencies": missing, "allow_partial": allow_partial,
            "complete_sweep": len(runs) == len(set(concurrencies)) and not missing,
            "requirements": {"input_tokens": 260096, "output_tokens": 2048, "native_context_tokens": 262144,
                             "target_tps_per_stream": target_tps, "minimum_common_seconds": min_overlap},
            "highest_tested_full_native_concurrency_meeting_target": max(passing) if passing else None,
            "runs": runs, "quality_verified": False,
            "interpretation": [
                "Highest tested passing concurrency is not a proven hardware maximum; intermediate values may be untested.",
                "260096 input tokens occupy 99.21875% of the native budget; 262144 is reached only after 2048 outputs, not throughout decoding.",
                "Common rates count accepted output token IDs at client receive times in (latest first token, earliest last token]; MTP chunks may contain multiple tokens.",
                "Shorter common intervals retain exploratory rates but cannot pass the sustained-duration gate.",
                "Raw TTFT uses request_end.start and the first raw token event; it includes prefill, admission, transport, and any unrecorded pauses.",
                "Scheduler metrics are sampled; zero observed queueing/preemptions is not proof against sub-sample transients.",
                "Synthetic prose, native thinking, and repeated planning may change MTP acceptance; these rates are not a real-world workload guarantee.",
                "Token counts, repetition, and finish reasons do not establish useful-answer quality, multimodal performance, or 97% base-model accuracy retention."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--out", type=Path, help="Write JSON here; otherwise use stdout")
    parser.add_argument("--concurrencies", default="1,2,4,8,16", help="Comma-separated expected concurrency values")
    parser.add_argument("--allow-partial", action="store_true", help="Audit available pairs without failing for missing expected cases")
    parser.add_argument("--target-tps", type=float, default=18.0)
    parser.add_argument("--min-overlap-seconds", type=float, default=60.0)
    args = parser.parse_args()
    try:
        concurrencies = [int(v) for v in args.concurrencies.split(",")]
        if not concurrencies or min(concurrencies) < 1:
            raise ValueError("concurrencies must be positive")
        if not number(args.target_tps) or not number(args.min_overlap_seconds) or min(args.target_tps, args.min_overlap_seconds) <= 0:
            raise ValueError("thresholds must be positive finite numbers")
    except ValueError as exc:
        parser.error(str(exc))
    try:
        result = audit_archive(args.archive, concurrencies, args.target_tps, args.min_overlap_seconds, args.allow_partial)
    except (OSError, tarfile.TarError) as exc:
        result = {"audit_pass": False, "audit_errors": [f"Cannot read archive: {exc}"], "quality_verified": False}
    output = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(output, encoding="utf-8")
        print(f"Audit {'PASS' if result['audit_pass'] else 'FAIL'}: {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(output)
    return 0 if result["audit_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
