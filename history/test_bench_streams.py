"""A tiny local server validates multi-token SSE accounting and strict failures."""
import argparse
import asyncio
import hashlib
import json
import tempfile
from pathlib import Path

from aiohttp import ClientConnectionError, web

import bench_streams as bench


async def main():
    state = {"running": 0, "waiting": 0, "mode": "ids", "paused": False,
             "queued": 0, "resume_count": 0}
    resumed = asyncio.Event()
    resumed.set()

    async def pause(request):
        assert request.query["mode"] == "keep"
        assert request.query["clear_cache"] == "false"
        state["paused"] = True
        resumed.clear()
        return web.json_response({"status": "paused"})

    async def resume(_):
        state["paused"] = False
        state["resume_count"] += 1
        resumed.set()
        return web.json_response({"status": "resumed"})

    async def is_paused(_):
        return web.json_response({"is_paused": state["paused"]})

    async def models(_):
        return web.json_response({"data": [{"id": "mock"}]})

    async def tokenize(request):
        data = await request.json()
        if 'messages' in data:
            data['prompt'] = ('CHAT_START ' + data['messages'][0]['content'] + ' USER_START ' +
                              data['messages'][1]['content'] + ' USER_END ASSISTANT_START')
            assert 'chat_template_kwargs' not in data  # Native thinking remains the default.
        ids = [int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "big")
               for word in data["prompt"].split()]
        return web.json_response({"tokens": ids, "count": len(ids), "max_model_len": 200000})

    async def metrics(_):
        return web.Response(text=(f'vllm:num_requests_running{{model_name="mock"}} {state["running"]}\n'
                                  f'vllm:num_requests_waiting{{model_name="mock"}} {state["waiting"] + state["queued"]}\n'
                                  'vllm:num_preemptions_total{model_name="mock"} 0\n'
                                  'vllm:kv_cache_usage_perc{model_name="mock"} 0.5\n'))

    async def completion(request):
        data = await request.json()
        p, count = len(data["prompt"]), data["max_tokens"]
        if not data["stream"]:
            return web.json_response({"choices": [{"text": "warm"}],
                                      "usage": {"prompt_tokens": p, "completion_tokens": 1}})
        if state["mode"] == "early":
            assert data["ignore_eos"] is False
            count = min(count, 6)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        if state["paused"]:
            state["queued"] += 1
            await resumed.wait()
            state["queued"] -= 1
        state["running"] += 1
        await asyncio.sleep(0.03)
        for n in range(0, count, 2):
            delta = min(2, count - n)
            choice = {"index": 0, "text": "two words "}
            if state["mode"] in ("ids", "early"):
                choice["token_ids"] = [10] * delta
                if n == 0:
                    # Real vLLM echoes all prompt IDs in the first SSE event.
                    # This exceeds aiohttp's default line-reader limit.
                    choice["prompt_token_ids"] = [123456] * 131072
            event = {"choices": [choice], "usage": {
                "prompt_tokens": p, "completion_tokens": n + delta}}
            if state["mode"] == "unknown":
                event.pop("usage")
            try:
                await response.write(("data: " + json.dumps(event) + "\n\n").encode())
            except ClientConnectionError:
                state["running"] -= 1
                return response  # Expected when testing failed-barrier cleanup.
            await asyncio.sleep(0.03)
        final = {"choices": [], "usage": {"prompt_tokens": p, "completion_tokens": count}}
        if state["mode"] == "early":
            terminal = {"choices": [{"index": 0, "text": "", "token_ids": [], "finish_reason": "stop"}]}
            await response.write(("data: " + json.dumps(terminal) + "\n\n").encode())
        await response.write(("data: " + json.dumps(final) + "\n\ndata: [DONE]\n\n").encode())
        state["running"] -= 1
        return response

    app = web.Application()
    app.add_routes([web.get("/v1/models", models), web.post("/tokenize", tokenize),
                    web.get("/metrics", metrics), web.post("/v1/completions", completion),
                    web.post("/pause", pause), web.post("/resume", resume),
                    web.get("/is_paused", is_paused)])
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    try:
        mixed = bench.parse_metrics('vllm:num_requests_waiting 3\n'
                                    'vllm:num_requests_waiting_by_reason{reason="capacity"} 3\n'
                                    'vllm:num_preemptions_total 2\n'
                                    'vllm:num_preemptions_created 1000000\n')
        assert bench.metric_sum({"samples": mixed}, "num_requests_waiting") == 3
        assert bench.metric_sum({"samples": mixed}, "num_preemptions") == 2
        with tempfile.TemporaryDirectory() as directory:
            for mode, waiting, expected, barrier in [("ids", 0, True, False), ("usage", 0, True, False),
                    ("ids", 1, False, False), ("unknown", 0, False, False),
                    ("early", 0, False, False), ("ids", 0, True, True)]:
                state.update(mode=mode, waiting=waiting)
                out = Path(directory) / f"{mode}-{waiting}.json"
                args = argparse.Namespace(base_url=f"http://127.0.0.1:{port}", model="mock",
                    concurrency=2, prompt_tokens=256, output_tokens=20, out=str(out),
                    timeout=10, metrics_interval=0.01, target_tps=40,
                    min_overlap_seconds=0.1, seed=1, api_key_env="NO_SUCH_SECRET",
                    image_smoke=None, image_question="test", warm_prefixes=True,
                    admission_barrier=barrier, admission_timeout=1,
                    chat_prompt=True, respect_eos=mode == "early", disable_thinking=False)
                code = await bench.run(args)
                result = json.loads(out.read_text())
                assert result["fatal_error"] is None, result
                assert result["summary"]["strict_throughput_pass"] is expected, result
                assert (code == 0) is expected
                assert len(result["requests"]) == 2
                assert result["prefix_warming"]["seconds"] > 0
                assert all(p["chat_prompt"] and p["assistant_suffix_tokens"] > 0
                           and p["input_tokens"] == 256 for p in result["prompts"])
                if mode == "early":
                    assert all(r["natural_stop_before_requested_length"] for r in result["requests"])
                    assert all(r["decode_tps"] is not None for r in result["requests"])
                if barrier:
                    assert result["admission_barrier"]["all_enqueued_confirmed"]
                    assert result["admission_barrier"]["resume_confirmed"]
                    assert not state["paused"]
                    assert result["summary"]["metrics"]["max_waiting"] == 2
                    assert all(r["first_token"] > result["admission_barrier"]["resumed_at"]
                               for r in result["requests"])
                if expected:
                    # Two tokens per SSE event must count as two, not one.
                    assert all(45 < r["decode_tps"] < 90 for r in result["requests"])
                    assert all(r["stream_token_count"] == 20 for r in result["requests"])
                else:
                    key = ("no_observed_queueing_during_overlap" if waiting else
                           "all_requests_completed_with_exact_token_counts")
                    assert not result["summary"]["checks"][key]
                print(f"PASS mode={mode} waiting={waiting} barrier={barrier}")
            synthetic = [1, 2, 3, 4] * 100 + [99, 5, 98]
            diagnostics = bench.output_diagnostics(synthetic, 'user\nassistant\n',
                                                   {'<|im_start|>': 99, '<|im_end|>': 98}, [])
            assert diagnostics['new_turn_start_count'] == 1
            assert diagnostics['special_turn_markers']['<|im_end|>']['first_output_token_index'] == 402
            assert diagnostics['repetition']['duplicate_window_fraction'] > 0.9
            assert diagnostics['visible_role_label_matches'] == 2
            print('PASS generated turn markers and repetition diagnostics')
            # Force a queue-confirmation failure without relying on wall-clock
            # scheduling; the finally path must resume before returning failure.
            original_wait = bench.wait_for_admission
            async def fail_admission(*_):
                raise RuntimeError("injected admission failure")
            bench.wait_for_admission = fail_admission
            before_resumes = state["resume_count"]
            args.out = str(Path(directory) / "barrier-failure.json")
            try:
                code = await bench.run(args)
                failed = json.loads(Path(args.out).read_text())
                assert code == 2 and "injected admission failure" in failed["fatal_error"]
                assert not state["paused"] and state["resume_count"] > before_resumes
                assert failed["admission_barrier"]["resume_confirmed"]
                print("PASS barrier failure resumes server")
            finally:
                bench.wait_for_admission = original_wait
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
