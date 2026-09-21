#!/usr/bin/env python3
"""CPU contract tests: no inference, sockets or real credentials."""
import asyncio
import os
from pathlib import Path
import tempfile
from private_auth import BearerMiddleware


async def test():
    with tempfile.TemporaryDirectory() as tmp:
        key = b"synthetic-local-test-credential-00000000"
        keyfile = Path(tmp) / "key"
        keyfile.write_bytes(key + b"\n")
        os.environ["PRIVATE_API_KEY_FILE"] = str(keyfile)
        calls = []
        async def app(scope, receive, send):
            calls.append(scope)
            assert (await receive())["body"] == b"image-and-prompt"
            await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/event-stream")]})
            await send({"type": "http.response.body", "body": b"data: one\n\n", "more_body": True})
            await send({"type": "http.response.body", "body": b"data: [DONE]\n\n", "more_body": False})
        protected = BearerMiddleware(app)
        async def request(path, headers=(), method="POST", kind="http"):
            sent = []
            async def receive():
                return {"type": "http.request", "body": b"image-and-prompt"}
            async def send(value):
                sent.append(value)
            await protected({"type": kind, "path": path, "method": method, "headers": headers}, receive, send)
            return sent
        for path in ("/v1/chat/completions", "/invocations", "/v2/models", "/inference", "/metrics", "/start_profile", "/health", "/unknown"):
            for method in ("GET", "POST", "OPTIONS"):
                if path == "/health" and method == "GET":
                    continue
                for headers in ([], [(b"authorization", b"Bearer wrong")], [(b"authorization", b"Bearer " + key), (b"authorization", b"Bearer " + key)]):
                    result = await request(path, headers, method)
                    assert result[0]["status"] == 401, (path, method)
        assert not calls
        result = await request("/v1/chat/completions", [(b"authorization", b"Bearer " + key)])
        assert result[0]["status"] == 200 and result[1]["more_body"] is True and result[2]["body"] == b"data: [DONE]\n\n"
        assert (await request("/health", method="GET"))[0]["status"] == 200
        assert (await request("/health", method="HEAD"))[0]["status"] == 200
        assert (await request("/v1/realtime", kind="websocket"))[0]["code"] == 1008
        keyfile.write_text("short")
        try:
            BearerMiddleware(app)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Short key accepted")
    print("PASS: unauthorized methods/routes, duplicate headers, stream passthrough, health exception, websocket refusal, missing-strength key")


asyncio.run(test())
