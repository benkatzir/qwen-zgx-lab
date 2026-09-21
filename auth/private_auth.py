"""All-route ASGI authentication, loaded with vLLM's supported --middleware.

vLLM's built-in --api-key guards only the /v1 routes, which leaves /invocations and, with
VLLM_SERVER_DEV_MODE=1, the development admin endpoints open. This outer middleware
protects every HTTP method/path except GET/HEAD /health and rejects websockets. It passes ASGI events
through unchanged, so SSE and long multimodal request bodies remain streamed.
"""
import hmac
import os
from pathlib import Path


class BearerMiddleware:
    def __init__(self, app):
        self.app = app
        key = Path(os.environ.get("PRIVATE_API_KEY_FILE", "/run/secrets/api-key")).read_bytes().strip()
        if len(key) < 32 or len(key) > 512 or not key.isascii() or any(c <= 32 for c in key):
            raise RuntimeError("API key must contain 32..512 non-whitespace ASCII bytes")
        self.expected = b"Bearer " + key

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if scope.get("method") in ("GET", "HEAD") and scope.get("path") == "/health":
            await self.app(scope, receive, send)
            return
        auth = [value for name, value in scope.get("headers", []) if name.lower() == b"authorization"]
        if len(auth) != 1 or not hmac.compare_digest(auth[0], self.expected):
            body = b'{"error":"Unauthorized"}'
            await send({"type": "http.response.start", "status": 401, "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()), (b"www-authenticate", b"Bearer")]})
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)
