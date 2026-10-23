"""The local language model (Ollama). Free, offline, and optional: every caller has a fallback.

FORMA_LLM=ollama:<model> (default ollama:qwen3:8b) or none. Thinking is switched off for speed, and
structured calls pass a JSON schema so the model can only answer in the expected shape.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

import httpx

from .config import settings


class LLM:
    def __init__(self, spec: str | None = None, url: str | None = None):
        s = settings()
        spec = spec or s.llm
        self.model = spec.split(":", 1)[1] if spec.startswith("ollama:") else None
        self.url = (url or s.ollama_url).rstrip("/")
        self._checked: tuple[float, bool] | None = None
        self.calls: list[dict] = []  # per-call latency and tokens, for the dashboard

    @property
    def available(self) -> bool:
        if not self.model:
            return False
        if self._checked and time.time() - self._checked[0] < 30:
            return self._checked[1]
        try:
            ok = httpx.get(f"{self.url}/api/tags", timeout=2).status_code == 200
        except httpx.HTTPError:
            ok = False
        self._checked = (time.time(), ok)
        return ok

    def _log(self, kind: str, started: float, data: dict) -> None:
        self.calls.append({"kind": kind, "ms": round((time.time() - started) * 1000),
                           "prompt_tokens": data.get("prompt_eval_count"), "output_tokens": data.get("eval_count")})

    async def json(self, system: str, user: str, schema: dict, timeout: float = 120) -> dict | None:
        if not self.available:
            return None
        started = time.time()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(f"{self.url}/api/chat", json={
                    "model": self.model, "stream": False, "think": False, "format": schema,
                    "options": {"temperature": 0}, "keep_alive": "30m",
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
            data = r.json()
            self._log("json", started, data)
            return json.loads(data["message"]["content"])
        except (httpx.HTTPError, KeyError, ValueError):
            return None

    async def stream(self, system: str, user: str, timeout: float = 180) -> AsyncIterator[str]:
        """Yield the reply token by token. Yields nothing if the model is unavailable."""
        if not self.available:
            return
        started = time.time()
        body = {"model": self.model, "stream": True, "think": False, "options": {"temperature": 0.3}, "keep_alive": "30m",
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        async with httpx.AsyncClient(timeout=timeout) as client, client.stream("POST", f"{self.url}/api/chat", json=body) as r:
            async for line in r.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("message", {}).get("content"):
                    yield chunk["message"]["content"]
                if chunk.get("done"):
                    self._log("stream", started, chunk)
