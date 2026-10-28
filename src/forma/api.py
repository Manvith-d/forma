"""The web app: the dashboard, a REST API for sessions, and a WebSocket that streams each turn live.

    forma serve            http://localhost:8000
    POST /api/sessions     start a conversation
    WS   /api/sessions/{id}/ws   send {"text": "..."}; receive agent, handoff, tool, token and state events
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .agents import Coach, view
from .config import ROOT

WEB = ROOT / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    coach = Coach()
    await asyncio.to_thread(coach.kb.search, "warm up the embedding model", 1)  # load ONNX once, not on the first question
    app.state.coach = coach
    app.state.locks = {}
    yield


app = FastAPI(title="Forma", version="0.1.0", lifespan=lifespan,
              description="An evidence-based multi-agent fitness coach.")


@app.middleware("http")
async def revalidate(request, call_next):  # the page and its assets: browsers check for a newer deploy (a cheap 304)
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def coach_of(ws_or_app) -> Coach:
    return ws_or_app.state.coach


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health() -> dict:
    c = coach_of(app)
    return {"status": "ok", "llm": {"model": c.llm.model, "available": c.llm.available}, "knowledge": c.kb.stats(),
            "demo": os.environ.get("FORMA_DEMO") == "1"}  # the public demo: rules-only, says so on the page


@app.get("/api/sources")
def sources() -> list[dict]:
    c = coach_of(app)
    return sorted(({k: d.get(k) for k in ("doc_id", "title", "publisher", "year", "url", "license", "topic")}
                   for d in c.kb.documents.values()), key=lambda d: (d["topic"] or "", d["title"]))


@app.post("/api/sessions")
def new_session() -> dict:
    c = coach_of(app)
    return view(c.new(), c)


@app.get("/api/sessions/{sid}")
def get_session(sid: str) -> dict:
    c = coach_of(app)
    s = c.get(sid)
    if not s:
        raise HTTPException(404, "session not found")
    return view(s, c)


@app.websocket("/api/sessions/{sid}/ws")
async def converse(ws: WebSocket, sid: str) -> None:
    c = coach_of(ws.app)
    s = c.get(sid)
    await ws.accept()
    if not s:
        await ws.send_json({"type": "error", "text": "session not found"})
        await ws.close()
        return
    lock = ws.app.state.locks.setdefault(sid, asyncio.Lock())
    try:
        while True:
            msg = await ws.receive_json()
            text = str(msg.get("text", "")).strip()[:2000]
            if not text:
                continue
            async with lock:  # one turn at a time per conversation
                try:
                    async for event in c.send(s, text):
                        await ws.send_json(event)
                except Exception as e:  # noqa: BLE001 - report the failure to the page instead of dropping the socket
                    await ws.send_json({"type": "error", "text": f"Something went wrong: {e}"})
    except WebSocketDisconnect:
        return


if (WEB / "assets").exists():
    app.mount("/assets", StaticFiles(directory=Path(WEB / "assets")), name="assets")
