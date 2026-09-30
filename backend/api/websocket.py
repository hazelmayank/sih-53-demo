"""WebSocket stream: binary processed frames out, small JSON commands in.

Client -> server messages (JSON text):
    {"type": "subscribe", "points": true, "uniform": false, "max_points": 60000}
    {"type": "get_frame", "frame_id": 42}      # re-send a frame from the history buffer
Server -> client: binary frames (see protocol.py). Only the newest frame is sent
when a client falls behind, so slow clients never build a backlog.
"""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.api.protocol import encode_frame
from backend.core.engine import Subscriber

router = APIRouter()


@router.websocket("/ws/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    eng = ws.app.state.engine
    sub = Subscriber()
    eng.subscribers.add(sub)
    if not eng.started_once or eng.auto_paused:
        eng.auto_paused = False
        await eng.start()
    if eng.latest is not None:
        sub.event.set()

    async def reader():
        while True:
            msg = json.loads(await ws.receive_text())
            kind = msg.get("type")
            if kind == "subscribe":
                sub.options.update({k: msg[k] for k in ("points", "uniform", "max_points") if k in msg})
                sub.event.set()
            elif kind == "get_frame":
                sub.pending_frame = int(msg["frame_id"])
                sub.event.set()

    async def writer():
        last_sent = None
        while True:
            await sub.event.wait()
            sub.event.clear()
            pf = None
            if sub.pending_frame is not None:
                pf = eng.get_frame(sub.pending_frame)
                sub.pending_frame = None
            if pf is None:
                pf = eng.latest
            if pf is None:
                continue
            key = (id(pf), json.dumps(sub.options, sort_keys=True), eng.running)
            if key == last_sent:
                continue
            payload = await asyncio.to_thread(encode_frame, pf, dict(sub.options), eng.status())
            await ws.send_bytes(payload)
            last_sent = key

    tasks = [asyncio.create_task(reader()), asyncio.create_task(writer())]
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for t in done:
            exc = t.exception()
            if exc and not isinstance(exc, WebSocketDisconnect):
                raise exc
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        for t in tasks:
            t.cancel()
        eng.subscribers.discard(sub)
        if not eng.subscribers and eng.running:
            # nobody is watching: pause to save CPU; resumes when a viewer reconnects
            eng.auto_paused = True
            await eng.stop()
