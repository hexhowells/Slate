from __future__ import annotations

from flask import Flask, send_from_directory
from flask_socketio import SocketIO
import asyncio
import threading
import json
import websockets
from pathlib import Path
import logging

logging.getLogger('werkzeug').disabled = True


# Resolve static assets directory (prefer repo dev path if present)
_PKG_DIR = Path(__file__).parent
_REPO_STATIC = _PKG_DIR.parent.parent / "server" / "static"
_PKG_STATIC = _PKG_DIR / "static"
STATIC_DIR = _REPO_STATIC if _REPO_STATIC.exists() else _PKG_STATIC


app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    static_url_path="/static",
)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


ML_WS_PORT = 8765
ml_loop = asyncio.new_event_loop()
ml_clients: set[asyncio.Future] = set()


@app.route("/")
def index():
    _send_to_ml({"type": "send_checkpoints"})
    return send_from_directory(str(STATIC_DIR), "index.html")


async def ml_handler(ws):
    ml_clients.add(ws)
    try:
        async for msg in ws:
            data = json.loads(msg)
            evt = data.get("type", "frame_update")
            socketio.emit(evt, data)
    finally:
        ml_clients.discard(ws)


async def ml_server():
    async with websockets.serve(ml_handler, "127.0.0.1", ML_WS_PORT):
        print(f"[Slate] waiting for ML on ws://127.0.0.1:{ML_WS_PORT}")
        await asyncio.Future()


def _run_ml_loop():
    asyncio.set_event_loop(ml_loop)
    ml_loop.run_until_complete(ml_server())


def _send_to_ml(payload: dict):
    if not ml_clients:
        return
    txt = json.dumps(payload)
    for ws in list(ml_clients):
        asyncio.run_coroutine_threadsafe(ws.send(txt), ml_loop)


# @socketio.on("connect")
# def on_browser_connect():
#     print("[Browser] connected")


@socketio.on("step")
def on_step():
    _send_to_ml({"type": "step"})


@socketio.on("run")
def on_run():
    _send_to_ml({"type": "run"})


@socketio.on("pause")
def on_pause():
    _send_to_ml({"type": "pause"})


@socketio.on("reset")
def on_reset():
    _send_to_ml({"type": "reset"})


@socketio.on("select_checkpoint")
def on_select_checkpoint(data):
    _send_to_ml({"type": "select_checkpoint", "checkpoint": data.get("checkpoint", "")})


@socketio.on("send_checkpoints")
def on_send_checkpoints():
    _send_to_ml({"type": "send_checkpoints"})


def start_local_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """
    Start the local Slate dashboard server and the ML WebSocket bridge in background threads.

    Args:
        host: HTTP host for the dashboard.
        port: HTTP port for the dashboard.
    """
    # Start ML websocket bridge in background
    threading.Thread(target=_run_ml_loop, name="slate-ml-ws", daemon=True).start()

    # Start Flask-SocketIO web server in background
    threading.Thread(
        target=lambda: socketio.run(app, host=host, port=port),
        name="slate-web",
        daemon=True,
    ).start()
