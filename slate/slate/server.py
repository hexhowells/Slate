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

# Server-side run history storage
MAX_HISTORY_SIZE = 5
run_history: list[dict] = []
run_data_storage: dict[int, dict] = {}  # Store full run data separately


@app.route("/")
def index():
    """Serve the dashboard and request the latest checkpoints from the ML side.

    Returns:
        A Flask response that serves `index.html` from the resolved static dir.
    """
    _send_to_ml({"type": "send_checkpoints"})
    return send_from_directory(str(STATIC_DIR), "index.html")


async def ml_handler(ws) -> None:
    """Handle a single ML runtime WebSocket connection.

    Adds the connection to the client set and forwards any JSON messages from
    the ML runtime to the browser via Socket.IO, using the `type` field as the
    event name. Removes the connection on disconnect.

    Args:
        ws: The connected WebSocket server protocol instance.
    """
    ml_clients.add(ws)
    try:
        async for msg in ws:
            data = json.loads(msg)
            msg_type = data.get("type", None)
            
            match msg_type:
                case "frame_update":
                    socketio.emit(msg_type, data)
                case "run_completed":
                    run_data = data.get("payload", {})
                    add_run_to_history(run_data)
                case _:
                    print(f"ML Websocket received unknown message type: {msg_type}, ignoring")
                
    finally:
        ml_clients.discard(ws)


async def ml_server(ml_host: str = "127.0.0.1") -> None:
    """Start the ML WebSocket server and wait indefinitely.

    The server listens on 127.0.0.1 at `ML_WS_PORT` and spawns `ml_handler`
    for each incoming connection.

    Args:
        ml_host: the HTTP web host of the websocket
    """
    async with websockets.serve(ml_handler, ml_host, ML_WS_PORT):
        print(f"[Slate] waiting for ML on ws://{ml_host}:{ML_WS_PORT}")
        await asyncio.Future()


def add_run_to_history(run_data: dict) -> None:
    """Add a completed run to the server-side history.
    
    Args:
        run_data: Dictionary containing run information and frames
    """
    global run_history, run_data_storage
    
    # Assign server-side ID
    run_id = len(run_history)
    run_data["id"] = run_id
    
    # Store full data separately (including frames)
    run_data_storage[run_id] = run_data
    
    # Create metadata-only version for history list
    run_metadata = {
        "id": run_id,
        "timestamp": run_data["timestamp"],
        "duration": run_data["duration"],
        "total_steps": run_data["total_steps"],
        "total_reward": run_data["total_reward"],
        "checkpoint": run_data["checkpoint"]
    }
    
    # Add to history
    run_history.append(run_metadata)
    
    # Maintain max history size
    if len(run_history) > MAX_HISTORY_SIZE:
        old_run = run_history.pop(0)
        # Clean up stored data for removed run
        if old_run["id"] in run_data_storage:
            del run_data_storage[old_run["id"]]
    
    # Broadcast updated history to all connected clients
    socketio.emit("run_history_update", {"run_history": run_history})


def _send_to_ml(payload: dict) -> None:
    """Send a JSON payload to all connected ML runtime WebSocket clients.

    Args:
        payload: Dictionary that will be JSON-encoded and sent.
    """
    if not ml_clients:
        return
    txt = json.dumps(payload)
    for ws in list(ml_clients):
        asyncio.run_coroutine_threadsafe(ws.send(txt), ml_loop)


@socketio.on("step")
def on_step() -> None:
    """Request a single environment step from the ML runtime."""
    _send_to_ml({"type": "step"})


@socketio.on("run")
def on_run() -> None:
    """Start continuous stepping on the ML runtime."""
    _send_to_ml({"type": "run"})


@socketio.on("pause")
def on_pause() -> None:
    """Pause continuous stepping on the ML runtime."""
    _send_to_ml({"type": "pause"})


@socketio.on("reset")
def on_reset() -> None:
    """Reset the environment on the ML runtime."""
    _send_to_ml({"type": "reset"})


@socketio.on("select_checkpoint")
def on_select_checkpoint(data) -> None:
    """Ask the ML runtime to load a specific checkpoint.

    Args:
        data: Dict containing a `checkpoint` key with the identifier/path.
    """
    _send_to_ml({"type": "select_checkpoint", "checkpoint": data.get("checkpoint", "")})


@socketio.on("send_checkpoints")
def on_send_checkpoints() -> None:
    """Request the list of available checkpoints from the ML runtime."""
    _send_to_ml({"type": "send_checkpoints"})


@socketio.on("send_run_history")
def on_send_run_history() -> None:
    """Request the run history from the ML runtime."""
    _send_to_ml({"type": "send_run_history"})


@socketio.on("playback_run")
def on_playback_run(data) -> None:
    """Send playback data for a specific run directly to the requesting client.
    Uses chunked transfer for large runs to avoid WebSocket size limits.
    
    Args:
        data: Dict containing a `run_id` key with the run identifier.
    """
    run_id = data.get("run_id", 0)
    if run_id in run_data_storage:
        run_data = run_data_storage[run_id]
        
        # Check if data is too large (estimate > 500KB)
        estimated_size = len(str(run_data))
        if estimated_size > 500000:  # 500KB threshold
            # Send in chunks
            frames = run_data.get("frames", [])
            metadata = run_data.get("metadata", [])
            
            # Send run info first
            run_info = {
                "id": run_data["id"],
                "timestamp": run_data["timestamp"],
                "duration": run_data["duration"],
                "total_steps": run_data["total_steps"],
                "total_reward": run_data["total_reward"],
                "checkpoint": run_data["checkpoint"],
                "chunked": True,
                "total_chunks": len(frames)
            }
            socketio.emit("playback_data_start", {"payload": run_info})
            
            # Send frames in chunks
            chunk_size = 10  # Send 10 frames per chunk
            for i in range(0, len(frames), chunk_size):
                chunk_frames = frames[i:i+chunk_size]
                chunk_metadata = metadata[i:i+chunk_size]
                socketio.emit("playback_data_chunk", {
                    "chunk_index": i // chunk_size,
                    "frames": chunk_frames,
                    "metadata": chunk_metadata
                })
        else:
            # Send normally for smaller runs
            socketio.emit("playback_data", {"payload": run_data})


@socketio.on("get_run_history")
def on_get_run_history() -> None:
    """Send current run history to the requesting client."""
    socketio.emit("run_history_update", {"run_history": run_history})


def _run_ml_loop(ml_host: str) -> None:
    """Run the ML WebSocket server inside a dedicated asyncio event loop."""
    asyncio.set_event_loop(ml_loop)
    ml_loop.run_until_complete(ml_server(ml_host))


def start_local_server(
        host: str = "0.0.0.0",
        port: int = 8000,
        ml_host: str = "0.0.0.0",
    ) -> None:
    """
    Start the local Slate dashboard server and the ML WebSocket bridge in background threads.

    Args:
        host: HTTP host for the dashboard.
        port: HTTP port for the dashboard.
        ml_host: HTTP host for ML websocket
    """
    # Start ML websocket bridge in background
    threading.Thread(target=lambda: _run_ml_loop(ml_host), name="slate-ml-ws", daemon=True).start()

    # Start Flask-SocketIO web server in background
    threading.Thread(
        target=lambda: socketio.run(app, host=host, port=port),
        name="slate-web",
        daemon=True,
    ).start()
