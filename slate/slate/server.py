from __future__ import annotations

from flask import Flask, send_from_directory, request
from flask_socketio import SocketIO
import asyncio
import threading
import json
import websockets
from pathlib import Path
import logging
import time

from slate.run_history import RunHistory
from slate.session import Session

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

MAX_HISTORY_SIZE = 5
run_history = RunHistory(MAX_HISTORY_SIZE)


sessions: dict[str, Session] = {}


def get_session(sid: str) -> Session:
    """
    Get session given an SID

    If a sesssion with the given SID is not available, a new Session object is created

    Args:
        sid: the session id to fetch
    
    Return:
        a Session object of the current session, or a new Session object if the session is new
    """
    sess = sessions.get(sid)
    if not sess:
        sess = Session(sid)
        sessions[sid] = sess
    
    return sess


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
                    if run_history.recording:
                        run_history.update_recording(data['payload'])
                    else:
                        run_history.new_recording(data['payload'])
                    socketio.emit(msg_type, data)
                case "checkpoints_update":
                    socketio.emit(msg_type, data)
                case "run_completed":
                    run_history.stop_recording()
                    socketio.emit("run_history_update", {"run_history": run_history.get_history_metadata()})
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


def get_request_id():
    return request.sid  # type: ignore


def stream_run(session):
    while True:
        with session.lock:
            if not session.streaming:
                break

            if session.paused:
                pass
            elif session.awaiting_ack:
                pass
            else:
                run_id = session.asset.get("id")
                cursor = session.cursor

                total_steps = session.asset.get("total_steps", 0)
                if cursor >= total_steps:
                    socketio.emit("playback:eos", {"cursor": cursor})
                    session.streaming = False
                    break

                session.last_sent_cursor = cursor
                session.cursor += 1
                session.awaiting_ack = True
        
        with session.lock:
            if not session.streaming:
                break

            paused = session.paused
            awaiting = session.awaiting_ack
            last_cursor = session.last_sent_cursor
            run_id = session.asset.get("id")
        
        if (not paused) and (awaiting) and (last_cursor is not None):
            frame_data = run_history.fetch_recording_frame(run_id, last_cursor)
            if frame_data:
                socketio.emit("playback:frame", {"frame_data": frame_data, "cursor": last_cursor})
            else:
                socketio.emit("playback:error", 
                            {
                                  "message": 
                                  f"No frame could be loaded for cursor position {last_cursor}"
                            })
        
        time.sleep(0.1)
    
    with session.lock:
        session.streaming = False
        session.awaiting_ack = False


def launch_stream(session: Session) -> None:
    """
    Start video stream if the stream has not already started

    Args:
        session: Session object containing the session information
    """
    with session.lock:
        if session.streaming:
            return
        session.streaming = True
    socketio.start_background_task(stream_run, session)


@socketio.on("step")
def on_step() -> None:
    print(get_request_id())
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


@socketio.on("playback:load")
def on_playback_load(data) -> None:
    """
    Load a playback video given an run ID

    Verifies the run ID is valid and returns the run metadata to the client

    Args:
        data: Dict containing a `run_id` key with the run identifier.
    """
    run_id = data.get("run_id", 0)
    sid = get_request_id()
    sess = get_session(sid)

    if run_history.check_id(run_id):
        on_pause()

        run_data = run_history.fetch_recording(run_id)

        run_info = {
            "id": run_data["id"],
            "timestamp": run_data["timestamp"],
            "duration": run_data["duration"],
            "total_steps": run_data["total_steps"],
            "total_reward": run_data["total_reward"],
            "checkpoint": run_data["checkpoint"]
        }

        with sess.lock:
            sess.asset = run_info
            sess.cursor = 0
            sess.paused = True
            sess.awaiting_ack = False
            sess.last_sent_cursor = None

        socketio.emit("playback:loaded", {"payload": run_info})
    else:
        socketio.emit("playback:error", {"message": f"Run ID {run_id} could not be found."})


@socketio.on("playback:seek")
def on_playback_seek(data) -> None:
    cursor = data.get("frame", None)
    if cursor is None:
        socketio.emit("playback:error", {"message": f"No frame index provided in message."})
        return
    
    sid = get_request_id()
    sess = get_session(sid)

    if not (0 <= cursor < sess.asset.get('total_steps', 0)):
        socketio.emit("playback:error", {"message": f"Frame index is out of range for the given video"})
        return
    
    with sess.lock:
        sess.cursor = cursor
        sess.awaiting_ack = False
        sess.last_sent_cursor = None
        resume_stream = not sess.paused
    
    if resume_stream:
        launch_stream(sess)
    
    socketio.emit("playback:seek:ok", {"cursor": sess.cursor})


@socketio.on("playback:pause")
def on_playback_pause(data) -> None:
    sid = get_request_id()
    sess = get_session(sid)
    with sess.lock:
        sess.paused = True


@socketio.on("playback:resume")
def on_playback_resume(data) -> None:
    sid = get_request_id()
    sess = get_session(sid)
    with sess.lock:
        sess.paused = False
    
    launch_stream(sess)


@socketio.on("playback:ack")
def on_playback_ack(data) -> None:
    sid = get_request_id()
    sess = get_session(sid)
    
    with sess.lock:
        sess.awaiting_ack = False


@socketio.on("get_run_history")
def on_get_run_history() -> None:
    """Send current run history to the requesting client."""
    socketio.emit("run_history_update", {"run_history": run_history.get_history_metadata()})


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
