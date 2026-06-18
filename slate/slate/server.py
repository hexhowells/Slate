from __future__ import annotations

import asyncio
import json
import logging
import uuid
from io import BytesIO
from pathlib import Path
from typing import TypeAlias

import websockets
from websockets.asyncio.server import serve, ServerConnection
from flask import Flask, send_from_directory, send_file
from flask import Response
import threading

from slate.run_history import RunHistory
from slate.session import Session
from slate.video.codec import encode_video_to_s4
from slate.router import Router

logging.getLogger('werkzeug').disabled = True

main_loop: asyncio.AbstractEventLoop | None = None

_PKG_DIR = Path(__file__).parent
_REPO_STATIC = _PKG_DIR.parent.parent / "server" / "static"
_PKG_STATIC = _PKG_DIR / "static"
STATIC_DIR = _REPO_STATIC if _REPO_STATIC.exists() else _PKG_STATIC

ML_WS_PORT = 8765
WEB_WS_PORT = 8766

sid: TypeAlias = str

ml_clients: set[ServerConnection] = set()
web_clients: dict[ServerConnection, sid] = {}
sessions: dict[sid, Session] = {}

run_history = RunHistory(max_history_size=5)

router = Router()

app = Flask(
    __name__,
    static_folder=str(STATIC_DIR),
    static_url_path="/static",
)

@app.route("/")
def index() -> Response:
    """
    Serve the dashboard and request the latest checkpoints from the ML side.

    Returns:
        A Flask response that serves `index.html` from the resolved static dir.
    """
    if (main_loop is not None) and main_loop.is_running():
        # Safely bridge from the Werkzeug thread to the main asyncio loop
        asyncio.run_coroutine_threadsafe(
            _send_to_ml({"type": "send_checkpoints"}), 
            main_loop
        )
    
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.route("/playback/<run_id>")
def download_playback(run_id: int) -> Response:
    """
    Flask route to download a playback object

    Args:
        run_id: the id of the playback object to fetch
    
    Returns:
        a flask response containing the playback file bytes
    """
    run_data = run_history.fetch_recording(int(run_id))
    video_bytes = encode_video_to_s4(run_data)
    
    return send_file(
        BytesIO(video_bytes),
        as_attachment=True,
        download_name=f"slate_run_{run_id}.s4",
    )


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


async def _send_to_ml(payload: dict) -> None:
    """
    Send a JSON payload to all connected ML runtime WebSocket clients.

    Args:
        payload: Dictionary that will be JSON-encoded and sent.
    """
    if not ml_clients:
        return
    txt = json.dumps(payload)
    websockets.broadcast(ml_clients, txt)


async def _broadcast_to_web(payload: dict) -> None:
    """
    Broadcast data to the UI client via a websocket

    Args:
        payload: the payload object to broadcast
    """
    if not web_clients:
        return
    txt = json.dumps(payload)
    websockets.broadcast(web_clients.keys(), txt)


async def stream_run(session: Session, ws: ServerConnection) -> None:
    """
    Start streaming a run to the client

    Handles interrupts from the client such as pausing, ack, etc

    Args:
        session: Session object storing information about the current session
    """
    while True:
        with session.lock:
            if not session.streaming:
                break

            if not session.paused and not session.awaiting_ack:
                cursor = session.cursor
                total_steps: int = session.asset.get("total_steps", 0)
                
                if cursor >= total_steps:
                    await ws.send(json.dumps({"type": "playback:eos", "cursor": cursor}))
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
            run_id = session.asset.get("id", 0)
        
        if not paused and awaiting and last_cursor is not None:
            frame_data = run_history.fetch_recording_frame(run_id, last_cursor)
            if frame_data:
                await ws.send(json.dumps({"type": "playback:frame", "frame_data": frame_data, "cursor": last_cursor}))
            else:
                await ws.send(json.dumps({
                    "type": "playback:error", 
                    "message": f"No frame could be loaded for cursor {last_cursor}"
                }))
        
        await asyncio.sleep(0.1)
    
    with session.lock:
        session.streaming = False
        session.awaiting_ack = False


async def launch_stream(session: Session, ws: ServerConnection) -> None:
    """
    Start video stream if the stream has not already started

    Args:
        session: Session object containing the session information
    """
    with session.lock:
        if session.streaming:
            return
        session.streaming = True
    asyncio.create_task(stream_run(session, ws))


async def web_handler(ws: ServerConnection) -> None:
    """
    Route handler for websocket messages

    Args:
        ws: the websocket connection
    """
    sid = str(uuid.uuid4())
    web_clients[ws] = sid
    
    try:
        async for msg in ws:
            data = json.loads(msg)
            msg_type = data.get("type")
            await router.dispatch(msg_type, sid, ws, data)
    finally:
        del web_clients[ws]


@router.on("step")
async def on_step(sid: str, ws: ServerConnection, _data=None) -> None:
    """Request a single environment step from the ML runtime."""
    await _send_to_ml({"type": "step"})


@router.on("run")
async def on_run(sid: str, ws: ServerConnection, _data=None) -> None:
    """Start continuous stepping on the ML runtime."""
    await _send_to_ml({"type": "run"})


@router.on("pause")
async def on_pause(sid: str, ws: ServerConnection, _data=None) -> None:
    """Pause continuous stepping on the ML runtime."""
    await _send_to_ml({"type": "pause"})


@router.on("reset")
async def on_reset(sid: str, ws: ServerConnection, _data=None) -> None:
    """Reset the environment on the ML runtime."""
    await _send_to_ml({"type": "reset"})


@router.on("select_checkpoint")
async def on_select_checkpoint(sid: str, ws: ServerConnection, data) -> None:
    """Ask the ML runtime to load a specific checkpoint.

    Args:
        data: Dict containing a `checkpoint` key with the identifier/path.
    """
    await _send_to_ml({"type": "select_checkpoint", "checkpoint": data.get("checkpoint", "")})


@router.on("send_checkpoints")
async def on_send_checkpoints(sid: str, ws: ServerConnection, _data=None) -> None:
    """Request the list of available checkpoints from the ML runtime."""
    await _send_to_ml({"type": "send_checkpoints"})


@router.on("send_run_history")
async def on_send_run_history(sid: str, ws: ServerConnection, _data=None) -> None:
    """Request the run history from the ML runtime."""
    await _send_to_ml({"type": "send_run_history"})


@router.on("playback:save")
async def on_playback_save(sid: str, ws: ServerConnection, data) -> None:
    """
    Generate a download id to download a playback video from

    Args:
        data: data sent from the client
    """
    sess = get_session(sid)

    run_id = sess.asset["id"]
    await ws.send(json.dumps({
        "type": "playback:save:ready", 
        "run_id": run_id, 
        "download_url": f"/playback/{run_id}"
    }))


@router.on("playback:load")
async def on_playback_load(sid: str, ws: ServerConnection, data) -> None:
    """
    Load a playback video given an run ID

    Verifies the run ID is valid and returns the run metadata to the client

    Args:
        data: Dict containing a `run_id` key with the run identifier.
    """
    sess = get_session(sid)

    run_id = data.get("run_id", 0)
    if run_history.check_id(run_id):
        await _send_to_ml({"type": "pause"})
        run_data = run_history.fetch_recording(run_id)
        run_info = {
            "id": run_data["id"],
            "timestamp": run_data["timestamp"],
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
        await ws.send(json.dumps({"type": "playback:loaded", "payload": run_info}))
    else:
        await ws.send(json.dumps({"type": "playback:error", "message": f"Run ID {run_id} not found."}))


@router.on("playback:seek")
async def on_playback_seek(sid: str, ws: ServerConnection, data) -> None:
    """
    Seek to a given frame in the playback video

    Args:
        data: data sent from the client - including the frame to seek to
    """
    sess = get_session(sid)

    cursor = data.get("frame")
    if cursor is None:
        await ws.send(json.dumps({"type": "playback:error", "message": "No frame index provided."}))
        return
    
    if not (0 <= cursor < sess.asset.get('total_steps', 0)):
        await ws.send(json.dumps({"type": "playback:error", "message": "Frame index out of range."}))
        return
    
    with sess.lock:
        sess.cursor = cursor
        sess.awaiting_ack = False
        sess.last_sent_cursor = None
        resume_stream = not sess.paused
    
    if resume_stream:
        await launch_stream(sess, ws)
    
    await ws.send(json.dumps({"type": "playback:seek:ok", "cursor": sess.cursor}))


@router.on("playback:pause")
async def on_playback_pause(sid: str, ws: ServerConnection, data) -> None:
    """
    Pause the current VOD stream

    Args:
        data: data sent from the client
    """
    sess = get_session(sid)
    with sess.lock:
        sess.paused = True


@router.on("playback:resume")
async def on_playback_resume(sid: str, ws: ServerConnection, data) -> None:
    """
    Resume the current VOD stream

    Args:
        data: data sent from the client
    """
    sess = get_session(sid)
    with sess.lock:
        sess.paused = False
    
    await launch_stream(sess, ws)


@router.on("playback:ack")
async def on_playback_ack(sid: str, ws: ServerConnection, data) -> None:
    """
    Handler for client ACK messages

    Acknowleges that the previous frame sent in playback mode was received
    and processed by the client

    Args:
        data: data sent from the client
    """
    sess = get_session(sid)
    
    with sess.lock:
        sess.awaiting_ack = False


@router.on("get_run_history")
async def on_get_run_history(sid: str, ws: ServerConnection, _data=None) -> None:
    """Send current run history to the requesting client."""
    await ws.send(json.dumps({
        "type": "run_history_update", 
        "run_history": run_history.get_history_metadata()
    }))


async def ml_handler(ws: ServerConnection) -> None:
    """
    Handle a single ML runtime WebSocket connection.

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
            msg_type = data.get("type")
            
            match msg_type:
                case "frame_update":
                    if run_history.recording:
                        run_history.update_recording(data['payload'])
                    else:
                        run_history.new_recording(data['payload'])
                    await _broadcast_to_web(data)
                    
                case "checkpoints_update":
                    await _broadcast_to_web(data)
                    
                case "run_completed":
                    run_history.stop_recording()
                    await _broadcast_to_web({
                        "type": "run_history_update", 
                        "run_history": run_history.get_history_metadata()
                    })
                    
                case _:
                    logging.warning(f"ML Websocket received unknown message type: {msg_type}")
    finally:
        ml_clients.discard(ws)


async def run_servers(ml_host: str, web_host: str) -> None:
    """
    Run both WebSocket servers concurrently
    
    Args:
        ml_host: HTTP host for ML websocket
        web_host: HTTP host for the dashboard
    """
    global main_loop
    main_loop = asyncio.get_running_loop()  # Capture the background thread's loop for Flask
    
    async with serve(ml_handler, ml_host, ML_WS_PORT):
        async with serve(web_handler, web_host, WEB_WS_PORT):
            print(f"[Slate] ML ws listening on ws://{ml_host}:{ML_WS_PORT}")
            print(f"[Slate] Web ws listening on ws://{web_host}:{WEB_WS_PORT}")
            await asyncio.Future()  # run forever


def _run_async_servers(ml_host: str, web_host: str) -> None:
    """
    Bootstraps a fresh asyncio loop for the background thread
    
    Args:
        ml_host: HTTP host for ML websocket
        web_host: HTTP host for the dashboard
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_servers(ml_host, web_host))


def start_local_server(
        host: str = "0.0.0.0",
        port: int = 8000,
        ml_host: str = "0.0.0.0",
    ) -> None:
    """
    Start the local Slate dashboard server and the ML WebSocket bridge in background threads.

    Args:
        host: HTTP host for the dashboard
        port: HTTP port for the dashboard
        ml_host: HTTP host for ML websocket
    """
    # start HTTP server 
    threading.Thread(
        target=lambda: app.run(host=host, port=port, use_reloader=False),
        name="slate-http",
        daemon=True,
    ).start()

    # start websocket server
    threading.Thread(
        target=lambda: _run_async_servers(ml_host, host),
        name="slate-ws",
        daemon=True,
    ).start()
