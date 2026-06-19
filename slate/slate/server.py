from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import TypeAlias

import threading
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles

from slate.run_history import RunHistory
from slate.session import Session
from slate.video.codec import encode_video_to_s4
from slate.router import Router

_PKG_DIR = Path(__file__).parent
_REPO_STATIC = _PKG_DIR.parent.parent / "server" / "static"
_PKG_STATIC = _PKG_DIR / "static"
STATIC_DIR = _REPO_STATIC if _REPO_STATIC.exists() else _PKG_STATIC

sid: TypeAlias = str

ml_clients: set[WebSocket] = set()
web_clients: dict[WebSocket, sid] = {}
sessions: dict[sid, Session] = {}

run_history = RunHistory(max_history_size=5)

router = Router()

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> FileResponse:
    """
    Serve the dashboard and request the latest checkpoints from the ML side.

    Returns:
        A FastAPI response that serves `index.html` from the resolved static dir.
    """
    await _send_to_ml({"type": "send_checkpoints"})
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/playback/{run_id}")
async def download_playback(run_id: int) -> Response:
    """
    HTTP route to download a playback object

    Args:
        run_id: the id of the playback object to fetch
    
    Returns:
        a response containing the playback file bytes
    """
    run_data = run_history.fetch_recording(int(run_id))
    video_bytes = encode_video_to_s4(run_data)
    
    return Response(
        content=video_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=slate_run_{run_id}.s4"}
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
    
    clients_list = list(ml_clients)
    
    results = await asyncio.gather(
        *(ws.send_text(txt) for ws in clients_list),
        return_exceptions=True
    )
    
    disconnected = {
        ws for ws, result in zip(clients_list, results) 
        if isinstance(result, Exception)
    }
            
    if disconnected:
        ml_clients.difference_update(disconnected)


async def _broadcast_to_web(payload: dict) -> None:
    """
    Broadcast data to the UI client via a websocket

    Args:
        payload: the payload object to broadcast
    """
    if not web_clients:
        return
    txt = json.dumps(payload)
    
    results = await asyncio.gather(
        *(ws.send_text(txt) for ws in web_clients.keys()),
        return_exceptions=True
    )
    
    for ws, result in zip(web_clients.keys(), results):
        if isinstance(result, Exception):
            web_clients.pop(ws, None)


async def stream_run(session: Session, ws: WebSocket) -> None:
    """
    Start streaming a run to the client

    Handles interrupts from the client such as pausing, ack, etc

    Args:
        session: Session object storing information about the current session
        ws: the websocket connection
    """
    while True:
        with session.lock:
            if not session.streaming:
                break

            if not session.paused and not session.awaiting_ack:
                cursor = session.cursor
                total_steps: int = session.asset.get("total_steps", 0)
                
                if cursor >= total_steps:
                    await ws.send_text(json.dumps({"type": "playback:eos", "cursor": cursor}))
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
                await ws.send_text(json.dumps({"type": "playback:frame", "frame_data": frame_data, "cursor": last_cursor}))
            else:
                await ws.send_text(json.dumps({
                    "type": "playback:error", 
                    "message": f"No frame could be loaded for cursor {last_cursor}"
                }))
        
        await asyncio.sleep(0.1)
    
    with session.lock:
        session.streaming = False
        session.awaiting_ack = False


async def launch_stream(session: Session, ws: WebSocket) -> None:
    """
    Start video stream if the stream has not already started

    Args:
        session: Session object containing the session information
        ws: the websocket connection
    """
    with session.lock:
        if session.streaming:
            return
        session.streaming = True
    asyncio.create_task(stream_run(session, ws))


@app.websocket("/ws/ui")
async def web_handler(ws: WebSocket) -> None:
    """
    Route handler for websocket messages from the UI

    Args:
        ws: the websocket connection
    """
    await ws.accept()
    client_sid = str(uuid.uuid4())
    web_clients[ws] = client_sid
    
    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            msg_type = data.get("type")
            await router.dispatch(msg_type, client_sid, ws, data)
    except WebSocketDisconnect:
        pass
    finally:
        web_clients.pop(ws, None)


@router.on("step")
async def on_step(
    sid: str, 
    ws: WebSocket, 
    _data: dict|None=None
) -> None:
    """
    Request a single environment step from the ML runtime.
    
    Args:
        sid: the SID of the request
        ws: the websocket connection
        _data: data from the websocket message
    """
    await _send_to_ml({"type": "step"})


@router.on("run")
async def on_run(
    sid: str, 
    ws: WebSocket, 
    _data: dict|None=None
) -> None:
    """
    Start continuous stepping on the ML runtime.
    
    Args:
        sid: the SID of the request
        ws: the websocket connection
        _data: data from the websocket message
    """
    await _send_to_ml({"type": "run"})


@router.on("pause")
async def on_pause(
    sid: str, 
    ws: WebSocket, 
    _data: dict|None=None
) -> None:
    """
    Pause continuous stepping on the ML runtime.
    
    Args:
        sid: the SID of the request
        ws: the websocket connection
        _data: data from the websocket message
    """
    await _send_to_ml({"type": "pause"})


@router.on("reset")
async def on_reset(
    sid: str, 
    ws: WebSocket, 
    _data: dict|None=None
) -> None:
    """
    Reset the environment on the ML runtime.
    
    Args:
        sid: the SID of the request
        ws: the websocket connection
        _data: data from the websocket message
    """
    await _send_to_ml({"type": "reset"})


@router.on("select_checkpoint")
async def on_select_checkpoint(
    sid: str, 
    ws: WebSocket, 
    data: dict
) -> None:
    """Ask the ML runtime to load a specific checkpoint.

    Args:
        sid: the SID of the request
        ws: the websocket connection
        data: Dict containing a `checkpoint` key with the identifier/path.
    """
    await _send_to_ml({"type": "select_checkpoint", "checkpoint": data.get("checkpoint", "")})


@router.on("send_checkpoints")
async def on_send_checkpoints(
    sid: str, 
    ws: WebSocket, 
    _data: dict|None=None
) -> None:
    """
    Request the list of available checkpoints from the ML runtime.
    
    Args:
        sid: the SID of the request
        ws: the websocket connection
        _data: data from the websocket message
    """
    await _send_to_ml({"type": "send_checkpoints"})


@router.on("send_run_history")
async def on_send_run_history(
    sid: str, 
    ws: WebSocket, 
    _data: dict|None=None
) -> None:
    """
    Request the run history from the ML runtime.
    
    Args:
        sid: the SID of the request
        ws: the websocket connection
        _data: data from the websocket message
    """
    await _send_to_ml({"type": "send_run_history"})


@router.on("playback:save")
async def on_playback_save(
    sid: str, 
    ws: WebSocket, 
    data: dict
) -> None:
    """
    Generate a download id to download a playback video from

    Args:
        sid: the SID of the request
        ws: the websocket connection
        data: data sent from the client
    """
    sess = get_session(sid)

    run_id = sess.asset["id"]
    await ws.send_text(json.dumps({
        "type": "playback:save:ready", 
        "run_id": run_id, 
        "download_url": f"/playback/{run_id}"
    }))


@router.on("playback:load")
async def on_playback_load(
    sid: str, 
    ws: WebSocket, 
    data: dict
) -> None:
    """
    Load a playback video given an run ID

    Verifies the run ID is valid and returns the run metadata to the client

    Args:
        sid: the SID of the request
        ws: the websocket connection
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
        await ws.send_text(json.dumps({"type": "playback:loaded", "payload": run_info}))
    else:
        await ws.send_text(json.dumps({"type": "playback:error", "message": f"Run ID {run_id} not found."}))


@router.on("playback:seek")
async def on_playback_seek(
    sid: str, 
    ws: WebSocket, 
    data: dict
) -> None:
    """
    Seek to a given frame in the playback video

    Args:
        sid: the SID of the request
        ws: the websocket connection
        data: data sent from the client - including the frame to seek to
    """
    sess = get_session(sid)

    cursor = data.get("frame")
    if cursor is None:
        await ws.send_text(json.dumps({"type": "playback:error", "message": "No frame index provided."}))
        return
    
    if not (0 <= cursor < sess.asset.get('total_steps', 0)):
        await ws.send_text(json.dumps({"type": "playback:error", "message": "Frame index out of range."}))
        return
    
    with sess.lock:
        sess.cursor = cursor
        sess.awaiting_ack = False
        sess.last_sent_cursor = None
        resume_stream = not sess.paused
    
    if resume_stream:
        await launch_stream(sess, ws)
    
    await ws.send_text(json.dumps({"type": "playback:seek:ok", "cursor": sess.cursor}))


@router.on("playback:pause")
async def on_playback_pause(
    sid: str, 
    ws: WebSocket, 
    data: dict
) -> None:
    """
    Pause the current VOD stream

    Args:
        sid: the SID of the request
        ws: the websocket connection
        data: data sent from the client
    """
    sess = get_session(sid)
    with sess.lock:
        sess.paused = True


@router.on("playback:resume")
async def on_playback_resume(
    sid: str, 
    ws: WebSocket, 
    data: dict
) -> None:
    """
    Resume the current VOD stream

    Args:
        sid: the SID of the request
        ws: the websocket connection
        data: data sent from the client
    """
    sess = get_session(sid)
    with sess.lock:
        sess.paused = False
    
    await launch_stream(sess, ws)


@router.on("playback:ack")
async def on_playback_ack(
    sid: str, 
    ws: WebSocket, 
    data: dict
) -> None:
    """
    Handler for client ACK messages

    Acknowleges that the previous frame sent in playback mode was received
    and processed by the client

    Args:
        sid: the SID of the request
        ws: the websocket connection
        data: data sent from the client
    """
    sess = get_session(sid)
    
    with sess.lock:
        sess.awaiting_ack = False


@router.on("get_run_history")
async def on_get_run_history(
    sid: str, 
    ws: WebSocket, 
    _data: dict|None=None
) -> None:
    """
    Send current run history to the requesting client.
    
    Args:
        sid: the SID of the request
        ws: the websocket connection
        _data: data from the websocket message
    """
    await ws.send_text(json.dumps({
        "type": "run_history_update", 
        "run_history": run_history.get_history_metadata()
    }))


@app.websocket("/ws/ml")
async def ml_handler(ws: WebSocket) -> None:
    """
    Handle a single ML runtime WebSocket connection.

    Adds the connection to the client set and forwards any JSON messages from
    the ML runtime to the browser via the unified broadcast, using the `type` field 
    as the event name. Removes the connection on disconnect.

    Args:
        ws: The connected WebSocket instance.
    """
    await ws.accept()
    ml_clients.add(ws)
    try:
        while True:
            msg = await ws.receive_text()
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
    except WebSocketDisconnect:
        pass
    finally:
        ml_clients.discard(ws)


def start_local_server(
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
    """
    Start the unified FastAPI server handling both HTTP and WebSockets in a background thread.

    Args:
        host: HTTP host for the application
        port: HTTP port for the application
    """
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    
    # Run the ASGI server in a background thread so it doesn't block 
    # the ML client from launching its own asyncio loop on the main OS thread.
    threading.Thread(
        target=server.run,
        name="slate-server",
        daemon=True,
    ).start()