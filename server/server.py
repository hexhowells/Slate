from flask import Flask, send_from_directory
from flask_socketio import SocketIO
import asyncio, threading, json, websockets


app = Flask(__name__, static_folder="static")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

ML_WS_PORT = 8765
ml_loop = asyncio.new_event_loop()
ml_clients = set()


@app.route("/")
def index():
    """
    Serve the main HTML interface from the static directory.

    Returns:
        Response: The index.html file for the web UI
    """
    _send_to_ml({"type": "send_checkpoints"})
    return send_from_directory("static", "index.html")


async def ml_handler(ws):
    """
    Handles communication from an ML client over a raw WebSocket connection.

    Args:
        ws (WebSocketServerProtocol): The connected ML client socket

    Emits:
        "frame_update": Message forwarded to browser clients
    """
    ml_clients.add(ws)
    print("[ML] connected")
    try:
        async for msg in ws:
            data = json.loads(msg)
            evt = data.get("type", "frame_update")
            socketio.emit(evt, data)
    finally:
        ml_clients.discard(ws)
        print("[ML] disconnected")


async def ml_server():
    """
    Starts the ML WebSocket server and listens for incoming ML connections.

    Raises:
        RuntimeError: If the server cannot bind to the port
    """
    async with websockets.serve(ml_handler, "0.0.0.0", ML_WS_PORT):
        print(f"[Slate] waiting for ML on ws://0.0.0.0:{ML_WS_PORT}")
        await asyncio.Future()


def start_ml_server():
    """
    Initializes the asyncio event loop for ML clients and starts the server in that loop.
    """
    asyncio.set_event_loop(ml_loop)
    ml_loop.run_until_complete(ml_server())


def _send_to_ml(payload: dict):
    """
    Sends a dictionary payload to all connected ML clients.

    Args:
        payload (dict): The message to send to the ML client(s)

    Logs:
        A warning if no ML clients are connected
    """
    if not ml_clients:
        print("\u26a0  No ML client connected – command ignored")
        return
    txt = json.dumps(payload)
    for ws in list(ml_clients):
        asyncio.run_coroutine_threadsafe(ws.send(txt), ml_loop)


@socketio.on("connect")
def on_browser_connect():
    """
    Event handler triggered when a browser connects via SocketIO.
    """
    print("[Browser] connected")


@socketio.on("step")
def on_step():
    """
    Forwards a 'step' command from the browser to the ML client.
    """
    _send_to_ml({"type": "step"})


@socketio.on("run")
def on_run():
    """
    Forwards a 'run' command from the browser to the ML client.
    """
    _send_to_ml({"type": "run"})


@socketio.on("pause")
def on_pause():
    """
    Forwards a 'pause' command from the browser to the ML client.
    """
    _send_to_ml({"type": "pause"})


@socketio.on("reset")
def on_reset():
    """
    Forwards a 'reset' command from the browser to the ML client.
    """
    _send_to_ml({"type": "reset"})


@socketio.on("select_checkpoint")
def on_select_checkpoint(data):
    _send_to_ml({"type": "select_checkpoint", "checkpoint": data.get("checkpoint", "")})


if __name__ == "__main__":
    """
    Starts the Flask-SocketIO server for browser communication on port 8000.
    """
    threading.Thread(target=start_ml_server, daemon=True).start()
    socketio.run(app, host="0.0.0.0", port=8000)
