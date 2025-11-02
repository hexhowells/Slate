import asyncio
import json
import base64
import cv2
import websockets
import threading
import os
import numpy as np
import torchvision.transforms as T
from datetime import datetime

from .agent import Agent
from .utils import FrameBuffer


default_transform = transform = T.Compose([
	T.ToPILImage(),
	T.ToTensor()
])


class SlateClient:
    """
    Handles real-time interaction between a reinforcement learning environment, an agent,
    and a WebSocket server for visualizing and controlling the training loop.

    Args:
        env: A Gym-like environment that supports reset, step, and render methods
        agent: Agent with get_action(env) and optionally get_q_values(obs)
        endpoint: the server endpoint that the client with connect to
        run_local: whether to run the slate server locally or connect to a cloud server
        frame_rate: Delay (in seconds) between steps during continuous run
        buffer_len: Length of the frame buffer (detault = 1)
        transform: Optional transform function to transform the input frames
        checkpoints_dir: the directory which the agent checkpoints are stored

    Attributes:
        current_frame: Base64-encoded JPEG of the latest environment render
        q_values: List of Q-values returned by the agent
        reward: Latest reward obtained
        done: Boolean indicating whether the last episode ended
        info: Dictionary with environment-specific metadata
        high_score: Highest reward observed so far
        checkpoint: Filename of the saved model checkpoint
    """
    def __init__(
            self, 
            env, 
            agent: Agent,
            endpoint: str|None = None,
            run_local: bool = False,
            frame_rate: float=0.1,
            buffer_len: int=1,
            transform=default_transform,
            checkpoints_dir: str = ""
        ) -> None:
        self.env = env
        self.agent = agent
        self.frame_rate = frame_rate
        self.running = False
        self.step_mode = False
        self.state_lock = threading.Lock()
        self.loop_task = None
        self.ws_endpoint = "ws://localhost:8765"
        self.ui_endpoint = endpoint or "127.0.0.1"

        obs, _ = self.env.reset()
        self.current_frame = None
        self.q_values = []
        self.action_str = "None"
        self.action_meanings = env.unwrapped.get_action_meanings()
        self.reward = 0
        self.done = False
        self.info = {}
        self.high_score = 0
        self.run_local = run_local

        self.ckpt_dir = checkpoints_dir
        self.checkpoints: list[str] = []
        self._rescan_checkpoints()
        self.checkpoint = (self.checkpoints[-1] if self.checkpoints else None) or ""
        
        # Recording functionality
        self.current_recording: list[dict] = []
        self.is_recording = False
        self.run_start_time = None
        self.start_recording()

        # frame buffer
        self.frame_buffer = FrameBuffer(obs, buffer_len, transform)
        

    def _rescan_checkpoints(self) -> None:
        """
        Lists all files in the specified checkpoints folder with the file extension .pth

        Updates the checkpoints class variable with the new list
        """
        if not self.ckpt_dir:
            return
        self.checkpoints = sorted(
            [f for f in os.listdir(self.ckpt_dir) if f.endswith(".pth")]
        )


    def start_recording(self) -> None:
        """
        Start recording a new run session.
        """
        self.is_recording = True
        self.current_recording = []
        self.run_start_time = datetime.now()


    async def stop_recording(self) -> None:
        """
        Stop recording and send the current recording to the server.
        """
        if not self.is_recording or not self.current_recording:
            return
        
        assert self.run_start_time is not None, "run_start_time timestamp variable is None"
            
        self.is_recording = False
        
        # Create run summary
        run_data = {
            "id": len(self.current_recording),  # Will be updated by server
            "timestamp": self.run_start_time.isoformat(),
            "duration": (datetime.now() - self.run_start_time).total_seconds(),
            "total_steps": len(self.current_recording),
            "total_reward": sum(step["metadata"].get("reward", 0) for step in self.current_recording),
            "checkpoint": self.checkpoint,
            "frames": [step["frame"] for step in self.current_recording],
            "metadata": [step["metadata"] for step in self.current_recording]
        }
        
        # Send to server via websocket
        await self.websocket.send(json.dumps({
            "type": "run_completed",
            "payload": run_data
        }))
        
        # Clear current recording
        self.current_recording = []
        self.run_start_time = None


    def record_step(self, frame: str, reward: float, done: bool, info: dict, q_values: list) -> None:
        """
        Record a single step in the current recording.
        
        Args:
            frame: Base64-encoded frame
            reward: Step reward
            done: Whether episode is done
            info: Environment info
            q_values: Q-values from agent
        """
        if not self.is_recording:
            return
            
        step_data = {
            "frame": frame,
            "metadata": {
                "reward": reward,
                "done": done,
                "info": info,
                "q_values": q_values,
                "action": self.action_str,
                "timestamp": datetime.now().isoformat()
            }
        }
        self.current_recording.append(step_data)


    def encode_frame(self, frame: np.ndarray) -> str:
        """
        Encode an RGB image frame into a base64-encoded JPEG string.

        Args:
            frame: RGB image from the environment

        Returns:
            str: Base64 string of the JPEG-encoded frame
        """
        _, img = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return base64.b64encode(img).decode('utf-8')


    async def run_step(self) -> None:
        """
        Execute a single step in the environment using the agent or random policy,
        and update internal state values.

        Raises:
            Any exception from the environment or rendering is propagated
        """
        frame = self.env.render()
        action = self.agent.get_action(self.frame_buffer.state())
        obs, reward, done, truncated, info = self.env.step(action)
        self.frame_buffer.append(obs)

        self.action_str = self.action_meanings[action]

        with self.state_lock:
            self.current_frame = self.encode_frame(frame)
            self.reward = reward
            self.done = done
            
            self.info = info
            self.q_values = self.agent.get_q_values()
            self.high_score = max(self.high_score, reward)
            
            # Record the step if recording is active
            self.record_step(self.current_frame, reward, done, info, self.q_values)

        if done:
            await self.stop_recording()
            self.env.reset()
            self.running = False
            self.start_recording()
            

    async def send_state(self) -> None:
        """
        Send the current environment state, including encoded frame and metadata,
        over the active WebSocket connection.

        Raises:
            websockets.exceptions.ConnectionClosed: If the connection is closed
        """
        with self.state_lock:
            await self.websocket.send(json.dumps({
                "type": "frame_update",
                "payload": {
                    "frame": self.current_frame,
                    "reward": self.reward,
                    "done": self.done,
                    "info": self.info,
                    "q_values": self.q_values,
                    "action": self.action_str,
                    "high_score": self.high_score,
                    "checkpoint": self.checkpoint
                }
            }))


    async def _send_checkpoints(self) -> None:
        """
        Send checkpoints to the server via a websocket
        """
        await self.websocket.send(
            json.dumps(
                {
                    "type": "checkpoints_update",
                    "payload": {"checkpoints": self.checkpoints},
                }
            )
        )


    async def watch_checkpoints(self) -> None:
        """
        Continuously watch the checkpoints folder for additional checkpoints

        If additional checkpoints are found, then send the new checkpoint values
        to the server via the websocket
        """
        if not self.ckpt_dir:
            return

        all_checkpoints = set(self.checkpoints)

        while True:
            await asyncio.sleep(1.0)
            self._rescan_checkpoints()
            new_checkpoints = set(self.checkpoints)

            if new_checkpoints != all_checkpoints:
                all_checkpoints = new_checkpoints
                await self._send_checkpoints()


    async def run_loop(self) -> None:
        """
        Continuously execute steps in the environment and send updated state
        to the WebSocket server as long as `self.running` is True.
        """
        while self.running:
            await self.run_step()
            await self.send_state()
            await asyncio.sleep(self.frame_rate)


    async def ws_handler(self, websocket) -> None:
        """
        Handle incoming WebSocket messages and perform actions like step, run, pause, and reset.

        Args:
            websocket: Connected WebSocket client

        Raises:
            websockets.exceptions.ConnectionClosed: If the WebSocket connection is terminated
        """
        self.websocket = websocket
        print("[SlateRunner] connected to Slate server")
        await self.send_state()
        await self._send_checkpoints()

        # start watcher
        if self.ckpt_dir:
            asyncio.create_task(self.watch_checkpoints())

        try:
            async for msg in websocket:
                data = json.loads(msg)
                command = data.get("type")
                #print(f'Client received command: {command}')

                match command:
                    case "step":
                        await self.run_step()
                        await self.send_state()
                    case "run":
                        self.running = True
                        if not self.loop_task or self.loop_task.done():
                            self.loop_task = asyncio.create_task(self.run_loop())
                    case "pause":
                        self.running = False
                    case "reset":
                        await self.stop_recording()
                        self.start_recording()
                        obs, _ = self.env.reset()
                        self.frame_buffer.reset(obs)
                        self.running = False
                        await self.send_state()
                    case "select_checkpoint":
                        self.checkpoint = data.get("checkpoint", "")
                        self.agent.load_checkpoint(os.path.join(self.ckpt_dir, self.checkpoint))
                        await self.send_state()
                    case "send_checkpoints":
                        await self._send_checkpoints()
        except websockets.ConnectionClosed:
            print("[SlateRunner] connection lost")


    async def _dial_and_serve(self, url: str) -> None:
        """
        Attempt to connect to the WebSocket server and handle interaction.
        Will automatically retry connection on failure.

        Args:
            url: The WebSocket server URL (e.g., ws://localhost:8765)
        """
        for _ in range(10):
            try:
                print(f"[SlateRunner] dialing {url}")
                async with websockets.connect(url) as ws:
                    await self.ws_handler(ws)
            except (ConnectionRefusedError, websockets.WebSocketException) as e:
                print(f"   failed ({e}) – retry in 1 s")
                await asyncio.sleep(1)


    def start_client(self) -> None:
        """
        Start the client and block the main thread to handle interaction with the WebSocket server.
        """
        # If configured to run locally, start the embedded server and point endpoint to localhost
        if self.run_local:
            try:
                from .server import start_local_server
                # Start local dashboard on ui_endpoint:8000 and ML WS bridge  ws_endpoint:8765
                start_local_server(host=self.ui_endpoint, port=8000)
                print(f"\033[95m[Slate] Open dashboard at http://{self.ui_endpoint}:8000\033[0m")
            except Exception as e:
                print(f"[Slate] Failed to start local server: {e}")
        asyncio.run(self._dial_and_serve(self.ws_endpoint))
    