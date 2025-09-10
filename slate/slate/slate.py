import asyncio
import json
import base64
import cv2
import websockets
import threading
import time
import os


class SlateClient:
    """
    Handles real-time interaction between a reinforcement learning environment, an agent,
    and a WebSocket server for visualizing and controlling the training loop.

    Args:
        env: A Gym-like environment that supports reset, step, and render methods
        agent: Agent with get_action(env) and optionally get_q_values(obs)
        frame_rate: Delay (in seconds) between steps during continuous run

    Attributes:
        current_frame: Base64-encoded JPEG of the latest environment render
        q_values: List of Q-values returned by the agent
        reward: Latest reward obtained
        done: Boolean indicating whether the last episode ended
        info: Dictionary with environment-specific metadata
        high_score: Highest reward observed so far
        checkpoint: Filename of the saved model checkpoint
    """
    def __init__(self, env, agent, frame_rate=0.1, checkpoints_dir = None) -> None:
        self.env = env
        self.agent = agent
        self.frame_rate = frame_rate
        self.running = False
        self.step_mode = False
        self.state_lock = threading.Lock()
        self.loop_task = None

        self.env.reset()
        self.current_frame = None
        self.q_values = []
        self.reward = 0
        self.done = False
        self.info = {}
        self.high_score = 0

        self.ckpt_dir = checkpoints_dir
        self.checkpoints: list[str] = []
        self._rescan_checkpoints()
        self.checkpoint = (self.checkpoints[-1] if self.checkpoints else None) or ""


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


    def encode_frame(self, frame) -> None:
        """
        Encode an RGB image frame into a base64-encoded JPEG string.

        Args:
            frame (np.ndarray): RGB image from the environment

        Returns:
            str: Base64 string of the JPEG-encoded frame
        """
        _, img = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return base64.b64encode(img).decode('utf-8')


    def run_step(self) -> None:
        """
        Execute a single step in the environment using the agent or random policy,
        and update internal state values.

        Raises:
            Any exception from the environment or rendering is propagated
        """
        action = self.agent.get_action(self.env) if self.agent else self.env.action_space.sample()
        obs, reward, done, truncated, info = self.env.step(action)
        frame = self.env.render()

        with self.state_lock:
            self.current_frame = self.encode_frame(frame)
            self.reward = reward
            self.done = done
            self.info = info
            self.q_values = getattr(self.agent, "get_q_values", lambda x: [])(obs)
            self.high_score = max(self.high_score, reward)

        if done:
            self.env.reset()


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
            self.run_step()
            await self.send_state()
            await asyncio.sleep(self.frame_rate)


    async def ws_handler(self, websocket) -> None:
        """
        Handle incoming WebSocket messages and perform actions like step, run, pause, and reset.

        Args:
            websocket (WebSocketClientProtocol): Connected WebSocket client

        Raises:
            websockets.exceptions.ConnectionClosed: If the WebSocket connection is terminated
        """
        self.websocket = websocket
        print("[SlateRunner] connected to Slate server")
        await self.send_state()
        print(self.checkpoints)
        await self._send_checkpoints()

        # start watcher
        if self.ckpt_dir:
            asyncio.create_task(self.watch_checkpoints())

        try:
            async for msg in websocket:
                data = json.loads(msg)
                command = data.get("type")
                print(f'Client received command: {command}')

                if command == "step":
                    self.run_step()
                    await self.send_state()
                elif command == "run":
                    self.running = True
                    if not self.loop_task or self.loop_task.done():
                        self.loop_task = asyncio.create_task(self.run_loop())
                elif command == "pause":
                    self.running = False
                elif command == "reset":
                    self.env.reset()
                    await self.send_state()
                elif command == "select_checkpoint":
                    self.checkpoint = data.get("checkpoint", "")
                    await self.send_state()
                elif command == "send_checkpoints":
                    await self._send_checkpoints()
        except websockets.ConnectionClosed:
            print("[SlateRunner] connection lost")


    async def _dial_and_serve(self, url: str) -> None:
        """
        Attempt to connect to the WebSocket server and handle interaction.
        Will automatically retry connection on failure.

        Args:
            url (str): The WebSocket server URL (e.g., ws://localhost:8765)
        """
        for _ in range(10):
            try:
                print(f"[SlateRunner] dialing {url}")
                async with websockets.connect(url) as ws:
                    await self.ws_handler(ws)
            except (ConnectionRefusedError, websockets.WebSocketException) as e:
                print(f"   failed ({e}) – retry in 1 s")
                await asyncio.sleep(1)


    def start_client(self, url: str="ws://localhost:8765") -> None:
        """
        Start the client and block the main thread to handle interaction with the WebSocket server.

        Args:
            url (str): WebSocket server address
        """
        asyncio.run(self._dial_and_serve(url))