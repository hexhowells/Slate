import threading
from typing import Any


class Session:
    def __init__(self, sid: str):
        self.sid: str = sid
        self.asset: dict[str, Any] = {}
        self.cursor: int = 0
        self.paused: bool = True
        self.awaiting_ack: bool = False
        self.streaming: bool = False
        self.last_sent_cursor: int | None = None
        self.lock = threading.Lock()