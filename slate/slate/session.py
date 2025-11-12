import threading


class Session:
    def __init__(self, sid: str):
        self.sid = sid
        self.cursor: dict[str, int] = {}
        self.paused: bool = True
        self.lock = threading.Lock()