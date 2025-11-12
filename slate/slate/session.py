import threading


class Session:
    def __init__(self, sid: str):
        self.sid = sid
        self.cursor: int = 0
        self.paused: bool = True
        self.lock = threading.Lock()