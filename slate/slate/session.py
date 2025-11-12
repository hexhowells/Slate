import threading


class Session:
    def __init__(self, sid: str):
        self.sid: str = sid
        self.asset: dict[str, str|int|list] = {}
        self.cursor: int = 0
        self.paused: bool = True
        self.lock = threading.Lock()