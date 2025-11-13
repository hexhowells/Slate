from dataclasses import dataclass


@dataclass
class EnvInfo:
    lives: int
    episode_frame_number: int
    frame_number: int


@dataclass
class FramePayload:
    frame: str
    reward: float
    done: bool
    info: EnvInfo
    q_values: list[float]
    action: str
    high_score: int
    checkpoint: str
