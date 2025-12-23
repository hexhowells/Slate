import torch
from collections import deque
import json
import numpy as np
import torch


class FrameBuffer:
	def __init__(self, frame, buffer_len, transform):
		self.transform = transform
		self.buffer_len = buffer_len

		self.reset(frame)


	def append(self, frame):
		frame_tensor = self.transform(frame).unsqueeze(0)
		self.frame_stack.append(frame_tensor)


	def stack_frames(self):
		return torch.cat(list(self.frame_stack), dim=1)


	def state(self):
		return self.stack_frames()
	
	
	def reset(self, frame):
		frame_tensor = self.transform(frame).unsqueeze(0)
		self.frame_stack = deque([frame_tensor] * self.buffer_len, maxlen=self.buffer_len)


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.detach().cpu().tolist()
        if isinstance(o, torch.dtype):
              return str(o)
        return super().default(o)
