from abc import ABC, abstractmethod
import numpy as np


class Agent(ABC):
	@abstractmethod
	def get_action(self, frame: np.ndarray) -> int:
		"""Get next action given current frame"""
		pass
