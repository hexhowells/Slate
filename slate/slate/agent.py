from abc import ABC, abstractmethod
import numpy as np
import torch


class Agent(ABC):
	@abstractmethod
	def get_action(self, frame: np.ndarray|torch.Tensor) -> int:
		"""
		Get next action given current frame

		Args:
			frame: the latest environment frame

		Returns:
			the numerical action code
		"""
		pass


	@abstractmethod
	def load_checkpoint(self, checkpoint: str) -> None:
		"""
		Load checkpoint from a given checkpoint file

		Args:
			checkpoint: the filepath to load the checkpoint from
		"""
		pass
	

	@abstractmethod
	def get_q_values(self) -> list|torch.Tensor:
		"""
		Gets the models current q-values from the previously processed frame
		"""
		pass
