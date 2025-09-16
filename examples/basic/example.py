import argparse
import ale_py
import gymnasium as gym
import numpy as np
from slate import SlateClient
from slate import Agent


class RandomAgent(Agent):
    def __init__(self, env):
        self.env = env
    def get_action(self, frame):
        return self.env.action_space.sample()
    
    def load_checkpoint(self, checkpoint: str) -> None:
        return super().load_checkpoint(checkpoint)

    def get_q_values(self, obs):
        return np.random.rand(env.action_space.n).tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_dir",
        default="checkpoints",
        help="Folder that contains .pth model checkpoints",
    )
    parser.add_argument(
        "--server",
        default="ws://localhost:8765",
        help="Slate server websocket URL",
    )
    args = parser.parse_args()

    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    agent = RandomAgent(env)
    runner = SlateClient(env, agent, checkpoints_dir=args.ckpt_dir)
    runner.init(endpoint=args.server)
    runner.start_client()

