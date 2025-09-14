import argparse
import ale_py
import gymnasium as gym
import numpy as np
from slate import SlateClient


class RandomAgent:
    def get_action(self, env):
        return env.action_space.sample()

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
    agent = RandomAgent()
    runner = SlateClient(env, agent, checkpoints_dir=args.ckpt_dir)
    runner.init(endpoint=args.server)
    runner.start_client()

