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
    env = gym.make("ALE/Breakout-v5", render_mode='rgb_array')
    agent = RandomAgent()
    runner = SlateClient(env, agent)
    runner.start_client(url="ws://localhost:8765")
