import gym
from zydash import ZyDash


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, env):
        return self.action_space.sample()


env = gym.make("Breakout-v4", render_mode="rgb_array")
agent = RandomAgent(env.action_space)

server = ZyDash(env, agent)
server.run()
