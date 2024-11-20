import gym
from zydash import ZyDash, Agent


class RandomAgent(Agent):
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def get_action(self, env):
        return self.action_space.sample()


env = gym.make("Breakout-v4", render_mode="rgb_array")
agent = RandomAgent(env.action_space)

server = ZyDash(env, agent)
server.run()
