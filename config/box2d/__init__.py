from core.config import BaseConfig
from core.env import GymEnv

ENVS = ['LunarLanderContinuous-v2', 'BipedalWalker-v3', 'BipedalWalkerHardcore-v3']


class Box2DConfig(BaseConfig):
    def __init__(self):
        super(Box2DConfig, self).__init__()
        self.seed_steps = 5000
        self.max_env_steps = 1000000
        self.env_itr_steps = 1000
        self.test_interval_steps = 10000

    def new_game(self, seed=None):
        env = GymEnv(self.args.env, self.args.symbolic_env, seed, self.args.max_episode_length,
                     1, self.args.bit_depth)
        return env


run_config = Box2DConfig()