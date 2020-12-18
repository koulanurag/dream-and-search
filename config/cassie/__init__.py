from core.config import BaseConfig
from core.env import GymEnv

ENVS = ['Cassie-v0', 'Cassie-v1']
ACTION_SCALE = {_: 3 for _ in ENVS}
import numpy as np


class CassieConfig(BaseConfig):
    def __init__(self):
        super(CassieConfig, self).__init__()
        self.seed_steps = 1000
        self.max_env_steps = 500000
        self.env_itr_steps = 200
        self.test_interval_steps = 2500

    def new_game(self, seed=None):
        import gym_cassie
        env = GymEnv(self.args.env, self.args.symbolic_env, seed, self.args.max_episode_length,
                     1, self.args.bit_depth, action_scale=ACTION_SCALE[self.args.env])

        env._env.action_space.low = -1 * np.array(env.action_space.low.shape)
        env._env.action_space.high = np.array(env.action_space.high.shape)
        return env


run_config = CassieConfig()
