from core.config import BaseConfig
from core.env import GymEnv

ENVS = ['Cassie-v0', 'Cassie-v1']
ACTION_SCALE = {_: 3 for _ in ENVS}
import numpy as np


class CassieConfig(BaseConfig):
    def __init__(self):
        super(CassieConfig, self).__init__()
        self.seed_steps = 1000
        self.max_env_steps = 2000000
        self.env_itr_steps = 200
        self.test_interval_steps = 2500

    def new_game(self, seed=None):
        import gym_cassie
        env = GymEnv(self.args.env, self.args.symbolic_env, seed, self.args.max_episode_length,
                     1, self.args.bit_depth)

        from gym import spaces
        env._env.action_space = spaces.Box(low=-1 * np.ones(env.action_space.low.shape),
                                           high=np.ones(env.action_space.high.shape),
                                           dtype=np.float32)

        return env


run_config = CassieConfig()
