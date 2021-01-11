from core.config import BaseConfig
from core.env import GymEnv

ENVS = ['Cassie-v2']
ACTION_SCALE = {_: 3 for _ in ENVS}
import numpy as np


class CassieV2GymEnv(GymEnv):
    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth, action_scale=1):
        from .cassie import CassieEnv_v2
        from gym import spaces

        self.symbolic = symbolic
        self._env = CassieEnv_v2()
        self._env.seed(seed)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth
        self.action_scale = action_scale
        self._env.action_space = spaces.Box(low=-1 * np.ones(self._env.action_space.shape),
                                            high=np.ones(self._env.action_space.shape),
                                            dtype=np.float32)


class CassieConfigV2(BaseConfig):
    def __init__(self):
        super(CassieConfigV2, self).__init__()
        self.seed_steps = 1000
        self.max_env_steps = 2000000
        self.env_itr_steps = 200
        self.test_interval_steps = 2500

    def new_game(self, seed=None):
        env = CassieV2GymEnv(self.args.env, self.args.symbolic_env, seed, self.args.max_episode_length,
                             1, self.args.bit_depth, action_scale=ACTION_SCALE[self.args.env])
        return env


run_config = CassieConfigV2()
