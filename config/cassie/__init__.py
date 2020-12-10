from core.config import BaseConfig
from core.env import GymEnv

ENVS = ['']


class CassieConfig(BaseConfig):
    def __init__(self):
        super(CassieConfig, self).__init__()
        self.seed_steps = 5000
        self.max_env_steps = 500000
        self.env_itr_steps = 1000
        self.test_interval_steps = 5000

    def new_game(self, seed=None):
        import gym_cassie
        env = GymEnv(self.args.env, self.args.symbolic_env, seed, self.args.max_episode_length,
                     1, self.args.bit_depth)
        return env


run_config = CassieConfig()
