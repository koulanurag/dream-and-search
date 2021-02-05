from core.config import BaseConfig
from core.env import GymEnv
from collections import defaultdict

ENVS = []
ACTION_SCALE = defaultdict(lambda: 1)
ACTION_REPEAT = defaultdict(lambda: 1)


class D4RLConfig(BaseConfig):
    def __init__(self):
        super(D4RLConfig, self).__init__()
        self.seed_steps = 500
        self.max_env_steps = 500000
        self.env_itr_steps = 200
        self.test_interval_steps = 5000

    def new_game(self, seed=None):
        # d4rl setup
        import os
        os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
        os.environ['D4RL_DATASET_DIR'] = str(self.args.d4rl_dataset_dir)
        import d4rl
        env = GymEnv(self.args.env, self.args.symbolic_env, seed, self.args.max_episode_length,
                     ACTION_REPEAT[self.args.env], self.args.bit_depth)
        return env

    @property
    def action_scale(self):
        return ACTION_SCALE[self.args.env]


run_config = D4RLConfig()
