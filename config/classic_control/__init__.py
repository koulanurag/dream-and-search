from core.config import BaseConfig
from core.env import GymEnv

ENVS = ['Pendulum-v0', 'MountainCarContinuous-v0']
ACTION_SCALE = {'Pendulum-v0': 2, 'MountainCarContinuous-v0': 1}
ACTION_REPEAT = {'Pendulum-v0': 1, 'MountainCarContinuous-v0': 1}


class ClassicControlConfig(BaseConfig):
    def __init__(self):
        super(ClassicControlConfig, self).__init__()
        self.seed_steps = 500
        self.max_env_steps = 500000
        self.env_itr_steps = 200
        self.test_interval_steps = 5000

    def new_game(self, seed=None):
        assert self.args.env in ENVS
        env = GymEnv(self.args.env, self.args.symbolic_env, seed, self.args.max_episode_length,
                     ACTION_REPEAT[self.args.env], self.args.bit_depth)
        return env

    @property
    def action_scale(self):
        return ACTION_SCALE[self.args.env]


run_config = ClassicControlConfig()
