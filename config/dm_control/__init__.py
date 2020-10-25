import cv2
import numpy as np
import torch
from core.config import BaseConfig
from core.env import _images_to_observation


class ControlSuiteEnv:
    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
        from dm_control import suite
        from dm_control.suite.wrappers import pixels
        domain, task = env.split('-')
        self.symbolic = symbolic
        self._env = suite.load(domain_name=domain, task_name=task, task_kwargs={'random': seed})
        if not symbolic:
            self._env = pixels.Wrapper(self._env)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth

    def reset(self):
        self.t = 0  # Reset internal timer
        state = self._env.reset()
        if self.symbolic:
            _state = [np.asarray([obs]) if (isinstance(obs, float) or obs.shape == ()) else obs
                      for obs in state.observation.values()]
            return torch.tensor(np.concatenate(_state, axis=0), dtype=torch.float32).unsqueeze(dim=0)
        else:
            return _images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth)

    def step(self, action):
        action = action.detach().numpy()
        reward = 0
        for k in range(self.action_repeat):
            state = self._env.step(action)
            reward += state.reward
            self.t += 1  # Increment internal timer
            done = state.last() or self.t == self.max_episode_length
            if done:
                break
        if self.symbolic:
            _state = [np.asarray([obs]) if (isinstance(obs, float) or obs.shape == ()) else obs
                      for obs in state.observation.values()]
            observation = torch.tensor(np.concatenate(_state, axis=0), dtype=torch.float32).unsqueeze(dim=0)
        else:
            observation = _images_to_observation(self._env.physics.render(camera_id=0), self.bit_depth)
        return observation, reward, done

    def render(self):
        cv2.imshow('screen', self._env.physics.render(camera_id=0)[:, :, ::-1])
        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()
        self._env.close()

    @property
    def observation_size(self):
        return sum([(1 if len(obs.shape) == 0 else obs.shape[0]) for obs in
                    self._env.observation_spec().values()]) if self.symbolic else (3, 64, 64)

    @property
    def action_size(self):
        return self._env.action_spec().shape[0]

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        spec = self._env.action_spec()
        return torch.from_numpy(np.random.uniform(spec.minimum, spec.maximum, spec.shape))


class DmControlConfig(BaseConfig):
    def __init__(self):
        super(DmControlConfig, self).__init__()
        self.seed_steps = 5000
        self.max_env_steps = 500000
        self.env_itr_steps = 1000
        self.test_interval_steps = 10000

    def new_game(self, seed=None):
        env = ControlSuiteEnv(self.args.env, self.args.symbolic_env, seed, self.args.max_episode_length,
                              1, self.args.bit_depth)
        return env


run_config = DmControlConfig()
