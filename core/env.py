import cv2
import numpy as np
import torch

GYM_ENVS = ['Pendulum-v0', 'MountainCarContinuous-v0', 'Ant-v2', 'HalfCheetah-v2', 'Hopper-v2', 'Humanoid-v2',
            'HumanoidStandup-v2', 'InvertedDoublePendulum-v2', 'InvertedPendulum-v2', 'Reacher-v2', 'Swimmer-v2',
            'Walker2d-v2']
CONTROL_SUITE_ENVS = ['cartpole-balance', 'cartpole-swingup', 'reacher-easy', 'finger-spin', 'cheetah-run',
                      'ball_in_cup-catch', 'walker-walk', 'reacher-hard', 'walker-run', 'humanoid-stand',
                      'humanoid-walk', 'fish-swim', 'acrobot-swingup']
CONTROL_SUITE_ACTION_REPEATS = {'cartpole': 8, 'reacher': 4, 'finger': 2, 'cheetah': 4, 'ball_in_cup': 6, 'walker': 2,
                                'humanoid': 2, 'fish': 2, 'acrobot': 4}


# Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
def preprocess_observation_(observation, bit_depth):
    # Quantise to given bit depth and centre
    observation.div_(2 ** (8 - bit_depth)).floor_().div_(2 ** bit_depth).sub_(0.5)
    # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)
    observation.add_(torch.rand_like(observation).div_(2 ** bit_depth))


# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
    _obs = np.clip(np.floor((observation + 0.5) * 2 ** bit_depth) * 2 ** (8 - bit_depth), 0, 2 ** 8 - 1)
    return _obs.astype(np.uint8)


def _images_to_observation(images, bit_depth):
    images = torch.tensor(cv2.resize(images, (64, 64), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1),
                          dtype=torch.float32)  # Resize and put channel first
    preprocess_observation_(images, bit_depth)  # Quantise, centre and dequantise inplace
    return images.unsqueeze(dim=0)  # Add batch dimension


class GymEnv:
    def __init__(self, env, symbolic, seed, max_episode_length, action_repeat, bit_depth):
        import gym
        self.symbolic = symbolic
        self._env = gym.make(env)
        self._env.seed(seed)
        self.max_episode_length = max_episode_length
        self.action_repeat = action_repeat
        self.bit_depth = bit_depth

    def reset(self):
        self.t = 0  # Reset internal timer
        state = self._env.reset()
        if self.symbolic:
            return torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
        else:
            return _images_to_observation(self._env.render(mode='rgb_array'), self.bit_depth)

    def step(self, action):
        action = action.detach().numpy()
        reward = 0
        for k in range(self.action_repeat):
            state, reward_k, done, _ = self._env.step(action)
            reward += reward_k
            self.t += 1  # Increment internal timer
            done = done or self.t == self.max_episode_length
            if done:
                break
        if self.symbolic:
            observation = torch.tensor(state, dtype=torch.float32).unsqueeze(dim=0)
        else:
            observation = _images_to_observation(self._env.render(mode='rgb_array'), self.bit_depth)
        return observation, reward, done

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return self._env.observation_space.shape[0] if self.symbolic else (3, 64, 64)

    @property
    def action_size(self):
        return self._env.action_space.shape[0]

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return torch.from_numpy(self._env.action_space.sample())


# Wrapper for batching environments together
class EnvBatcher():
    def __init__(self, env_fn, n):
        self.n = n
        self.envs = [env_fn() for _ in range(n)]
        self.dones = [True] * n

    # Resets every environment and returns observation
    def reset(self):
        observations = [env.reset() for env in self.envs]
        self.dones = [False] * self.n
        return torch.cat(observations)

    # Steps/resets every environment and returns (observation, reward, done)
    def step(self, actions):
        # Done mask to blank out observations and zero rewards for previously terminated environments
        done_mask = torch.nonzero(torch.tensor(self.dones))[:, 0]
        observations, rewards, dones = zip(*[env.step(action) for env, action in zip(self.envs, actions)])

        # Env should remain terminated if previously terminated
        dones = [d or prev_d for d, prev_d in zip(dones, self.dones)]
        self.dones = dones

        observations = torch.cat(observations)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.uint8)

        observations[done_mask] = 0
        rewards[done_mask] = 0
        return observations, rewards, dones

    def close(self):
        [env.close() for env in self.envs]

    def render(self):
        return [env.render() for env in self.envs]
