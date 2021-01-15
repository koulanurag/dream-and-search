import numpy as np
import torch

from core.env import postprocess_observation, preprocess_observation_


class ExperienceReplay:
    def __init__(self, size, symbolic_env, observation_size, action_size, bit_depth, device,
                 enforce_absorbing_state=False, chunk_size=50):
        self.device = device
        self.symbolic_env = symbolic_env
        self.chunk_size = chunk_size
        self.chunks = int(size / self.chunk_size)
        self.observations = np.empty(
            (self.chunks, self.chunk_size, observation_size) if symbolic_env else (size, 3, 64, 64),
            dtype=np.float32 if symbolic_env else np.uint8)
        self.actions = np.empty((self.chunks, self.chunk_size, action_size), dtype=np.float32)
        self.rewards = np.empty((self.chunks, self.chunk_size,), dtype=np.float32)
        self.nonterminals = np.empty((self.chunks, self.chunk_size, 1), dtype=np.float32)
        self.idx = 0
        self.full = False  # Tracks if memory has been filled/all slots are valid
        self.steps, self.episodes = 0, 0  # Tracks how much experience has been used in total
        self.bit_depth = bit_depth
        self.enforce_absorbing_state = enforce_absorbing_state

        self.chunk_sub_priorities = np.empty((self.chunks, self.chunk_size,), dtype=np.float32)
        self.chunk_priorities = np.empty((self.chunks,), dtype=np.float32)

        self._chunk_observations = []
        self._chunk_actions = []
        self._chunk_rewards = []
        self._chunk_nonterminals = []

    def append(self, observation, action, reward, done):

        if self.symbolic_env:
            self._chunk_observations.append(observation.numpy())
        else:
            # Decentre and discretise visual observations (to save memory)
            self._chunk_observations.append(postprocess_observation(observation.numpy(), self.bit_depth))

        self._chunk_actions.append(action.numpy())
        self._chunk_rewards.append(reward)
        self._chunk_nonterminals.append(not done)
        self.steps, self.episodes = self.steps + 1, self.episodes + (1 if done else 0)

        if len(self._chunk_actions) % self.chunk_size == 0:
            self.observations[self.idx] = np.array(self._chunk_observations).squeeze(1)
            self.actions[self.idx] = np.array(self._chunk_actions)
            self.rewards[self.idx] = np.array(self._chunk_rewards)
            self.nonterminals[self.idx] = np.expand_dims(self._chunk_nonterminals, -1)

            self.idx = (self.idx + 1) % self.chunks
            self.full = self.full or self.idx == 0

            self._chunk_observations = []
            self._chunk_actions = []
            self._chunk_rewards = []
            self._chunk_nonterminals = []

    def _sample_idx(self, use_priroty=False):
        if use_priroty:
            chunk_idx = np.random.randint(0, self.chunks if self.full else self.idx)
        else:
            chunk_idx = np.random.randint(0, self.chunks if self.full else self.idx)
        return chunk_idx

    def _retrieve_batch(self, idxs, n):
        vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
        observations = torch.as_tensor(self.observations[vec_idxs].astype(np.float32))
        if not self.symbolic_env:
            preprocess_observation_(observations, self.bit_depth)  # Undo discretisation for visual observations

        obs = observations.reshape(self.chunk_size, n, *observations.shape[2:])
        actions = self.actions[vec_idxs].reshape(self.chunk_size, n, -1)
        rewards = self.rewards[vec_idxs].reshape(self.chunk_size, n)
        non_terminals = self.nonterminals[vec_idxs].reshape(self.chunk_size, n, 1)

        if self.enforce_absorbing_state:
            for chunk_idx in range(non_terminals.shape[1]):
                terminal_idxs = np.where(non_terminals[:, chunk_idx, 0] == 0)[0]
                if len(terminal_idxs) > 0:
                    first_terminal_idx = terminal_idxs[0]
                    rewards[first_terminal_idx + 1:, chunk_idx] = 0.0  # absorbing reward
                    non_terminals[first_terminal_idx + 1:, chunk_idx, 0] = 0  # terminal states
                    # terminal observations
                    obs[first_terminal_idx + 1:, chunk_idx, :] = obs[first_terminal_idx, chunk_idx, :]

        return obs, actions, rewards, non_terminals

    # Returns a batch of sequence chunks uniformly sampled from the memory
    def sample(self, n):
        batch = self._retrieve_batch(np.asarray([self._sample_idx() for _ in range(n)]), n)
        return [torch.as_tensor(item).to(device=self.device) for item in batch]

    def __len__(self):
        return self.chunks if self.full else (self.idx)
