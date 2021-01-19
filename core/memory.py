import numpy as np
import torch

from core.env import postprocess_observation, preprocess_observation_


class ExperienceReplay:
    def __init__(self, size, symbolic_env, observation_size, action_size, bit_depth, device,
                 enforce_absorbing_state=False, chunk_size=50, prob_alpha=0.6):
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

        self.prob_alpha = prob_alpha

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
            if self.full:
                max_prio = self.chunk_priorities.max()
            else:
                max_prio = self.chunk_priorities[:self.idx].max() if self.idx > 0 else 1.0
            self.chunk_priorities[self.idx] = max_prio

            self.idx = (self.idx + 1) % self.chunks
            self.full = self.full or self.idx == 0
            self._chunk_observations = []
            self._chunk_actions = []
            self._chunk_rewards = []
            self._chunk_nonterminals = []

    def _sample_idx(self, batch_size=1, beta=0.4, use_priority=False):
        if use_priority:

            if self.full:
                prios = self.chunk_priorities
            else:
                prios = self.chunk_priorities[:self.idx]

            probs = prios ** self.prob_alpha
            probs /= probs.sum()
            chunk_idx = np.random.choice(len(prios), batch_size, p=probs)

            total = len(prios)
            weights = (total * probs[chunk_idx]) ** (-beta)
            weights /= weights.max()
            weights = np.array(weights, dtype=np.float32)
        else:
            chunk_idx = np.random.randint(0, self.chunks if self.full else self.idx, (batch_size,))
            weights = np.ones(chunk_idx.shape)

        return chunk_idx, weights

    def _retrieve_batch(self, idxs, priorities, n):
        vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
        observations = torch.as_tensor(self.observations[vec_idxs].astype(np.float32))
        if not self.symbolic_env:
            preprocess_observation_(observations, self.bit_depth)  # Undo discretisation for visual observations

        obs = observations.reshape(self.chunk_size, n, *observations.shape[2:])
        actions = self.actions[vec_idxs].reshape(self.chunk_size, n, -1)
        rewards = self.rewards[vec_idxs].reshape(self.chunk_size, n)
        non_terminals = self.nonterminals[vec_idxs].reshape(self.chunk_size, n, 1)
        absorbing_states = torch.zeros(non_terminals.shape)

        if self.enforce_absorbing_state:
            for chunk_idx in range(non_terminals.shape[1]):
                terminal_idxs = np.where(non_terminals[:, chunk_idx, 0] == 0)[0]
                if len(terminal_idxs) > 0:
                    first_terminal_idx = terminal_idxs[0]
                    rewards[first_terminal_idx + 1:, chunk_idx] = 0.0  # absorbing reward
                    non_terminals[first_terminal_idx + 1:, chunk_idx, 0] = 0  # terminal states
                    # terminal observations
                    obs[first_terminal_idx + 1:, chunk_idx, :] = obs[first_terminal_idx, chunk_idx, :]
                    absorbing_states[first_terminal_idx + 1:, chunk_idx, 0] = 1
        return obs, actions, rewards, non_terminals, absorbing_states, vec_idxs, priorities

    # Returns a batch of sequence chunks uniformly sampled from the memory
    def sample(self, n, use_priority=True, beta=0.4):
        chunk_idxs, weights = self._sample_idx(batch_size=n, beta=beta, use_priority=use_priority)
        batch = self._retrieve_batch(chunk_idxs, weights, n)
        return [torch.as_tensor(item).to(device=self.device) for item in batch]

    def update_priorities(self, idxs, priorities):
        self.chunk_priorities[idxs] = priorities

    def __len__(self):
        return self.size if self.full else self.idx
