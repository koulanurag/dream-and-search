import math
from typing import NamedTuple
import numpy as np
import torch
from torch import Tensor

from torch.distributions import Normal


class ActionCapsule(NamedTuple):
    action: Tensor
    repeat: int
    repeat_one_hot: Tensor

    def __eq__(self, other):
        if (self.action == other.action).all().item() and self.repeat == other.repeat:
            return True
        else:
            return False

    def __gt__(self, other):
        if (self.action < other.action).all().item():
            return True
        else:
            return False

    def __ge__(self, other):
        if (self.action < other.action).all().item():
            return True
        else:
            return False


class MinMaxStats(object):
    """A class that holds the min-max values of the tree."""

    def __init__(self, min_value_bound=None, max_value_bound=None):
        self.maximum = min_value_bound if min_value_bound else -float('inf')
        self.minimum = max_value_bound if max_value_bound else float('inf')

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node(object):

    def __init__(self, action_log_prob: float, root=False):
        self.visit_count = 0
        self.root = root
        self.prior = None
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0
        self.actor_output = None
        self.action_log_prob = action_log_prob

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, model, belief, state, reward, proposal_action_sample_n=20, action_noise=0.3, progressive=False):

        self.hidden_state = (belief, state)
        self.reward = reward

        if progressive:
            action = model.actor.get_action(belief, state)
            self.children[action] = Node(0)
        else:
            actions = model.actor.get_action(belief.repeat(proposal_action_sample_n, 1),
                                             state.repeat(proposal_action_sample_n, 1))
            actions = torch.clamp(Normal(actions, action_noise).rsample(), -1, 1)

            for child_i, child_action in enumerate(actions):
                self.children[child_action.unsqueeze(0)] = Node(0)

        self.update_prior()

    def update_prior(self):
        exp_sum = sum([(math.e ** 0.25) ** (child.action_log_prob) for action_cap, child in self.children.items()])
        for child in self.children.values():
            child.prior = ((math.e ** 0.25) ** (child.action_log_prob) / exp_sum)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class MCTS(object):
    def __init__(self, config, model, exploration=False, progressive=False):
        self.config = config
        self.exploration = exploration
        self.progressive = progressive
        self.model = model

    def __call__(self, belief, state):
        reward = self.model.reward(belief, state)
        root = Node(0, root=True)
        root.expand(self.model, belief, state, reward, self.config.args.mcts_fixed_proposal_action,
                    self.config.args.action_noise, self.progressive)
        self.run(root, self.config.args.mcts_num_simulations)
        _, action, child = max(((root.reward + self.config.args.discount * child.value(), action, child)
                                for action, child in root.children.items()), key=lambda t: t[0])
        return action

    def run(self, root, num_simulations=50):
        min_max_stats = MinMaxStats()

        for _ in range(num_simulations):
            node = root
            search_path = [node]

            while node.expanded():
                action, node = self.select_child(self.model, node, min_max_stats)
                search_path.append(node)

            # Inside the search tree we use the dynamics function to obtain the next
            # hidden state given an action and the previous hidden state.
            parent = search_path[-2]
            belief, state = parent.hidden_state
            transition_output = self.model.transition(state, action.unsqueeze(0), belief)
            next_belief, next_state = transition_output.beliefs.squeeze(0), transition_output.prior_states.squeeze(0)

            value = self.model.value(next_belief, next_state)
            reward = self.model.reward(next_belief, next_state)

            node.expand(self.model, next_belief, next_state, reward.item(),
                        self.config.args.mcts_fixed_proposal_action,
                        self.config.args.action_noise, self.progressive)

            self.backpropagate(search_path, value.item(), min_max_stats)

    def select_child(self, model, node, min_max_stats):
        p = self.config.args.mcts_cpw * (node.visit_count) ** self.config.args.mcts_alpha
        if (self.progressive and p <= len(node.children.keys())) or (not self.progressive):
            _, action, child = max(((self.ucb_score(node, child, min_max_stats), action, child)
                                    for action, child in node.children.items()), key=lambda t: t[0])
        else:
            belief, state = node.hidden_state
            actor_output = node.actor_output

            # add an action to the node
            action = model.actor.action_sample(actor_output, deterministic=False)
            action_sample_attempt = 0

            rounded = lambda arr: np.around(arr.cpu().numpy(), 2).tolist()

            while action_sample_attempt < 100 and \
                    (rounded(action) in [rounded(action_cap.action.squeeze(0)) for action_cap in node.children.keys()]):
                action = model.actor.action_sample(actor_output, deterministic=False)
                action_sample_attempt += 1

            if action_sample_attempt == 99:
                print('Something is wrong in sampling')

            actor_repeat_output = model.actor_repeat(belief, state, action)
            action_repeat_one_hot, action_repeat = model.actor_repeat.sample(actor_repeat_output, deterministic=False)
            action_cap = ActionCapsule(action.unsqueeze(0), action_repeat.int().item(),
                                       action_repeat_one_hot.unsqueeze(0))

            actor_dist = model.actor.action_dist(actor_output)
            action_log_prob = - actor_dist.entropy(action)

            child = Node(action_log_prob.item())
            node.children[action_cap] = child
            node.update_prior()

            if self.exploration and node.root:
                node.add_exploration_noise(self.config.root_dirichlet_alpha, self.config.root_exploration_fraction)

        return action, child

    def ucb_score(self, parent, child, min_max_stats) -> float:
        pb_c = math.log((parent.visit_count + self.config.args.pb_c_base + 1) / self.config.args.pb_c_base)
        pb_c += self.config.args.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = min_max_stats.normalize(child.value())

        return prior_score + value_score

    def backpropagate(self, search_path, value, min_max_stats):
        for node in search_path:
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.value())

            value = node.reward + self.config.args.discount * value
