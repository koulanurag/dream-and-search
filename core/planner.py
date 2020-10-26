import torch
from torch import jit
from .utils import imagine_ahead, lambda_return
from .model import bottle


# Model-predictive control planner with cross-entropy method and learned transition model
class MPCPlanner(jit.ScriptModule):
    __constants__ = ['action_size', 'planning_horizon', 'optimisation_iters', 'candidates', 'top_candidates']

    def __init__(self, action_size, planning_horizon, optimisation_iters, candidates, top_candidates, transition_model,
                 reward_model):
        super().__init__()
        self.transition_model, self.reward_model = transition_model, reward_model
        self.action_size = action_size
        self.planning_horizon = planning_horizon
        self.optimisation_iters = optimisation_iters
        self.candidates, self.top_candidates = candidates, top_candidates

    @jit.script_method
    def forward(self, belief, state):
        B, H, Z = belief.size(0), belief.size(1), state.size(1)
        belief = belief.unsqueeze(dim=1).expand(B, self.candidates, H).reshape(-1, H)
        state = state.unsqueeze(dim=1).expand(B, self.candidates, Z).reshape(-1, Z)

        # Initialize factorized belief over action sequences q(a_t:t+H) ~ N(0, I)
        action_mean = torch.zeros(self.planning_horizon, B, 1, self.action_size, device=belief.device)
        action_std_dev = torch.ones(self.planning_horizon, B, 1, self.action_size, device=belief.device)

        for _ in range(self.optimisation_iters):
            # Evaluate J action sequences from the current belief (over entire sequence at once, batched over particles)
            noise = torch.randn(self.planning_horizon, B, self.candidates, self.action_size, device=action_mean.device)
            noise = noise.view(self.planning_horizon, B * self.candidates, self.action_size)
            actions = (action_mean + action_std_dev * noise)  # Sample actions (time x (batch x candidates) x actions)

            # Sample next states
            # [12, 1000, 200] [12, 1000, 30] : 12 horizon steps; 1000 candidates
            beliefs, states, _, _ = self.transition_model(state, actions, belief)

            # Calculate expected returns (technically sum of rewards over planning horizon)
            returns = self.reward_model(beliefs.view(-1, H), states.view(-1, Z))
            # output from r-model[12000]->view[12, 1000]->sum[1000]
            returns = returns.view(self.planning_horizon, -1).sum(dim=0)

            # Re-fit belief to the K best action sequences
            _, topk = returns.reshape(B, self.candidates).topk(self.top_candidates, dim=1, largest=True, sorted=False)
            # Fix indices for unrolled actions
            topk += self.candidates * torch.arange(0, B, dtype=torch.int64, device=topk.device).unsqueeze(dim=1)
            best_actions = actions[:, topk.view(-1)].reshape(self.planning_horizon, B, self.top_candidates,
                                                             self.action_size)
            # Update belief with new means and standard deviations
            action_mean = best_actions.mean(dim=2, keepdim=True)
            action_std_dev = best_actions.std(dim=2, unbiased=False, keepdim=True)

        # Return first action mean µ_t
        return action_mean[0].squeeze(dim=1)


class RolloutPlanner():
    __constants__ = ['proposal_action_count', 'uniform_action_count', 'planning_horizon', 'gamma', 'disclam']

    def __init__(self, proposal_action_count, uniform_action_count, planning_horizon, model, gamma, disclam):
        super().__init__()
        self.proposal_action_count = proposal_action_count
        self.uniform_action_count = uniform_action_count
        self.model = model
        self.planning_horizon = planning_horizon
        self.gamma = gamma
        self.disclam = disclam

    # @jit.script_method
    def __call__(self, belief, state):
        batch_size, belief_size = belief.shape
        _, state_size = state.shape

        total_actions = self.proposal_action_count + self.uniform_action_count
        root_uniform_action_mask = torch.zeros((batch_size, total_actions)).to(belief.device)
        root_uniform_action_mask[:, torch.arange(self.proposal_action_count, total_actions)] = 1
        root_uniform_action_mask = root_uniform_action_mask.flatten().long()

        # reshape states to accommodate multiple actions for each root state.
        _belief = belief.repeat(1, 1, total_actions)
        _belief = _belief.reshape((1, batch_size * total_actions, belief_size))

        _state = state.repeat(1, 1, total_actions)
        _state = _state.reshape((1, batch_size * total_actions, state_size))

        # imagine rollouts
        imagination_output = imagine_ahead(_state, _belief, self.model.actor, self.model.transition,
                                           self.planning_horizon, det=True,
                                           root_uniform_action_mask=root_uniform_action_mask)

        imged_reward = bottle(self.model.reward, (imagination_output.belief, imagination_output.prior_state))
        value_pred = bottle(self.model.value, (imagination_output.belief, imagination_output.prior_state))

        returns = lambda_return(imged_reward, value_pred,
                                bootstrap=value_pred[-1],
                                discount=self.gamma,
                                lambda_=self.disclam)

        # get value of root childs
        q_values = returns[0, :].reshape((batch_size, total_actions))

        # determine actions
        root_actions = imagination_output.actions[0, :, :]
        root_actions = root_actions.reshape((batch_size, total_actions, root_actions.shape[-1]))

        greedy_actions_idx = torch.argmax(q_values, dim=1)
        greedy_actions_idx = greedy_actions_idx.view(-1, 1, 1).expand(root_actions.size(0), 1, root_actions.size(2))
        greedy_action = root_actions.gather(1, greedy_actions_idx).squeeze(1)
        return greedy_action
