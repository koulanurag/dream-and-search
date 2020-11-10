import torch

from ..model import bottle
from ..utils import imagine_ahead, lambda_return


# Rollout with a proposal policy.
class RolloutPlanner:
    __constants__ = ['proposal_action_count', 'uniform_action_count', 'planning_horizon', 'gamma', 'disclam']

    def __init__(self, proposal_action_count, uniform_action_count, planning_horizon, model, discount, disclam, pcont):
        super().__init__()
        self.proposal_action_count = proposal_action_count
        self.uniform_action_count = uniform_action_count
        self.model = model
        self.planning_horizon = planning_horizon
        self.discount = discount
        self.pcont = pcont
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
        if self.pcont:
            pcont_pred = bottle(self.model.pcont, (imagination_output.belief, imagination_output.prior_state))
        else:
            pcont_pred = self.discount * torch.ones_like(imged_reward)

        returns = lambda_return(imged_reward, value_pred, pcont_pred,
                                bootstrap=value_pred[-1],
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
