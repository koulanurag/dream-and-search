import logging
import os
from typing import Iterable

import torch
from torch.nn import Module
from torch.nn import functional as F
from torch.distributions import Normal
from typing import NamedTuple
from torch import Tensor


class ImaginationOutput(NamedTuple):
    belief: Tensor
    prior_state: Tensor
    prior_mean: Tensor
    prior_std_dev: Tensor


def update_belief(transition_model, encoder, belief, posterior_state, action, observation):
    # Action and observation need extra time dimension
    belief, _, _, _, posterior_state, _, _ = transition_model(posterior_state, action.unsqueeze(dim=0),
                                                              belief, encoder(observation).unsqueeze(dim=0))
    # Remove time dimension from belief/state
    belief, posterior_state = belief.squeeze(dim=0), posterior_state.squeeze(dim=0)
    return belief, posterior_state


def select_action(config, env, planner, belief, posterior_state, random=False, explore=False):
    if random:  # collect data for seeding
        action = env.sample_random_action()
        action = action.unsqueeze(0).float()
    else:
        if config.args.search_mode == "no-search":
            action = planner.get_action(belief, posterior_state, det=not (explore))
        else:
            action = planner(belief, posterior_state)  # Get action from planner(q(s_t|o≤t,a<t), p)
        if explore:
            # Add gaussian exploration noise on top of the sampled action
            action = torch.clamp(Normal(action, config.args.action_noise).rsample(), -1, 1)
    return action


def init_logger(base_path):
    formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s][%(filename)s>%(funcName)s] ==> %(message)s')
    for mode in ['train', 'test', 'train_eval', 'root']:
        file_path = os.path.join(base_path, mode + '.log')
        logger = logging.getLogger(mode)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        handler = logging.FileHandler(file_path, mode='a')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)


def imagine_ahead(prev_state, prev_belief, policy, transition_model, planning_horizon=12):
    '''
    imagine_ahead is the function to draw the imaginary tracjectory using the dynamics model, actor, critic.
    Input: current state (posterior), current belief (hidden), policy, transition_model  # torch.Size([50, 30]) torch.Size([50, 200])
    Output: generated trajectory of features includes beliefs, prior_states, prior_means, prior_std_devs
            torch.Size([49, 50, 200]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30]) torch.Size([49, 50, 30])
    '''
    flatten = lambda x: x.view([-1] + list(x.size()[2:]))
    prev_belief = flatten(prev_belief)
    prev_state = flatten(prev_state)

    # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
    T = planning_horizon
    beliefs, prior_states, prior_means, prior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [
        torch.empty(0)] * T, [torch.empty(0)] * T
    beliefs[0], prior_states[0] = prev_belief, prev_state

    # Loop over time sequence
    for t in range(T - 1):
        _state = prior_states[t]
        actions = policy.get_action(beliefs[t].detach(), _state.detach())
        # Compute belief (deterministic hidden state)
        hidden = transition_model.act_fn(transition_model.fc_embed_state_action(torch.cat([_state, actions], dim=1)))
        beliefs[t + 1] = transition_model.rnn(hidden, beliefs[t])
        # Compute state prior by applying transition dynamics
        hidden = transition_model.act_fn(transition_model.fc_embed_belief_prior(beliefs[t + 1]))
        prior_means[t + 1], _prior_std_dev = torch.chunk(transition_model.fc_state_prior(hidden), 2, dim=1)
        prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + transition_model.min_std_dev
        prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])
        # Return new hidden states

    return ImaginationOutput(torch.stack(beliefs[1:], dim=0),
                             torch.stack(prior_states[1:], dim=0),
                             torch.stack(prior_means[1:], dim=0),
                             torch.stack(prior_std_devs[1:], dim=0))


def lambda_return(imged_reward, value_pred, bootstrap, discount=0.99, lambda_=0.95):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    next_values = torch.cat([value_pred[1:], bootstrap[None]], 0)
    discount_tensor = discount * torch.ones_like(imged_reward)  # pcont
    inputs = imged_reward + discount_tensor * next_values * (1 - lambda_)
    last = bootstrap
    indices = reversed(range(len(inputs)))
    outputs = []
    for index in indices:
        inp, disc = inputs[index], discount_tensor[index]
        last = inp + disc * lambda_ * last
        outputs.append(last)
    outputs = list(reversed(outputs))
    outputs = torch.stack(outputs, 0)
    returns = outputs
    return returns


class ActivateParameters:
    def __init__(self, modules: Iterable[Module]):
        """
        Context manager to locally Activate the gradients.
        example:
        ```
        with ActivateParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            # print(param.requires_grad)
            param.requires_grad = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]


def get_parameters(modules: Iterable[Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


class FreezeParameters:
    def __init__(self, modules: Iterable[Module]):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]
