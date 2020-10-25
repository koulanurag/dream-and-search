from .utils import update_belief, select_action
import numpy as np
from typing import NamedTuple


class TestOutput(NamedTuple):
    score: float
    steps: float


def test(config, env, model, planner, mode, render=False):
    observation = env.reset()
    dones = [False for _ in range(env.n)]

    # init
    belief = model.init_belief(env.n).to(config.args.device)
    posterior_state = model.init_state(env.n).to(config.args.device)
    action = model.init_action(env.n).to(config.args.device)

    # test trackers
    episode_rewards = []
    episode_steps = np.zeros(env.n)

    while not all(dones):

        # update belief and determine action
        belief, posterior_state = update_belief(model.transition, model.encoder, belief, posterior_state,
                                                action, observation.to(device=config.args.device))
        action = select_action(config, env, planner, belief, posterior_state, mode=mode)

        # step in the environment
        for _ in range(config.args.action_repeat):
            next_observation, reward, dones = env.step(action)
            episode_rewards.append(reward.cpu().numpy() * (1 - dones.cpu().int().numpy()))
            episode_steps += (1 - dones.cpu().int().numpy())

            if render:
                env.render()

            if all(dones):
                break

        observation = next_observation

    episode_steps += 1  # adding for last step.
    return TestOutput(np.array(episode_rewards).sum(axis=0).mean(), np.mean(episode_rewards))
