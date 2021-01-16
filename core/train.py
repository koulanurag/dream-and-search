import logging
from collections import defaultdict

import torch
from torch.distributions import Normal, kl_divergence, Bernoulli
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from .config import BaseConfig
from .env import EnvBatcher
from .memory import ExperienceReplay
from .model import bottle
from .planner import MPCPlanner, RolloutPlanner, MCTS
from .test import test
from .utils import imagine_ahead, lambda_return, FreezeParameters
from .utils import update_belief, select_action
import wandb

train_logger = logging.getLogger('train')
test_logger = logging.getLogger('train_eval')
count_tracker = {'updates': 0}


def update_params(config, model, optimizers, D, free_nats, global_prior, writer, total_env_steps):
    dynamics_optimizer, value_optimizer, policy_optimizer = optimizers
    losses = defaultdict(lambda: 0)
    data_distribution = {'reward': [], 'terminal': []}

    for update_step in range(config.args.collect_interval):
        # sample batch
        # Transitions start at time t = 0
        observations, actions, rewards, non_terminals, absorbing_states = D.sample(config.args.batch_size,
                                                                                   config.args.chunk_size)

        # ##################
        # Dynamics learning
        # ##################

        init_belief = model.init_belief(config.args.batch_size).to(config.args.device)
        init_state = model.init_state(config.args.batch_size).to(config.args.device)

        transition_output = model.transition(init_state, actions[:-1], init_belief,
                                             bottle(model.encoder, (observations[1:],)),
                                             non_terminals[:-1])

        # observation and reward loss
        predicted_obs = bottle(model.observation, (transition_output.beliefs, transition_output.posterior_states))
        predicted_reward_1 = bottle(model.reward_1, (transition_output.beliefs, transition_output.posterior_states))
        predicted_reward_2 = bottle(model.reward_2, (transition_output.beliefs, transition_output.posterior_states))
        if config.args.worldmodel_LogProbLoss:
            observation_dist = Normal(predicted_obs, 1)
            reward_dist_1 = Normal(predicted_reward_1, 1)
            reward_dist_2 = Normal(predicted_reward_2, 1)

            observation_loss = -observation_dist.log_prob(observations[1:])
            reward_loss_1 = -reward_dist_1.log_prob(rewards[:-1]).mean(dim=(0, 1))
            reward_loss_2 = -reward_dist_2.log_prob(rewards[:-1]).mean(dim=(0, 1))
        else:
            observation_loss = F.mse_loss(predicted_obs, observations[1:], reduction='none')
            reward_loss_1 = F.mse_loss(predicted_reward_1, rewards[:-1], reduction='none').mean(dim=(0, 1))
            reward_loss_2 = F.mse_loss(predicted_reward_2, rewards[:-1], reduction='none').mean(dim=(0, 1))

        observation_loss = observation_loss.sum(dim=2 if config.args.symbolic_env else (2, 3, 4)).mean(dim=(0, 1))

        # transition loss
        div = kl_divergence(Normal(transition_output.posterior_means, transition_output.posterior_std_devs),
                            Normal(transition_output.prior_means, transition_output.prior_std_devs)).sum(dim=2)
        kl_loss = torch.max(div, free_nats).mean(dim=(0, 1))

        # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
        if config.args.global_kl_beta != 0:
            kl_loss += config.global_kl_beta * kl_divergence(Normal(transition_output.posterior_means,
                                                                    transition_output.posterior_std_devs),
                                                             global_prior).sum(dim=2).mean(dim=(0, 1))

        dynamics_loss = observation_loss + reward_loss_1 + reward_loss_2 + kl_loss

        # discount loss
        pcont_loss = torch.tensor([0])
        if config.args.pcont:
            pcont_pred = bottle(model.pcont, (transition_output.beliefs, transition_output.posterior_states))
            pcont_dist = Bernoulli(probs=pcont_pred)
            pcont_target = config.args.discount * non_terminals[:-1].squeeze(-1)
            pcont_loss = - pcont_dist.log_prob(pcont_target).mean(dim=(0, 1))
            pcont_loss *= config.args.pcont_scale

        # Update dynamics parameters
        dynamics_optimizer.zero_grad()
        if config.args.pcont:
            (dynamics_loss + pcont_loss).backward()
        else:
            dynamics_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.transition.parameters(), config.args.grad_clip_norm, norm_type=2)
        torch.nn.utils.clip_grad_norm_(model.reward_1.parameters(), config.args.grad_clip_norm, norm_type=2)
        torch.nn.utils.clip_grad_norm_(model.reward_2.parameters(), config.args.grad_clip_norm, norm_type=2)
        torch.nn.utils.clip_grad_norm_(model.observation.parameters(), config.args.grad_clip_norm, norm_type=2)
        torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), config.args.grad_clip_norm, norm_type=2)
        if config.args.pcont:
            torch.nn.utils.clip_grad_norm_(model.pcont.parameters(), config.args.grad_clip_norm, norm_type=2)
        dynamics_optimizer.step()

        # ##################
        # Policy Learning
        # ##################

        with torch.no_grad():
            actor_states = transition_output.posterior_states.detach()
            actor_beliefs = transition_output.beliefs.detach()

        with FreezeParameters([model.transition, model.encoder, model.reward_1, model.reward_2,
                               model.observation, model.pcont]):
            imagination_output = imagine_ahead(actor_states, actor_beliefs, model.actor, model.transition,
                                               config.args.planning_horizon)
        with FreezeParameters([model.transition, model.encoder, model.reward_1, model.reward_2,
                               model.observation, model.pcont]):
            with FreezeParameters([model.value_1, model.value_2]):
                imged_reward_1 = bottle(model.reward_1, (imagination_output.belief, imagination_output.prior_state))
                imged_reward_2 = bottle(model.reward_2, (imagination_output.belief, imagination_output.prior_state))
                imged_reward = torch.min(imged_reward_1, imged_reward_2)
                value_pred_1 = bottle(model.value_1, (imagination_output.belief, imagination_output.prior_state))
                value_pred_2 = bottle(model.value_2, (imagination_output.belief, imagination_output.prior_state))
                value_pred = torch.min(value_pred_1, value_pred_2)
                if config.args.pcont:
                    pcont_pred = bottle(model.pcont, (imagination_output.belief, imagination_output.prior_state))
                else:
                    pcont_pred = config.args.discount * torch.ones_like(imged_reward)

        returns = lambda_return(imged_reward, value_pred, pcont_pred, bootstrap=value_pred[-1],
                                lambda_=config.args.disclam)
        if config.args.enforce_absorbing_state:
            returns[:, absorbing_states[:-1].flatten() == 1] = 0
        policy_loss = -torch.mean(returns)

        # Update policy parameters
        policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.actor.parameters(), config.args.grad_clip_norm, norm_type=2)
        policy_optimizer.step()

        # ##################
        # Value learning
        # ##################
        with torch.no_grad():
            value_beliefs = imagination_output.belief.detach()
            value_prior_states = imagination_output.prior_state.detach()
            target_return = returns.detach()

        # detach the input tensor from the transition network.
        value_dist_1 = Normal(bottle(model.value_1, (value_beliefs, value_prior_states)), 1)
        value_dist_2 = Normal(bottle(model.value_2, (value_beliefs, value_prior_states)), 1)
        value_loss_1 = -value_dist_1.log_prob(target_return).mean(dim=(0, 1))
        value_loss_2 = -value_dist_2.log_prob(target_return).mean(dim=(0, 1))

        # Update value parameters
        value_optimizer.zero_grad()
        (value_loss_1 + value_loss_2).backward()
        torch.nn.utils.clip_grad_norm_(model.value_1.parameters(), config.args.grad_clip_norm, norm_type=2)
        torch.nn.utils.clip_grad_norm_(model.value_2.parameters(), config.args.grad_clip_norm, norm_type=2)
        value_optimizer.step()

        # store for logging
        losses['actor'] += policy_loss.item()
        losses['value_1'] += value_loss_1.item()
        losses['value_2'] += value_loss_2.item()
        losses['obs'] += observation_loss.item()
        losses['reward_1'] += reward_loss_1.item()
        losses['reward_2'] += reward_loss_2.item()
        losses['kl'] += kl_loss.item()
        losses['pcont'] += pcont_loss.item()

        # log distribution
        count_tracker['updates'] += 1
        data_distribution['terminal'] += (1 - non_terminals).flatten().data.cpu().numpy().tolist()

    losses = {k: v / config.args.collect_interval for k, v in losses.items()}

    # Log
    for key, value in losses.items():
        writer.add_scalar('train/{}'.format(key), value, total_env_steps)
    writer.add_scalar('train/network_updates', count_tracker['updates'], total_env_steps)
    if config.args.pcont:
        writer.add_scalar('train/pcont_loss', losses['pcont'], total_env_steps)

    # log sampling information of data.
    writer.add_scalar('train_data/terminal_batch_percent',
                      sum(data_distribution['terminal']) / len(data_distribution['terminal']),
                      total_env_steps)

    _msg = 'env steps #{:<10}'.format(total_env_steps)
    _msg += ' '.join(['{}:{:<8.3f} loss'.format(key, value) for key, value in losses.items()])
    train_logger.info(_msg)


def _test(config, model, env, base_policy, planner, writer, best_test_score, total_env_steps):
    no_search_test_output = test(config, env, model, base_policy, mode='no-search')
    if config.args.search_mode == 'no-search':
        with_search_test_output = no_search_test_output
    else:
        with_search_test_output = test(config, env, model, planner, mode=config.args.search_mode)

    # save best model
    if no_search_test_output.score >= best_test_score['score']['no_search']:
        torch.save(model.state_dict(), config.best_model_path['no_search'])
        best_test_score['score']['no_search'] = no_search_test_output.score

    if with_search_test_output.score >= best_test_score['score']['with_search']:
        torch.save(model.state_dict(), config.best_model_path['with_search'])
        best_test_score['score']['with_search'] = with_search_test_output.score

    # Test Log
    writer.add_scalar('test/with_search/episode_reward', with_search_test_output.score, total_env_steps)
    writer.add_scalar('test/no_search/episode_reward', no_search_test_output.score, total_env_steps)

    msg = '#{:<10} |'
    msg += 'WITH SEARCH [ test score: {} best score:{} ] ||'
    msg += 'NO SEARCH [ test score: {} best score:{} ]'
    msg = msg.format(total_env_steps,
                     with_search_test_output.score, best_test_score['score']['with_search'],
                     no_search_test_output.score, best_test_score['score']['no_search'])

    test_logger.info(msg)


def train(config: BaseConfig, writer: SummaryWriter):
    # create envs
    env = config.new_game(seed=config.args.seed)
    test_envs = EnvBatcher(config.new_game, 1)

    # create memory
    D = ExperienceReplay(config.args.experience_size, config.args.symbolic_env, env.observation_size,
                         env.action_size, config.args.bit_depth, config.args.device,
                         config.args.enforce_absorbing_state)

    # create networks
    model = config.get_uniform_network().to(config.args.device)
    test_model = config.get_uniform_network().to(config.args.device)
    model.train()
    test_model.eval()

    # create optimizers
    dynamics_optimizer = Adam([{'params': model.transition.parameters()},
                               {'params': model.observation.parameters()},
                               {'params': model.reward_1.parameters()},
                               {'params': model.reward_2.parameters()},
                               {'params': model.encoder.parameters()}] +
                              ([{'params': model.pcont.parameters()}] if config.args.pcont else []),
                              lr=config.args.dynamics_lr)
    value_optimizer = Adam([{'params': model.value_1.parameters()},
                            {'params': model.value_2.parameters()}], lr=config.args.value_lr)
    policy_optimizer = Adam([{'params': model.actor.parameters()}], lr=config.args.actor_lr)
    optimizer = (dynamics_optimizer, value_optimizer, policy_optimizer)

    if config.args.use_wandb:
        assert '.p' in config.model_path
        torch.save(model.state_dict(), config.model_path)
        torch.save({'model': model.state_dict(),
                    'dynamics_optimizer': dynamics_optimizer.state_dict(),
                    'value_optimizer': value_optimizer.state_dict(),
                    'policy_optimizer': policy_optimizer.state_dict()},
                   config.checkpoint_path)

        import wandb
        wandb.save(config.checkpoint_path, policy='live')
        wandb.save(config.model_path, policy='live')

    # Select Planner
    base_policy, test_base_policy = model.actor, test_model.actor  # dreamer
    if config.args.search_mode == 'no-search':
        planner, test_planner = base_policy, test_base_policy
    elif config.args.search_mode == 'rollout':
        planner = RolloutPlanner(config.args.rollout_proposal_action, config.args.rollout_uniform_action,
                                 config.args.planning_horizon, model, config.args.discount, config.args.disclam,
                                 config.args.pcont)
        test_planner = RolloutPlanner(config.args.rollout_proposal_action, config.args.rollout_uniform_action,
                                      config.args.planning_horizon, test_model, config.args.discount,
                                      config.args.disclam, config.args.pcont)
    elif config.args.search_mode == 'mpc':
        planner = MPCPlanner(env.action_size, config.args.planning_horizon, config.args.optimisation_iters,
                             config.args.candidates, config.args.top_candidates, model.transition, model.reward)
        test_planner = MPCPlanner(env.action_size, config.args.planning_horizon, config.args.optimisation_iters,
                                  config.args.candidates, config.args.top_candidates,
                                  test_model.transition, test_model.reward)
    elif 'mcts' in config.args.search_mode:
        planner = MCTS(config, model, exploration=True, progressive='progressive' in config.args.search_mode)
        test_planner = MCTS(config, test_model, exploration=False, progressive='progressive' in config.args.search_mode)
    else:
        raise ValueError('--search-mode {} is not implemented'.format(config.args.search_mode))

    # training constraints
    free_nats = torch.full((1,), config.args.free_nats, device=config.args.device)
    global_prior = Normal(torch.zeros_like(model.init_state()).to(config.args.device),
                          torch.ones_like(model.init_state()).to(config.args.device))

    # training trackers
    done = True
    total_env_steps = 0
    episodes = 0
    best_test_score = {'score': {'with_search': float('-inf'), 'no_search': float('-inf')}}

    while True:
        # Learning
        if len(D) >= (config.args.batch_size * config.args.chunk_size) and total_env_steps > config.seed_steps:
            update_params(config, model, optimizer, D, free_nats, global_prior, writer, total_env_steps)

        # Environment Interaction
        with torch.no_grad():
            for _ in range(config.env_itr_steps // config.args.action_repeat):
                if done:
                    belief = model.init_belief().to(config.args.device)
                    posterior_state = model.init_state().to(config.args.device)
                    action = model.init_action().to(config.args.device)

                    observation, done, episode_steps, episode_reward = env.reset(), False, 0, 0
                    episodes += 1

                # update belief and determine action
                belief, posterior_state = update_belief(model.transition, model.encoder, belief, posterior_state,
                                                        action, observation.to(device=config.args.device))
                action = select_action(config, env, planner, belief, posterior_state, explore=True,
                                       mode=('random' if (total_env_steps < config.seed_steps)
                                             else config.args.search_mode))

                # step in the environment
                step_action = action[0].cpu()
                step_reward = 0
                for _ in range(config.args.action_repeat):
                    next_observation, reward, done = env.step(step_action)
                    step_reward += reward
                    episode_steps += 1
                    total_env_steps += 1

                    # ########
                    # Test
                    # ########
                    # Note : This is kept inside env step for-loop to keep test intervals sync. across multiple seeds.
                    if total_env_steps % config.test_interval_steps == 0 and total_env_steps > config.seed_steps:
                        test_model.load_state_dict(model.state_dict())
                        _test(config, test_model, test_envs, test_base_policy, test_planner, writer, best_test_score,
                              total_env_steps)

                    if done:
                        break

                episode_reward += step_reward
                # add to memory
                D.append(observation, step_action, step_reward, done)
                observation = next_observation

                # ################
                # Log & save model
                # ################
                if done:
                    writer.add_scalar('train/episode_reward', episode_reward, total_env_steps)
                    writer.add_scalar('train/episode_steps', episode_steps, total_env_steps)
                    writer.add_scalar('train/episodes', episodes, total_env_steps)
                    writer.add_scalar('train/replay_memory_size', len(D), total_env_steps)

                    _msg = 'total-steps #{:<10}|| train score:{:<8.3f} eps steps: {:<10} episodes: {:<10}'
                    _msg = _msg.format(total_env_steps, episode_reward, episode_steps, episodes)
                    train_logger.info(_msg)

                    # save model
                    if (episodes % config.args.checkpoint_interval == 0) or total_env_steps >= config.max_env_steps:
                        print('model saved!')
                        assert '.p' in config.model_path
                        torch.save(model.state_dict(), config.model_path)
                        torch.save({'model': model.state_dict(),
                                    'dynamics_optimizer': dynamics_optimizer.state_dict(),
                                    'value_optimizer': value_optimizer.state_dict(),
                                    'policy_optimizer': policy_optimizer.state_dict()},
                                   config.checkpoint_path)

                        if config.args.checkpoint_experience:
                            # Warning: will fail with MemoryError with large memory sizes
                            torch.save(D, config.experiance_path)

        # check if max. env steps reached.
        if total_env_steps >= config.max_env_steps:
            train_logger.info('max env. steps reached!!')
            break

    env.close()
    test_envs.close()
