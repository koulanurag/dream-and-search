import argparse
import logging.config
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from core.env import EnvBatcher
from core.planner import MPCPlanner, RolloutPlanner, MCTS
from core.test import test
from core.train import train
from core.utils import init_logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Performing Search for Control with Dreamer')
    parser.add_argument('--search-mode', type=str, default='no-search', help='Search Mode to be used for planning',
                        choices=['no-search', 'rollout', 'mcts+fixed', 'mcts+progressive', 'mpc'])
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
    parser.add_argument('--no-cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--env', type=str, default='cartpole-balance', help='Gym/Control Suite environment')
    parser.add_argument('--case', type=str, default='dm_control', choices=['dm_control','cassie', 'box2d', 'classic_control'],
                        help="It's used for switching between different domains(default: %(default)s)")
    parser.add_argument('--results-dir', type=Path, default=os.path.join(os.getcwd(), 'results'),
                        help="Directory Path to store results (default: %(default)s)")
    parser.add_argument('--wandb-dir', type=Path, default=os.path.join(os.getcwd(), 'wandb'),
                        help="Directory Path to store results (default: %(default)s)")
    parser.add_argument('--opr', required=True, choices=['train', 'test'], help='operation to be performed')
    parser.add_argument('--symbolic-env', action='store_true', help='Symbolic features')
    parser.add_argument('--max-episode-length', type=int, default=1000, metavar='T', help='Max episode length')
    # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
    parser.add_argument('--experience-size', type=int, default=1000000, metavar='D', help='Experience replay size')
    # Note that the default encoder for visual observations outputs a 1024D vector;
    # for other embedding sizes an additional fully-connected layer is used
    parser.add_argument('--embedding-size', type=int, default=1024, metavar='E', help='Observation embedding size')
    parser.add_argument('--hidden-size', type=int, default=200, metavar='H', help='Hidden size')
    parser.add_argument('--belief-size', type=int, default=200, metavar='H', help='Belief/hidden size')
    parser.add_argument('--state-size', type=int, default=30, metavar='Z', help='State/latent size')
    parser.add_argument('--action-repeat', type=int, default=2, metavar='R', help='Action repeat')
    parser.add_argument('--action-noise', type=float, default=0.3, metavar='ε', help='Action noise')
    parser.add_argument('--collect-interval', type=int, default=100, metavar='C', help='Collect interval')
    parser.add_argument('--batch-size', type=int, default=50, metavar='B', help='Batch size')
    parser.add_argument('--chunk-size', type=int, default=50, metavar='L', help='Chunk size')
    parser.add_argument('--worldmodel-LogProbLoss', action='store_true',
                        help='use LogProb loss for observation_model and reward_model training')
    parser.add_argument('--global-kl-beta', type=float, default=0, metavar='βg', help='Global KL weight (0 to disable)')
    parser.add_argument('--free-nats', type=float, default=3.0, metavar='F', help='Free nats')
    parser.add_argument('--bit-depth', type=int, default=5, metavar='B', help='Image bit depth (quantisation)')
    parser.add_argument('--dynamics_lr', type=float, default=1e-3, metavar='α', help='Learning rate for Dynamics')
    parser.add_argument('--actor_lr', type=float, default=8e-5, metavar='α', help='Learning rate for Actor')
    parser.add_argument('--value_lr', type=float, default=8e-5, metavar='α', help='Learning rate for Value')
    parser.add_argument('--learning-rate-schedule', type=int, default=0, metavar='αS',
                        help='Linear learning rate schedule (optimisation steps from 0 to final learning rate;'
                             ' 0 to disable)')
    parser.add_argument('--adam-epsilon', type=float, default=1e-7, metavar='ε', help='Adam optimizer epsilon value')
    # Note that original has a linear learning rate decay,
    # but it seems unlikely that this makes a significant difference
    parser.add_argument('--grad-clip-norm', type=float, default=100.0, metavar='C', help='Gradient clipping norm')
    parser.add_argument('--planning-horizon', type=int, default=15, metavar='H', help='Planning horizon distance')
    parser.add_argument('--discount', type=float, default=0.99, metavar='H', help='Planning horizon distance')
    parser.add_argument('--disclam', type=float, default=0.95, metavar='H', help='discount rate to compute return')
    parser.add_argument('--optimisation-iters', type=int, default=10, metavar='I',
                        help='Planning optimisation iterations')
    parser.add_argument('--candidates', type=int, default=1000, metavar='J', help='Candidate samples per iteration')
    parser.add_argument('--top-candidates', type=int, default=100, metavar='K', help='Number of top candidates to fit')
    parser.add_argument('--test-interval', type=int, default=25, metavar='I', help='Test interval (episodes)')
    parser.add_argument('--test-episodes', type=int, default=10, metavar='E', help='Number of test episodes')
    parser.add_argument('--checkpoint-interval', type=int, default=50, metavar='I',
                        help='Checkpoint interval (episodes)')
    parser.add_argument('--checkpoint-experience', action='store_true', help='Checkpoint experience replay')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--use-wandb', action='store_true', default=False,
                        help='Use Weight and bias visualization lib for logging. (default: %(default)s)')
    parser.add_argument('--pcont', action='store_true', default=False,
                        help=' Learning the discount factor. (default: %(default)s)')
    parser.add_argument('--pcont-scale', type=float, default=10.0, help='Scale for pcont')
    parser.add_argument('--rollout-uniform-action', type=int, default=100,
                        help='No. of uniform actions to be sampled for rollout planning (default: %(default)s)')
    parser.add_argument('--rollout-proposal-action', type=int, default=50,
                        help='No. of proposal actions to be sampled for rollout planning (default: %(default)s)')
    parser.add_argument('--mcts-fixed-proposal-action', type=int, default=20,
                        help='No. of proposal actions to be sampled for mcts for each child. '
                             'This is applying only for mcts+fixed mode(default: %(default)s)')
    parser.add_argument('--mcts-num-simulations', type=int, default=50,
                        help='No. of proposal actions to be sampled for rollout planning (default: %(default)s)')
    parser.add_argument('--mcts-cpw', type=float, default=1.0,
                        help='MCTS cpw for progressive mode (default: %(default)s)')
    parser.add_argument('--mcts-alpha', type=float, default=0.5,
                        help='MCTS alpha for progressive mode (default: %(default)s)')
    parser.add_argument('--pb-c-init', type=float, default=1.25,
                        help='MCTS ucb estimation attribute (default: %(default)s)')
    parser.add_argument('--pb-c-base', type=float, default=19652,
                        help='MCTS ucb estimation attribute (default: %(default)s)')
    parser.add_argument('--root-dirichlet-alpha', type=float, default=0.25,
                        help='Exploration noise for root of MCTS (default: %(default)s)')
    parser.add_argument('--root-exploration-fraction', type=float, default=0.25,
                        help='Exploratiom fraction for root of MCTS (default: %(default)s)')

    # Process arguments
    args = parser.parse_args()
    args.device = 'cuda' if (not args.no_cuda) and torch.cuda.is_available() else 'cpu'

    # import corresponding configuration , neural networks and envs
    if args.case == 'classic_control':
        from config.classic_control import run_config
    elif args.case == 'box2d':
        from config.box2d import run_config
    elif args.case == 'cassie':
        from config.cassie import run_config
    elif args.case == 'dm_control':
        from config.dm_control import run_config
    else:
        raise Exception('Invalid --case option.')

    # seeding random iterators
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    # set config as per cmd arguments
    run_config.set_config(args)

    # set-up logger
    init_logger(run_config.logs_path)
    logging.getLogger('root').info('cmd args:{}'.format(' '.join(sys.argv[1:])))  # log command line arguments.

    try:
        if args.opr == 'train':
            if args.use_wandb:
                import wandb

                wandb.init(dir=args.wandb_dir, group=args.case + ':' + args.env, project="dream-and-search",
                           config=run_config.get_hparams(), sync_tensorboard=True)

            summary_writer = SummaryWriter(run_config.logs_path, flush_secs=60 * 1)  # flush every 1 minutes
            train(run_config, summary_writer)
            summary_writer.flush()
            summary_writer.close()

            if args.use_wandb:
                wandb.join()
        elif args.opr == 'test':
            model_path = run_config.model_path
            assert os.path.exists(model_path), 'model not found: {}'.format(model_path)

            model = run_config.get_uniform_network()
            model = model.to('cpu')
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            env_batch = EnvBatcher(run_config.new_game, args.test_episodes)

            base_policy = model.actor  # dreamer
            if run_config.args.search_mode == 'no-search':
                planner = base_policy
            elif run_config.args.search_mode == 'rollout':
                planner = RolloutPlanner(run_config.args.rollout_proposal_action,
                                         run_config.args.rollout_uniform_action,
                                         run_config.args.planning_horizon, model, run_config.args.discount,
                                         run_config.args.disclam)
            elif run_config.args.search_mode == 'mpc':
                planner = MPCPlanner(run_config.action_size, run_config.args.planning_horizon,
                                     run_config.args.optimisation_iters, run_config.args.candidates,
                                     run_config.args.top_candidates, model.transition, model.reward)
            elif 'mcts' in run_config.args.search_mode:
                planner = MCTS(run_config, model, exploration=False, progressive='progressive')
            else:
                raise NotImplementedError

            test_output = test(run_config, env_batch, model, planner, render=args.render, mode=args.search_mode)
            env_batch.close()
            logging.getLogger('test').info('Test Score: {}'.format(test_output.score))
        else:
            raise NotImplementedError('"--opr {}" is not implemented ( or invalid)'.format(args.opr))

    except Exception as e:
        logging.getLogger('root').error(e, exc_info=True)

    logging.shutdown()
