import os

from core.model import DreamerNetwork


class BaseConfig(object):
    def __init__(self):
        pass

    def new_game(self, seed=None):
        raise NotImplementedError

    def get_uniform_network(self):
        return DreamerNetwork(obs_size=self.observation_size,
                              belief_size=200,
                              state_size=30,
                              hidden_size=200,
                              embedding_size=200,
                              action_size=self.action_size,
                              sample_random_action_fn=self.sample_random_action,
                              symbolic=self.args.symbolic_env)

    def get_hparams(self):
        hparams = {k: v for k, v in vars(self.args).items() if v is not None}
        for k, v in self.__dict__.items():
            if 'path' not in k and 'args' not in k and 'sample_random_action' not in k and (v is not None):
                hparams[k] = v
        return hparams

    def set_config(self, args):
        self.args = args

        # paths
        self.exp_path = os.path.join(args.results_dir, args.case, args.env, str(hash(frozenset(vars(args).items()))))
        self.logs_path = os.path.join(self.exp_path, 'logs')
        self.model_path = os.path.join(self.exp_path, 'model.p')
        self.best_model_path = {'with_search': os.path.join(self.exp_path, 'with_search_best_model.p'),
                                'no_search': os.path.join(self.exp_path, 'no_search_best_model.p')}
        self.checkpoint_path = os.path.join(self.exp_path, 'checkpoint.p')
        os.makedirs(self.logs_path, exist_ok=True)

        # store env attributes
        env = self.new_game()
        self.observation_size = env.observation_size
        self.action_size = env.action_size
        self.sample_random_action = env.sample_random_action
        # env.close()
