import time
import pickle
import numpy as np
from typing import Optional, Type
from collections.abc import Iterable
from stable_baselines3.ppo import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from jarvis import BaseJob
from jarvis.hashable import to_hashable
from jarvis.utils import flatten, nest, time_str, progress_str, numpy_dict, tensor_dict

from . import __version__ as VERSION
from .distributions import BaseDistribution
from .models import BeliefModel
from .utils import Array, GymEnv, SB3Policy, SB3Algo


class BeliefAgent:

    def __init__(self,
        model: BeliefModel,
        algo: SB3Algo,
        gamma: float = 0.99,
    ):
        self.model = model
        self.algo = algo
        self.gamma = gamma

    def episode_likelihood(self,
        actions: Array,
        obss: Array,
    ):
        raise NotImplementedError

    def state_dict(self):
        return {
            'model_state': self.model.state_dict(),
            'policy_state': self.algo.policy.state_dict(),
        }

    def _return(self, rewards):
        w = self.gamma**np.flip(np.arange(len(rewards)))
        g = (w*rewards).sum()
        return g

    def load_state_dict(self, state):
        self.model.load_state_dict(state['model_state'])
        self.algo.policy.load_state_dict(state['policy_state'])

    def evaluate(self, num_episodes=10, num_steps=80):
        returns = []
        for _ in range(num_episodes):
            episode = self.run_one_episode(num_steps=num_steps)
            rewards = episode['rewards']
            returns.append(self._return(rewards))
        return {
            'num_episodes': num_episodes,
            'num_steps': num_steps,
            'returns': returns,
            }

    def run_one_episode(self,
        env: Optional[GymEnv] = None,
        num_steps: int = 40,
    ):
        _to_restore_train = self.algo.policy.training # policy will be set to evaluation mode temporarily
        self.algo.policy.set_training_mode(False)
        actions, rewards, states, obss, beliefs = [], [], [], [], []
        belief = self.model.reset()
        t = 0
        while True:
            action, _ = self.algo.predict(belief)
            actions.append(action)
            belief, reward, done, info = self.model.step(action, env)
            rewards.append(reward)
            states.append(info['state'])
            obss.append(info['obs'])
            beliefs.append(belief)
            t += 1
            if done or t==num_steps:
                break
        episode = {
            'num_steps': t,
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'states': np.array(states),
            'obss': np.array(obss),
            'beliefs': np.array(beliefs),
        }
        self.algo.policy.set_training_mode(_to_restore_train)
        return episode


class BeliefAgentFamily(BaseJob):

    def __init__(self,
        env_class: Type[GymEnv],
        store_dir: str = 'cache',
        env_kwargs: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
        belief_class: Optional[Type[BaseDistribution]] = None,
        belief_kwargs: Optional[dict] = None,
        algo_class: Optional[Type[SB3Algo]] = None,
        algo_kwargs: Optional[dict] = None,
        policy_class: Optional[Type[SB3Policy]] = None,
        policy_kwargs: Optional[dict] = None,
        learn_kwargs: Optional[dict] = None,
        eval_interval: int = 5,
        save_interval: int = 10,
        **kwargs,
    ):
        super(BeliefAgentFamily, self).__init__(store_dir, **kwargs)
        self.env_class = env_class
        self.env_kwargs = env_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.model_kwargs['belief_class'] = belief_class
        self.model_kwargs['belief_kwargs'] = belief_kwargs or {}
        self.algo_class = algo_class or PPO
        self.algo_kwargs = algo_kwargs or {}
        self.algo_kwargs['policy'] = policy_class or ActorCriticPolicy
        self.algo_kwargs['policy_kwargs'] = policy_kwargs or {}
        self._fill_default_vals(
            self.algo_kwargs,
            n_steps=64, batch_size=16,
            )
        self.learn_kwargs = learn_kwargs or {}
        self._fill_default_vals(
            self.learn_kwargs,
            total_timesteps=256,
            )
        self.eval_interval = eval_interval
        self.save_interval = save_interval

        self.catalog_path = f'{self.store_dir}/class_catalog.pickle'
        self._register_classes()

    @staticmethod
    def _fill_default_vals(spec, **kwargs):
        for key, val in kwargs.items():
            if key not in spec:
                spec[key] = val

    def _register_classes(self):
        try:
            with open(self.catalog_path, 'rb') as f:
                self.catalog = pickle.load(f)
        except:
            self.catalog = {}
        updated = False
        self._config = flatten({
            'env_class': self.env_class,
            'env_kwargs': self.env_kwargs,
            'model_kwargs': self.model_kwargs,
            'algo_class': self.algo_class,
            'algo_kwargs': self.algo_kwargs,
            'learn_kwargs': self.learn_kwargs,
        })
        for c_key, c_val in self._config.items():
            c_str = str(c_val)
            if c_str.startswith('<class '):
                self.catalog[c_str] = c_val
                updated = True
                self._config[c_key] = c_str
        self._config = nest(self._config)
        if updated:
            with open(self.catalog_path, 'wb') as f:
                pickle.dump(self.catalog, f)

    def _raw_config(self, config):
        if isinstance(config, dict): # only dict object are recreated
            raw_config = {}
            for key, val in config.items():
                if isinstance(val, str) and val in self.catalog:
                    raw_config[key] = self.catalog[val]
                else:
                    raw_config[key] = self._raw_config(val)
            return raw_config
        else:
            return config

    def init_agent(self, config):
        config = self._raw_config(config)
        env = config['env_class'](**config['env_kwargs'])
        env.set_env_param(config['env_param'])
        model = BeliefModel(env=env, **config['model_kwargs'])
        algo = config['algo_class'](env=model, **config['algo_kwargs'])
        agent = BeliefAgent(model, algo)
        return agent

    def main(self, config, num_epochs, verbose=1):
        agent = self.init_agent(config)
        if verbose>0:
            print("Belief agent initiated for environment parameter:")
            print("({})".format(', '.join(['{:g}'.format(p) for p in config['env_param']])))

        try:
            ckpt = self.load_ckpt(config, verbose=0)
            epoch = ckpt['epoch']
            train_times = ckpt['train_times']
            eval_records = ckpt['eval_records']
            agent_state = ckpt['agent_state']
            agent.load_state_dict(tensor_dict(agent_state))
        except:
            epoch = 0
            train_times = []
            eval_records = []
        while epoch<num_epochs:
            tic = time.time()
            agent.algo.learn(**config['learn_kwargs'], log_interval=None, reset_num_timesteps=False)
            # TODO check algo loggers for info to save
            epoch += 1
            toc = time.time()
            train_times.append(toc-tic)

            if epoch%self.eval_interval==0:
                eval_record = agent.evaluate()
                eval_records.append(eval_record)
                if verbose>0:
                    print("Epoch {}".format(progress_str(epoch, num_epochs)))
                    print("Episode return {:.3f} ({:.2f})".format(
                        np.mean(eval_record['returns']),
                        np.std(eval_record['returns']),
                    ))
                    print("{} per epoch".format(time_str(np.mean(train_times))))

            if epoch%self.save_interval==0 or epoch==num_epochs:
                ckpt = {
                    'train_times': train_times,
                    'eval_records': eval_records,
                    'agent_state': numpy_dict(agent.state_dict()),
                }
                preview = {}
                self.save_ckpt(config, epoch, ckpt, preview)
                if verbose>0:
                    print(f"Checkpoint {epoch} saved.")

    def to_config(self, env_param):
        return to_hashable(dict(env_param=env_param, **self._config))

    def train_agents(self,
        env_params: Iterable[Array],
        **kwargs,
    ):
        configs = (self.to_config(env_param) for env_param in env_params)
        self.batch(configs, **kwargs)

    def optimal_agent(self,
        env_param: Array,
        num_epochs: int = 48,
        verbose: int = 0,
    ):
        ckpt, _ = self.main(self.to_config(env_param), num_epochs, verbose)
        agent = self.init_agent(self.to_config(env_param))
        agent.load_state_dict(tensor_dict(ckpt['agent_state']))
        return agent
