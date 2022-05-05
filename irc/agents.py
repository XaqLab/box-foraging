import time
import pickle
import numpy as np
import torch
from typing import Optional, Type
from collections.abc import Iterable
from stable_baselines3.ppo import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from jarvis import BaseJob
from jarvis.hashable import to_hashable
from jarvis.utils import flatten, nest, fill_defaults, time_str, progress_str, numpy_dict, tensor_dict

from . import __version__ as VERSION
from .distributions import BaseDistribution
from .models import BeliefModel
from .utils import Array, GymEnv, SB3Policy, SB3Algo


class BeliefAgent:
    r"""An agent with an internal model and uses belief for decision."""

    def __init__(self,
        model: BeliefModel,
        algo: SB3Algo,
        gamma: float = 0.99,
    ):
        r"""
        Args
        ----
        model:
            The internal model of assumed environment. Belief about the states
            is updated by it.
        algo:
            The reinfocement learning algorithm for learning the policy based on
            belief.
        gamma:
            Reward discount factor. It does not have to be the same one in `algo`
            for calculating advantages.

        """
        self.model = model
        self.algo = algo
        self.gamma = gamma

    def state_dict(self):
        r"""Returns state dictionary."""
        return {
            'model_state': self.model.state_dict(),
            'policy_state': self.algo.policy.state_dict(),
        }

    def load_state_dict(self, state):
        r"""Loads state dictionary."""
        self.model.load_state_dict(state['model_state'])
        self.algo.policy.load_state_dict(state['policy_state'])

    def _return(self, rewards):
        r"""Returns cumulative discounted reward."""
        w = self.gamma**np.flip(np.arange(len(rewards)))
        g = (w*rewards).sum()
        return g

    def evaluate(self, num_episodes=10, num_steps=40):
        r"""Evaluates current policy with respect to internal model.

        Args
        ----
        num_episodes:
            Number of evaluation episodes.

        Returns
        -------
        eval_record: dict
            Evaluation record, used for keeping track of training progress.

        """
        returns, optimalities = [], []
        for _ in range(num_episodes):
            episode = self.run_one_episode(num_steps=num_steps)
            rewards = episode['rewards']
            returns.append(self._return(rewards))
            optimalities.append(np.nanmean(episode['optimalities']))
        eval_record = {
            'num_episodes': num_episodes,
            'num_steps': num_steps,
            'returns': returns,
            'optimalities': optimalities,
            }
        return eval_record

    def run_one_episode(self,
        env: Optional[GymEnv] = None,
        num_steps: int = 40,
    ):
        r"""Runs one episode.

        Args
        ----
        env:
            The environment to interact with.
        num_steps:
            Maximum number of time steps of each episode.

        Returns
        -------
        episode: dict
            Results of one episode.

        """
        _to_restore_train = self.algo.policy.training # policy will be set to evaluation mode temporarily
        self.algo.policy.set_training_mode(False)
        actions, rewards, states, obss, beliefs = [], [], [], [], []
        optimalities, fvus = [], []
        belief, info = self.model.reset(return_info=True)
        states.append(info['state'])
        obss.append(info['obs'])
        beliefs.append(belief)
        t = 0
        while True:
            action, _ = self.algo.predict(belief)
            actions.append(action)
            belief, reward, done, info = self.model.step(action, env)
            rewards.append(reward)
            states.append(info['state'])
            obss.append(info['obs'])
            beliefs.append(belief)
            optimalities.append(self.model.p_s.est_stats['optimality'])
            fvus.append(self.model.p_s.est_stats['fvu'])
            t += 1
            if done or t==num_steps:
                break
        episode = {
            'num_steps': t,
            'actions': np.array(actions), # [0, t)
            'rewards': np.array(rewards), # [0, t)
            'states': np.array(states), # [0, t]
            'obss': np.array(obss), # [0, t]
            'beliefs': np.array(beliefs), # [0, t]
            'optimalities': np.array(optimalities),
            'fvus': np.array(fvus),
        }
        self.algo.policy.set_training_mode(_to_restore_train)
        return episode

    def query_probs(self, belief: Array, states: Array):
        r"""Returns probabilities of queried states.

        Args
        ----
        belief: (num_vars_belief)
            Parameters of state distribution.
        query_states: (num_samples, num_vars_state)
            Queried states.

        Returns
        -------
        probs: (num_samples,) Array
            Probability mass/density values of queried states.

        """
        device = self.model.p_s.get_param_vec().device
        self.model.p_s.set_param_vec(torch.tensor(belief, device=device))
        with torch.no_grad():
            probs = np.exp(self.model.p_s.loglikelihood(states).cpu().numpy())
        return probs

    def episode_likelihood(self,
        actions: Array,
        obss: Array,
        num_repeats: int = 4,
    ):
        r"""Returns the likelihood of given episode.

        Args
        ----
        actions:
            Actions taken by the agent, in [0, t).
        obss:
            Observations to the agent, in [0, t]. The last one will not be used.
        num_repeats:
            The number of sampled belief trajectories.

        Returns
        -------
        logps: (num_repeats,) Array
            Log likelihood p(actions, obss|model) for each sampled belief
            trajectory.

        """
        num_steps = len(actions)
        logps = np.zeros((num_repeats, num_steps))
        device = self.model.p_s.get_param_vec().device
        self.algo.policy.eval().to(device)
        for i in range(num_repeats):
            for t in range(len(actions)):
                obs = obss[t]
                if t==0:
                    with torch.no_grad():
                        self.model.p_s.set_param_vec(
                            self.model.p_s_o.param_net(np.array(obs)[None])[0]
                        )
                else:
                    self.model.update_belief(actions[t-1], obs)
                belief = self.model.p_s.get_param_vec()
                pi = self.algo.policy.get_distribution(belief[None].to(device))
                logps[i, t] = pi.log_prob(torch.tensor(actions[t], dtype=torch.long, device=device)).item()
        return logps.sum(axis=1)


class BeliefAgentFamily(BaseJob):
    r"""A family of belief agents.

    The class uses a folder for saving training checkpoints. Batch processing
    over a set of assumed environment parameters is implemented.

    """

    def __init__(self,
        env_class: Type[GymEnv],
        store_dir: str = 'cache',
        env_kwargs: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
        state_dist_class: Optional[Type[BaseDistribution]] = None,
        state_dist_kwargs: Optional[dict] = None,
        obs_dist_class: Optional[Type[BaseDistribution]] = None,
        obs_dist_kwargs: Optional[dict] = None,
        algo_class: Optional[Type[SB3Algo]] = None,
        algo_kwargs: Optional[dict] = None,
        policy_class: Optional[Type[SB3Policy]] = None,
        policy_kwargs: Optional[dict] = None,
        learn_kwargs: Optional[dict] = None,
        eval_interval: int = 5,
        save_interval: int = 10,
        device: str = 'cuda',
        **kwargs,
    ):
        super(BeliefAgentFamily, self).__init__(store_dir, **kwargs)
        self.env_class = env_class
        self.env_kwargs = env_kwargs or {}
        self.model_kwargs = model_kwargs or {}
        self.model_kwargs['state_dist_class'] = state_dist_class
        self.model_kwargs['state_dist_kwargs'] = state_dist_kwargs
        self.model_kwargs['obs_dist_class'] = obs_dist_class
        self.model_kwargs['obs_dist_kwargs'] = obs_dist_kwargs
        self.algo_class = algo_class or PPO
        self.algo_kwargs = algo_kwargs or {}
        self.algo_kwargs['policy'] = policy_class or ActorCriticPolicy
        self.algo_kwargs['policy_kwargs'] = policy_kwargs or {}
        self.algo_kwargs = fill_defaults(self.algo_kwargs, {'n_steps': 64, 'batch_size': 16})
        self.learn_kwargs = learn_kwargs or {}
        self.learn_kwargs = fill_defaults(self.learn_kwargs, {'total_timesteps': 256})

        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')

        self.catalog_path = f'{self.store_dir}/class_catalog.pickle'
        self._register()

    def _register(self):
        r"""Registers base configuration and class objects."""
        self._config = flatten({
            'env_class': self.env_class,
            'env_kwargs': self.env_kwargs,
            'model_kwargs': self.model_kwargs,
            'algo_class': self.algo_class,
            'algo_kwargs': self.algo_kwargs,
            'learn_kwargs': self.learn_kwargs,
        })
        try:
            with open(self.catalog_path, 'rb') as f:
                self.catalog = pickle.load(f)
        except:
            self.catalog = {}
        updated = False
        for c_key, c_val in self._config.items():
            c_str = str(c_val)
            if c_str.startswith('<class '):
                if c_str not in self.catalog:
                    updated = True
                self.catalog[c_str] = c_val
                self._config[c_key] = c_str
        self._config = nest(self._config)
        if updated:
            with open(self.catalog_path, 'wb') as f:
                pickle.dump(self.catalog, f)

    def _raw_config(self, config):
        r"""Returns the raw configuration with class objects as values."""
        config = flatten(config)
        for key, val in config.items():
            if isinstance(val, str) and val in self.catalog:
                config[key] = self.catalog[val]
        return nest(config)

    def init_agent(self, config):
        r"""Returns a new agent."""
        config = self._raw_config(config)
        env = config['env_class'](**config['env_kwargs'])
        env.set_env_param(config['env_param'])
        model = BeliefModel(env=env, **config['model_kwargs'])
        algo = config['algo_class'](env=model, **config['algo_kwargs'])
        agent = BeliefAgent(model, algo)
        return agent

    def main(self, config, num_epochs=40, verbose=1):
        r"""Trains an agent."""
        agent = self.init_agent(config)
        if verbose>0:
            print("Belief agent initialized for environment parameter:")
            print("({})".format(', '.join(['{:g}'.format(p) for p in config['env_param']])))

        try:
            epoch, ckpt = self.load_ckpt(config)
            eval_records = ckpt['eval_records']
            agent.load_state_dict(tensor_dict(ckpt['agent_state']))
            if verbose>0:
                print(f"Checkpoint ({epoch}) loaded.")
        except:
            epoch = 0
            eval_records = {}
            tic = time.time()
            _, info = agent.model.reset(return_info=True)
            toc = time.time()
            if verbose>0:
                print("Initial state distribution estimation optimality {:.1%}".format(
                    info['p_s_o_est_stats']['optimality'],
                ))
                print("Conditional observation distribution estimation optimality {:.1%}".format(
                    info['p_o_s_est_stats']['optimality'],
                ))
                print("{} elapsed.".format(time_str(toc-tic)))
            ckpt = {
                'p_s_o_est_stats': info['p_s_o_est_stats'],
                'p_o_s_est_stats': info['p_o_s_est_stats'],
            }
        t_train, t_eval = None, None
        while epoch<num_epochs:
            tic = time.time()
            agent.algo.learn(**config['learn_kwargs'], log_interval=None, reset_num_timesteps=False)
            toc = time.time()
            t_train = toc-tic if t_train is None else 0.9*t_train+0.1*(toc-tic)
            epoch += 1

            if epoch%self.eval_interval==0:
                tic = time.time()
                eval_record = agent.evaluate()
                toc = time.time()
                t_eval = toc-tic if t_eval is None else 0.9*t_eval+0.1*(toc-tic)
                eval_records[epoch] = eval_record
                if verbose>0:
                    print("Epoch {}".format(progress_str(epoch, num_epochs)))
                    print("Episode return {:.3f} ({:.2f})".format(
                        np.mean(eval_record['returns']),
                        np.std(eval_record['returns']),
                    ))
                    print("Belief update optimality {:.1%}".format(
                        np.mean(eval_record['optimalities']),
                    ))
                    print('Average training time {}/epoch, evaluation time {}'.format(
                        time_str(t_train), time_str(t_eval),
                    ))

            if epoch%self.save_interval==0 or epoch==num_epochs:
                ckpt['eval_records'] = eval_records
                ckpt['agent_state'] = numpy_dict(agent.state_dict())
                preview = {}
                self.save_ckpt(config, epoch, ckpt, preview)
                if verbose>0:
                    print(f"Checkpoint {epoch} saved.")

    def to_config(self, env_param):
        r"""Converts environment parameter to configuration."""
        return to_hashable(dict(env_param=env_param, **self._config))

    def train_agents(self,
        env_params: Iterable[Array],
        **kwargs,
    ):
        r"""Train agents for a list of environment parameters."""
        configs = (self.to_config(env_param) for env_param in env_params)
        self.batch(configs, **kwargs)

    def optimal_agent(self,
        env_param: Array,
        num_epochs: int = 0,
        verbose: int = 0,
    ):
        r"""Returns the optimal agent of given environment parameter."""
        self.train_agents([env_param], num_epochs=num_epochs, verbose=verbose, patience=0)
        config = self.to_config(env_param)
        _, ckpt = self.load_ckpt(config)
        agent = self.init_agent(config)
        agent.load_state_dict(tensor_dict(ckpt['agent_state']))
        return agent
