import numpy as np
import gym

from typing import Optional, Type, Union
from gym.spaces import MultiDiscrete, Box
from stable_baselines3.ppo import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from jarvis.utils import flatten, nest

from .distributions import BaseDistribution, DiscreteDistribution
from .utils import Tensor, RandGen, GymEnv, SB3Policy, SB3Algo

class BeliefModel(gym.Env):
    r"""Base class for internal belief model.

    Agent makes decision based on the belief of environment states instead of
    the direct observation. The update of belief is based on the assumed
    environment by the agent, and will not be changed by reinforcement learning
    algorithm.

    """

    def __init__(self,
        env: GymEnv,
        belief_class: Optional[Type[BaseDistribution]] = None,
        belief_kwargs: Optional[dict] = None,
        b_param_init: Optional[Tensor] = None,
        p_o_s: Optional[BaseDistribution] = None,
        est_spec: Optional[dict] = None,
        rng: Union[RandGen, int, None] = None,
    ):
        r"""
        Args
        ----
        env:
            The assumed environment, with a few utility methods implemented,
            such as `get_state`, `set_state` etc. Detailed requirements are
            specified in 'README.md'.
        belief_class, belief_kwargs:
            Class and key-word arguments for belief.
        b_param_init:
            Initial parameters of belief. If not provided, will be estimated
            using the assumed environment.
        p_o_s:
            Conditional distribution p(o|s) of observation given environment
            state. If not provided, will be estimated using the assumed
            environment.
        est_spec:
            Specifications for estimating state prior p(s), observation
            conditional distribution p(o|s) and parameter of the new belief.
        rng:
            Random generator.

        """
        self.env = env
        self.action_space = self.env.action_space

        if belief_class is None:
            if isinstance(self.env.state_space, MultiDiscrete):
                belief_class = DiscreteDistribution
            if isinstance(self.env.state_space, Box):
                raise NotImplementedError
        self.belief = belief_class(self.env.state_space, **(belief_kwargs or {}))
        self.belief_space = Box(-np.inf, np.inf, shape=self.belief.get_param_vec().shape)
        self.observation_space = self.belief_space

        self.est_spec = self._get_est_spec(**(est_spec or {}))
        if b_param_init is None:
            self.b_param_init = self.estimate_state_prior()
        else:
            assert b_param_init.shape==self.belief.get_param_vec().shape
        self.p_o_s = self.estimate_obs_conditional() if p_o_s is None else p_o_s

        self.rng = rng if isinstance(rng, RandGen) else np.random.default_rng(rng)

    @staticmethod
    def _get_est_spec(**kwargs):
        r"""Returns full estimation specification."""
        est_spec = flatten({
            'state_prior': {
                'num_samples': 1000,
                'optim_kwargs': {
                    'batch_size': 32, 'num_epochs': 10,
                    'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4,
                    'verbose': 1,
                },
            },
            'obs_conditional': {
                'dist_class': None, 'dist_kwargs': None,
                'num_samples': 10000,
                'optim_kwargs': {
                    'batch_size': 32, 'num_epochs': 10,
                    'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4,
                    'verbose': 1,
                },
            },
            'belief': {
                'num_samples': 100,
                'optim_kwargs': {
                    'batch_size': 32, 'num_epochs': 20,
                    'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4,
                    'verbose': 0,
                },
            }
        })
        for key, val in flatten(kwargs).items():
            if key in est_spec:
                est_spec[key] = val
        return nest(est_spec)

    def estimate_state_prior(self):
        r"""Estimates initial belief."""
        # TODO consider change it to p(s|o)
        # use env to collect initial states
        _state_to_restore = self.env.get_state()
        states = []
        for _ in range(self.est_spec['state_prior']['num_samples']):
            self.env.reset()
            states.append(self.env.get_state())
        self.env.set_state(_state_to_restore)
        # estimate the initial belief
        if self.est_spec['state_prior']['optim_kwargs']['verbose']>0:
            print("estimating prior distribution p(s)")
        self.belief.estimate(
            xs=np.array(states), **self.est_spec['state_prior']['optim_kwargs'],
        )
        return self.belief.get_param_vec().clone()

    def estimate_obs_conditional(self):
        r"""Estimates conditional distribution of observation."""
        # use env to collect state/observation pairs
        _state_to_restore = self.env.get_state()
        obss, states = [], []
        for _ in range(self.est_spec['obs_conditional']['num_samples']):
            # TODO specify state and action distributions
            state = self.env.state_space.sample()
            self.env.set_state(state)
            action = self.env.action_space.sample()
            obs, *_ = self.env.step(action)
            obss.append(obs)
            states.append(self.env.get_state())
        self.env.set_state(_state_to_restore)
        # estimate the conditional distribution
        if self.est_spec['obs_conditional']['optim_kwargs']['verbose']>0:
            print("estimating conditional distribution p(o|s)")
        if self.est_spec['obs_conditional']['dist_class'] is None:
            if isinstance(self.env.observation_space, MultiDiscrete):
                dist_class = DiscreteDistribution
            if isinstance(self.env.observation_space, Box):
                raise NotImplemented
        else:
            dist_class = self.est_spec['obs_conditional']['dist_class']
        dist = dist_class(
            x_space=self.env.observation_space, y_space=self.env.state_space,
            **(self.est_spec['obs_conditional']['dist_kwargs'] or {}),
        )
        dist.estimate(
            np.array(obss), np.array(states),
            **self.est_spec['obs_conditional']['optim_kwargs'],
            )
        return dist

    def reset(self, env=None):
        if env is None:
            env = self.env
        env.reset()
        self.belief.set_param_vec(self.b_param_init)
        # TODO append env_param optionally
        return self.b_param_init.cpu().numpy()

    def step(self, action, env=None):
        r"""Runs one step.

        Args
        ----
        action:
            Action at time t.
        env:
            The actual environment to interact with. It can differ from the
            assumed environment `self.env`, though the latter is always used for
            belief updating.

        """
        if env is None:
            env = self.env
        obs, reward, done, info = env.step(action)
        info.update({
            'obs': obs, 'state': env.get_state(),
        })
        self.update_belief(action, obs)
        return self.belief.get_param_vec().cpu().numpy(), reward, done, info

    def update_belief(self, action, obs):
        r"""Updates belief.

        Draw s_t samples from current belief, use env to get s_tp1 samples.
        These samples are weighted according to estimated conditional
        distribution p(o|s), and used to estimate the new belief.

        Args
        ----
        action:
            Action at time t.
        obs:
            Observation at time t+1.

        """
        _state_to_restore = self.env.get_state()
        states, weights = [], []
        for _ in range(self.est_spec['belief']['num_samples']):
            state = self.belief.sample()
            self.env.set_state(state)
            self.env.step(action)
            next_state = self.env.get_state()
            states.append(next_state)
            weights.append(np.exp(self.p_o_s.loglikelihood(
                np.array(obs)[None], np.array(next_state)[None]).item()))
        self.env.set_state(_state_to_restore)
        self.belief.estimate( # TODO add running statistics for inspection
            xs=np.array(states), ws=np.array(weights),
            **self.est_spec['belief']['optim_kwargs'],
        )

    def train_agent(self,
        policy_class: Optional[Type[SB3Policy]] = None,
        policy_kwargs: Optional[dict] = None,
        algo_class: Optional[Type[SB3Algo]] = None,
        algo_kwargs: Optional[dict] = None,
        learn_kwargs: Optional[dict] = None,
        verbose: int = 1,
    ):
        if policy_class is None:
            policy_class = ActorCriticPolicy
        if policy_kwargs is None:
            policy_kwargs = {}
        if algo_class is None:
            algo_class = PPO
        if algo_kwargs is None:
            algo_kwargs = {'n_steps': 128, 'batch_size': 32}
        if learn_kwargs is None:
            learn_kwargs = {'total_timesteps': 5120, 'log_interval': 10}

        algo = algo_class(
            policy=(policy_class or ActorCriticPolicy), env=self,
            policy_kwargs=(policy_kwargs or {}),
            verbose=verbose, **algo_kwargs,
        )
        algo.learn(**learn_kwargs)
        return algo

    def run_one_trial(self,
        *,
        algo: Optional[SB3Algo] = None,
        env: Optional[GymEnv] = None,
        num_steps: int = 40,
    ):
        r"""Runs one trial.

        Args
        ----
        algo:
            An RL algorithm compatible with stable-baselines3.
        env:
            The actual environment to interact with.
        num_steps:
            Number of time steps of one trial.

        Returns
        -------
        trial: dict
            A dictionary containing actions, rewards, states, observations and
            beliefs in one trial.

        """
        if algo is not None:
            algo.policy.set_training_mode(False)
        actions, rewards, states, obss, beliefs = [], [], [], [], []
        b_param = self.reset()
        for _ in range(num_steps): # TODO deal with episodic environment
            if algo is None:
                action = self.action_space.sample()
            else:
                action, _ = algo.predict(b_param)
            actions.append(action)

            belief, reward, _, info = self.step(action, env)
            rewards.append(reward)
            states.append(info['state'])
            obss.append(info['obs'])
            beliefs.append(belief)
        trial = {
            'num_steps': num_steps,
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'states': np.array(states),
            'obss': np.array(obss),
            'beliefs': np.array(beliefs),
        }
        return trial
