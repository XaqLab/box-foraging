import time
import gym
import numpy as np
from gym.spaces import Box
from .estimators import MaximumLikelihoodEstimator

from typing import Optional
from gym.spaces import Discrete
from .distributions import VarSpace
from .distributions import EnergyBasedDistribution as Distribution
RandomGenerator = np.random.Generator


class TransitModel:
    r"""Base class for transition model."""

    def __init__(self,
        state_space: VarSpace,
        action_space: Discrete,
    ):
        self.state_space = state_space
        self.action_space = action_space

    def __call__(self, state, action):
        r"""Forward pass of transition model.

        Args
        ----
        state:
            Environment state s_t.
        action:
            Agent action a_t.

        Returns
        -------
        state_dist: Distribution
            Conditional probability of next state, p(s_tp1|s_t, a_t).

        """
        raise NotImplementedError

    def reward_func(self, state, action, next_state):
        r"""Returns reward value.

        Args
        ----
        state:
            Environment state s_t.
        action:
            Agent action a_t.
        next_state:
            Environment state s_tp1.

        Returns
        reward: float
            Reward r_t.

        """
        raise NotImplementedError

    def done_func(self, state):
        r"""Returns termination status.

        Args
        ----
        state:
            Environment state s_t.

        Returns
        -------
        done: bool
            ``True`` if `state` is a termination state.

        """
        done = False # non-episodic by default
        return done


class ObsModel:
    r"""Base class for observation model."""

    def __init__(self,
        obs_space,
    ):
        self.obs_space = obs_space

    def __call__(self, state):
        r"""Forward pass of observation model.

        Args
        ----
        state:
            State sample s_t.

        Returns
        -------
        obs_dist: Distribution
            Conditional probability p(o_t|s_t).

        """
        raise NotImplementedError


class BeliefMDPEnvironment(gym.Env):
    r"""Base class for belief MDP environment.

    Agent makes decision based on the belief of environment states instead of
    the direct observation. The update of belief is based on the internal
    transition model and observation model of the agent, and will not be changed
    by reinforcement learning algorithm.

    If transition model, observation model or belief update function is not
    provided, default methods of the class will estimate them from simulators.

    """

    def __init__(self,
        state_space: VarSpace,
        action_space: Discrete,
        obs_space: VarSpace,
        belief: Distribution,
        transit_model: Optional[TransitModel] = None,
        obs_model: Optional[ObsModel] = None,
        est_spec: Optional[dict] = None,
    ):
        r"""
        Args
        ----
        state_space, action_space, obs_space:
            State, action and observation space of the environment.
        belief:
            Belief of state.
        transit_model:
            The transition model that returns probability distribution
            p(s_tp1|s_t, a_t).
        obs_model:
            The observation model that returns probability distribution
            p(o_t|s_t).
        est_spec:
            Specifications for estimating transition model, observation model or
            the belief update function.

        """
        super(BeliefMDPEnvironment, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.obs_space = obs_space
        self.belief = belief

        # set up belief space as the 'observation_space' for gym.Env
        b_param = self.belief.get_param_vec()
        self.observation_space = Box(-np.inf, np.inf, shape=b_param.shape)

        self.transit_model = transit_model
        self.obs_model = obs_model
        self.est_spec = self._get_est_spec(est_spec or {})

    @staticmethod
    def _get_est_spec(est_spec):
        r"""Returns estimation specification by filling default values."""
        _est_spec = {
            'transit': {
                # TODO set default spec
            },
            'obs': {
                # TODO set default spec
            },
            'belief': {
                'estimator': MaximumLikelihoodEstimator(),
                'num_samples': 100, # number of state samples
                'use_obs_model': True,
                'kwargs': {'num_batches': 20, 'num_epochs': 5,},
            },
        }
        for module_key in _est_spec:
            if module_key in est_spec:
                for key in _est_spec[module_key]:
                    if key in est_spec[module_key]:
                        _est_spec[module_key][key] = est_spec[module_key][key]
        return _est_spec

    def get_state(self):
        raise NotImplementedError

    def set_state(self, state):
        raise NotImplementedError

    def transit_step(self, state, action):
        r"""Simulates one step of transition.

        The default method calls `transition_model` and samples the new state
        from the conditional distribution p(s_tp1|s_t, a_t). The immediate
        reward and termination status is determiend by the sampled next state.

        This method can be overrode by child class.

        """
        state_dist = self.transit_model(state, action)
        next_state = state_dist.sample()
        reward = self.transit_model.reward_func(state, action, next_state)
        done = self.transit_model.done_func(next_state)
        return next_state, reward, done

    def obs_step(self, state):
        r"""Simulates one step of observation.

        The default method calls `obs_model` and samples the observation from
        the conditional distribution p(o_t|s_t).

        This method can be overrode by child class.

        """
        obs_dist = self.obs_model(state)
        obs = obs_dist.sample()
        return obs

    def init_belief(self, obs):
        r"""Initializes belief.

        The default method uses `obs_model` to estimate a posterior distribution
        p(s_0|o_0), and sets it as the intial belief.

        This method can be overrode by child class.

        """
        spec = self.est_spec['belief']

        states, weights = [], []
        for _ in range(spec['num_samples']):
            state = self.state_space.sample()
            states.append(state)
            obs_dist = self.obs_model(state)
            weights.append(np.exp(obs_dist.loglikelihood(obs[None]).item()))
        spec['estimator'].estimate(
                self.belief, np.stack(states), np.array(weights), **spec['kwargs'],
                )

    def update_belief(self, action, obs):
        r"""Updates belief based on action and observation.

        The default method samples next states from p(s_tp1|s_t, a_t), and
        assigns weight p(o_tp1|s_tp1) to each sample. New belief is estimated
        from these samples using specified estimator, keeping the factorization
        structure unchanged.

        This method can be overrode by child class.

        Args
        ----
        action:
            Action at time t.
        obs:
            Observation at time t+1.

        Returns
        -------
        log: dict
            Auxiliary information about the update process.

        """
        spec = self.est_spec['belief']
        log = {}

        tic = time.time()
        s_t = self.belief.sample(num_samples=spec['num_samples'])
        if spec['use_obs_model']:
            s_tp1, weights = [], []
            for _s_t in s_t:
                _s_tp1, _, _ = self.transit_step(_s_t, action)
                s_tp1.append(_s_tp1)
                _o_dist = self.obs_model(_s_tp1)
                weights.append(np.exp(_o_dist.loglikelihood(obs[None]).item()))
            spec['estimator'].estimate(
                self.belief, np.stack(s_tp1), np.array(weights), **spec['kwargs'],
                )
        else: # TODO to remove the simulator method
            s_tp1 = []
            for _s_t in s_t:
                _s_tp1, _, _ = self.transit_step(_s_t, action)
                _o_tp1 = self.obs_step(_s_tp1)
                if np.all(_o_tp1==obs):
                    s_tp1.append(_s_tp1)
            spec['estimator'].estimate(
                self.belief, np.stack(s_tp1), **spec['kwargs'],
            )
            log['num_valid_samples'] = len(s_tp1)
        toc = time.time()
        log['t_elapse'] = toc-tic
        return log

    def step(self, action):
        r"""Simulates one step of environment.

        This method can be overrode by the child class. Aside from being
        compatible with gym.Env requirements, the `info` dictionary must
        contains the new environment state at key 'state' and observation at key
        'obs'.

        Args
        ----
        action:
            Action at time t. It is supposed to be decided based on the belief
            at time t.

        Returns
        -------
        b_param: Array
            Parameter vector of the belief at time t+1, treated as the input for
            reinforcement learning algorithm.
        reward: float
            Reward at time t.
        done: bool
            Termination signal.
        info: dict
            The step information containing the state and observation at t+1.

        """
        state = self.get_state()
        state, reward, done = self.transit_step(state, action)
        self.set_state(state)
        obs = self.obs_step(state)
        log = self.update_belief(action, obs)
        b_param = self.belief.get_param_vec()
        info = {'state': state, 'obs': obs, 'log': log}
        return b_param, reward, done, info

    def reset(self, obs):
        r"""Resets environment.

        Child class needs to override this method as `reset()`, the `obs`
        returned by normal reset method will be passed to the parent method.

        Args
        ----
        obs:
            Observation returned by a normal gym environment.

        Returns
        -------
        b_param:
            Parameter vector of the initial belief.

        """
        self.init_belief(obs)
        b_param = self.belief.get_param_vec()
        return b_param
