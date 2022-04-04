import gym
import numpy as np

from typing import Union, Optional
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

    """

    def __init__(self,
        state_space: VarSpace,
        action_space: Discrete,
        obs_space: VarSpace,
        transit_model: Optional[TransitModel] = None,
        obs_model: Optional[ObsModel] = None,
    ):
        super(BeliefMDPEnvironment, self).__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.obs_space = obs_space

        self.transit_model = transit_model
        self.obs_model = obs_model

    def get_state(self):
        raise NotImplementedError

    def set_state(self):
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

    def step(self, action):
        r"""Simulates one step of environment.

        This method can be overrode by the child class. Aside from being
        compatible with gym.Env requirements, the `info` dictionary must
        contains the environment state at key 'state'.

        """
        state = self.get_state()
        state, reward, done = self.transit_step(state, action)
        obs = self.obs_step(state)
        self.set_state(state)
        info = {'state': state}
        return obs, reward, done, info

    def reset(self):
        r"""Resets environment."""
        raise NotImplementedError
