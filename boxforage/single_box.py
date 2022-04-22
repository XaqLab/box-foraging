import numpy as np
import gym
from gym.spaces import Discrete, MultiDiscrete
from irc.distributions import BaseDistribution as Distribution
from irc.distributions import DiscreteDistribution
from irc.environments import BeliefMDPEnvironment

from typing import Optional, Union
RandGen = np.random.Generator

from jarvis.utils import flatten, nest


class SingleBoxForaging(gym.Env):
    r"""Single box foraging environment.

    A minimal example of foraging experiment. Only one box exists and food will
    appear with fixed probability at each time step if it is not already there.
    The agent needs to decide whether to open the box or not based on the
    observed color cue of the box. Monochromatic cues are draw from a binomial
    distribution with probability p if food exists, and 1-p otherwise.

    """

    def __init__(self,
        *,
        env_spec: Optional[dict] = None,
        rng: Union[RandGen, int, None] = None,
    ):
        r"""
        Args
        ----
        env_spec:
            Environment specification.
        rng:
            Random generator.

        """
        self.env_spec = self._get_env_spec(**(env_spec or {}))
        self.state_space = MultiDiscrete([2]) # box state
        self.action_space = Discrete(2) # wait and fetch
        self.observation_space = MultiDiscrete([self.env_spec['box']['num_shades']+1]) # color cue

        self.rng = rng if isinstance(rng, RandGen) else np.random.default_rng(rng)

    @staticmethod
    def _get_env_spec(**kwargs):
        r"""Returns full environment specification."""
        env_spec = flatten({
            'box': {
                'num_shades': 3, 'p_appear': 0.2, 'p_cue': 0.8,
            },
            'reward': {
                'food': 10., 'fetch': -1.,
            },
        })
        for key, val in flatten(kwargs).items():
            if key in env_spec:
                env_spec[key] = val
        return nest(env_spec)

    def get_state(self):
        state = (self.has_food,)
        return state

    def set_state(self, state):
        self.has_food, = state

    def reset(self):
        self.has_food = 0
        obs = self.observe_step()
        return obs

    def step(self, action):
        reward, done = self.transition_step(action)
        obs = self.observe_step()
        info = {'state': self.get_state()}
        return obs, reward, done, info

    def transition_step(self, action):
        r"""Transition step."""
        reward, done = 0., False
        if action==0: # wait
            if self.has_food==0 and self.rng.random()<self.env_spec['box']['p_appear']:
                self.has_food = 1
        else: # fetch
            reward += self.env_spec['reward']['fetch']
            if self.has_food==1:
                reward += self.env_spec['reward']['food']
                self.has_food = 0
        return reward, done

    def observe_step(self):
        r"""Observation step."""
        color = self.rng.binomial(
            self.env_spec['box']['num_shades'],
            self.env_spec['box']['p_cue'] if self.has_food==1 else 1-self.env_spec['box']['p_cue'],
        )
        obs = (color,)
        return obs
