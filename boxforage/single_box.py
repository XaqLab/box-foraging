import numpy as np
from gym.spaces import Discrete, MultiDiscrete
from irc.distributions import EnergyBasedDistribution as Distribution
from irc.environments import BeliefMDPEnvironment

from typing import Optional, TypeVar
RandomGenerator = np.random.Generator

from jarvis.utils import flatten, nest


class SingleBoxForaging(BeliefMDPEnvironment):

    def __init__(self,
        *,
        env_spec: Optional[dict] = None,
        belief_class: Optional[TypeVar[Distribution]] = None,
        rng: Optional[RandomGenerator] = None,
    ):
        self.env_spec = self._get_env_spec(**env_spec)
        state_space, action_space, obs_space = self._get_spaces(self.env_spec)

        super(SingleBoxForaging, self).__init__(
            state_space, action_space, obs_space, belief_class,
            rng=rng,
            )

    @staticmethod
    def _get_env_spec(**kwargs):
        r"""Returns full environment specification."""
        env_spec = flatten({
            'box': {
                'num_grades': 3, 'p_appear': 0.2, 'p_cue': 0.8,
            },
            'reward': {
                'food': 10., 'fetch': -1.,
            },
        })
        for key, val in flatten(kwargs).items():
            if key in env_spec:
                env_spec[key] = val
        return nest(env_spec)

    @staticmethod
    def _get_spaces(env_spec):
        r"""Returns gym spaces."""
        state_space = MultiDiscrete([2]) # box state
        action_space = Discrete(2) # wait and fetch
        obs_space = MultiDiscrete([env_spec['box']['num_grades']+1]) # color cue
        return state_space, action_space, obs_space

    def transition_step(self, state, action):
        r"""Transition step."""
        reward, done = 0., False
        has_food, = state
        if action==0: # wait
            if has_food==0 and self.rng.random()<self.env_spec['box']['p_appear']:
                has_food = 1
        else: # fetch
            reward += self.env_spec['reward']['fetch']
            if has_food==1:
                reward += self.env_spec['reward']['food']
                has_food = 0
        next_state = (has_food,)
        return next_state, reward, done

    def observe_step(self, state):
        r"""Observation step."""
        has_food, = state
        color = self.rng.binomial(
            self.env_spec['box']['num_grades'],
            self.env_spec['box']['p_cue'] if has_food==1 else 1-self.env_spec['box']['p_cue'],
        )
        obs = (color,)
        return obs
