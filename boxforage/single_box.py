import numpy as np
import gym
from gym.spaces import Discrete, MultiDiscrete
from typing import Optional, Union
RandGen = np.random.Generator

from jarvis.utils import fill_defaults


class SingleBoxForaging(gym.Env):
    r"""Single box foraging environment.

    A minimal example of foraging experiment. Food will exist in a single box
    and the agent needs to decide whether to open the box based on the
    monochromatic cue outside the box. Food appears with a fixed probability if
    it does not already exist and the agent does not open the box at this time
    step. Color cue is drawn from a binomial distribution with probability p if
    food exists, and 1-p otherwise.

    """
    D_ENV_SPEC = {
        'box': {
            'num_shades': 5, 'p_appear': 0.2, 'p_cue': 0.8,
        },
        'reward': {
            'food': 10., 'fetch': -1.,
        },
    }

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
            Random generator or seed.

        """
        self.env_spec = fill_defaults(env_spec or {}, self.D_ENV_SPEC)
        self.state_space = MultiDiscrete([2]) # box state
        self.action_space = Discrete(2) # wait and fetch
        self.observation_space = MultiDiscrete([self.env_spec['box']['num_shades']+1]) # color cue

        self.rng = rng if isinstance(rng, RandGen) else np.random.default_rng(rng)
        self.reset()

    def get_env_param(self):
        r"""Returns environment parameters."""
        env_param = (
            self.env_spec['box']['p_appear'],
            self.env_spec['box']['p_cue'],
            self.env_spec['reward']['food'],
        )
        return env_param

    def set_env_param(self, env_param):
        r"""Updates environment with parameters."""
        self.env_spec['box']['p_appear'] = env_param[0]
        self.env_spec['box']['p_cue'] = env_param[1]
        self.env_spec['reward']['food'] = env_param[2]

    def get_state(self):
        r"""Returns environment state."""
        state = (self.has_food,)
        return state

    def set_state(self, state):
        r"""Sets environment state."""
        self.has_food, = state

    def reset(self):
        self.has_food = 0
        return self.observe_step()

    def step(self, action):
        reward, done = self.transition_step(action)
        obs = self.observe_step()
        info = {}
        return obs, reward, done, info

    def transition_step(self, action):
        r"""Runs one transition step."""
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
        r"""Runs one observe step."""
        color = self.rng.binomial(
            self.env_spec['box']['num_shades'],
            self.env_spec['box']['p_cue'] if self.has_food==1 else 1-self.env_spec['box']['p_cue'],
        )
        obs = (color,)
        return obs
