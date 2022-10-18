from itertools import product
import numpy as np
import gym
from gym.spaces import Discrete, MultiDiscrete
from typing import Optional, Union
RandGen = np.random.Generator

from jarvis.config import Config


class MultiBoxForaging(gym.Env):
    r"""Multiple boxes foraging experiment.

    Multiple boxes are placed in the room, and food can appear or vanish at
    fixed probabilities at each time step. The agent needs to travel to
    locations of boxes to open and fetch the food. In addition, the agent can
    also travel to the location 'CENTER', which is equally distanced to each
    box. Food availability of each box is stochastically indicated by the
    monochromatic color cue outside the box, which is drawn from different
    binomial distributions based on box state.

    """
    D_ENV_SPEC = {
        'boxes': {
            'num_boxes': 2, 'num_shades': 5,
            'p_appear': 0.2, 'p_vanish': 0.05,
            'p_true': 0.8, 'p_false': 0.2,
        },
        'reward': {
            'food': 10., 'fetch': -2.,
            'move_box': -1., 'move_center': -0.5,
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
        self.env_spec = Config(env_spec).fill(self.D_ENV_SPEC)
        self.num_boxes = self.env_spec['boxes']['num_boxes']
        for key in ['p_appear', 'p_vanish', 'p_true', 'p_false']:
            self.env_spec['boxes'][key] = self._get_array(self.env_spec['boxes'][key], self.num_boxes)
        self.env_spec['reward']['move'] = np.concatenate([
            self._get_array(self.env_spec['reward']['move_box'], self.num_boxes),
            np.array([self.env_spec['reward']['move_center']]),
        ])

        self.state_space = MultiDiscrete( # box states and agent position
            [2]*self.num_boxes+[self.num_boxes+1]
        )
        self.action_space = Discrete(self.num_boxes+2) # move and fetch
        self.observation_space = MultiDiscrete( # color cues and agent position
            [self.env_spec['boxes']['num_shades']+1]*self.num_boxes+[self.num_boxes+1]
        )

        self.rng = rng if isinstance(rng, RandGen) else np.random.default_rng(rng)
        self.reset()

    @staticmethod
    def _get_array(val, n):
        r"""Returns an array of desired length."""
        if isinstance(val, float):
            return np.ones(n)*val
        else:
            assert len(val)==n
            return np.array(val)

    def get_env_param(self):
        r"""Returns environment parameters."""
        env_param = tuple(np.concatenate([
            self.env_spec['boxes']['p_appear'],
            self.env_spec['boxes']['p_vanish'],
            self.env_spec['boxes']['p_true'],
            self.env_spec['boxes']['p_false'],
            [self.env_spec['reward']['food']],
            self.env_spec['reward']['move'],
        ]))
        return env_param

    def set_env_param(self, env_param):
        r"""Updates environment with parameters."""
        c_p, n_p = 0, self.num_boxes
        env_param = np.array(env_param)
        self.env_spec['boxes']['p_appear'] = env_param[c_p:c_p+n_p]
        c_p += n_p
        self.env_spec['boxes']['p_vanish'] = env_param[c_p:c_p+n_p]
        c_p += n_p
        self.env_spec['boxes']['p_true'] = env_param[c_p:c_p+n_p]
        c_p += n_p
        self.env_spec['boxes']['p_false'] = env_param[c_p:c_p+n_p]
        c_p += n_p
        self.env_spec['reward']['food'] = env_param[c_p]
        c_p += 1
        n_p = self.num_boxes+1
        self.env_spec['reward']['move'] = env_param[c_p:c_p+n_p]

    def get_state(self):
        r"""Returns environment state."""
        state = (*self.has_foods, self.agent_loc)
        return state

    def set_state(self, state):
        r"""Sets environment state."""
        self.has_foods = state[:-1]
        self.agent_loc = state[-1]

    def query_states(self):
        return [
            (*has_foods, self.agent_loc) for has_foods in
            product(range(2), repeat=self.num_boxes)
        ]

    def seed(self, seed):
        r"""Sets random generator."""
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.has_foods = tuple(self.rng.choice(2, self.num_boxes))
        self.agent_loc = self.rng.choice(self.num_boxes+1)
        return self.observe_step()

    def step(self, action):
        reward, done = self.transition_step(action)
        obs = self.observe_step()
        info = {}
        return obs, reward, done, info

    def transition_step(self, action):
        r"""Runs one transition step."""
        reward, done = 0., False
        _has_foods = list(self.has_foods)
        if action<=self.num_boxes: # 'MOVE'
            self.agent_loc = action
            reward += self.env_spec['reward']['move'][self.agent_loc]
            for b_idx in range(self.num_boxes): # box state change
                if _has_foods[b_idx]==0 and self.rng.random()<self.env_spec['boxes']['p_appear'][b_idx]:
                    _has_foods[b_idx] = 1
                if _has_foods[b_idx]==1 and self.rng.random()<self.env_spec['boxes']['p_vanish'][b_idx]:
                    _has_foods[b_idx] = 0
        else: # 'FETCH'
            reward += self.env_spec['reward']['fetch']
            if self.agent_loc<self.num_boxes and _has_foods[self.agent_loc]==1:
                _has_foods[self.agent_loc] = 0
                reward += self.env_spec['reward']['food']
        self.has_foods = tuple(_has_foods)
        return reward, done

    def observe_step(self):
        r"""Runs one observe step."""
        cues = []
        for b_idx in range(self.num_boxes):
            if self.has_foods[b_idx]==1:
                p = self.env_spec['boxes']['p_true'][b_idx]
            else:
                p = self.env_spec['boxes']['p_false'][b_idx]
            cues.append(self.rng.binomial(
                n=self.env_spec['boxes']['num_shades'], p=p,
            ))
        obs = (*cues, self.agent_loc)
        return obs


class IdenticalBoxForaging(MultiBoxForaging):
    r"""Identical boxes foraging experiment."""

    def __init__(self,
        *,
        env_spec: Optional[dict] = None,
        **kwargs,
    ):
        super(IdenticalBoxForaging, self).__init__(env_spec=env_spec, **kwargs)
        for key in ['p_appear', 'p_vanish', 'p_true', 'p_false']:
            assert len(np.unique(self.env_spec['boxes'][key]))==1

    def get_env_param(self):
        r"""Returns environment parameters."""
        env_param = tuple(np.concatenate([
            [
                self.env_spec['boxes']['p_appear'][0],
                self.env_spec['boxes']['p_vanish'][0],
                self.env_spec['boxes']['p_true'][0],
                self.env_spec['boxes']['p_false'][0],
                self.env_spec['reward']['food'],
            ],
            self.env_spec['reward']['move'],
        ]))
        return env_param

    def set_env_param(self, env_param):
        r"""Updates environment with parameters."""
        self.env_spec['boxes']['p_appear'] = self._get_array(env_param[0], self.num_boxes)
        self.env_spec['boxes']['p_vanish'] = self._get_array(env_param[1], self.num_boxes)
        self.env_spec['boxes']['p_true'] = self._get_array(env_param[2], self.num_boxes)
        self.env_spec['boxes']['p_false'] = self._get_array(env_param[3], self.num_boxes)
        self.env_spec['reward']['food'] = env_param[4]
        self.env_spec['reward']['move'] = env_param[5:(self.num_boxes+6)]
