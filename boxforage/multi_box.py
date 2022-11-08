from pathlib import Path
import yaml
from itertools import product
import numpy as np
import gym
from gym.spaces import Discrete, MultiDiscrete
from typing import Optional, Union

from jarvis import Config

from .alias import RandGen


with open(Path(__file__).parent/'defaults.yaml') as f:
    D_SPEC = Config(yaml.safe_load(f))

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

    def __init__(self,
        *,
        spec: Optional[dict] = None,
        rng: Union[RandGen, int, None] = None,
    ):
        r"""
        Args
        ----
        spec:
            Environment specification.
        rng:
            Random number generator or seed.

        """
        self.spec = Config(spec).fill(D_SPEC.multi_box)
        self.num_boxes = self.spec.boxes.num_boxes
        for key in ['p_appear', 'p_vanish', 'p_true', 'p_false']:
            self.spec.boxes[key] = self._get_array(
                self.spec.boxes[key], self.num_boxes,
            )
        self.spec.reward.move = np.concatenate([
            self._get_array(self.spec.reward.pop('move_box'), self.num_boxes),
            np.array([self.spec.reward.pop('move_center')]),
        ])

        self.state_space = MultiDiscrete( # box states and agent position
            [2]*self.num_boxes+[self.num_boxes+1]
        )
        self.action_space = Discrete(self.num_boxes+2) # move and fetch
        self.observation_space = MultiDiscrete( # color cues and agent position
            [self.spec.boxes.num_shades+1]*self.num_boxes+[self.num_boxes+1]
        )

        self.rng = rng if isinstance(rng, RandGen) else np.random.default_rng(rng)

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
            self.spec.boxes.p_appear,
            self.spec.boxes.p_vanish,
            self.spec.boxes.p_true,
            self.spec.boxes.p_false,
            [self.spec.reward.food],
            self.spec.reward.move,
        ]))
        return env_param

    def set_env_param(self, env_param):
        r"""Updates environment with parameters."""
        c_p, n_p = 0, self.num_boxes
        env_param = np.array(env_param)
        self.spec.boxes.p_appear = env_param[c_p:c_p+n_p]
        c_p += n_p
        self.spec.boxes.p_vanish = env_param[c_p:c_p+n_p]
        c_p += n_p
        self.spec.boxes.p_true = env_param[c_p:c_p+n_p]
        c_p += n_p
        self.spec.boxes.p_false = env_param[c_p:c_p+n_p]
        c_p += n_p
        self.spec.reward.food = env_param[c_p]
        c_p += 1
        n_p = self.num_boxes+1
        self.spec.reward.move = env_param[c_p:c_p+n_p]

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

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.has_foods = self.rng.choice(2, self.num_boxes)
        self.agent_loc = self.rng.choice(self.num_boxes+1)
        observation = self.observe_step()
        info = {}
        return observation, info

    def step(self, action):
        reward, terminated = self.transition_step(action)
        observation = self.observe_step()
        truncated, info = False, {}
        return observation, reward, terminated, truncated, info

    def transition_step(self, action):
        r"""Runs one transition step."""
        reward, terminated = 0., False
        if action<=self.num_boxes: # 'MOVE'
            self.agent_loc = action
            reward += self.spec.reward.move[self.agent_loc]
            for b_idx in range(self.num_boxes): # box state change
                if self.has_foods[b_idx]==0 and self.rng.random()<self.spec.boxes.p_appear[b_idx]:
                    self.has_foods[b_idx] = 1
                if self.has_foods[b_idx]==1 and self.rng.random()<self.spec.boxes.p_vanish[b_idx]:
                    self.has_foods[b_idx] = 0
        else: # 'FETCH'
            reward += self.spec.reward.fetch
            if self.agent_loc<self.num_boxes and self.has_foods[self.agent_loc]==1:
                self.has_foods[self.agent_loc] = 0
                reward += self.spec.reward.food
        return reward, terminated

    def observe_step(self):
        r"""Runs one observe step."""
        cues = []
        for b_idx in range(self.num_boxes):
            if self.has_foods[b_idx]==1:
                p = self.spec.boxes.p_true[b_idx]
            else:
                p = self.spec.boxes.p_false[b_idx]
            cues.append(self.rng.binomial(
                n=self.spec.boxes.num_shades, p=p,
            ))
        observation = (*cues, self.agent_loc)
        return observation


class IdenticalBoxForaging(MultiBoxForaging):
    r"""Identical boxes foraging experiment."""

    def __init__(self,
        *,
        spec: Optional[dict] = None,
        **kwargs,
    ):
        super(IdenticalBoxForaging, self).__init__(spec=spec, **kwargs)
        for key in ['p_appear', 'p_vanish', 'p_true', 'p_false']:
            assert len(np.unique(self.spec.boxes[key]))==1
        assert len(np.unique(self.spec.reward.move[:-1]))==1

    def get_env_param(self):
        r"""Returns environment parameters."""
        env_param = tuple(np.concatenate([
            [
                self.spec.boxes.p_appear[0],
                self.spec.boxes.p_vanish[0],
                self.spec.boxes.p_true[0],
                self.spec.boxes.p_false[0],
                self.spec.reward.food,
            ],
            self.spec.reward.move[-2:],
        ]))
        return env_param

    def set_env_param(self, env_param):
        r"""Updates environment with parameters."""
        self.spec.boxes.p_appear = self._get_array(env_param[0], self.num_boxes)
        self.spec.boxes.p_vanish = self._get_array(env_param[1], self.num_boxes)
        self.spec.boxes.p_true = self._get_array(env_param[2], self.num_boxes)
        self.spec.boxes.p_false = self._get_array(env_param[3], self.num_boxes)
        self.spec.reward.food = env_param[4]
        self.spec.reward.move = np.array([env_param[5]]*self.num_boxes+[env_param[6]])


class CoupledBoxForaging(gym.Env):

    def __init__(self, env: MultiBoxForaging, spec=None):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.state_space = env.state_space
        self.rng = env.rng
        self.spec = env.spec.clone()
        self.spec.couple = Config(spec).fill(D_SPEC.couple)

    def get_env_param(self):
        return (
            *self.env.get_env_param(), self.spec.couple.p,
        )

    def set_env_param(self, env_param):
        self.env.set_env_param(env_param[:-1])
        self.spec.couple.p = env_param[-1]

    def get_state(self):
        return self.env.get_state()

    def set_state(self, state):
        self.env.set_state(state)

    def query_states(self):
        return self.env.query_states()

    def reset(self, seed=None):
        return self.env.reset(seed)

    def step(self, action):
        reward, terminated = self.env.transition_step(action)
        if action<=self.env.num_boxes:
            self.couple_step()
        observation = self.env.observe_step()
        truncated, info = False, {}
        return observation, reward, terminated, truncated, info

    def couple_step(self):
        for _ in range(self.spec.couple.num_steps):
            i, j = self.rng.choice(self.env.num_boxes, 2, replace=False)
            if self.spec.couple.p>0 and self.has_foods[i]!=self.has_foods[j]:
                p = self.spec.couple.p
                if self.rng.random()<p:
                    if self.rng.random()<0.5:
                        self.has_foods[i] = 0
                        self.has_foods[j] = 0
                    else:
                        self.has_foods[i] = 1
                        self.has_foods[j] = 1
            if self.spec.couple.p<0 and self.has_foods[i]==self.has_foods[j]:
                p = -self.spec.couple.p
                if self.rng.random()<p:
                    if self.rng.random()<0.5:
                        self.has_foods[i] = 0
                        self.has_foods[j] = 1
                    else:
                        self.has_foods[i] = 1
                        self.has_foods[j] = 0


def make_coupled_env(env_name, env_spec):
    config = Config({'_target_': env_name})
    env_spec = Config(env_spec)
    couple_spec = env_spec.pop('couple', None)
    env = config.instantiate(spec=env_spec)
    env = CoupledBoxForaging(env, couple_spec)
    return env
