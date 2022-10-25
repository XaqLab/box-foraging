from pathlib import Path
import yaml
import numpy as np
from gym.spaces import Discrete, MultiDiscrete
from typing import Optional, Union

from jarvis.config import Config

from .alias import GymEnv, RandGen


with open(Path(__file__).parent/'defaults.yaml') as f:
    D_ENV_SPEC = Config(yaml.safe_load(f)).single_box

class SingleBoxForaging(GymEnv):
    r"""Single box foraging environment.

    A minimal example of foraging experiment. Food will exist in a single box
    and the agent needs to decide whether to open the box based on the
    monochromatic cue outside the box. Food appears with a fixed probability if
    it does not already exist and the agent does not open the box at this time
    step. Color cue is drawn from a binomial distribution with probability p if
    food exists, and 1-p otherwise.

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
            Random generator or seed.

        """
        self.spec = Config(spec).fill(D_ENV_SPEC)
        self.state_space = MultiDiscrete([2]) # box state
        self.action_space = Discrete(2) # wait and fetch
        self.observation_space = MultiDiscrete([self.spec.box.num_shades+1]) # color cue

        self.rng = rng if isinstance(rng, RandGen) else np.random.default_rng(rng)

    def get_env_param(self):
        r"""Returns environment parameters."""
        env_param = (
            self.spec.box.p_appear,
            self.spec.box.p_cue,
            self.spec.reward.food,
        )
        return env_param

    def set_env_param(self, env_param):
        r"""Updates environment with parameters."""
        self.spec.box.p_appear = env_param[0]
        self.spec.box.p_cue = env_param[1]
        self.spec.reward.food = env_param[2]

    def get_state(self):
        r"""Returns environment state."""
        state = (self.has_food,)
        return state

    def set_state(self, state):
        r"""Sets environment state."""
        self.has_food, = state

    def query_states(self):
        r"""Query states for visualization."""
        return [(1,)]

    def reset(self, seed=None):
        self.has_food = 0
        self.rng = np.random.default_rng(seed)
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
        if action==0: # wait
            if self.has_food==0 and self.rng.random()<self.spec.box.p_appear:
                self.has_food = 1
        else: # fetch
            reward += self.spec.reward.fetch
            if self.has_food==1:
                reward += self.spec.reward.food
                self.has_food = 0
        return reward, terminated

    def observe_step(self):
        r"""Runs one observation step."""
        color = self.rng.binomial(
            self.spec.box.num_shades,
            self.spec.box.p_cue if self.has_food==1 else 1-self.spec.box.p_cue,
        )
        observation = (color,)
        return observation
