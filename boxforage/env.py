import numpy as np

from typing import Union

from jarvis.config import Config
from irc.examples import FoodBoxesEnv


class IdenticalBoxesEnv(FoodBoxesEnv):
    r"""Foraging environment with identical boxes.

    Parameters of the boxes are the same, while food availability and color cues
    are still independent.

    """

    def __init__(self,
        p_appear: float = 0.2,
        p_vanish: float = 0.05,
        p_cue: float = 0.6,
        **kwargs,
    ):
        super().__init__(p_appear=p_appear, p_vanish=p_vanish, p_cue=p_cue, **kwargs)

    def get_param(self):
        env_param = np.array([
            self.p_appear[0], self.p_vanish[0], self.p_cue[0],
            self.lambda_center, self.r_food, self.r_move,
        ])
        return env_param

    def set_param(self, env_param):
        assert len(env_param)==6
        self.p_appear = np.full(self.num_boxes, fill_value=env_param[0])
        self.p_vanish = np.full(self.num_boxes, fill_value=env_param[1])
        self.p_cue = np.full(self.num_boxes, fill_value=env_param[2])
        self.lambda_center = env_param[3]
        self.r_food = env_param[4]
        self.r_move = env_param[5]

class CoupledWrapper:
    r"""A wrapper to couple food boxes.

    At each time step, the untouched boxes will be coupled with certain
    probability, implemented as a sequential coupling operation between randomly
    selected pair of boxes.

    """

    def __init__(self,
        env: Union[FoodBoxesEnv, dict, None] = None,
        p_couple: float = 0.,
        num_ops: int = 1,
    ):
        r"""
        Args
        ----
        env:
            A foraging environment or a configuration to instantiate one.
        p_couple:
            Coupling parameter, in (-1, 1). 0 means no coupling, 1 means
            positive coupling and -1 means negative coupling.
        num_ops:
            Number of coupling operations on random pair of boxes.

        """
        if env is None or isinstance(env, dict):
            env = Config(env)
            if '_target_' not in env:
                env._target_ = 'boxforage.env.IdenticalBoxesEnv'
            env = env.instantiate()
        else:
            assert isinstance(env, FoodBoxesEnv)
        self.env: FoodBoxesEnv = env

        self.state_space = self.env.state_space
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.p_couple = p_couple
        self.num_ops = num_ops

    def reset(self, seed=None):
        return self.env.reset(seed)

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        self.couple_step(action)
        observation = self.env.observe_step()
        return observation, reward, terminated, truncated, info

    def couple_step(self, action):
        r"""Couples untouched boxes.

        Args
        ----
        action:
            Current action. If the agent opens one box at this step, that box
            will not be changed.

        """
        b_idxs = list(range(self.env.num_boxes))
        if action>self.env.num_boxes and self.env.pos<self.env.num_boxes:
            b_idxs.remove(self.env.pos)
        if len(b_idxs)>1:
            for _ in range(self.num_ops):
                i, j = self.env.rng.choice(b_idxs, 2, replace=False)
                if (
                    (self.p_couple>0 and self.env.has_food[i]!=self.env.has_food[j])
                    or
                    (self.p_couple<0 and self.env.has_food[i]==self.env.has_food[j])
                ) and self.env.rng.random()<np.abs(self.p_couple):
                    if self.env.rng.random()<0.5: # randomly flip one box
                        self.env.has_food[i] = 1-self.env.has_food[i]
                    else:
                        self.env.has_food[j] = 1-self.env.has_food[j]

    def get_param(self):
        env_param = np.array([
            *self.env.get_param(), self.p_couple,
        ])
        return env_param

    def set_param(self, env_param):
        self.p_couple = env_param[-1]
        self.env.set_param(env_param[:-1])

    def get_state(self):
        return self.env.get_state()

    def set_state(self, state):
        self.env.set_state(state)

    def query_states(self):
        return self.env.query_states()
