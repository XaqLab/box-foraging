# -*- coding: utf-8 -*-
"""
Created on Wed May  5 11:51:12 2021

@author: Zhe
"""

from itertools import product
from scipy.stats import binom


class DefaultValDict(dict):
    r"""Custom dictionary with a default value.

    Args
    ----
    default: float
        The default value if a key does not exist.
    eps: float
        A small number to determine whether a value is close enough to the
        default value.

    """

    def __init__(self, default=0., eps=1e-5):
        self.default = default
        self.eps = eps

    def __getitem__(self, key):
        if key in self:
            return super(DefaultValDict, self).__getitem__(key)
        else:
            return self.default

    def __setitem__(self, key, val):
        if abs(val-self.default)>self.eps:
            super(DefaultValDict, self).__setitem__(key, val)
        else:
            self.pop(key)


class Distribution(DefaultValDict):
    r"""Probability distribution.

    The positive terms are stored as dictionary items.

    """

    def __init__(self):
        super(Distribution, self).__init__(default=0., eps=0.)

    def add(self, key, val):
        r"""Adds value to a probability term.

        Args
        ----
        key: Immutable
            The random variable.
        val: float
            The positive value to be added.

        """
        assert val>0
        if key in self:
            self[key] += val
        else:
            self[key] = val

    def normalize(self):
        r"""Normalizes the distribution.

        """
        z = sum(self.values())
        for key in self:
            self[key] /= z


def state_reward_prob(state, action, dynamics_params, reward_params):
    r"""Returns transition probability distribution.

    Details of definitions can be found in ForagingEnvironment.

    Args
    ----
    state: tuple
        The foraging environment state.
    action: int
        The foraging agent action.
    dynamics_params: dict
        The dynamics parameters.
    reward_params: dict
        The reward parameters.

    Returns
    -------
    p_sr_sa: Distribution
        The probability distribution :math:`p(s_{t+1}, r_t|s_t, a_t)`.

    """
    p_appear = dynamics_params['p_appear']
    p_vanish = dynamics_params['p_vanish']
    p_fetch = dynamics_params['p_fetch']

    r_time = reward_params['r_time']
    r_food = reward_params['r_food']
    r_invalid = reward_params['r_invalid']
    r_move = reward_params['r_move']

    p_sr_sa = Distribution()
    box_num = len(state)-1
    for box_states in product(range(2), repeat=box_num):
        p = 1.
        for b_idx in range(box_num):
            if state[b_idx]==1:
                if box_states[b_idx]==0:
                    p *= p_vanish[b_idx]
                else:
                    p *= 1-p_vanish[b_idx]
            else:
                if box_states[b_idx]==1:
                    p *= p_appear[b_idx]
                else:
                    p *= 1-p_appear[b_idx]

        # _state = list(box_states)+[state[-1]]
        loc_idx = state[-1]
        if action==(box_num+1): # 'PUSH'
            if loc_idx==box_num: # agent at 'CENTER'
                reward = r_time+r_invalid
                p_sr_sa.add((box_states+(loc_idx,), reward), p)
            else:
                if state[loc_idx]==1:
                    # agent does not fetch the food
                    reward = r_time
                    p_sr_sa.add((box_states+(loc_idx,), reward), p*(1-p_fetch))
                    # agent fetches the food
                    reward = r_time+r_food
                    _box_states = tuple([
                        0 if b_idx==loc_idx else box_states[b_idx] for b_idx in range(box_num)
                        ])
                    p_sr_sa.add((_box_states+(loc_idx,), reward), p*p_fetch)
                else:
                    reward = r_time
                    p_sr_sa.add((box_states+(loc_idx,), reward), p)
        else: # ('MOVE', location)
            reward = r_time+r_move[loc_idx, action]
            p_sr_sa.add((box_states+(action,), reward), p)
    return p_sr_sa


def observation_prob(state, observation_params):
    r"""Returns observation probability distribution.

    Args
    ----
    state: tuple
        The foraging environment state.
    observation_params: dict
        The observation parameters.

    Returns
    -------
    p_o_s: Distribution
        The probability distribution :math:`p(o_t|s_t)`.

    """
    max_color = observation_params['max_color']
    p_high = observation_params['p_high']
    p_low = observation_params['p_low']

    p_o_s = Distribution()
    box_num = len(state)-1
    for box_colors in product(*[range(mc+1) for mc in max_color]):
        p = 1.
        for b_idx in range(box_num):
            if state[b_idx]==1:
                p_binomial = p_high[b_idx]
            else:
                p_binomial = p_low[b_idx]
            p *= binom.pmf(
                box_colors[b_idx], max_color[b_idx], p_binomial
                )
        p_o_s.add(box_colors, p)
    return p_o_s
