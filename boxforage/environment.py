# -*- coding: utf-8 -*-
"""
Created on Sat May  1 15:33:44 2021

@author: Zhe
"""

import numpy as np


class Box:
    r"""The box containing food.

    Args
    ----
    location: str
        The label of the box location.
    appear_rate: float
        The transition probability from 'no food' state to 'has food' state.
    vanish_rate: float
        The transition probability from 'has food' state to 'no food' state.

    """

    def __init__(self, location, p_appear, p_vanish,
                 max_color, p_low=0.2, p_high=0.8):
        self.location = location
        self.p_appear = p_appear
        self.p_vanish = p_vanish
        self.max_color = max_color
        self.p_low, self.p_high = p_low, p_high

        self.has_food = False
        self.color = 0

    def step(self):
        r"""Updates the box for one time step.

        """
        if self.has_food and np.random.rand()<self.vanish_rate:
            self.has_food = False
        if not self.has_food and np.random.rand()<self.appear_rate:
            self.has_food = True

        self.color = np.random.binomial(
            self.max_color, self.p_high if self.has_food else self.p_low,
            )
