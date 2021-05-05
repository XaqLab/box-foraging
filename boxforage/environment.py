# -*- coding: utf-8 -*-
"""
Created on Sat May  1 15:33:44 2021

@author: Zhe
"""

import numpy as np
rng = np.random.default_rng()


class Box:
    r"""Food box with color cue.

    The box color is a random varialbe drawn from a binomial distribution.

    Args
    ----
    location: str
        The label of the box location.
    p_appear: float
        The probability of food appearing.
    p_vanish: float
        The probability of food vanishing.
    max_color: int
        The maximum number of color cue. Color cue is an integer in
        ``[0, max_color]``.
    p_low, p_high:
        The binomial distribution parameters for color cue, corresponding to
        'no food' and 'has food' conditions respectively.

    """

    def __init__(self, location, p_appear, p_vanish,
                 max_color=5, p_low=0.4, p_high=0.6):
        self.location = location
        self.p_appear = p_appear
        self.p_vanish = p_vanish
        self.max_color = max_color
        self.p_low, self.p_high = p_low, p_high

        self.has_food = False
        self.color = None

    def step(self):
        r"""Updates the box for one time step.

        """
        # update food availability
        if self.has_food and rng.random()<self.p_vanish:
            self.has_food = False
        if not self.has_food and rng.random()<self.p_appear:
            self.has_food = True
        # update color cue
        self.color = rng.binomial(
            self.max_color, self.p_high if self.has_food else self.p_low,
            )
