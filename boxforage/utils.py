# -*- coding: utf-8 -*-
"""
Created on Wed May  5 11:51:12 2021

@author: Zhe
"""

class Distribution(dict):
    r"""Probability distribution.

    The positive terms are stored as dictionary items.

    """

    def add(self, key, val):
        r"""Adds value to an item.

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
