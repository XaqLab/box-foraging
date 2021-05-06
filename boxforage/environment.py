# -*- coding: utf-8 -*-
"""
Created on Sat May  1 15:33:44 2021

@author: Zhe
"""

from itertools import product
import numpy as np
from scipy.special import softmax

from .utils import (
    DefaultValDict, Distribution,
    state_reward_prob, observation_prob,
    )

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

    def __init__(self, location, p_appear, p_vanish, *,
                 max_color=5, p_low=0.4, p_high=0.6):
        self.location = location
        self.p_appear = p_appear
        self.p_vanish = p_vanish
        self.max_color = max_color
        self.p_low, self.p_high = p_low, p_high

        self.has_food = False
        self.color = None

    def step(self):
        r"""Updates the box for one time step."""
        # update food availability
        if self.has_food and rng.random()<self.p_vanish:
            self.has_food = False
        if not self.has_food and rng.random()<self.p_appear:
            self.has_food = True
        # update color cue
        self.color = rng.binomial(
            self.max_color, self.p_high if self.has_food else self.p_low,
            )


class Agent:
    r"""Agent performing the foraging task.

    Args
    ----
    gamma: float
        The discounting parameter for cumulative return.
    beta: float
        The temperature parameter for a softmax policy.
    p_fetch: float
        The probability of fetching the food after seeing it in the box.
    p_prior: float
        The prior probability of believing a box has food.
    b_resol: float
        The resolution of belief discretization. The belief probability is
        assigned to bins like [0, 1/b_resol), [1/b_resol, 2/b_resol), etc.

    """

    def __init__(self, *, gamma=0.9, beta=10., p_fetch=0.95, p_prior=0.1, b_resol=0.1):
        self.gamma, self.beta = gamma, beta
        self.p_fetch = p_fetch
        self.p_prior = p_prior
        self.b_resol = b_resol

        self.environment = None
        self.location = None

        self.observations = [] # List[tuple]
        self.beliefs = [] # List[Distribution]
        self.actions = [] # List[tuple]
        self.rewards = [] # List[float]

        self.Q = DefaultValDict() # action value dictionary

    def _observation_prob(self, state):
        return observation_prob(state, self.environment.observation_params)

    def _state_reward_prob(self, state, action):
        return state_reward_prob(
            state, action,
            self.environment.dynamics_params,
            self.environment.reward_params,
            )

    def discretize_belief(self, belief):
        r"""Returns discretized belief."""
        return tuple(sorted(
            [(self.environment.get_state_idx(s), int(-(-p//self.b_resol)))
             for s, p in belief.items()], key=lambda x: x[0]
            ))

    def decide_action(self):
        r"""Decides an action using softmax policy."""
        d_belief = self.discretize_belief(self.beliefs[-1])
        p = self.policy(d_belief)
        action = rng.choice(len(p), p=p)
        self.actions.append(action)

    def observe(self):
        r"""Observes the environment."""
        self.observations.append(tuple([box.color for box in self.environment.boxes]))

    def prior_belief(self):
        r"""Returns prior belief.

        Returns
        -------
        belief: Distribution
            The prior belief of the environment state.

        """
        belief = Distribution()
        box_num = self.environment.box_num
        loc_idx = self.environment.get_loc_idx(self.location)
        for box_states in product(range(2), repeat=box_num):
            count = sum(box_states)
            p = self.p_prior**count*(1-self.p_prior)**(box_num-count)
            state = box_states+(loc_idx,)
            belief.add(state, p)
        return belief

    def policy(self, d_belief):
        r"""Returns the policy given the discretized belief.

        Args
        ----
        d_belief: tuple
            The tuple representation of discretized belief.

        Returns
        -------
        p: array_like, (action_num,)
            The probability for each action.

        """
        logits = self.beta*np.array([
            self.Q[(d_belief, a)] for a in self.environment.all_actions
            ])
        p = softmax(logits)
        return p

    def step(self):
        r"""Updates the agent for one time step.

        The agent observes the environment, and updates its belief and decides
        an action.

        """
        self.observe()
        self.update_belief()
        self.decide_action()

    def update_belief(self):
        r"""Updates the belief.

        The belief for :math:`s_{t+1}` is calculated and stored. When there is
        no previous belief, :math:`b_{t+1}(s_{t+1})=p(s_{t+1}|o_{t+1})`,
        otherwise :math:`b_{t+1}(s_{t+1})=\sum_{s_t}p(s_{t+1}|s_t,a_t,r_t,o_{t+1})b_t(s_t)`.

        """
        o_tp1 = self.observations[-1]
        if len(self.beliefs)==0: # calculate the belief from prior and observation
            p_s = self.prior_belief()

            b_tp1 = Distribution()
            for s_tp1 in p_s:
                p_o_s = self._observation_prob(s_tp1)
                b_tp1.add(s_tp1, p_o_s[o_tp1]*p_s[s_tp1])
            b_tp1.normalize()
        else: # update the belief from action, reward and observation
            b_t = self.beliefs[-1]
            a_t = self.actions[-1]
            r_t = self.rewards[-1]

            b_tp1 = Distribution()
            for s_t in b_t:
                p_s_saro = Distribution()
                p_sr_sa = self._state_reward_prob(s_t, a_t)
                for _s_tp1, _r_t in p_sr_sa:
                    if _r_t==r_t:
                        p_o_s = self._observation_prob(_s_tp1)
                        p_s_saro.add(_s_tp1, p_sr_sa[(_s_tp1, r_t)]*p_o_s[o_tp1])
                p_s_saro.normalize()

                for s_tp1 in p_s_saro:
                    b_tp1.add(s_tp1, p_s_saro[s_tp1]*b_t[s_t])
        self.beliefs.append(b_tp1)


class ForagingEnvironment:
    r"""Foraging environment.

    The foraging environment contains several boxes, each of which possibly
    contains food. The food availability is indicated by the color cue of each
    box. An agent moves around in the environment, and can push a button when
    it is at the location of a box.

    The environment state is encoded by a tuple of size B+1, where B is the
    number of boxes. The first B values are either 0 or 1, indicating whether
    the food is available in each box. The last value is an integer in [0, B],
    indicating the agent location, can be at any of the B boxes or the center
    of the room.

    The agent action is encoded by an integer in [0, B+1]. If it is no greater
    than B, it means moving to the desired location. B+1 means pushing the
    button.

    Args
    ----
    boxes: list of Box
        The boxes in the environment.
    agent: Agent
        The agent performing the foraging task.
    reward_params: dict
        A dictionary of reward parameters. Rewards has to be integer for
        discrete processing.

    """

    def __init__(self, boxes, agent, *, reward_params=None):
        self.boxes = boxes
        self.box_num = len(self.boxes)

        self.agent = agent
        self.agent.environment = self
        self.agent.location = 'CENTER'

        # define dynamics parameters
        self.dynamics_params = {
            'p_appear': [box.p_appear for box in self.boxes],
            'p_vanish': [box.p_vanish for box in self.boxes],
            'p_fetch': self.agent.p_fetch,
            }

        # define observation parameters
        self.observation_params = {
            'max_color': [box.max_color for box in self.boxes],
            'p_low': [box.p_low for box in self.boxes],
            'p_high': [box.p_high for box in self.boxes],
            }

        # define default reward parameters and update from the argument
        self.reward_params = {
            'r_time': -1, # reward of spending one time step
            'r_food': 10, # reward of getting food from a box
            'r_invalid': -1000, # reward of taking an invalid action
            'r_move': np.zeros(
                (self.box_num+1, self.box_num+1), dtype=int,
                ), # reward of moving to different locations
            }
        for i in range(self.box_num):
            for j in range(self.box_num):
                if i!=j:
                    self.reward_params['r_move'][i, j] = -3
        for i in range(self.box_num):
            self.reward_params['r_move'][i, -1] = -1
            self.reward_params['r_move'][-1, i] = -1
        if reward_params is None:
            reward_params = {}
        for key in self.reward_params:
            if key in reward_params:
                self.reward_params[key] = reward_params[key]

        self.all_states = list(product(
            *[range(2) for _ in range(self.box_num)], range(self.box_num+1)
            ))
        self.all_actions = list(range(self.box_num+2))

        self.all_locations = [box.location for box in self.boxes]+['CENTER']
        assert len(set(self.all_locations))==len(self.all_locations)

        self.states = [self.state()]

    def get_loc_idx(self, location):
        r"""Returns integer index of a location."""
        return self.all_locations.index(location)

    def get_state_idx(self, state):
        r"""Returns integer index of a state"""
        return np.ravel_multi_index(state, [2]*self.box_num+[self.box_num+1])

    def state(self):
        r"""Returns the current environment state."""
        return tuple(1 if box.has_food else 0 for box in self.boxes)+(self.agent.location,)

    def step(self):
        r"""Updates the environment for one time step."""
        # passive dynamics of the boxes
        for box in self.boxes:
            box.step()
        # agent observe the environment and decide how to act
        self.agent.step()
        # the environment get updated by the action
        self.execute(self.agent.actions[-1])

        self.states.append(self.state())

    def execute(self, action):
        r"""Executes the agent action.

        Args
        ----
        action:
            The agent action.

        """
        assert action in self.all_actions
        loc_idx = self.get_loc_idx(self.agent.location)
        reward = self.reward_params['r_time']
        if action==self.box_num+1: # 'PUSH'
            if loc_idx==self.box_num: # agent at 'CENTER'
                reward += self.reward_params['r_invalid']
            else:
                box = self.boxes[loc_idx]
                if box.has_food and rng.random()<self.agent.p_fetch:
                    box.has_food = False
                    reward += self.reward_params['r_food']
        else: # ('MOVE', location)
            reward += self.reward_params['r_move'][loc_idx, action]
            self.agent.location = self.all_locations[action]
        self.agent.rewards.append(reward) # reward feedback to the agent
