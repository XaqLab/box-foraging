import gym
from gym.spaces import Discrete, MultiDiscrete
import numpy as np

from typing import Optional
RandomGenerator = np.random.Generator

class Box:
    """Food box with a monochromatic cue.

    The availability of food in the box follows as telegraph process. The grade
    of color cue is draw from a binomial distribution dependent on whether the
    food is available or not.

    """

    def __init__(
        self,
        p_appear: float,
        p_vanish: float,
        *,
        num_grades: int = 5,
        p_true: float = 0.6,
        p_false: float = 0.2,
        rng: Optional[RandomGenerator] = None,
    ):
        r"""
        Args
        ----
        p_appear: float
            The probability of food appearing.
        p_vanish: float
            The probability of food vanishing.
        num_grades: int
            The number of color grades, also serves the 'n' parameter of
            binomial distribution.
        p_true, p_false: float
            The 'p' parameter of binomial distributions for 'has food' and 'no
            food' conditions.
        rng: RandomGenerator
            Random number generator for the box.

        """
        self.p_appear, self.p_vanish = p_appear, p_vanish
        self.num_grades = num_grades
        self.p_true, self.p_false = p_true, p_false
        self.rng = rng or np.random.default_rng()

        self.reset()

    def _sample_color(self, p):
        r"""Samples color cue from binomial distribution."""
        return self.rng.binomial(self.num_grades, p)

    def reset(self):
        r"""Resets box state"""
        self.has_food = False
        self.color = self._sample_color(self.p_false)

    def step(self):
        r"""Updates the box for one time step."""
        # update food availability
        if self.has_food and self.rng.random()<self.p_vanish:
            self.has_food = False
        if not self.has_food and self.rng.random()<self.p_appear:
            self.has_food = True
        # update color cue
        self.color = self._sample_color(self.p_true if self.has_food else self.p_false)


class ForagingEnvironment(gym.Env):
    r"""Foraging environment.

    The foraging environment contains several boxes, each of which possibly
    contains food. The food availability is indicated by the color cue of each
    box. An agent moves around in the environment, and can push a button when
    it is at the position of a box.

    Aside from the B box positions, the agent can also be at 'CENTER'. Agent
    position is indicated by an integer in `[0, B]`, with `B` as the 'CENTER'.

    """
    MAX_BOX_SEED = 1000000

    def __init__(
        self,
        boxes_spec: dict = None,
        reward_spec: dict = None,
        *,
        episodic: bool = True,
        seed: Optional[int] = None,
    ):
        r"""
        Args
        ----
        boxes_spec: dict
            Specification of boxes. The number of grades are the same for all
            boxes, while other parameters can be individually set.
        reward_spec: dict
            Specification of rewards. `reward_spec['move']` describes the moving
            cost to each box and the center.
        episodic: bool
            Whether the episode terminates after food is fetched.
        seed: int
            Random seed for the environment.

        """
        super(ForagingEnvironment, self).__init__()
        self.num_boxes = boxes_spec['num_boxes']
        self.num_grades = boxes_spec['num_grades']
        self.p_appear = self._get_array(boxes_spec['p_appear'], self.num_boxes)
        self.p_vanish = self._get_array(boxes_spec['p_vanish'], self.num_boxes)
        self.p_true = self._get_array(boxes_spec['p_true'], self.num_boxes)
        self.p_false = self._get_array(boxes_spec['p_false'], self.num_boxes)

        self.r_food = reward_spec['food']
        self.r_move = self._get_array(reward_spec['move'], self.num_boxes+1)
        self.r_time = reward_spec['time']

        self.state_space = MultiDiscrete(
            [2]*self.num_boxes+[self.num_boxes+1] # food availability and agent position
        )
        self.observation_space = MultiDiscrete(
            [self.num_grades+1]*self.num_boxes+[self.num_boxes+1] # color cues and agent position
        )
        self.action_space = Discrete(self.num_boxes+2) # move and fetch

        self.episodic = episodic
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        r"""Resets environment."""
        self.boxes = []
        for i in range(self.num_boxes):
            self.boxes.append(Box(
                self.p_appear[i], self.p_vanish[i],
                num_grades=self.num_grades,
                p_true=self.p_true[i], p_false=self.p_false[i],
                rng=np.random.default_rng(self.rng.integers(self.MAX_BOX_SEED))
            ))
        self.agent_pos = self.rng.integers(self.num_boxes+1)
        return self.get_obs()

    def get_state(self):
        r"""Returns state of the environment.

        Returns
        -------
        state: tuple
            A tuple of length `num_boxes+1`. The first `num_boxes` elements are
            the food availability of each box. The last element is the agent
            position.

        """
        state = tuple(int(box.has_food) for box in self.boxes)+(self.agent_pos,)
        return state

    def get_obs(self):
        r"""Returns observation of the environment.

        Returns
        -------
        obs: tuple
            A tuple of length `num_boxes+1`. The first `num_boxes` elements are
            the color cue of each box. The last element is the agent position.

        """
        obs = tuple(box.color for box in self.boxes)+(self.agent_pos,)
        return obs

    def step(self, action):
        r"""Runs one time step."""
        # agent interaction
        reward, done = self.r_time, False
        if action<=self.num_boxes: # 'MOVE'
            dest_pos = action
            reward += self.r_move[dest_pos] if dest_pos!=self.agent_pos else 0.
            self.agent_pos = dest_pos
        else: # 'FETCH'
            if self.agent_pos<self.num_boxes:
                box = self.boxes[self.agent_pos]
                if box.has_food:
                    box.has_food = False
                    reward += self.r_food
                    if self.episodic:
                        done = True
        # passive dynamics
        for box in self.boxes:
            box.step()

        obs = self.get_obs()
        info = {'state': self.get_state()}
        return obs, reward, done, info

    @staticmethod
    def _get_array(val, n):
        r"""Returns an array of desired length."""
        if isinstance(val, float):
            return np.ones(n)*val
        else:
            assert len(val)==n
            return np.array(val)
