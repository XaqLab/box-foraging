from gym.spaces import Discrete, MultiDiscrete
import numpy as np
import torch
from scipy.stats import binom
from irc.distributions import IndependentDiscreteDistribution
from irc.environments import TransitModel, ObsModel, BeliefMDPEnvironment

from typing import Optional
from irc.distributions import VarSpace
from irc.distributions import EnergyBasedDistribution as Distribution
RandomGenerator = np.random.Generator

from .utils import get_spec


def _get_spaces(boxes_spec):
    state_space = MultiDiscrete( # box states and agent position
        [2]*boxes_spec['num_boxes']+[boxes_spec['num_boxes']+1]
    )
    action_space = Discrete(boxes_spec['num_boxes']+2) # move and fetch
    obs_space = MultiDiscrete( # color cues and agent position
        [boxes_spec['num_grades']+1]*boxes_spec['num_boxes']+[boxes_spec['num_boxes']+1]
    )
    return state_space, action_space, obs_space


class BoxForagingTransitModel(TransitModel):

    def __init__(self,
        boxes_spec: dict,
        reward_spec: dict,
    ):
        state_space, action_space, _ = _get_spaces(boxes_spec)
        super(BoxForagingTransitModel, self).__init__(state_space, action_space)
        self.num_boxes = boxes_spec['num_boxes']
        self.p_appear = boxes_spec['p_appear']
        self.p_vanish = boxes_spec['p_vanish']
        self.r_food = reward_spec['food']
        self.r_move = reward_spec['move']
        self.r_fetch = reward_spec['fetch']
        self.r_time = reward_spec['time']

    def __call__(self, state, action):
        state_dist = IndependentDiscreteDistribution(self.state_space)
        has_foods, agent_pos = state[:-1], state[-1]
        # agent interaction
        to_update_boxes = True
        if action<=self.num_boxes: # 'MOVE'
            agent_pos = action
        else: # 'FETCH'
            if agent_pos<self.num_boxes:
                has_foods[agent_pos] = 0
                to_update_boxes = False # box states do not change when a valid fetch is triggered
        state = [*has_foods, agent_pos]
        # construct p(s_tp1|s_t, a_t)
        prob_dicts = []
        if to_update_boxes:
            for i in range(self.num_boxes): # passive dynamics of boxes
                if has_foods[i]==0:
                    prob_dicts.append({
                        (0,): 1-self.p_appear[i],
                        (1,): self.p_appear[i],
                    })
                else:
                    prob_dicts.append({
                        (0,): self.p_vanish[i],
                        (1,): 1-self.p_vanish[i],
                    })
        else:
            for i in range(self.num_boxes):
                prob_dicts.append({
                    (has_foods[i],): 1.,
                })
        prob_dicts.append({(agent_pos,): 1.})
        state_dist.set_probs(prob_dicts)
        return state_dist

    def reward_func(self, state, action, next_state):
        reward = self.r_time
        if action<=self.num_boxes: # 'MOVE'
            orig = state[-1]
            dest = next_state[-1]
            reward += self.r_move[dest] if dest!=orig else 0.
        else: # 'FETCH'
            reward += self.r_fetch
            agent_pos = state[-1]
            if agent_pos<self.num_boxes and state[agent_pos]==1 and next_state[agent_pos]==0:
                reward += self.r_food
        return reward

    def done_func(self, state):
        return False


class BoxForagingObsModel(ObsModel):

    def __init__(self,
        boxes_spec,
    ):
        _, _, obs_space = _get_spaces(boxes_spec)
        super(BoxForagingObsModel, self).__init__(obs_space)
        self.num_boxes = boxes_spec['num_boxes']
        self.num_grades = boxes_spec['num_grades']
        self.p_true = boxes_spec['p_true']
        self.p_false = boxes_spec['p_false']

    def __call__(self, state):
        obs_dist = IndependentDiscreteDistribution(self.obs_space)
        has_foods, agent_pos = state[:-1], state[-1]
        prob_dicts = []
        for i in range(self.num_boxes):
            p = self.p_true[i] if has_foods[i] else self.p_false[i]
            vals = binom.pmf(np.arange(self.num_grades+1), self.num_grades, p)
            prob_dicts.append(dict(
                ((k,), val) for k, val in enumerate(vals)
            ))
        prob_dicts.append({(agent_pos,): 1.})
        obs_dist.set_probs(prob_dicts)
        return obs_dist


class BoxForagingEnvironment(BeliefMDPEnvironment):

    def __init__(self,
        boxes_spec: Optional[dict] = None,
        reward_spec: Optional[dict] = None,
        belief: Optional[Distribution] = None,
        **kwargs,
    ):
        boxes_spec = get_spec('boxes', **(boxes_spec or {}))
        self.num_boxes = boxes_spec['num_boxes']
        boxes_spec['p_appear'] = self._get_array(boxes_spec['p_appear'], self.num_boxes)
        boxes_spec['p_vanish'] = self._get_array(boxes_spec['p_vanish'], self.num_boxes)
        boxes_spec['p_true'] = self._get_array(boxes_spec['p_true'], self.num_boxes)
        boxes_spec['p_false'] = self._get_array(boxes_spec['p_false'], self.num_boxes)
        reward_spec = get_spec('reward', **(reward_spec or {}))
        reward_spec['move'] = self._get_array(reward_spec['move'], self.num_boxes+1)
        self.boxes_spec, self.reward_spec = boxes_spec, reward_spec

        state_space, action_space, obs_space = _get_spaces(boxes_spec)
        if belief is None:
            belief = IndependentDiscreteDistribution(state_space)
        if 'transit_model' not in kwargs:
            kwargs['transit_model'] = BoxForagingTransitModel(boxes_spec, reward_spec)
        if 'obs_model' not in kwargs:
            kwargs['obs_model'] = BoxForagingObsModel(boxes_spec)
        super(BoxForagingEnvironment, self).__init__(
            state_space, action_space, obs_space, belief, **kwargs,
            )

    @staticmethod
    def _get_array(val, n):
        r"""Returns an array of desired length."""
        if isinstance(val, float):
            return np.ones(n)*val
        else:
            assert len(val)==n
            return np.array(val)

    def get_state(self):
        state = [*self.has_foods, self.agent_pos]
        return state

    def set_state(self, state):
        self.has_foods = state[:-1]
        self.agent_pos = state[-1]

    def reset(self):
        state = [0]*self.num_boxes+[self.num_boxes]
        self.set_state(state)
        obs = self.obs_step(state)
        return super(BoxForagingEnvironment, self).reset(obs)

    def run_one_trial(self, algo=None, num_steps=100):
        r"""Runs a test trial.

        Args
        ----
        algo: SB3.Algo
            A stable-baselines 3 algorithm. Random policy is used when `algo` is
            ``None``.
        num_steps: int
            The number of steps in the trial.

        Returns
        -------
        trial: dict
            Information of the trial.

        """
        b_params, actions, rewards = [], [], []
        box_beliefs = []
        has_foods, color_cues, agent_poss = [], [], []

        if algo is not None:
            algo.policy.set_training_mode(False)

        b_param = self.reset()
        for _ in range(num_steps):
            b_params.append(b_param)
            if isinstance(self.belief, IndependentDiscreteDistribution):
                box_belief = []
                for i in range(self.num_boxes):
                    phi = self.belief.phis[i]
                    p = torch.softmax(phi.embed.weight.data, dim=0)[1].item()
                    box_belief.append(p)
                box_beliefs.append(box_belief)

            if algo is None: # random policy
                action = self.action_space.sample()
            else:
                action, _ = algo.predict(b_param)
            actions.append(action)

            b_param, reward, _, info = self.step(action)
            rewards.append(reward)
            has_foods.append(info['state'][:-1])
            color_cues.append(info['obs'][:-1])
            agent_poss.append(info['state'][-1])

        trial = {
            'b_params': np.array(b_params),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'box_beliefs': np.array(box_beliefs),
            'has_foods': np.array(has_foods),
            'color_cues': np.array(color_cues),
            'agent_poss': np.array(agent_poss),
        }
        return trial
