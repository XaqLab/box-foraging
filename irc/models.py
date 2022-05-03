import numpy as np
import gym

from typing import Optional, Type, Union
from gym.spaces import MultiDiscrete, Box
from jarvis.utils import fill_defaults

from .distributions import BaseDistribution, DiscreteDistribution
from .utils import Tensor, RandGen, GymEnv

class BeliefModel(gym.Env):
    r"""Base class for internal belief model.

    Agent makes decision based on the belief of environment states instead of
    the direct observation. The update of belief is based on the assumed
    environment by the agent, and will not be changed by reinforcement learning
    algorithm.

    """
    D_EST_SPEC = { # default specification for distribution estimation
        'state_prior': {
            'num_samples': 100,
            'optim_kwargs': {
                'batch_size': 32, 'num_epochs': 10,
                'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4,
            },
        },
        'obs_conditional': {
            'num_samples': 100,
            'optim_kwargs': {
                'batch_size': 32, 'num_epochs': 10,
                'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4,
            },
        },
        'belief': {
            'num_samples': 20,
            'optim_kwargs': {
                'batch_size': 16, 'num_epochs': 5,
                'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4,
            },
        },
    }

    def __init__(self,
        env: GymEnv,
        state_dist_class: Optional[Type[BaseDistribution]] = None,
        state_dist_kwargs: Optional[dict] = None,
        obs_dist_class: Optional[Type[BaseDistribution]] = None,
        obs_dist_kwargs: Optional[dict] = None,
        p_s_o: Optional[BaseDistribution] = None,
        p_o_s: Optional[BaseDistribution] = None,
        est_spec: Optional[dict] = None,
        rng: Union[RandGen, int, None] = None,
    ):
        r"""
        Args
        ----
        env:
            The assumed environment, with a few utility methods implemented,
            such as `get_state`, `set_state` etc. Detailed requirements are
            specified in 'README.md'.
        state_dist_class, state_dist_kwargs:
            Class and key word arguments for state distributions, including the
            running belief and a conditional distribution p(s|o) at initial
            state.
        obs_dist_class, obs_dist_kwargs:
            Class and key word arguments for observation distribution, used for
            conditional distribution p(o|s).
        p_s_o:
            Prior distribution of state given initial observation. If not
            provided, it will be estimated using assumed environment `env`.
        p_o_s:
            Conditional distribution of observation. If not provided, it will be
            estimated using assumed environment `env`.
        est_spec:
            Estimation specifications.
        rng:
            Random seed or generator.

        """
        self.env = env
        self.action_space = self.env.action_space

        self.state_dist_class = state_dist_class or self._get_default_dist_class(self.env.state_space)
        self.state_dist_kwargs = state_dist_kwargs or {}
        self.obs_dist_class = obs_dist_class or self._get_default_dist_class(self.env.observation_space)
        self.obs_dist_kwargs = obs_dist_kwargs or {}

        self.p_s = self.state_dist_class(self.env.state_space, **self.state_dist_kwargs)
        self.belief_space = Box(-np.inf, np.inf, shape=self.p_s.get_param_vec().shape)
        self.observation_space = self.belief_space

        self.p_s_o = p_s_o or self.state_dist_class(
            x_space=self.env.state_space, y_space=self.env.observation_space,
            **self.state_dist_kwargs,
            )
        assert self.p_s_o.x_space==self.env.state_space and self.p_s_o.y_space==self.env.observation_space
        assert len(self.p_s_o.get_param_vec())==len(self.p_s.get_param_vec())
        self.p_o_s = p_o_s or self.obs_dist_class(
            x_space=self.env.observation_space, y_space=self.env.state_space,
            **self.obs_dist_kwargs,
            )
        assert self.p_o_s.x_space==self.env.observation_space and self.p_o_s.y_space==self.env.state_space

        self.to_estimate = True
        self.est_spec = fill_defaults((est_spec or {}), self.D_EST_SPEC)
        self.rng = rng if isinstance(rng, RandGen) else np.random.default_rng(rng)

    @staticmethod
    def _get_default_dist_class(space):
        if isinstance(space, MultiDiscrete):
            return DiscreteDistribution
        if isinstance(space, Box):
            raise NotImplementedError
        raise RuntimeError(f"Default class for {space} is not defined.")

    def estimate_state_prior(self):
        r"""Estimates initial belief."""
        # use env to collect initial states
        _state_to_restore = self.env.get_state()
        states, obss = [], []
        for _ in range(self.est_spec['state_prior']['num_samples']):
            obs = self.env.reset()
            states.append(self.env.get_state())
            obss.append(obs)
        self.env.set_state(_state_to_restore)
        # estimate the initial belief
        self.p_s_o.estimate(
            np.array(states), np.array(obss), verbose=0,
            **self.est_spec['state_prior']['optim_kwargs'],
        )

    def estimate_obs_conditional(self):
        r"""Estimates conditional distribution of observation."""
        # use env to collect state/observation pairs
        _state_to_restore = self.env.get_state()
        obss, states = [], []
        for _ in range(self.est_spec['obs_conditional']['num_samples']):
            # TODO specify state and action distributions
            state = self.env.state_space.sample()
            self.env.set_state(state)
            action = self.env.action_space.sample()
            obs, *_ = self.env.step(action)
            obss.append(obs)
            states.append(self.env.get_state())
        self.env.set_state(_state_to_restore)
        # estimate the conditional distribution
        self.p_o_s.estimate(
            np.array(obss), np.array(states), verbose=0,
            **self.est_spec['obs_conditional']['optim_kwargs'],
            )

    def state_dict(self):
        r"""Returns state dictionary."""
        state = {
            'p_s_o_state': self.p_s_o.state_dict(),
            'p_o_s_state': self.p_o_s.state_dict(),
        }
        return state

    def load_state_dict(self, state):
        r"""Loads state dictionary."""
        self.p_s_o.load_state_dict(state['p_s_o_state'])
        self.p_o_s.load_state_dict(state['p_o_s_state'])
        self.to_estimate = False

    def reset(self, env=None, keep_obs=False):
        r"""Resets the environment.

        Args
        ----
        env:
            The actual environment to interact with.
        keep_obs:
            Whether to return raw observation or not, necessary for plotting
            full trajectory.

        Returns
        -------
        belief: Array
            Parameter of initial belief.
        obs: Array
            Initial observation.

        """
        if self.to_estimate:
            self.estimate_state_prior()
            self.estimate_obs_conditional()
            self.to_estimate = False
        if env is None:
            env = self.env
        obs = env.reset()
        belief = self.p_s_o.param_net(np.array(obs)[None])[0].data.cpu().numpy()
        if keep_obs:
            return belief, obs
        else:
            return belief

    def step(self, action, env=None):
        r"""Runs one step.

        Args
        ----
        action:
            Action at time t, which is supposed to be determined by belief at
            time t.
        env:
            The actual environment to interact with. It can differ from the
            assumed environment `self.env`, though the latter is always used for
            belief updating.

        Returns
        -------
        belief: Array
            Belief at time t+1.
        reward: float
            Reward at time t.
        done: bool
            Whether to terminate the episode.
        info: dict
            Auxiliary information containing observation and state at t+1.

        """
        if env is None:
            env = self.env
        obs, reward, done, info = env.step(action)
        info.update({
            'obs': obs, 'state': env.get_state(),
        })
        self.update_belief(action, obs)
        belief = self.p_s.get_param_vec().cpu().numpy()
        return belief, reward, done, info

    def update_belief(self, action, obs):
        r"""Updates belief.

        Draw s_t samples from current belief, use env to get s_tp1 samples.
        These samples are weighted according to estimated conditional
        distribution p(o|s), and used to estimate the new belief.

        Args
        ----
        action:
            Action at time t.
        obs:
            Observation at time t+1.

        """
        _state_to_restore = self.env.get_state()
        states, weights = [], []
        for _ in range(self.est_spec['belief']['num_samples']):
            state = self.p_s.sample()
            self.env.set_state(state)
            self.env.step(action)
            next_state = self.env.get_state()
            states.append(next_state)
            weights.append(np.exp(self.p_o_s.loglikelihood(
                np.array(obs)[None], np.array(next_state)[None]).item()))
        self.env.set_state(_state_to_restore)
        self.p_s.estimate( # TODO add running statistics for inspection
            xs=np.array(states), ws=np.array(weights), verbose=0,
            **self.est_spec['belief']['optim_kwargs'],
        )
