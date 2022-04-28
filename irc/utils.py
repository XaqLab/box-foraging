import numpy as np
import torch
from typing import Union
from gym import Env as GymEnv
from gym.spaces import MultiDiscrete, Box
from stable_baselines3.common.base_class import BaseAlgorithm as SB3Algo
from stable_baselines3.common.policies import BasePolicy as SB3Policy

Array = np.ndarray
Tensor = torch.Tensor
VarSpace = Union[MultiDiscrete, Box]
RandGen = np.random.Generator
