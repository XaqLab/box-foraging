import numpy as np
import torch
from typing import Union
from gym.spaces import MultiDiscrete, Box
from stable_baselines3.common.base_class import BaseAlgorithm as SB3Algo

Array = np.ndarray
Tensor = torch.Tensor
VarSpace = Union[MultiDiscrete, Box]
RandGen = np.random.Generator
