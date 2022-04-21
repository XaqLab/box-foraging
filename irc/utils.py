import numpy as np
import torch
from typing import Union
from gym.spaces import MultiDiscrete, Box

Array = np.ndarray
Tensor = torch.Tensor
VarSpace = Union[MultiDiscrete, Box]
RandGen = np.random.Generator
