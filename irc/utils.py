import numpy as np
from scipy.optimize import curve_fit
import torch
from typing import Optional, Union
from gym import Env as GymEnv
from gym.spaces import MultiDiscrete, Box
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy as SB3Policy

Array = np.ndarray
Tensor = torch.Tensor
VarSpace = Union[MultiDiscrete, Box]
SB3Algo = Union[OnPolicyAlgorithm, OffPolicyAlgorithm]
RandGen = np.random.Generator


def loss_summary(losses: Array, e_idxs: Optional[Array] = None):
    def exp_decay(x, a, b, k):
        y = a*np.exp(-k*x)+b
        return y
    num_epochs = len(losses)
    a = np.max(losses)-np.min(losses)
    b = np.min(losses)
    k = 1/((losses<(a*np.exp(-1)+b)).nonzero()[0][0]+1)
    if e_idxs is None:
        e_idxs = np.arange(1, num_epochs+1)
    try:
        (a, b, k), _ = curve_fit(
            exp_decay, e_idxs, losses, p0=(a, b, k),
            bounds=((0, -np.inf, 0), (np.inf, np.inf, np.inf)),
            )
        optimality = 1-np.exp(-k*num_epochs)
        fvu = ((losses-exp_decay(e_idxs, a, b, k))**2).mean()/losses.var()
    except:
        optimality, fvu = np.nan, np.nan
    return optimality, fvu
