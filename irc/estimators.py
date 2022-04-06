import numpy as np
import torch
from torch.optim import SGD, Adam
from torch.utils.data import TensorDataset, WeightedRandomSampler, DataLoader

from typing import Union, Optional
from torch.optim import Optimizer
from .distributions import EnergyBasedDistribution
Array = np.ndarray


class MaximumLikelihoodEstimator:
    r"""Maximum likelihood estimator.

    The estimator uses gradient-based optimizer to update the distrbution
    parameters. Data samples are split into training and validation sets, while
    the former is used for parameter learning and latter for early termination.

    """

    def __init__(self,
        optim_class: Union[Optimizer, str] = 'SGD',
        optim_kwargs: Optional[dict] = None,
    ):
        r"""
        Args
        ----
        optim_class:
            Optimizer class.
        optim_kwargs:
            Keyword arguments for optimizer.

        """
        if isinstance(optim_class, str):
            assert optim_class in ['SGD', 'Adam']
            if optim_class=='SGD':
                optim_class = SGD
                default_optim_kwargs = {
                    'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4,
                    }
            if optim_class=='Adam':
                optim_class = Adam
                default_optim_kwargs = {
                    'lr': 0.001, 'weight_decay': 1e-4, 'weight_decay': 1e-4,
                }
        self.optim_class = optim_class
        default_optim_kwargs.update(optim_kwargs or {})
        self.optim_kwargs = default_optim_kwargs

    def estimate(self,
        dist: EnergyBasedDistribution,
        xs: Array,
        weights: Optional[Array] = None,
        num_batches: int = 20,
        num_epochs: int = 5,
    ):
        r"""Estimates parameters of a distribution from data samples.

        Args
        ----
        dist:
            An energy based distribution.
        xs: (num_samples, num_vars)
            Data samples.
        weights: (num_samples,)
            Weights of samples.
        num_batches:
            Number of batches.
        num_epochs:
            Number of epochs.

        """
        if weights is None:
            weights = np.ones(len(xs))
        else:
            assert weights.shape==(len(xs),) and np.all(weights>0)
        dset = TensorDataset(torch.tensor(xs))
        sampler = WeightedRandomSampler(weights, len(xs)*num_epochs)
        batch_size = -(-len(dset)//num_batches)
        loader = DataLoader(dset, batch_size=batch_size, sampler=sampler, drop_last=True)
        optimizer = self.optim_class(dist.parameters(), **self.optim_kwargs)
        for _xs, in loader:
            loss = -dist.loglikelihood(_xs.numpy()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
