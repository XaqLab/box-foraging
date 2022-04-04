import numpy as np
import torch
from torch.optim import SGD, Adam
from torch.utils.data import TensorDataset, DataLoader

from typing import Union, Optional
from torch.optim import Optimizer
from .distributions import EnergyBasedDistribution
Array = np.ndarray
Tensor = torch.Tensor


class Estimator:
    r"""Maximum likelihood estimator.

    The estimator uses gradient-based optimizer to update the distrbution
    parameters. Data samples are split into training and validation sets, while
    the former is used for parameter learning and latter for early termination.

    """

    def __init__(self,
        optim_class: Union[Optimizer, str] = 'SGD',
        optim_kwargs: Optional[dict] = None,
        train_config: Optional[dict] = None,
    ):
        r"""
        Args
        ----
        optim_class:
            Optimizer class.
        optim_kwargs:
            Keyword arguments for optimizer.
        train_config:
            Configuration for gradient-based training.

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

        default_train_config = {
            'split_ratio': 0.95,
            'num_batches': 10,
            'num_epochs': 200,
        }
        for key, val in (train_config or {}).items():
            if key in default_train_config:
                default_train_config[key] = val
        self.train_config = default_train_config

    def estimate(self,
        xs: Array,
        dist: EnergyBasedDistribution,
    ):
        r"""Estimates parameters of a distribution from data samples.

        Args
        ----
        xs: (num_samples, num_vars)
            Data samples.
        dist:
            An energy based distribution.

        """
        # prepare training/validation sets
        dset = TensorDataset(torch.tensor(xs))
        train_size = int(len(dset)*self.train_config['split_ratio'])
        val_size = len(dset)-train_size
        assert train_size>0 and val_size>0, "Number of samples ({}) is too small.".format(len(dset))
        dset_train, dset_val = torch.utils.data.random_split(dset, [train_size, val_size])
        batch_size = -(-len(dset_train)//self.train_config['num_batches'])
        loader_train = DataLoader(dset_train, batch_size=batch_size, shuffle=True, drop_last=True)
        loader_val = DataLoader(dset_val, batch_size=batch_size)

        # stochastic gradient ascent
        optimizer = self.optim_class(dist.parameters(), **self.optim_kwargs)
        best_param, min_loss, epoch = None, np.inf, 0
        while True:
            # evaluation on validation set
            loss = 0.
            for _xs, in loader_val:
                with torch.no_grad():
                    loss += -dist.loglikelihood(_xs.numpy()).sum().item()
            loss /= len(dset_val)
            if loss<min_loss:
                best_param = dist.state_dict()
                loss = min_loss
            # loop control
            epoch += 1
            if epoch==self.train_config['num_epochs']:
                break
            # batch training
            for _xs, in loader_train:
                loss = -dist.loglikelihood(_xs.numpy()).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # update distribution parameters
        dist.load_state_dict(best_param)
