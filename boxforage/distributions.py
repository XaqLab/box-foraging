import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer, SGD, Adam
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.distributions.categorical import Categorical
from gym.spaces import MultiDiscrete, Box

from typing import Union, Optional
Array = np.ndarray
Tensor = torch.Tensor


class BasePotential(nn.Module):
    r"""Base class for potential."""

    def __init__(self):
        super(BasePotential, self).__init__()

    def forward(self, xs: Array):
        r"""Returns energy values of data samples.

        Args
        xs: (num_samples, dim)
            Data samples.

        Returns
        e: (num_samples,) Tensor
            Energy values of input samples.

        """
        raise NotImplementedError


class BasicDiscetePotential(BasePotential):
    r"""Basic potential for categorical variables.

    MultiDiscrete samples are represented by one-hot vectors. Probabilities are
    encoded by an embedding layer with parameters as logits.

    """

    def __init__(self,
        nvec: list[int],
    ):
        r"""
        Args
        ----
        nvec:
            Vector of counts of each categorical variable.

        """
        super(BasicDiscetePotential, self).__init__()
        self.nvec = nvec
        self.embed = nn.Embedding(np.prod(self.nvec), 1)

    def forward(self, xs: Array):
        xs = np.ravel_multi_index(xs.T, self.nvec)
        xs = torch.tensor(xs, device=self.embed.weight.device, dtype=torch.long)
        logits = self.embed(xs).squeeze(-1)
        return logits


class BasicContinuousPotential(BasePotential):
    r"""Basic potential for continuous variables.

    """

    def __init__(self,
        num_vars: int,
    ):
        r"""
        Args
        ----
        num_vars:
            Number of continuous variables.

        """
        super(BasicContinuousPotential, self).__init__()
        self.num_vars = num_vars


class EnergyBasedDistribution(nn.Module):
    r"""Energy based distributions.

    The probability (density) for each sample is characterized by a scalar of
    energy, i.e. p(x) is proportional to exp(energy(x)). The energy function is
    the summation of local potentials defined over a small subset of variables.

    """

    def __init__(self,
        space: Union[MultiDiscrete, Box],
        idxs: Optional[list[list[int]]] = None,
        phis: Optional[list[Optional[BasePotential]]] = None,
    ):
        r"""
        Args
        ----
        space:
            Variable space. Currently only supports MultiDiscrete and Box with
            dimension 1.
        idxs:
            Variable indices for each potential. When `idxs` is ``None``, it
            will be initialized as the full set.
        phis:
            Local potentials. If not provided, default basic potentials will be
            initialized.

        """
        super(EnergyBasedDistribution, self).__init__()
        if isinstance(space, MultiDiscrete):
            self.num_vars = len(space.nvec)
        if isinstance(space, Box):
            assert len(space.shape)==1, "only supports 1D box space"
            self.num_vars = space.shape[0]
        self.space = space
        if idxs is None:
            assert phis is None, "potential support is not specified"
            self.idxs = [list(range(self.num_vars))]
        else:
            for idx in idxs:
                assert set(idx).issubset(range(self.num_vars)), f"indices {idx} is invalid"
            self.idxs = idxs
        if phis is None:
            phis = [None]*len(self.idxs)
        else:
            assert len(phis)==len(self.idxs), "list of variable indices and potentials should of the same length"
        self.phis = nn.ModuleList()
        for idx, phi in zip(self.idxs, phis):
            if phi is None:
                if isinstance(space, MultiDiscrete):
                    self.phis.append(BasicDiscetePotential(self.space.nvec[idx]))
                if isinstance(space, Box):
                    self.phis.apend(BasicContinuousPotential(len(idx)))
            else:
                self.phis.append(phi)

    def energy(self, xs):
        r"""Returns energy values of data samples.

        Energy values from each local poentials are summed up.

        """
        e = 0
        for idx, phi in zip(self.idxs, self.phis):
            e += phi(xs[:, idx])
        return e

    def logpartition(self):
        r"""Returns log partition function.

        Returns
        -------
        logz: Tensor
            A scalar whose gradient with respect to distribution parameters is
            enabled.

        """
        raise NotImplementedError

    def loglikelihood(self, xs: Array):
        r"""Returns log likelihood of data samples.

        Args
        ----
        xs: (num_samples, dim)
            Data samples.

        Returns
        -------
        logp: (num_samples,) Tensor
            Log likelihood of data samples.

        """
        logp = self.energy(xs).mean(dim=0)-self.logpartition()
        return logp

    def sample(self,
        num_samples: Optional[int] = None,
    ):
        r"""Returns data samples of the distribution.

        Args
        ----
        num_samples:
            Number of samples returned. When `num_samples` is ``None``, only one
            sample is returned.

        Returns
        -------
        xs: (num_vars,) or (num_samples, num_vars) Array
            Samples drawn from the distribution.

        """
        # TODO, Gibbs sampling
        raise NotImplementedError


class DiscreteDistribution(EnergyBasedDistribution):
    r"""Distribution for discrete variables."""

    def __init__(
        self,
        space: MultiDiscrete,
        **kwargs,
    ):
        super(DiscreteDistribution, self).__init__(space, **kwargs)

    def _all_xs(self):
        r"""Returns all possible variable values.

        Returns
        -------
        xs: (*, num_vars) Array
            All possible variable values, used for calculating partition
            function.

        """
        xs = np.arange(np.prod(self.space.nvec))
        xs = np.stack(np.unravel_index(xs, self.space.nvec)).T
        return xs

    def logpartition(self):
        r"""Returns log partition function.

        Calculate `logz = log(sum(exp(energy(x))))` directly over all possible
        variable values.

        """
        xs = self._all_xs()
        logz = torch.logsumexp(self.energy(xs), dim=0)
        return logz

    def sample(self,
        num_samples: Optional[int] = None
    ):
        dist = Categorical(logits=self.energy(self._all_xs()))
        if num_samples is None:
            xs = dist.sample()
        else:
            xs = dist.sample(sample_shape=(num_samples,))
        xs = np.stack(np.unravel_index(xs, self.space.nvec)).T
        return xs
