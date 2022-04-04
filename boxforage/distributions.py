import numpy as np
import torch
from gym.spaces import MultiDiscrete, Box

from typing import Union, Optional
Array = np.ndarray
Tensor = torch.Tensor
VarSpace = Union[MultiDiscrete, Box]
RandomGenerator = np.random.Generator


class BasePotential(torch.nn.Module):
    r"""Base class for potential."""

    def __init__(self):
        super(BasePotential, self).__init__()

    def forward(self, xs: Array):
        r"""Returns energy values of data samples.

        Args
        ----
        xs: (num_samples, num_vars)
            Data samples.

        Returns
        -------
        e: (num_samples,) Tensor
            Energy values of data samples.

        """
        raise NotImplementedError


class BasicDiscetePotential(BasePotential):
    r"""Basic potential for categorical variables.

    MultiDiscrete samples are treated as one-hot vectors. Probabilities are thus
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
        self.embed = torch.nn.Embedding(np.prod(self.nvec), 1)

    def forward(self, xs: Array):
        xs = np.ravel_multi_index(xs.T, self.nvec)
        xs = torch.tensor(xs, device=self.embed.weight.device, dtype=torch.long)
        logits = self.embed(xs).squeeze(-1)
        return logits

    def set_prob(self,
        prob_dict: dict[tuple[int], float],
        eps: float = 1e-4,
    ):
        r"""Sets parameters for a given probability distribution.

        Args
        ----
        prob_dict:
            Relative probabilities for different variable values. Values not
            included are assigned with a small probability.
        eps:
            The probability that all values not in `prob_dict` account for.

        """
        self.embed.weight.data = torch.ones_like(self.embed.weight.data)
        n_small = np.prod(self.nvec)-len(prob_dict)
        if n_small>0: # initiate probabilities with small values
            self.embed.weight.data *= np.log(eps/n_small)
        z = sum(prob_dict.values())
        for x, p in prob_dict.items():
            idx = np.ravel_multi_index(x, self.nvec)
            self.embed.weight.data[idx] = np.log(p/z)
        self.embed.weight.data -= self.embed.weight.data.mean() # shift baseline


class BasicContinuousPotential(BasePotential):
    r"""Basic potential for continuous variables."""

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

    def forward(self, xs: Array):
        raise NotImplementedError


class EnergyBasedDistribution(torch.nn.Module):
    r"""Energy based distributions.

    The probability (density) for each sample is characterized by a scalar of
    energy, i.e. p(x) is proportional to exp(energy(x)). The energy function is
    the summation of local potentials defined over a small subset of variables.

    """

    def __init__(self,
        space: VarSpace,
        idxs: Optional[list[list[int]]] = None,
        phis: Optional[list[Optional[BasePotential]]] = None,
        *,
        rng: Union[RandomGenerator, int, None] = None,
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
        rng:
            Random number generator, designed for reproducibility.

        """
        super(EnergyBasedDistribution, self).__init__()
        if isinstance(space, MultiDiscrete):
            self.num_vars = len(space.nvec)
        if isinstance(space, Box):
            assert len(space.shape)==1, "Only supports 1D box space."
            self.num_vars = space.shape[0]
        self.space = space
        if idxs is None:
            assert phis is None, "Variable indices of potentials are not specified."
            self.idxs = [list(range(self.num_vars))] # one global potential by default
        else:
            for idx in idxs:
                assert set(idx).issubset(range(self.num_vars)), f"Variable indices {idx} is invalid."
            self.idxs = idxs
        if phis is None:
            phis = [None]*len(self.idxs)
        else:
            assert len(phis)==len(self.idxs), "List of variable indices and potentials should be of the same length."
        self.phis = torch.nn.ModuleList()
        for idx, phi in zip(self.idxs, phis):
            if phi is None:
                if isinstance(space, MultiDiscrete):
                    self.phis.append(BasicDiscetePotential(self.space.nvec[idx]))
                if isinstance(space, Box):
                    self.phis.apend(BasicContinuousPotential(len(idx)))
            else:
                self.phis.append(phi)

        self.rng = rng if isinstance(rng, RandomGenerator) else np.random.default_rng(rng)

    def energy(self, xs):
        r"""Returns energy values of data samples.

        Energy values from each local poentials are summed up.

        Args
        ----
        xs: (num_samples, num_vars)
            Data samples.

        Returns
        -------
        e: (num_samples,) Tensor
            Energy values of data samples.

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
        xs: (num_samples, num_vars)
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

    def __init__(self,
        space: MultiDiscrete,
        **kwargs,
    ):
        r"""
        Args
        ----
        space:
            A MultiDiscrete space.

        """
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
        with torch.no_grad():
            p = torch.nn.functional.softmax(self.energy(self._all_xs()), dim=0)
        xs = self.rng.choice(
            np.prod(self.space.nvec),
            size=None if num_samples is None else (num_samples,),
            p=p.numpy(),
        )
        xs = np.stack(np.unravel_index(xs, self.space.nvec)).T
        return xs


class IndependentDiscreteDistribution(DiscreteDistribution):
    r"""Independent discrete distribution.

    Each individual variable is associated with one potential.

    """

    def __init__(self,
        space: MultiDiscrete,
    ):
        idxs = [[i] for i in range(len(space.nvec))]
        super(IndependentDiscreteDistribution, self).__init__(space, idxs=idxs)

    def set_probs(self,
        prob_dicts: list[dict[tuple[int], float]],
        eps: float = 1e-4,
    ):
        r"""Sets probabilities for each potential.

        prob_dicts:
            A list of length `self.num_vars`, containing the probability values
            of each variable.
        eps:
            A small positive number to deal with zero probability.

        """
        for phi, prob_dict in zip(self.phis, prob_dicts):
            phi.set_prob(prob_dict, eps)

    def assign_delta(self,
        x: Array,
        eps: float = 1e-4,
    ):
        r"""Sets the distribution as a delta function on given value.

        Args
        ----
        x:
            Data sample that has a probability of 1.
        eps:
            A small positive number to deal with zero probability.

        """
        assert len(x)==self.num_vars
        prob_dicts = []
        for i in range(self.num_vars):
            prob_dicts.append({(x[i],): 1.})
        self.set_probs(prob_dicts, eps/self.num_vars)

    def sample(self, num_samples=None):
        # TODO more efficient implementation
        return super(IndependentDiscreteDistribution, self).sample(num_samples)
