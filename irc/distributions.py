import numpy as np
import torch
from torch.nn.functional import embedding
from gym.spaces import MultiDiscrete, Box

from typing import Union, Optional
from .utils import Array, Tensor, VarSpace, RandGen


class BasePotential(torch.nn.Module):
    r"""Base class for potential."""

    def __init__(self):
        super(BasePotential, self).__init__()

    def get_param_vec(self):
        r"""Returns the parameter vector.

        Returns
        -------
        param_vec: (num_params,) Tensor

        """
        raise NotImplementedError

    def set_param_vec(self, param_vec: Tensor):
        r"""Sets the parameter vector.

        Args
        ----
        param_vec: (num_params,)
            Parameters vector.

        """
        raise NotImplementedError

    def forward(self,
        xs: Array,
        param_vec: Optional[Tensor] = None,
    ):
        r"""Returns energy values of data samples.

        Args
        ----
        xs: (num_samples, num_vars)
            Data samples.
        param_vec: (num_params,)
            Parameters of the potential. Use `get_param_vec` if it is ``None``.

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
            Vector of counts for each categorical variable.

        """
        super(BasicDiscetePotential, self).__init__()
        self.nvec = nvec
        self.num_vals = np.prod(self.nvec)
        self.param_vec = torch.nn.Parameter(torch.randn(self.num_vals))

    def get_param_vec(self):
        return self.param_vec.data

    def set_param_vec(self, param_vec):
        self.param_vec.data = param_vec

    def forward(self,
        xs: Array,
        param_vec: Optional[Tensor] = None,
    ):
        if param_vec is None:
            param_vec = self.get_param_vec()
        xs = np.ravel_multi_index(xs.T, self.nvec)
        xs = torch.tensor(xs, device=param_vec.device, dtype=torch.long)
        logits = embedding(xs, param_vec[:, None])[:, 0]
        return logits

    def set_from_prob(self,
        prob_dict: dict[tuple[int], float],
        eps: float = 1e-4,
    ):
        r"""Sets parameters from a given probability mass function.

        Args
        ----
        prob_dict:
            Relative probabilities for different variable values. Values not
            included are assigned with a small probability.
        eps:
            The probability that all values not in `prob_dict` account for.

        """
        param_vec = torch.ones(self.num_vals, device=self.param_vec.device)
        n_small = self.num_vals-len(prob_dict)
        if n_small>0:
            param_vec *= np.log(eps/n_small) # initiate probabilities with small values
        else:
            eps = 0 # all probabilities are specified
        z = sum(prob_dict.values())
        for x, p in prob_dict.items():
            idx = np.ravel_multi_index(x, self.nvec)
            param_vec[idx] = np.log(p/z*(1-eps))
        param_vec -= param_vec.mean()
        self.set_param_vec(param_vec)


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


class BaseDistribution(torch.nn.Module):
    r"""Base class for energy based distributions.

    The probability (density) for each sample is characterized by a scalar of
    energy, i.e. p(x) is proportional to exp(energy(x)). The energy function is
    the summation of local potentials defined over a small subset of variables.

    """

    def __init__(self,
        space: VarSpace,
        idxs: Optional[list[list[int]]] = None,
        phis: Optional[list[Optional[BasePotential]]] = None,
        *,
        rng: Union[RandGen, int, None] = None,
    ):
        r"""
        Args
        ----
        space:
            Variable space. Currently only supports MultiDiscrete or Box with
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
        super(BaseDistribution, self).__init__()

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
        self.num_params = []
        for idx, phi in zip(self.idxs, phis):
            if phi is None:
                if isinstance(space, MultiDiscrete):
                    self.phis.append(BasicDiscetePotential(self.space.nvec[idx]))
                if isinstance(space, Box):
                    self.phis.apend(BasicContinuousPotential(len(idx)))
            else:
                self.phis.append(phi)
            self.num_params.append(len(self.phis[-1].get_param_vec()))

        self.rng = rng if isinstance(rng, RandGen) else np.random.default_rng(rng)

    def get_param_vec(self):
        r"""Returns the concatenated parameter vector.

        Returns
        -------
        param_vec: (num_params,) Tensor

        """
        param_vec = torch.cat([phi.get_param_vec() for phi in self.phis])
        return param_vec

    def set_param_vec(self, param_vec):
        r"""Sets the concatenated parameter vector.

        Args
        ----
        param_vec: (num_params,)
            Concatenated parameters for all potentials.

        """
        c_p = 0
        for phi, n_p in zip(self.phis, self.num_params):
            phi.set_param_vec(param_vec[c_p:c_p+n_p])
            c_p += n_p

    def energy(self,
        xs: Array,
        param_vec: Optional[Tensor] = None,
    ):
        r"""Returns energy values of data samples.

        Energy values from each local poentials are summed up.

        Args
        ----
        xs: (num_samples, num_vars)
            Data samples.
        param_vec: (num_params,)
            Concatenated parameters for all potentials.

        Returns
        -------
        e: (num_samples,) Tensor
            Energy values of data samples.

        """
        e, c_p = 0, 0
        for idx, phi, n_p in zip(self.idxs, self.phis, self.num_params):
            e += phi(xs[:, idx], None if param_vec is None else param_vec[c_p:c_p+n_p])
            c_p += n_p
        return e

    def logpartition(self,
        param_vec: Optional[Tensor] = None,
    ):
        r"""Returns log partition function.

        Args
        ----
        param_vec: (num_params,)
            Concatenated parameters for all potentials.

        Returns
        -------
        logz: Tensor
            A scalar whose gradient with respect to distribution parameters is
            enabled.

        """
        raise NotImplementedError

    def loglikelihood(self,
        xs: Array,
        param_vec: Optional[Tensor] = None,
    ):
        r"""Returns log likelihood of data samples.

        Args
        ----
        xs: (num_samples, num_vars)
            Data samples.
        param_vec: (num_params,)
            Concatenated parameters for all potentials.

        Returns
        -------
        logp: (num_samples,) Tensor
            Log likelihood of data samples.

        """
        logp = self.energy(xs, param_vec)-self.logpartition(param_vec)
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
        # TODO Gibbs sampling
        raise NotImplementedError


class DiscreteDistribution(BaseDistribution):
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

    def __repr__(self):
        return 'Discrete distribution on space {}'.format(self.space.nvec)

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

    def logpartition(self,
        param_vec: Optional[Tensor] = None,
    ):
        r"""Returns log partition function.

        Calculate `logz = log(sum(exp(energy(x))))` directly over all possible
        variable values.

        """
        # TODO more efficient implementation by using phis structure
        xs = self._all_xs()
        logz = torch.logsumexp(self.energy(xs, param_vec), dim=0)
        return logz

    def sample(self,
        num_samples: Optional[int] = None
    ):
        with torch.no_grad():
            p = torch.nn.functional.softmax(self.energy(self._all_xs()), dim=0)
        xs = self.rng.choice(
            np.prod(self.space.nvec),
            size=None if num_samples is None else (num_samples,),
            p=p.cpu().numpy(),
        )
        xs = np.stack(np.unravel_index(xs, self.space.nvec)).T
        return xs
