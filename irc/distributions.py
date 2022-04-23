import time
import numpy as np
import torch
from torch.nn.functional import embedding
from gym.spaces import MultiDiscrete, Box
from typing import Optional, Type, Union
from jarvis.utils import progress_str, time_str
from .utils import Array, Tensor, VarSpace, RandGen


class BasePotential(torch.nn.Module):
    r"""Base class for potential."""

    def __init__(self):
        super(BasePotential, self).__init__()

    def get_param_vec(self):
        r"""Returns the parameter vector data.

        Returns
        -------
        param_vec: (num_params,) Tensor

        """
        raise NotImplementedError

    def set_param_vec(self, param_vec: Tensor):
        r"""Sets the parameter vector data.

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
            Parameters of the potential. If `param_vec` is ``None``, use the
            internal parameters with gradient enabled.

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
            param_vec = self.param_vec
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


class BaseParamNet(torch.nn.Module):
    r"""Base class for parameter network."""

    def __init__(self,
        space: VarSpace,
        n_out: int,
    ):
        r"""
        Args
        ----
        space:
            Input variable space.
        n_out:
            Output dimension, i.e. number of distribution parameters.

        """
        super(BaseParamNet, self).__init__()
        self.space = space
        self.n_out = n_out

    def forward(self,
        xs: Array,
    ):
        r"""
        Args
        ----
        xs: (num_samples, num_vars)
            Input variable samples.

        Returns
        -------
        param_vecs: (num_samples, num_params) Tensor
            Distribution parameters as outputs.

        """
        raise NotImplementedError


class CompleteEmbedParamNet(BaseParamNet):
    r"""Parameter network using complete embedding.

    MultiDiscrete variables are treated as a one-hot vector, and MLPs are added
    further optionally.

    """

    def __init__(self,
        space: MultiDiscrete,
        n_out: int,
        mlp_sizes: Optional[list[int]] = None,
    ):
        r"""
        Args
        ----
        mlp_sizes:
            Hidden layer sizes for the appended MLP.

        """
        super(CompleteEmbedParamNet, self).__init__(space, n_out)
        self.nvec = self.space.nvec
        self.n_in = np.prod(self.nvec)
        self.mlp_sizes = [self.n_in]+(mlp_sizes or [])+[self.n_out]

        self.layers = torch.nn.ModuleList()
        for l_idx in range(len(self.mlp_sizes)-1):
            _n_in, _n_out = self.mlp_sizes[l_idx], self.mlp_sizes[l_idx+1]
            if l_idx==0:
                self.layers.append(torch.nn.Embedding(_n_in, _n_out))
            else:
                self.layers.append(torch.nn.Sequential(
                    torch.nn.ReLU(),
                    torch.nn.Linear(_n_in, _n_out),
                ))

    def forward(self,
        xs: Array,
    ):
        device = self.layers[0].weight.device
        xs = np.ravel_multi_index(xs.T, self.nvec)
        out = torch.tensor(xs, device=device, dtype=torch.long)
        for layer in self.layers:
            out = layer(out)
        return out


class BaseDistribution(torch.nn.Module):
    r"""Base class for energy based distributions.

    The probability (density) for each sample is characterized by a scalar of
    energy, which is the summation of local potentials defined over a small
    subset of variables.

    When modeling distribution p(x), the parameters of local potentials are
    used. When modeling conditional distribution p(x|y), parameters of local
    potentials are determined by a parameter network that takes y as inputs.

    """

    def __init__(self,
        x_space: VarSpace,
        idxs: Optional[list[list[int]]] = None,
        phis: Optional[list[Optional[BasePotential]]] = None,
        y_space: Optional[VarSpace] = None,
        param_net_class: Optional[Type[BaseParamNet]] = None,
        param_net_kwargs: Optional[dict] = None,
        rng: Union[RandGen, int, None] = None,
    ):
        r"""
        Args
        ----
        x_space:
            Variable space. Currently only supports MultiDiscrete or Box with
            dimension 1.
        idxs:
            Variable indices for each potential. When `idxs` is ``None``, it
            will be initialized as the full set.
        phis:
            Local potentials. If not provided, default basic potentials will be
            initialized.
        y_space:
            Conditioned variable space.
        param_net_class, param_net_kwargs:
            Class and key-word arguments for parameter network.
        rng:
            Random number generator, designed for reproducibility.

        """
        super(BaseDistribution, self).__init__()

        if isinstance(x_space, MultiDiscrete):
            self.num_vars = len(x_space.nvec)
        if isinstance(x_space, Box):
            assert len(x_space.shape)==1, "Only supports 1D box space."
            self.num_vars = x_space.shape[0]
        self.x_space = x_space

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
                if isinstance(x_space, MultiDiscrete):
                    self.phis.append(BasicDiscetePotential(self.x_space.nvec[idx]))
                if isinstance(x_space, Box):
                    self.phis.apend(BasicContinuousPotential(len(idx)))
            else:
                self.phis.append(phi)
            self.num_params.append(len(self.phis[-1].get_param_vec()))

        self.y_space = y_space
        if self.y_space is None:
            self.param_net = None
        else:
            if param_net_class is None:
                if isinstance(y_space, MultiDiscrete):
                    param_net_class = CompleteEmbedParamNet
                if isinstance(y_space, Box):
                    raise NotImplementedError
            self.param_net = param_net_class(
                space=self.y_space, n_out=sum(self.num_params),
                **(param_net_kwargs or {}),
            )

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
        ys: Optional[Array] = None,
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
        if ys is None: # p(x)
            logp = self.energy(xs)-self.logpartition()
        else: # p(x|y)
            # TODO more efficient implementation by merging same y
            param_vecs = self.param_net(ys)
            logp = []
            for x, param_vec in zip(xs, param_vecs):
                logp.append(
                    self.energy(x[None], param_vec)-self.logpartition(param_vec)
                )
            logp = torch.cat(logp, dim=0)
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

    def estimate(self,
        xs: Array,
        ys: Optional[Array] = None,
        ws: Optional[Array] = None,
        batch_size: int = 32,
        num_epochs: int = 50,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        verbose: int = 2,
    ):
        r"""Estimates p(x) or p(x|y) from samples.

        Args
        ----
        xs: (num_samples, num_vars_x)
            Data samples.
        ys: (num_samples, num_vars_y)
            Conditioned values.
        ws: (num_samples,)
            Weights of samples.
        batch_size:
            Batch size for SGD training.
        num_epochs:
            The number of epochs. For easy implementation with numpy random
            generator, batch is sampled with replacement and epoch simply
            specifies the total number of batches.
        lr, momentum, weight_decay:
            Learning rate, momentum and weight decay of SGD optimizer.
        verbose:
            Level of information display.

        """
        num_samples = len(xs)
        if ws is None:
            ws = np.ones(num_samples)
        ws /= ws.sum()
        optimizer = torch.optim.SGD(
            self.parameters() if ys is None else self.param_net.parameters(),
            lr=lr, momentum=momentum, weight_decay=weight_decay,
        )
        num_batches = num_samples*num_epochs//batch_size
        tic = time.time()
        for b_idx in range(1, num_batches+1):
            s_idxs = self.rng.choice(num_samples, batch_size, p=ws)
            if ys is None:
                loss = -self.loglikelihood(xs[s_idxs]).mean()
            else:
                loss = -self.loglikelihood(xs[s_idxs], ys[s_idxs]).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if verbose>1 and (b_idx%(-(-num_batches//6))==0 or b_idx==num_batches):
                print("{} {:.3f}".format(
                    progress_str(b_idx, num_batches), loss.item()
                    ))
        toc = time.time()
        if verbose>0:
            print("{} epochs trained on {} samples, log likelihood {:.3f} ({})".format(
                num_epochs, num_samples, -loss.item(), time_str(toc-tic),
            ))


class DiscreteDistribution(BaseDistribution):
    r"""Distribution for discrete variables."""

    def __init__(self,
        x_space: MultiDiscrete,
        **kwargs,
    ):
        r"""
        Args
        ----
        x_space:
            A MultiDiscrete space.

        """
        super(DiscreteDistribution, self).__init__(x_space, **kwargs)

    def __repr__(self):
        return 'Discrete distribution on space {}'.format(self.x_space.nvec)

    def _all_xs(self):
        r"""Returns all possible variable values.

        Returns
        -------
        xs: (num_samples, num_vars) Array
            All possible variable values, used for calculating partition
            function.

        """
        xs = np.arange(np.prod(self.x_space.nvec))
        xs = np.stack(np.unravel_index(xs, self.x_space.nvec)).T
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
            np.prod(self.x_space.nvec),
            size=None if num_samples is None else (num_samples,),
            p=p.cpu().numpy(),
        )
        xs = np.stack(np.unravel_index(xs, self.x_space.nvec)).T
        return xs
