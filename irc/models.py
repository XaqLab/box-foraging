import time
import numpy as np
import torch
from typing import Optional, Type, Union
from jarvis.utils import progress_str, time_str

from .distributions import BaseDistribution, DiscreteDistribution
from .utils import MultiDiscrete, Box, VarSpace, Array, RandGen


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


class BaseModel:
    r"""Base class for probabilistic model.

    A model characterizes the conditional probability p(y|x). The distribution
    parameter for variable `y` is calculated by a torch module `param_net` from
    variable `x`.

    """

    def __init__(self,
        x_space: VarSpace,
        y_space: VarSpace,
        dist_class: Optional[Type[BaseDistribution]] = None,
        dist_kwargs: Optional[dict] = None,
        param_net_class: Optional[Type[BaseParamNet]] = None,
        param_net_kwargs: Optional[dict] = None,
        rng: Union[RandGen, int, None] = None,
    ):
        r"""
        Args
        ----
        x_space:
            Input variable space.
        y_space:
            Ouptut variable space.

        """
        super(BaseModel, self).__init__()
        self.x_space, self.y_space = x_space, y_space

        if dist_class is None:
            if isinstance(y_space, MultiDiscrete):
                dist_class = DiscreteDistribution
            if isinstance(y_space, Box):
                raise NotImplementedError
        self.dist = dist_class(space=self.y_space, **(dist_kwargs or {}))

        if param_net_class is None:
            if isinstance(x_space, MultiDiscrete):
                param_net_class = CompleteEmbedParamNet
            if isinstance(x_space, Box):
                raise NotImplementedError
        self.param_net = param_net_class(
            space=x_space, n_out=len(self.dist.get_param_vec()), **(param_net_kwargs or {}),
            )

        self.rng = rng if isinstance(rng, RandGen) else np.random.default_rng(rng)

    def estimate(self,
        xs: Array,
        ys: Array,
        batch_size: int = 32,
        num_epochs: int = 50,
        lr: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        verbose: int = 2,
    ):
        r"""Maximum likelihood estimation of conditional probability.

        Likelihood of observations at each given conditions are calculated and
        maximized so that `param_net` optimally describes the probability p(y|x)
        on the given dataset. Optimization is performed by stochastic gradient
        ascent on log likelihood.

        Args
        xs: (num_samples, num_vars_x)
            Input samples.
        ys: (num_samples, num_vars_y)
            Output samples.
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
        num_batches = num_samples*num_epochs//batch_size
        optimizer = torch.optim.SGD(
            self.param_net.parameters(),
            lr=lr, momentum=momentum, weight_decay=weight_decay,
        )
        tic = time.time()
        for b_idx in range(1, num_batches+1):
            loss = 0.
            for s_idx in self.rng.choice(num_samples, batch_size):
                # TODO more efficient implementation by grouping identical xs
                x, y = xs[s_idx], ys[s_idx]
                loss += -self.dist.loglikelihood(y[None], self.param_net(x[None])[0])[0]
            loss /= batch_size
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
