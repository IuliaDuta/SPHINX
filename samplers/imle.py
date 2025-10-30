# -*- coding: utf-8 -*-

import math

import torch
from torch import Tensor, Size
from torch.distributions.gamma import Gamma
from torch.distributions.gumbel import Gumbel

from abc import ABC, abstractmethod

from typing import Optional

import logging

logger = logging.getLogger(__name__)


class BaseNoiseDistribution(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample(self,
               shape: Size) -> Tensor:
        raise NotImplementedError


class SumOfGammaNoiseDistribution(BaseNoiseDistribution):
    r"""
    Creates a generator of samples for the Sum-of-Gamma distribution [1], parameterized
    by :attr:`k`, :attr:`nb_iterations`, and :attr:`device`.

    [1] Mathias Niepert, Pasquale Minervini, Luca Franceschi - Implicit MLE: Backpropagating Through Discrete
    Exponential Family Distributions. NeurIPS 2021 (https://arxiv.org/abs/2106.01798)

    Example::

        >>> import torch
        >>> noise_distribution = SumOfGammaNoiseDistribution(k=5, nb_iterations=100)
        >>> noise_distribution.sample(torch.Size([5]))
        tensor([ 0.2504,  0.0112,  0.5466,  0.0051, -0.1497])

    Args:
        k (float): k parameter -- see [1] for more details.
        nb_iterations (int): number of iterations for estimating the sample.
        device (torch.devicde): device where to store samples.
    """
    def __init__(self,
                 k: float,
                 nb_iterations: int = 10,
                 device: Optional[torch.device] = None):
        super().__init__()
        self.k = k
        self.nb_iterations = nb_iterations
        self.device = device

    def sample(self,
               shape: Size) -> Tensor:
        samples = torch.zeros(size=shape, device=self.device)
        for i in range(1, self.nb_iterations + 1):
            concentration = torch.tensor(1. / self.k, device=self.device)
            rate = torch.tensor(i / self.k, device=self.device)

            gamma = Gamma(concentration=concentration, rate=rate)
            samples = samples + gamma.sample(sample_shape=shape).to(self.device)
        samples = (samples - math.log(self.nb_iterations)) / self.k
        return samples.to(self.device)


class GumbelDistribution(BaseNoiseDistribution):
    def __init__(self, loc: float = 0., scale: float = 1.0, device: torch.device = 'cpu'):
        super().__init__()
        self.loc = loc
        self._scale = scale
        self.device = device

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):
        self._scale = value

    def sample(self, shape: Size) -> Tensor:
        gumbel = Gumbel(loc=self.loc, scale=self.scale)
        samples = gumbel.sample(shape).to(self.device)
        return samples



# -*- coding: utf-8 -*-

from torch import Tensor
from abc import ABC, abstractmethod

import logging

logger = logging.getLogger(__name__)


class BaseTargetDistribution(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def params(self,
               theta: Tensor,
               dy: Tensor) -> Tensor:
        raise NotImplementedError


class TargetDistribution(BaseTargetDistribution):
    r"""
    Creates a generator of target distributions parameterized by :attr:`alpha` and :attr:`beta`.

    Example::

        >>> import torch
        >>> target_distribution = TargetDistribution(alpha=1.0, beta=1.0)
        >>> target_distribution.params(theta=torch.tensor([1.0]), dy=torch.tensor([1.0]))
        tensor([2.])

    Args:
        alpha (float): weight of the initial distribution parameters theta
        beta (float): weight of the downstream gradient dy
    """
    def __init__(self,
                 alpha: float = 1.0,
                 beta: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def params(self,
               theta: Tensor,
               dy: Tensor) -> Tensor:

        theta_prime = self.alpha * theta - self.beta * dy
        return theta_prime


import functools

import torch
from torch import Tensor

from typing import Callable, Optional

import logging

logger = logging.getLogger(__name__)


def imle(function: Callable[[Tensor], Tensor] = None,
         target_distribution: Optional[BaseTargetDistribution] = None,
         noise_distribution: Optional[BaseNoiseDistribution] = None,
         nb_samples: int = 1,
         input_noise_temperature: float = 1.0,
         target_noise_temperature: float = 1.0):
    r"""Turns a black-box combinatorial solver in an Exponential Family distribution via Perturb-and-MAP and I-MLE [1].

    The input function (solver) needs to return the solution to the problem of finding a MAP state for a constrained
    exponential family distribution -- this is the case for most black-box combinatorial solvers [2]. If this condition
    is violated though, the result would not hold and there is no guarantee on the validity of the obtained gradients.

    This function can be used directly or as a decorator.

    [1] Mathias Niepert, Pasquale Minervini, Luca Franceschi - Implicit MLE: Backpropagating Through Discrete
    Exponential Family Distributions. NeurIPS 2021 (https://arxiv.org/abs/2106.01798)
    [2] Marin Vlastelica, Anselm Paulus, Vít Musil, Georg Martius, Michal Rolínek - Differentiation of Blackbox
    Combinatorial Solvers. ICLR 2020 (https://arxiv.org/abs/1912.02175)

    Example::

        >>> from imle.wrapper import imle
        >>> from imle.target import TargetDistribution
        >>> from imle.noise import SumOfGammaNoiseDistribution
        >>> target_distribution = TargetDistribution(alpha=0.0, beta=10.0)
        >>> noise_distribution = SumOfGammaNoiseDistribution(k=21, nb_iterations=100)
        >>> @imle(target_distribution=target_distribution, noise_distribution=noise_distribution, nb_samples=100,
        >>>       input_noise_temperature=input_noise_temperature, target_noise_temperature=5.0)
        >>> def imle_solver(weights_batch: Tensor) -> Tensor:
        >>>     return torch_solver(weights_batch)

    Args:
        function (Callable[[Tensor], Tensor]): black-box combinatorial solver
        target_distribution (Optional[BaseTargetDistribution]): factory for target distributions
        noise_distribution (Optional[BaseNoiseDistribution]): noise distribution
        nb_samples (int): number of noise sammples
        input_noise_temperature (float): noise temperature for the input distribution
        target_noise_temperature (float): noise temperature for the target distribution
    """
    if target_distribution is None:
        target_distribution = TargetDistribution(alpha=1.0, beta=1.0)

    if function is None:
        return functools.partial(imle,
                                 target_distribution=target_distribution,
                                 noise_distribution=noise_distribution,
                                 nb_samples=nb_samples,
                                 input_noise_temperature=input_noise_temperature,
                                 target_noise_temperature=target_noise_temperature)

    @functools.wraps(function)
    def wrapper(input: Tensor, *args):
        class WrappedFunc(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input: Tensor, *args):
                # [BATCH_SIZE, ...]
                input_shape = input.shape
                dims = input.dim()

                batch_size = input_shape[0]
                instance_shape = input_shape[1:]

                # (B x n_sample x N x N x E) or (B x n_sample x N x E)
                perturbed_input_shape = [batch_size, nb_samples] + list(instance_shape)

                if noise_distribution is None:
                    noise = torch.zeros(size=perturbed_input_shape)
                else:
                    noise = noise_distribution.sample(shape=torch.Size(perturbed_input_shape))

                input_noise = noise * input_noise_temperature

                repeats = [1] * len(perturbed_input_shape)
                repeats[1] = nb_samples
                perturbed_input_3d = input[:, None, ...].repeat(repeats).view(perturbed_input_shape)
                perturbed_input_3d = perturbed_input_3d + input_noise

                # [BATCH_SIZE * N_SAMPLES, ...]
                perturbed_input_2d = perturbed_input_3d.view([-1] + perturbed_input_shape[2:])

                # [BATCH_SIZE * N_SAMPLES, ...]
                perturbed_output, aux_outputs = function(perturbed_input_2d)
                # [BATCH_SIZE, N_SAMPLES, ...]
                perturbed_output = perturbed_output.view(perturbed_input_shape)

                ctx.save_for_backward(input, noise, perturbed_output)

                # [BATCH_SIZE * N_SAMPLES, ...]
                if dims == 4:
                    res = perturbed_output.permute((1, 0, 2, 3, 4))
                elif dims == 3:
                    res = perturbed_output.permute((1, 0, 2, 3))
                else:
                    raise ValueError(f"Unexpected shape {perturbed_output.shape}")
                return res, aux_outputs

            @staticmethod
            def backward(ctx, dy, *args):
                # input: B x N x N x E
                # noise: B x VE x N x N x E
                # perturbed_output_3d: B x VE x N x N x E
                input, noise, perturbed_output_3d = ctx.saved_variables

                input_shape = input.shape

                dims = input.dim()
                if dims == 4:
                    dy = dy.permute((1, 0, 2, 3, 4))
                elif dims == 3:
                    dy = dy.permute((1, 0, 2, 3))
                else:
                    raise ValueError(f"Unexpected shape {dy.shape}")
                # B x VE x N x N x E
                dy_shape = dy.shape
                # B x VE x N x N x E
                noise_shape = noise.shape

                repeats = [1] * len(noise_shape)
                repeats[1] = nb_samples
                input_2d = input[:, None, ...].repeat(repeats).view(dy_shape)
                target_input_2d = target_distribution.params(input_2d, dy)

                # [BATCH_SIZE, NB_SAMPLES, ...]
                target_input_3d = target_input_2d.view(noise_shape)

                # [BATCH_SIZE, NB_SAMPLES, ...]
                target_noise = noise * target_noise_temperature

                # [BATCH_SIZE, N_SAMPLES, ...]
                perturbed_target_input_3d = target_input_3d + target_noise

                # [BATCH_SIZE * N_SAMPLES, ...]
                perturbed_target_input_2d = perturbed_target_input_3d.view((-1,) + input_shape[1:])

                # [BATCH_SIZE * N_SAMPLES, ...]
                target_output_2d, _ = function(perturbed_target_input_2d)

                # [BATCH_SIZE, N_SAMPLES, ...]
                target_output_3d = target_output_2d.view(noise_shape)

                # [BATCH_SIZE, ...]
                gradient = (perturbed_output_3d - target_output_3d)
                gradient = gradient.mean(axis=1)
                return gradient

        return WrappedFunc.apply(input, *args)
    return wrapper


LARGE_NUMBER = 1.e10

import pdb
def select_candidates(scores: torch.Tensor, k: int):
    Batch, Nmax, ensemble = scores.shape
    if k >= Nmax:
        return scores.new_ones(scores.shape)

    thresh = torch.topk(scores, k, dim=1, largest=True, sorted=True).values[:, -1, :][:, None, :]
    mask = (scores >= thresh).to(torch.float)
    return mask

class IMLEScheme:
    def __init__(self, sample_k):
        self.k = sample_k
        self.adj = None  # for potential usage

    @torch.no_grad()
    def torch_sample_scheme(self, logits: torch.Tensor):
        mask = select_candidates(logits, self.k)
        return mask, None

def get_sampler(sample_k, device, beta=1.0, noise_scale=1.0, num_train_ensemble=1.0, num_val_ensemble=1.0, nb_sample=1):
    imle_scheduler = IMLEScheme(sample_k)

    @imle( target_distribution=TargetDistribution(alpha=1.0, beta=beta),
            noise_distribution=GumbelDistribution(0., noise_scale, device),
            nb_samples=num_train_ensemble,
            input_noise_temperature=1.,
            target_noise_temperature=1., )
    def imle_train_scheme(logits: torch.Tensor):
        return imle_scheduler.torch_sample_scheme(logits)

    train_forward = imle_train_scheme

    # Do we need this??
    @imle(target_distribution=None,
                noise_distribution=GumbelDistribution(0., noise_scale, device),
                nb_samples=num_val_ensemble,
                input_noise_temperature=1. if nb_sample > 1 else 0.,
                # important
                target_noise_temperature=1., )
    def imle_val_scheme(logits: torch.Tensor):
                return imle_scheduler.torch_sample_scheme(logits)

    val_forward = imle_val_scheme
    return train_forward, val_forward

def get_sampler2(sample_k, device, beta=1.0, noise_scale=1.0, num_train_ensemble=1.0, num_val_ensemble=1.0):
    imle_scheduler = IMLEScheme(sample_k)

    def imle_train_scheme(logits: torch.Tensor):
        return imle_scheduler.torch_sample_scheme(logits)
    
    imle_train_scheme2 = imle(imle_train_scheme,
                    target_distribution=TargetDistribution(alpha=1.0, beta=beta),
                    noise_distribution=GumbelDistribution(0., noise_scale, device),
                    nb_samples=num_train_ensemble,
                    input_noise_temperature=1.,
                    target_noise_temperature=1.)
    train_forward = imle_train_scheme2

    def imle_val_scheme(logits: torch.Tensor):
        return imle_scheduler.torch_sample_scheme(logits)

    # # Do we need this??
    # @imle(target_distribution=None,
    #             noise_distribution=GumbelDistribution(0., noise_scale, device),
    #             nb_samples=num_train_ensemble,
    #             input_noise_temperature=1. if num_val_ensemble > 1 else 0.,
    #             # important
    #             target_noise_temperature=1., )
    imle_val_scheme2 = imle(imle_val_scheme,
                    target_distribution=None,
                    noise_distribution=GumbelDistribution(0., noise_scale, device),
                    nb_samples=num_val_ensemble,
                    input_noise_temperature=1. if num_val_ensemble > 1 else 0.,
                    target_noise_temperature=1.)

    val_forward = imle_val_scheme2
    return train_forward, val_forward