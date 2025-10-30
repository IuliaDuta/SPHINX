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

class BaseTargetDistribution(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def params(self,
               theta: Tensor,
               dy: Tensor) -> Tensor:
        raise NotImplementedError

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
        do_gradient_scaling (bool): whether to scale the gradient by 1/λ or not
    """
    def __init__(self,
                 alpha: float = 1.0,
                 beta: float = 1.0,
                 do_gradient_scaling: bool = False,
                 eps: float = 1e-7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.do_gradient_scaling = do_gradient_scaling
        self.eps = eps

    def params(self,
               theta: Tensor,
               dy: Optional[Tensor],
               alpha: Optional[float] = None,
               beta: Optional[float] = None,
               _is_minimization: bool = False) -> Tensor:
        alpha_ = self.alpha if alpha is None else alpha
        beta_ = self.beta if beta is None else beta

        if _is_minimization is True:
            theta_prime = alpha_ * theta + beta_ * (dy if dy is not None else 0.0)
        else:
            theta_prime = alpha_ * theta - beta_ * (dy if dy is not None else 0.0)
        return theta_prime

    def process(self,
                theta: Tensor,
                dy: Tensor,
                gradient_3d: Tensor) -> Tensor:
        scaling_factor = max(self.beta, self.eps)
        res = (gradient_3d / scaling_factor) if self.do_gradient_scaling is True else gradient_3d
        return res

class AdaptiveTargetDistribution(BaseTargetDistribution):
    def __init__(self,
                 initial_alpha: float = 1.0,
                 initial_beta: float = 1.0,
                 initial_grad_norm: float = 1.0,
                 # Pitch: the initial default hyperparams lead to very stable results,
                 # competitive with manually tuned ones -- E.g. try with 1e-3 for this hyperparam
                 beta_update_step: float = 0.0001,
                 beta_update_momentum: float = 0.0,
                 grad_norm_decay_rate: float = 0.9,
                 target_norm: float = 1.0):
        super().__init__()
        self.alpha = initial_alpha
        self.beta = initial_beta

        self.grad_norm = initial_grad_norm
        self.beta_update_step = beta_update_step
        self.beta_update_momentum = beta_update_momentum
        self.previous_beta_update = 0.0
        self.grad_norm_decay_rate = grad_norm_decay_rate
        self.target_norm = target_norm

    def _perturbation_magnitude(self,
                                theta: Tensor,
                                dy: Optional[Tensor]):
        norm_dy = torch.linalg.norm(dy).item() if dy is not None else 1.0
        return 0.0 if norm_dy <= 0.0 else self.beta * (torch.linalg.norm(theta) / norm_dy)

    def params(self,
               theta: Tensor,
               dy: Optional[Tensor],
               _is_minimization: bool = False) -> Tensor:
        pm = self._perturbation_magnitude(theta, dy)
        if _is_minimization is True:
            theta_prime = self.alpha * theta + pm * (dy if dy is not None else 0.0)
        else:
            theta_prime = self.alpha * theta - pm * (dy if dy is not None else 0.0)
        return theta_prime

    def process(self,
                theta: Tensor,
                dy: Tensor,
                gradient_3d: Tensor) -> Tensor:
        batch_size = gradient_3d.shape[0]
        nb_samples = gradient_3d.shape[1]
        pm = self._perturbation_magnitude(theta, dy)

        # We compute an exponentially decaying sum of the gradient norms
        grad_nnz = torch.count_nonzero(gradient_3d).float()
        nb_gradients = batch_size * nb_samples

        # print('GRAD', gradient_3d.shape, 'GRAD NNZ', grad_nnz, batch_size, nb_samples, grad_nnz / nb_gradients)
        # print(gradient_3d[0, 0].int())

        # Running estimate of the gradient norm (number of non-zero elements for every sample)
        self.grad_norm = self.grad_norm_decay_rate * self.grad_norm + \
                         (1.0 - self.grad_norm_decay_rate) * (grad_nnz / nb_gradients)

        # If the gradient norm is lower than 1, we increase beta; otherwise, we decrease beta.
        beta_update_ = (1.0 if self.grad_norm.item() < self.target_norm else - 1.0) * self.beta_update_step
        beta_update = (self.beta_update_momentum * self.previous_beta_update) + beta_update_

        # Enforcing \beta \geq 0
        self.beta = max(self.beta + beta_update, 0.0)
        self.previous_beta_update = beta_update

        # print(f'Gradient norm: {self.grad_norm:.5f}\tBeta: {self.beta:.5f}')
        res = gradient_3d / (pm if pm > 0.0 else 1.0)
        return res

import functools

import torch
from torch import Tensor

from typing import Callable, Optional

import logging

logger = logging.getLogger(__name__)


def aimle(function: Optional[Callable[[Tensor], Tensor]] = None,
          target_distribution: Optional[BaseTargetDistribution] = None,
          noise_distribution: Optional[BaseNoiseDistribution] = None,
          nb_samples: int = 1,
          nb_marginal_samples: int = 1,
          theta_noise_temperature: float = 1.0,
          target_noise_temperature: float = 1.0,
          symmetric_perturbation: bool = False,
          _is_minimization: bool = False):
    r"""Turns a black-box combinatorial solver in an Exponential Family distribution via Perturb-and-MAP and I-MLE [1].

    The theta function (solver) needs to return the solution to the problem of finding a MAP state for a constrained
    exponential family distribution -- this is the case for most black-box combinatorial solvers [2]. If this condition
    is violated though, the result would not hold and there is no guarantee on the validity of the obtained gradients.

    This function can be used directly or as a decorator.

    [1] Mathias Niepert, Pasquale Minervini, Luca Franceschi - Implicit MLE: Backpropagating Through Discrete
    Exponential Family Distributions. NeurIPS 2021 (https://arxiv.org/abs/2106.01798)
    [2] Marin Vlastelica, Anselm Paulus, Vít Musil, Georg Martius, Michal Rolínek - Differentiation of Blackbox
    Combinatorial Solvers. ICLR 2020 (https://arxiv.org/abs/1912.02175)

    Example::

        >>> from imle.aimle import aimle
        >>> from imle.target import TargetDistribution
        >>> from imle.noise import SumOfGammaNoiseDistribution
        >>> target_distribution = TargetDistribution(alpha=0.0, beta=10.0)
        >>> noise_distribution = SumOfGammaNoiseDistribution(k=21, nb_iterations=100)
        >>> @aimle(target_distribution=target_distribution, noise_distribution=noise_distribution, nb_samples=100,
        >>>        theta_noise_temperature=theta_noise_temperature, target_noise_temperature=5.0)
        >>> def aimle_solver(weights_batch: Tensor) -> Tensor:
        >>>     return torch_solver(weights_batch)

    Args:
        function (Callable[[Tensor], Tensor]): black-box combinatorial solver
        target_distribution (Optional[BaseTargetDistribution]): factory for target distributions
        noise_distribution (Optional[BaseNoiseDistribution]): noise distribution
        nb_samples (int): number of noise samples
        nb_marginal_samples (int): number of noise samples used to compute the marginals
        theta_noise_temperature (float): noise temperature for the theta distribution
        target_noise_temperature (float): noise temperature for the target distribution
        symmetric_perturbation (bool): whether it uses the symmetric version of IMLE
        _is_minimization (bool): whether MAP is solving an argmin problem
    """
    # if target_distribution is None:
    #     target_distribution = TargetDistribution(alpha=1.0, beta=1.0)

    if function is None:
        return functools.partial(aimle,
                                 target_distribution=target_distribution,
                                 noise_distribution=noise_distribution,
                                 nb_samples=nb_samples,
                                 nb_marginal_samples=nb_marginal_samples,
                                 theta_noise_temperature=theta_noise_temperature,
                                 target_noise_temperature=target_noise_temperature,
                                 symmetric_perturbation=symmetric_perturbation,
                                 _is_minimization=_is_minimization)

    @functools.wraps(function)
    def wrapper(theta: Tensor, *args):
        class WrappedFunc(torch.autograd.Function):
            @staticmethod
            def forward(ctx, theta: Tensor, *args):
                # [BATCH_SIZE, ...]
                theta_shape = theta.shape

                batch_size = theta_shape[0]
                instance_shape = theta_shape[1:]

                nb_total_samples = nb_samples * nb_marginal_samples

                # [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                perturbed_theta_shape = [batch_size, nb_total_samples] + list(instance_shape)

                # ε ∼ ρ(ε)
                # [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                if noise_distribution is None:
                    noise = torch.zeros(size=torch.Size(perturbed_theta_shape))
                else:
                    noise = noise_distribution.sample(shape=torch.Size(perturbed_theta_shape))

                # [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                eps = noise * theta_noise_temperature

                # [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                perturbed_theta_3d = theta.view(batch_size, 1, -1).repeat(1, nb_total_samples, 1).view(perturbed_theta_shape)
                perturbed_theta_3d = perturbed_theta_3d + eps

                # [BATCH_SIZE * N_TOTAL_SAMPLES, ...]
                perturbed_theta_2d = perturbed_theta_3d.view([-1] + perturbed_theta_shape[2:])

                perturbed_theta_2d_shape = perturbed_theta_2d.shape
                assert perturbed_theta_2d_shape[0] == batch_size * nb_total_samples

                # z = MAP(θ + ε)
                # [BATCH_SIZE * N_TOTAL_SAMPLES, ...]
                z_2d = function(perturbed_theta_2d)
             
                assert z_2d.shape == perturbed_theta_2d_shape

                # [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                z_3d = z_2d.view(perturbed_theta_shape)


                ctx.save_for_backward(theta, noise, z_3d)

                # [BATCH_SIZE * N_TOTAL_SAMPLES, ...]
                return z_2d

            @staticmethod
            def backward(ctx, dy):
                # theta: [BATCH_SIZE, ...]
                # noise: [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                # z_3d: [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                theta, noise, z_3d = ctx.saved_tensors

                nb_total_samples = nb_samples * nb_marginal_samples
                assert noise.shape[1] == nb_total_samples

                theta_shape = theta.shape
                instance_shape = theta_shape[1:]

                batch_size = theta_shape[0]

                # dy is [BATCH_SIZE * N_TOTAL_SAMPLES, ...]
                dy_shape = dy.shape
                # noise is [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                noise_shape = noise.shape

                assert noise_shape == z_3d.shape

                # [BATCH_SIZE * NB_SAMPLES, ...]
                theta_2d = theta.view(batch_size, 1, -1).repeat(1, nb_total_samples, 1).view(dy_shape)
                # θ'_R = θ - λ dy
                target_theta_r_2d = target_distribution.params(theta_2d, dy,
                                                               _is_minimization=_is_minimization)
                # θ'_L = θ + λ dy -- if symmetric_perturbation is False, then this reduces to θ'_L = θ
                target_theta_l_2d = target_distribution.params(theta_2d, - dy if symmetric_perturbation else None,
                                                               _is_minimization=_is_minimization)

                # [BATCH_SIZE, NB_SAMPLES, ...]
                target_theta_r_3d = target_theta_r_2d.view(noise_shape)
                target_theta_l_3d = target_theta_l_2d.view(noise_shape)

                # [BATCH_SIZE, NB_SAMPLES, ...]
                eps = noise * target_noise_temperature

                # [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                perturbed_target_theta_r_3d = target_theta_r_3d + eps
                perturbed_target_theta_l_3d = target_theta_l_3d + eps

                # [BATCH_SIZE * N_TOTAL_SAMPLES, ...]
                perturbed_target_theta_r_2d = perturbed_target_theta_r_3d.view(dy_shape)
                perturbed_target_theta_l_2d = perturbed_target_theta_l_3d.view(dy_shape)

                # [BATCH_SIZE * N_TOTAL_SAMPLES, ...]

                with torch.inference_mode():
                    # z'_R = MAP(θ'_R + ε)
                    z_r_2d = function(perturbed_target_theta_r_2d)

                    # z'_L = MAP(θ'_L + ε)
                    z_l_2d = function(perturbed_target_theta_l_2d)

                # [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                z_r_3d = z_r_2d.view(noise_shape)
                z_l_3d = z_l_2d.view(noise_shape)

                if nb_marginal_samples > 1:
                    assert batch_size == z_l_3d.shape[0] == z_r_3d.shape[0]
                    assert nb_total_samples == z_l_3d.shape[1] == z_r_3d.shape[1]

                    # [BATCH_SIZE, N_SAMPLES, N_MARGINAL_SAMPLES, ...]
                    z_l_4d = z_l_3d.view([batch_size, nb_samples, nb_marginal_samples] + list(instance_shape))
                    z_r_4d = z_r_3d.view([batch_size, nb_samples, nb_marginal_samples] + list(instance_shape))

                    z_l_3d = torch.mean(z_l_4d, dim=2)
                    z_r_3d = torch.mean(z_r_4d, dim=2)

                # g = z'_L - z'_R
                # Note that if symmetric_perturbation is False, then z'_L = z
                # [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                gradient_3d = z_l_3d - z_r_3d

                if symmetric_perturbation is True:
                    gradient_3d = gradient_3d / 2.0

                # [BATCH_SIZE, N_TOTAL_SAMPLES, ...]
                gradient_3d = target_distribution.process(theta, dy, gradient_3d)

                # [BATCH_SIZE, ...]
                gradient = gradient_3d.mean(dim=1)

                return (- gradient) if _is_minimization is True else gradient

        return WrappedFunc.apply(theta, *args)
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
        # print(mask.shape)
        return mask

def get_sampler_aimle(sample_k, device, beta=1.0, noise_scale=1.0, num_train_ensemble=1.0, num_val_ensemble=1.0, noise_distribution=None, nb_sample=1):
    aimle_scheduler = IMLEScheme(sample_k)

    # target_distribution=target_distribution, noise_distribution=noise_distribution, nb_samples=100,
    #     >>>        theta_noise_temperature=theta_noise_temperature, target_noise_temperature=5.0

    if noise_distribution == 'gumbel':
        train_noise_distribution = GumbelDistribution(0., noise_scale, device)
    elif noise_distribution == 'sog':
        train_noise_distribution = SumOfGammaNoiseDistribution(5, 10, device)
        # train_noise_distribution = SumOfGammaNoiseDistribution(20, 50, device)


    @aimle(target_distribution=AdaptiveTargetDistribution(initial_alpha=1.0,
                                                         initial_beta=beta,
                                                         beta_update_step=1e-4),
            noise_distribution=train_noise_distribution,
            nb_samples=num_train_ensemble,
            theta_noise_temperature=1.,
            target_noise_temperature=1., )
    # aimle(target_distribution= TargetDistribution(alpha=1.0,
    #                                                      beta=beta),
    #         noise_distribution=GumbelDistribution(0., noise_scale, device),
    #         nb_samples=num_train_ensemble,
    #         theta_noise_temperature=1.,
    #         target_noise_temperature=1., )
    def aimle_train_scheme(logits: torch.Tensor):
        return aimle_scheduler.torch_sample_scheme(logits)

    train_forward = aimle_train_scheme

    if noise_distribution == 'gumbel':
        val_noise_distribution = GumbelDistribution(0., noise_scale, device)
    elif noise_distribution == 'sog':
        val_noise_distribution = SumOfGammaNoiseDistribution(5, 10, device)
        # val_noise_distribution = SumOfGammaNoiseDistribution(20, 1000, device)

    # Do we need this??
    @aimle(target_distribution=None,
                noise_distribution=val_noise_distribution,
                nb_samples=num_val_ensemble,
                theta_noise_temperature=1. if nb_sample > 1 else 0.,
                # important
                target_noise_temperature=1., )
    def aimle_val_scheme(logits: torch.Tensor):
                return aimle_scheduler.torch_sample_scheme(logits)

    val_forward = aimle_val_scheme
    return train_forward, val_forward


    
