"""Basic posterior models (sampling functions + variational parameters)."""

import torch
from torch.nn.functional import softplus
from torch.distributions import Normal, MultivariateNormal

from typing import Tuple, Callable, Dict


def create_factorized_gaussian_sampler(
    parameter: torch.Tensor,
    device=None,
    epsilon_scale=1e-8,
    loc_initalization=lambda parameter: parameter.clone().detach(),
    uscale_initialization=lambda parameter: torch.randn_like(parameter),
    **ignored_params,
) -> Tuple[Callable, Dict[str, torch.Tensor], Dict[str, object]]:
    """Creates a function that samples from Normal(loc, softplus(unnormalized_scale)).

    Returns:
        sampling function: which takes n_samples and outputs tuple(sample, NLL)
        dictionary {name: tensor} with variational parameters (loc and unnormalized_scale)
        dictionary with auxiliary objects
    """
    loc = loc_initalization(parameter)
    loc = loc.requires_grad_(True).to(device or parameter.device)

    unnormalized_scale = uscale_initialization(parameter)
    unnormalized_scale = unnormalized_scale.requires_grad_(True).to(
        device or parameter.device
    )

    def sample_factorized_gaussian(n_samples=1):
        q = Normal(loc, softplus(unnormalized_scale) + epsilon_scale)
        sample = q.rsample(torch.Size([n_samples]))

        # calc total NLL for all params (out shape==n_samples)
        data_dims = list(range(1, len(sample.shape)))
        nll = -q.log_prob(sample).sum(dim=data_dims)
        nll = nll.to(sample.device)

        return sample, nll

    variational_params = {"loc": loc, "unnormalized_scale": unnormalized_scale}
    return sample_factorized_gaussian, variational_params, {}


def create_gaussian_tril_sampler(
    parameter: torch.Tensor,
    device=None,
    **ignored_params,
) -> Tuple[Callable, Dict[str, torch.Tensor], Dict[str, object]]:
    """Creates a function that samples from MultivariateNormal(loc, cov).

    The covariance matrix is composed as cov := diag(softplus(unnormalized_diag)) + tril(cov_tril)
    Returns:
        sampling function: which takes n_samples and outputs tuple(sample, NLL)
        dictionary {name: tensor} with variational parameters (loc, cov parameters)
        dictionary with auxiliary objects (currently empty)
    """
    loc = (
        parameter.flatten()
        .clone()
        .detach()
        .requires_grad_(True)
        .to(device or parameter.device)
    )
    unnormalized_diag = (
        torch.randn_like(loc).requires_grad_(True).to(device or parameter.device)
    )
    cov_tril = (
        torch.randn(torch.Size([loc.shape[0], loc.shape[0]]))
        .requires_grad_(True)
        .to(device or parameter.device)
    )

    def sample_gaussian_tril(n_samples=1):
        cov = torch.tril(cov_tril, diagonal=-1) + torch.diag(
            softplus(unnormalized_diag) + 1e-8
        )
        q = MultivariateNormal(loc=loc, scale_tril=cov)
        sample = q.rsample(torch.Size([n_samples]))

        # calc total NLL for all params (shape==n_samples)
        nll = -q.log_prob(sample)
        nll = nll.to(sample.device)

        sample = sample.reshape(torch.Size([n_samples]) + parameter.shape)
        return sample, nll

    variational_params = {
        "loc": loc,
        "unnormalized_diag": unnormalized_diag,
        "cov_tril": cov_tril,
    }
    return sample_gaussian_tril, variational_params, {}
