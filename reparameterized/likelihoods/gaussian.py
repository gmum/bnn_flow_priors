import torch
from torch.distributions import MultivariateNormal


def factorized_gaussian_with_fixed_scale_sample(
    logits: torch.Tensor, n_samples: int = 1, scale: float = 1.0
) -> torch.Tensor:
    event_dims = len(logits.shape) - 1
    d = MultivariateNormal(logits, scale**2 * torch.eye(event_dims))
    samples = d.rsample(torch.Size([n_samples]))
    return samples


def factorized_gaussian_with_fixed_scale_log_prob(
    logits: torch.Tensor, output_y: torch.Tensor, scale: float = 1.0
) -> torch.Tensor:
    event_dims = len(logits.shape) - 1
    d = MultivariateNormal(logits, scale**2 * torch.eye(event_dims))
    logliks = d.log_prob(output_y)
    return logliks
