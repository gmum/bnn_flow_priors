import torch
from typing import Tuple, Callable, Dict


def create_delta_distribution_sampler(
    parameter: torch.Tensor,
    device=None,
    loc_initalization=lambda parameter: parameter.clone().detach(),
    **ignored_params,
) -> Tuple[Callable, Dict[str, torch.Tensor], Dict[str, object]]:
    """Creates a function that samples from delta distribution.

    Returns:
        sampling function: which takes n_samples and outputs tuple(sample, NLL)
        dictionary {name: tensor} with variational parameters (loc)
        dictionary with auxiliary objects (currently empty)
    """
    loc = loc_initalization(parameter)
    loc = loc.requires_grad_(True).to(device or parameter.device)

    def sample_delta_distribution(n_samples=1):
        sample = loc
        sample_shape = torch.Size([n_samples] + [-1 for _ in sample.shape])
        sample = sample.expand(sample_shape)

        nll = torch.zeros(n_samples, dtype=sample.dtype)
        nll = nll.to(sample.device)

        return sample, nll

    variational_params = {"loc": loc}
    return sample_delta_distribution, variational_params, {}
