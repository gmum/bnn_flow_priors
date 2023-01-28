"""Modeling posteriors with flows."""
import logging
import sys
from typing import Callable
from .realnvp import build_realnvp
import torch


def create_flow_sampler(
    parameter: torch.Tensor,
    device=None,
    build_flow_func: Callable = build_realnvp,
    **build_flow_kwargs
):
    """Creates a function that samples from a normalizing flow.

    Args:
        parameter: tensor which we are sampling for
        build_flow_func: a function that takes args
            (output_dim=parameter.numel(), **build_flow_kwargs) and creates a flow.
            The flow must provide a method sample(n_samples, output_dim, calculate_nll).

    Returns:
        A tuple consisting of:
         - sampling function: which takes n_samples and outputs tuple(sample, NLL)
         - dictionary {name: tensor} with variational parameters (flow parameters)
         - dictionary with auxiliary objects: {"flow": flow object}
    """
    device = device or parameter.device
    flow = build_flow_func(output_dim=parameter.numel(), device=device, **build_flow_kwargs)

    def sample(n_samples=1):
        sample, nll = flow.sample(n_samples, parameter.numel(), calculate_nll=True)

        sample = sample.reshape(n_samples, *parameter.size())
        nll = nll.to(sample.device)

        return sample, nll

    variational_params = {n: p for n, p in flow.named_parameters()}
    return sample, variational_params, {"flow": flow}
