"""Sampling from predictive distribution."""
import torch

from .parameters import StateDict, take_parameters_sample, load_state_dict

from typing import Callable


def sample_predictive(
    input_x: torch.Tensor,
    model: torch.nn.Module,
    parameters_samples: StateDict,
    sampler: Callable,
    n_samples: int = 1,
    flatten_samples_dims: bool = True,
    **sampler_kwargs,
) -> torch.Tensor:
    predictive_samples = []
    for state_dict in take_parameters_sample(parameters_samples):

        model = load_state_dict(model, state_dict)
        logits = model.forward(input_x)

        predictive_samples1 = sampler(logits, n_samples=n_samples, **sampler_kwargs)
        assert predictive_samples1.shape == torch.Size([n_samples] + list(logits.shape))

        predictive_samples.append(predictive_samples1)
    # move ys to dim=0 and thetas to dim=1:
    predictive_samples = torch.stack(predictive_samples).transpose(0, 1)

    if flatten_samples_dims:
        predictive_samples = predictive_samples.flatten(start_dim=0, end_dim=1)
    return predictive_samples


def predictive_likelihoods(
    input_x: torch.Tensor,
    output_y: torch.Tensor,
    model: torch.nn.Module,
    parameters_samples: StateDict,
    likelihood_func: Callable,
    **likelihood_func_kwargs,
) -> torch.Tensor:
    batch_dim = torch.Size([len(output_y)])
    logliks = []
    for state_dict in take_parameters_sample(parameters_samples):

        model = load_state_dict(model, state_dict)
        logits = model.forward(input_x)

        logliks1 = likelihood_func(logits, output_y, **likelihood_func_kwargs)
        assert logliks1.shape == batch_dim

        logliks.append(logliks1)
    logliks = torch.stack(logliks)
    return logliks
