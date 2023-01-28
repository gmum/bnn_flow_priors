"""Joint sampling for multiple parameters."""

import torch

from typing import Callable, Iterable, Tuple


def _multiparameter_sampler_unpack(sampler, parameter2shape):
    """Splits flattened output of sampler into individual parameters according to parameter2shape."""

    def wrapped_sampler(n_samples=1):
        joint_samples, joint_nlls = sampler(n_samples)

        samples = []
        start_ix = 0
        for parameter_name, shape in parameter2shape:
            npositions = shape.numel()
            parameter_samples = joint_samples[
                :, start_ix : (start_ix + npositions), ...
            ]
            parameter_samples = parameter_samples.reshape(
                torch.Size([n_samples]) + shape
            )

            # samples[parameter_name] = parameter_samples   # return a dict
            samples.append(parameter_samples)  # return an ordered list

            start_ix += npositions
        return samples, joint_nlls

    return wrapped_sampler


def create_multiparameter_sampler(
    sampler_create_func: Callable,
    named_parameters: Iterable[Tuple[str, torch.Tensor]],
    **sampler_create_func_args
):
    named_parameters = list(named_parameters)

    # flatten to a single vector:
    parameters_shapes = [p.shape for _, p in named_parameters]
    parameters_jointly = torch.concat([p.flatten() for _, p in named_parameters])
    sampler, variational_params, aux_objs = sampler_create_func(
        parameters_jointly,
        parameters_shapes=parameters_shapes,  # additional information which may be used by samplers
        **sampler_create_func_args
    )

    parameter2shape = [(n, p.shape) for n, p in named_parameters]
    sampler = _multiparameter_sampler_unpack(sampler, parameter2shape)
    return sampler, variational_params, aux_objs
