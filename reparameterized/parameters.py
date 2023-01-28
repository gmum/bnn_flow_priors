"""Code for sampling parameters and loading them for native pytorch models."""
import torch

import warnings
from typing import Iterable, Union, Tuple, Callable, Dict, Generator


StateDict = Dict[str, torch.Tensor]

ParamsKey = Union[str, Iterable[str]]
NLLs = Dict[ParamsKey, torch.Tensor]
Samplers = Union[Dict[ParamsKey, Callable], Iterable[Tuple[ParamsKey, Callable]]]
DensityEsimators = Samplers  # an alias to avoid repeating the same structure


def load_state_dict(module, state_dict: StateDict, path=""):
    """Sets model params to samples from e.g. approximate posterior.

    Args:
        module: torch module instance
        state_dict: dictionary {parameter_name/path: sample_value (tensor)}
    """
    for name, m in module._modules.items():
        load_state_dict(m, state_dict, path=f"{path}.{name}")

    for name in module._parameters.keys():
        sample_path = f"{path}.{name}"[1:]  # skip the leading dot
        new_value = state_dict[sample_path]

        assert new_value.shape == module._parameters[name].shape, (
            f"sample_path={sample_path} shape={new_value.shape} "
            f"current shape={module._parameters[name].shape}"
        )
        module._parameters[name] = new_value

    return module


def sample_parameters(
    parameters2sampler: Samplers, n_samples: int = 1
) -> Tuple[StateDict, NLLs]:
    """Samples model parameters using predefined samplers.

    Args:
        parameters2sampler: pairs: parameter_name(s) => sampling function

    Returns:
        Two dictionaries: {parameter_name: samples} and {parameter_name(s): NLLs}
    """
    if hasattr(parameters2sampler, "items"):
        parameters2sampler = parameters2sampler.items()

    samples, nlls = {}, {}
    for parameters, sampler in parameters2sampler:
        parameters_samples, parameters_nlls = sampler(n_samples)

        nlls[parameters] = parameters_nlls

        if isinstance(parameters, str):
            samples[parameters] = parameters_samples

        elif isinstance(parameters_samples, dict):
            samples.update(parameters_samples)

        elif isinstance(parameters_samples, Iterable):
            parameters_samples = list(parameters_samples)
            assert len(parameters) == len(parameters_samples)
            for name, value in zip(parameters, parameters_samples):
                samples[name] = value

        else:
            raise Exception(
                f"I don't how to handle samples for parameters={parameters}!"
            )

    return samples, nlls


def take_parameters_sample(parameters_samples: StateDict):
    """Yields state dictionaries {parameter_name: next(parameters_sample)}.

    Args:
        parameters_samples: dictionary {parameter_name: tensor with samples in axis=0}
    """

    def _samples_len(samples):
        n_samples = min(len(v) for _, v in samples.items())
        if not all((len(v) == n_samples) for _, v in samples.items()):
            warnings.warn(
                "Not all samples have the same length. "
                f"Setting n_samples to the minimum = {n_samples}."
            )
        return n_samples

    for sample_no in range(_samples_len(parameters_samples)):
        yield {p: s[sample_no] for p, s in parameters_samples.items()}


def is_parameter_handled(parameters2sampler: Samplers, parameter_name: str) -> bool:
    if hasattr(parameters2sampler, "items"):
        parameters2sampler = parameters2sampler.items()

    for parameters, _ in parameters2sampler:
        if parameter_name in parameters or parameter_name == parameters:
            return True

    return False


def estimate_parameters_nll(
    parameters2nllfunc: DensityEsimators, state_dict: StateDict
) -> NLLs:
    if hasattr(parameters2nllfunc, "items"):
        parameters2nllfunc = parameters2nllfunc.items()

    parameters2nll = {}
    for parameters, nll_estimator in parameters2nllfunc:

        if isinstance(parameters, str):
            parameters_value = state_dict[parameters]

        elif isinstance(parameters, Iterable):
            # extract multiple parameters and pass them as a list
            parameters_value = [state_dict[parameter] for parameter in parameters]

        else:
            raise Exception(f"I don't how to handle parameters={parameters}!")

        parameters2nll[parameters] = nll_estimator(parameters_value)

    return parameters2nll
