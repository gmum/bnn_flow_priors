"""Wrapping """
import logging

import torch
from typing import Callable, Dict, Optional, Tuple

from .parameters import is_parameter_handled, sample_parameters, estimate_parameters_nll
from .parameters import StateDict, NLLs, ParamsKey
from .predictive import sample_predictive, predictive_likelihoods


class BayesianNeuralNetwork:
    """Manages sampling and NLL calculation for parameters of a native module."""

    def __init__(self, module: torch.nn.Module) -> None:
        self.parameters2sampler = {}  # posterior sampling
        self.variational_params = []  # parameters of the samplers

        self.parameters2nllfunc = {}  # prior densities
        self._module = module

        self.predictive_distribution_sampler = None
        self.predictive_distribution_log_lik = None

    def set_posterior_sampler(
        self,
        parameters: ParamsKey,
        sampler: Callable,
        variational_params: Dict[str, torch.tensor],
    ) -> None:
        """Register a sampler for a parameter or parameters."""
        if parameters in self.parameters2sampler:
            raise Exception(f"{parameters} is already handled!")
        self.parameters2sampler[parameters] = sampler
        prefix = parameters if isinstance(parameters, str) else "_".join(parameters)
        self.variational_params.extend(
            (prefix + ":" + vn, vp) for vn, vp in variational_params.items()
        )
        logging.info(
            f"[{self.__class__.__name__}] posterior for {parameters} set to {sampler}({variational_params.keys()})"
        )

    def set_posterior_samplers(
        self,
        create_sampler_func: Callable,
        filter: Callable = lambda parameter_name: True,
    ) -> None:
        """Register samplers for selected parameters (e.g. with 'bias' in name)."""
        for parameter_name, parameter_value in self._module.named_parameters():
            if filter(parameter_name) and not self.is_parameter_already_handled(
                parameter_name
            ):
                (
                    sampler,
                    variational_params,
                    _,
                ) = create_sampler_func(parameter_value)

                self.set_posterior_sampler(parameter_name, sampler, variational_params)

    def set_prior_density(self, parameters: ParamsKey, nll_func: Callable) -> None:
        """Register NLL calculation for a parameter or parameters."""
        if parameters in self.parameters2nllfunc:
            raise Exception(f"{parameters} is already handled!")
        self.parameters2nllfunc[parameters] = nll_func
        logging.info(
            f"[{self.__class__.__name__}] prior for {parameters} set to {nll_func}"
        )

    def set_prior_densities(
        self,
        create_density_func: Callable,
        filter: Callable = lambda parameter_name: True,
    ):
        """Register NLL calculation for selected parameters (e.g. with 'bias' in name).."""
        for parameter_name, parameter_value in self._module.named_parameters():
            if filter(parameter_name):
                nllfunc = create_density_func(parameter_value.shape)
                self.set_prior_density(parameter_name, nllfunc)

    def is_parameter_already_handled(self, parameter_name: str) -> bool:
        return is_parameter_handled(self.parameters2sampler.items(), parameter_name)

    def sample_posterior(self, n_samples: int = 1) -> Tuple[StateDict, NLLs]:
        """Returns samples + NLLs from pre-registered samplers."""
        parameters_samples, posterior_nlls = sample_parameters(
            self.parameters2sampler.items(), n_samples=n_samples
        )
        posterior_nlls = torch.stack(
            list(posterior_nlls.values())
        )  # out shape: n_param_groups x n_samples
        return parameters_samples, posterior_nlls

    def prior_nll(self, parameters_samples: StateDict) -> torch.tensor:
        """Returns samples' NLLs for pre-registered priors."""
        prior_nlls = estimate_parameters_nll(
            self.parameters2nllfunc, parameters_samples
        )
        prior_nlls = torch.stack(
            list(prior_nlls.values())
        )  # out shape: n_param_groups x n_posterior_samples
        return prior_nlls

    def _get_samples(self, parameters_samples, n_samples):
        if not parameters_samples:
            parameters_samples, _ = self.sample_posterior(n_samples)
        return parameters_samples

    def sample_predictive(
        self,
        input_x: torch.Tensor,
        parameters_samples: Optional[StateDict] = None,
        n_samples: int = 1,
        n_predictive_samples: int = 1,
        **sample_predictive_kwargs,
    ):
        parameters_samples = self._get_samples(parameters_samples, n_samples)

        return sample_predictive(
            input_x,
            self._module,
            parameters_samples,
            self.predictive_distribution_sampler,
            n_samples=n_predictive_samples,
            **sample_predictive_kwargs,
        )

    def predictive_likelihoods(
        self,
        input_x: torch.Tensor,
        output_x: torch.Tensor,
        parameters_samples: Optional[StateDict] = None,
        n_samples: int = 1,
        **predictive_likelihoods_kwargs,
    ):
        parameters_samples = self._get_samples(parameters_samples, n_samples)

        return predictive_likelihoods(
            input_x,
            output_x,
            self._module,
            parameters_samples,
            self.predictive_distribution_log_lik,
            **predictive_likelihoods_kwargs,
        )
