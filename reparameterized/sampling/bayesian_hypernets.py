"""Modeling posteriors with flows."""
from typing import Callable
import torch
from torch.nn.functional import normalize

from .realnvp import build_realnvp


def _normalize_v(v, dim=0):
    dim += 1
    u = normalize(v.flatten(start_dim=dim), p=2.0, dim=dim).reshape(v.shape)
    return u


def _weight_norm_our(v, g, dim=0):
    u = _normalize_v(v, dim=dim)
    return g * u


def create_bayesian_hypernet_sampler(
    parameter: torch.Tensor,
    device=None,
    only_flow_nll: bool = False,
    build_flow_func: Callable = build_realnvp,
    **build_flow_kwargs,
):
    device = device or parameter.device
    g_shape = torch.Size(list(parameter.shape[:1]) + [1 for _ in parameter.shape[1:]])
    flow = build_flow_func(output_dim=g_shape.numel(), **build_flow_kwargs, device=device)
    v_loc = parameter.clone().detach().requires_grad_(True).to(device)
    # TODO initialization of flows and v_loc

    # print(f"g_shape={g_shape} v_shape={v_loc.shape}")

    theta_dim_per_g = (
        parameter.numel() / g_shape.numel()
    )  # how many thetas are created from each g

    def sampler(n_samples=1):
        v = v_loc.expand(torch.Size([n_samples] + [-1 for _ in v_loc.shape]))
        assert v.shape == torch.Size([n_samples] + list(parameter.shape))

        g, nll = flow.sample(n_samples, g_shape.numel(), calculate_nll=True)
        g = g.reshape(n_samples, *g_shape)
        assert g.shape == torch.Size(
            [n_samples] + list(g_shape)
        ), f"{g.shape}!={n_samples},{g_shape}"

        sample = _weight_norm_our(v, g, dim=1)
        nll = (nll * theta_dim_per_g).to(sample.device)

        if not only_flow_nll:
            u = _normalize_v(v, dim=1)
            log_det_J = -torch.abs(u).log().flatten(start_dim=1).sum(dim=1)
            nll += -log_det_J

        assert nll.shape == torch.Size([n_samples])
        return sample, nll

    variational_params = {n: p for n, p in flow.named_parameters()}
    variational_params.update({"v": v_loc})
    return sampler, variational_params, {"flow": flow}
