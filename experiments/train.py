import json
import logging
import os
import warnings
from itertools import chain
from pathlib import Path

import neptune.new as neptune
import torch as t
from neptune.new.integrations.python_logger import NeptuneHandler
from neptune.new.integrations.sacred import NeptuneObserver
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from torch.distributions import Normal, MultivariateNormal
from torch.nn import Sequential, Linear, LeakyReLU, Tanh
from torch.nn.functional import softplus, normalize
from torch.nn.utils import weight_norm

from bnn_priors import exp_utils
from bnn_priors.data import Synthetic
from bnn_priors.exp_utils import evaluate_ood
from bnn_priors.flow import RealNVP
from bnn_priors.models import layers
from bnn_priors.utils import get_module_by_name


def main():
    # setup torch
    if t.cuda.is_available():
        t.backends.cudnn.benchmark = True

    # setup neptune
    neptune_run = neptune.init(source_files=["*.py", "**/*.py"])
    neptune_run_id = neptune_run["sys/id"].fetch()

    # setup local paths
    run_dir = Path(os.environ["PROJECT_PATH"]) / "runs" / neptune_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / "state.pth"
    logs_path = run_dir / "run.log"

    # setup logging
    logger = logging.getLogger("myLogger")
    formatter = logging.Formatter(
        "[%(asctime)s %(levelname)s %(module)s:%(lineno)d] %(message)s"
    )
    neptune_formatter = logging.Formatter(
        "%(levelname)s %(module)s:%(lineno)d - %(message)s"
    )
    file_handler = logging.FileHandler(logs_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    neptune_handler = NeptuneHandler(run=neptune_run)
    neptune_handler.setFormatter(neptune_formatter)
    logger.addHandler(neptune_handler)
    logger.setLevel(logging.INFO)
    logger.info(f"Created logger")

    # setup sacred
    ex = Experiment("mfvi_training")
    ex.captured_out_filter = apply_backspaces_and_linefeeds
    ex.observers.append(NeptuneObserver(run=neptune_run))
    ex.logger = logger

    # config
    @ex.config
    def config():
        # the dataset to be trained on, e.g., "mnist", "cifar10", "UCI_boston"
        data = "mnist"
        # model to be used, e.g., "classificationdensenet", "classificationconvnet", "googleresnet"
        # model = "googleresnet"
        model = "classificationconvnet"
        # width of the model (might not have an effect in some models)
        width = 50
        # depth of the model (might not have an effect in some models)
        depth = 3
        # weight prior, e.g., "gaussian", "laplace", "student-t"
        weight_prior = "gaussian"
        # bias prior, same as above
        bias_prior = "gaussian"
        # location parameter for the weight prior
        weight_loc = 0.0
        # scale parameter for the weight prior
        weight_scale = 2.0 ** 0.5
        # location parameter for the bias prior
        bias_loc = 0.0
        # scale parameter for the bias prior
        bias_scale = 1.0
        # additional keyword arguments for the weight prior
        weight_prior_params = {}
        # additional keyword arguments for the bias prior
        bias_prior_params = {}
        if not isinstance(weight_prior_params, dict):
            weight_prior_params = json.loads(weight_prior_params)
        if not isinstance(bias_prior_params, dict):
            bias_prior_params = json.loads(bias_prior_params)
        # total number of samples to be drawn from the posterior for evaluation
        n_samples = 100
        # total number of samples for training
        n_samples_training = 1
        # number of epochs to skip between computing metrics
        metrics_skip = 10
        # learning rate
        lr = 0.001
        # epochs
        epochs = 100
        # scheduler type to use
        schedule = None
        # initialization method for the network weights
        init_method = "he"
        # previous samples to be loaded to initialize the chain
        load_samples = None
        # batch size for the training
        batch_size = 128
        # whether to use batch normalization
        batchnorm = True
        # device to use, "cpu", "cuda:0", "try_cuda"
        device = "try_cuda"
        # whether the samples should be saved
        # TODO implement?
        save_samples = True
        # OOD dataset
        ood_data = None
        # whether to apply weight normalization on the weights
        weight_normalization = False
        # whether to learn distributions (None leaves only log likelihood loss)
        # choices: None, 'realnvp', 'mfvi'
        weights_posterior = None
        # whether to learn distributions on biases (None leaves only log likelihood loss)
        bias_posterior = None
        # whether to log statistics for mfvi
        log_mfvi = True
        # realnvp posterior settings
        realnvp_m = 256
        realnvp_num_layers = 8

    # decorators
    device = ex.capture(exp_utils.device)
    get_model = ex.capture(exp_utils.get_model)

    logger.info(f'Device: {device("try_cuda")}')

    @ex.capture
    def get_data(data, batch_size, _run):
        if data == "empty":
            dataset = exp_utils.get_data("UCI_boston", device())
            dataset.norm.train = [(None, None)]
            dataset.norm.test = [(None, None)]
            dataset.unnorm.train = [(None, None)]
            dataset.unnorm.test = [(None, None)]
            return dataset

        if data[:9] == "synthetic":
            _, data, prior = data.split(".")
            dataset = get_data(data)
            x_train = dataset.norm.train_X
            y_train = dataset.norm.train_y
            model = get_model(
                x_train=x_train,
                y_train=y_train,
                weight_prior=prior,
                weight_prior_params={},
            )
            model.sample_all_priors()
            data = Synthetic(
                dataset=dataset, model=model, batch_size=batch_size, device=device()
            )
            t.save(data, exp_utils.sneaky_artifact(_run, "synthetic_data.pt"))
            t.save(model, exp_utils.sneaky_artifact(_run, "true_model.pt"))
            return data
        else:
            return exp_utils.get_data(data, device())

    @ex.main
    def main(
            model,
            width,
            depth,
            weight_prior,
            weight_loc,
            weight_scale,
            bias_prior,
            bias_loc,
            bias_scale,
            batchnorm,
            weight_prior_params,
            bias_prior_params,
            load_samples,
            init_method,
            batch_size,
            lr,
            epochs,
            schedule,
            weight_normalization,
            weights_posterior,
            bias_posterior,
            n_samples,
            n_samples_training,
            ood_data,
            metrics_skip,
            log_mfvi,
            realnvp_m,
            realnvp_num_layers,
            _run,
            _log,
    ):
        def _log_warning(msg):
            print(f"[WARNING] {msg}")
            _log._log_warning(msg)

        def _log_info(msg):
            print(f"[INFO] {msg}")
            _log.info(msg)

        _log_info(f"Starting {neptune_run_id}")

        # loading data
        data = get_data()
        x_train = data.norm.train_X
        y_train = data.norm.train_y
        x_test = data.norm.test_X
        y_test = data.norm.test_y

        # model
        model = get_model(
            x_train,
            y_train,
            model,
            width,
            depth,
            weight_prior,
            weight_loc,
            weight_scale,
            bias_prior,
            bias_loc,
            bias_scale,
            batchnorm,
            weight_prior_params,
            bias_prior_params,
        )

        # init
        if load_samples is None:
            if init_method == "he":
                exp_utils.he_initialize(model)
            elif init_method == "he_uniform":
                exp_utils.he_uniform_initialize(model)
            elif init_method == "he_zerobias":
                exp_utils.he_zerobias_initialize(model)
            elif init_method == "prior":
                pass
            else:
                raise ValueError(f"unknown init_method={init_method}")
        else:
            state_dict = exp_utils.load_samples(load_samples, idx=-1, keep_steps=False)
            model_sd = model.state_dict()
            for k in state_dict.keys():
                if k not in model_sd:
                    _log_warning(f"key {k} not in model, ignoring")
                    del state_dict[k]
                elif model_sd[k].size() != state_dict[k].size():
                    _log_warning(
                        f"key {k} size mismatch, model={model_sd[k].size()}, loaded={state_dict[k].size()}"
                    )
                    state_dict[k] = model_sd[k]
            missing_keys = set(model_sd.keys()) - set(state_dict.keys())
            _log_warning(
                f"The following keys were not found in loaded state dict: {missing_keys}"
            )
            model_sd.update(state_dict)
            model.load_state_dict(model_sd)
            del state_dict
            del model_sd
        model.to(device("try_cuda"))

        _log_info(f"load_samples={load_samples}")
        _log_info(f"init_method={init_method}")

        # if save_samples:
        #     model_saver_fn = (lambda: exp_utils.HDF5ModelSaver(
        #         exp_utils.sneaky_artifact(_run, "samples.pt"), "w"))
        # else:
        #     @contextlib.contextmanager
        #     def model_saver_fn():
        #         yield None

        #######################################################################

        def build_realnvp(output_dim):
            d = output_dim
            flow_prior = MultivariateNormal(t.zeros(d), t.eye(d))
            m = realnvp_m
            net_s = lambda: Sequential(
                Linear(d // 2, m),
                LeakyReLU(),
                Linear(m, m),
                LeakyReLU(),
                Linear(m, d // 2),
                Tanh(),
            )
            net_t = lambda: Sequential(
                Linear(d // 2, m),
                LeakyReLU(),
                Linear(m, m),
                LeakyReLU(),
                Linear(m, d // 2),
            )
            realnvp = RealNVP(net_s, net_t, realnvp_num_layers, flow_prior)
            realnvp.to(device("try_cuda"))
            return realnvp

        #######################################################################

        # Currently supported posterior approximations:
        point_estimates = {}
        realnvps = {}
        gaussians = {}
        is_parameter_already_handled = (
            lambda name: name in realnvps
                         or name in point_estimates
                         or name in gaussians
        )
        if weight_normalization:
            for module_name, module in model.named_modules():
                if isinstance(module, (layers.Conv2d, layers.Linear)):
                    weight_norm(module.weight_prior, name="p")
        # Building RealNVP for selected layers
        v2g, g2v_dim = {}, {}  # matching flow variables and pointwise variables
        for module_name, module in model.named_modules():
            if isinstance(module, (layers.Conv2d, layers.Linear)):
                if weights_posterior == 'realnvp':
                    assert weight_normalization, f'RealNVP is currently handled only with weight_norm'
                    # add vs to point estimates
                    v_full_name = f"{module_name}.weight_prior.p_v"
                    v_params = get_module_by_name(model, v_full_name)
                    v_params = (
                        v_params.clone()
                        .detach()
                        .requires_grad_(True)
                        .to(device("try_cuda"))
                    )
                    point_estimates[v_full_name] = v_params
                    # add gs to flow targets
                    g_full_name = f"{module_name}.weight_prior.p_g"
                    g_params = get_module_by_name(model, g_full_name)
                    realnvps[g_full_name] = build_realnvp(g_params.numel())
                    #
                    _log_info(
                        f"Module {module_name}: v={v_params.numel()}, g={g_params.numel()} (RealNVP)"
                    )
                    g2v_dim[g_full_name] = v_params.shape.numel()
                    v2g[v_full_name] = g_full_name
                elif weights_posterior == 'mfvi':
                    if weight_normalization:
                        # add vs
                        v_full_name = f"{module_name}.weight_prior.p_v"
                        v_params = get_module_by_name(model, v_full_name)
                        loc = (
                            v_params.clone()
                            .detach()
                            .requires_grad_(True)
                            .to(device("try_cuda"))
                        )
                        unnormalized_scale = (
                            t.randn_like(loc).requires_grad_(True).to(device("try_cuda"))
                        )
                        gaussians[v_full_name] = (loc, unnormalized_scale)
                        # add gs
                        g_full_name = f"{module_name}.weight_prior.p_g"
                        g_params = get_module_by_name(model, g_full_name)
                        loc = (
                            g_params.clone()
                            .detach()
                            .requires_grad_(True)
                            .to(device("try_cuda"))
                        )
                        unnormalized_scale = (
                            t.randn_like(loc).requires_grad_(True).to(device("try_cuda"))
                        )
                        gaussians[g_full_name] = (loc, unnormalized_scale)
                        _log_info(
                            f"Module {module_name}: v={v_params.numel()} (MFVI), g={g_params.numel()} (MFVI)"
                        )
                    else:
                        # add params
                        p_full_name = f"{module_name}.weight_prior.p"
                        p_params = get_module_by_name(model, p_full_name)
                        loc = (
                            p_params.clone()
                            .detach()
                            .requires_grad_(True)
                            .to(device("try_cuda"))
                        )
                        unnormalized_scale = (
                            t.randn_like(loc).requires_grad_(True).to(device("try_cuda"))
                        )
                        gaussians[p_full_name] = (loc, unnormalized_scale)
                        _log_info(
                            f"Module {module_name}: p={p_params.numel()} (MFVI)"
                        )
                if bias_posterior == 'realnvp':
                    bias_module_name = f"{module_name}.bias_prior"
                    if get_module_by_name(model, bias_module_name) is not None:
                        bias_full_name = f"{module_name}.bias_prior.p"
                        bias_params = get_module_by_name(model, bias_full_name)
                        realnvps[bias_full_name] = build_realnvp(bias_params.numel())
                        _log_info(
                            f"Module {module_name}: bias={bias_params.numel()} (RealNVP)"
                        )
                elif bias_posterior == 'mfvi':
                    bias_module_name = f"{module_name}.bias_prior"
                    if get_module_by_name(model, bias_module_name) is not None:
                        bias_full_name = f"{module_name}.bias_prior.p"
                        bias_params = get_module_by_name(model, bias_full_name)
                        loc = (
                            bias_params.clone()
                            .detach()
                            .requires_grad_(True)
                            .to(device("try_cuda"))
                        )
                        unnormalized_scale = (
                            t.randn_like(loc).requires_grad_(True).to(device("try_cuda"))
                        )
                        gaussians[bias_full_name] = (loc, unnormalized_scale)
                        _log_info(
                            f"Module {module_name}: bias={bias_params.numel()} (MFVI)"
                        )
        # all parameters unaccounted for will be modeled as pointwise parameters
        for n, p in model.named_parameters():
            if is_parameter_already_handled(n):
                continue
            _log_info(f"Parameter {n} will be trained pointwise")
            # create a variable of the same shape:
            weights = p.clone().detach().requires_grad_(True)
            point_estimates[n] = weights

        # gather parameters to be trained for all approximation types
        approximation_params = list(
            chain(
                (p for module in realnvps.values() for p in module.parameters()),
                point_estimates.values(),
                (loc for loc, _ in gaussians.values()),
                (uscale for _, uscale in gaussians.values()),
            )
        )

        #######################################################################

        def sample_posterior(n_samples):
            samples = {}
            nlls = {}
            for name, p in model.named_parameters():

                if name in realnvps:
                    sample, nll = realnvps[name].sample(
                        n_samples, p.numel(), calculate_nll=True
                    )
                    print("name in realnvps")
                    print("nll:", nll)
                    sample = sample.reshape(n_samples, *p.size())
                    nll = nll.to(sample.device)

                    samples[name] = sample
                    theta_dim_per_g = (
                            g2v_dim.get(name, p.shape.numel()) / p.shape.numel()
                    )  # how many thetas are created from each g; if no associated v -> multiply by 1
                    print("theta_dim_per_g: ", theta_dim_per_g)
                    nlls[name] = nll * theta_dim_per_g
                    print("nll * theta_dim_per_g ", nll * theta_dim_per_g)

                elif name in point_estimates:
                    sample = point_estimates[name]
                    sample = sample.expand(
                        t.Size([n_samples] + [-1 for _ in sample.shape])
                    )

                    samples[name] = sample
                    nll = t.zeros(
                        n_samples, dtype=sample.dtype
                    )  # not used besides the below assert
                    nll = nll.to(sample.device)

                elif name in gaussians:
                    loc, unnormalized_scale = gaussians[name]
                    q = Normal(loc, softplus(unnormalized_scale) + 1e-8)
                    sample = q.rsample(t.Size([n_samples]))
                    data_dims = list(range(1, len(sample.shape)))

                    # calc total NLL for all params (shape==n_samples)
                    nll = -q.log_prob(sample).sum(dim=data_dims)
                    nll = nll.to(sample.device)

                    samples[name] = sample
                    nlls[name] = nll

                else:
                    raise Exception(f"I don't know how to sample posterior for {name}!")

                assert (
                        sample.shape[0] == n_samples
                ), f"parameter={name}({p.shape}) sample.shape={sample.shape} n_samples={n_samples}"
                assert (
                        sample.shape[1:] == p.shape
                ), f"parameter={name}({p.shape}) sample.shape=={sample.shape} p.shape={p.shape}"
                assert len(nll) == n_samples

            print("nlls before for")
            print(nlls)

            # add log determinants for changing variables g to theta
            for v_name, g_name in v2g.items():
                print("v_name: ", v_name)
                print("g_name: ", g_name)
                v = samples[v_name]
                print("v: ", v)
                data_dims = list(range(1, len(v.shape)))
                print("data_dims: ", data_dims)
                v_norm = normalize(v, p=2.0, dim=1)
                print("v_norm: ", v_norm)
                # log det J = \sum_i log 1/u_i = -\sum_i log u_
                u = v_norm  # TODO NORMALIZATION!
                print("u: ", u)
                print("-u: ", -u)
                print("-u.log(): ", -u.log())
                print("-u.log().sum(data_dims): ", -u.log().sum(data_dims))
                print("u.prod(): ", u.prod())
                print("u.prod().log(data_dims): ", u.prod().log(data_dims))
                print("-u.prod().log(data_dims): ", -u.prod().log(data_dims))
                log_det_J = -u.log().sum(data_dims)
                print("log_det_J: ", log_det_J)
                nlls[g_name] += -log_det_J

            print("nlls after for")
            print(nlls)
            return nlls, samples

        def sample_priors(n_samples):
            prior_samples = {
                name: t.cat(n_samples * [buffer[None, ...]])
                for name, buffer in model.named_buffers()
            }
            return prior_samples

        def overwrite_params(module, samples, path="", skip_biases=False):
            for name, m in module._modules.items():
                overwrite_params(
                    m, samples, path=f"{path}.{name}", skip_biases=skip_biases
                )
            for name in module._parameters.keys():
                if skip_biases and "bias" in name:
                    continue
                sample_path = f"{path}.{name}"[1:]  # skip the leading dot
                new_value = samples[sample_path]
                # assert (
                #     new_value.shape == module._parameters[name].shape
                # ), f"sample_path={sample_path} shape={new_value.shape} current shape={module._parameters[name].shape}"
                module._parameters[name] = new_value
            return module

        #######################################################################

        def evaluate_and_store_metrics(current_step, n_samples, calib=True, ood=False):
            _log_info("evaluate_and_store_metrics")
            training = model.training
            model.eval()

            _, posterior_samples = sample_posterior(n_samples)
            prior_samples = sample_priors(n_samples)
            samples = {**posterior_samples, **prior_samples}
            # _log_info(
            #     "[evaluate_and_store_metrics] samples ({n_samples}):"
            #     + ", ".join(f"{p}:{s.shape}" for p, s in samples.items())
            # )

            results = exp_utils.evaluate_model(
                model=model,
                dataloader_test=dataloader_test,
                samples=samples,
                likelihood_eval=True,
                accuracy_eval=True,
                calibration_eval=calib,
            )
            if dataloader_ood is not None and ood:
                results.update(
                    evaluate_ood(model, dataloader_test, dataloader_ood, samples)
                )
            for k, v in results.items():
                _run.log_scalar(f"eval.{k}", v, current_step)
                _log_info(
                    f"[evaluate_and_store_metrics][step={current_step}] eval.{k}={v}"
                )
            if log_mfvi:
                for name in gaussians.keys():
                    l, us = gaussians[name]
                    s = softplus(us) + 1e-8
                    r_s = s / l.abs()
                    _run.log_scalar(f"mfvi.{name}.miu.min", l.min(), current_step)
                    _run.log_scalar(f"mfvi.{name}.miu.max", l.max(), current_step)
                    _run.log_scalar(f"mfvi.{name}.miu.mean", l.mean(), current_step)
                    _run.log_scalar(f"mfvi.{name}.miu.median", l.median(), current_step)
                    _run.log_scalar(f"mfvi.{name}.miu.std", l.std(), current_step)
                    _run.log_scalar(f"mfvi.{name}.sigma.min", s.min(), current_step)
                    _run.log_scalar(f"mfvi.{name}.sigma.max", s.max(), current_step)
                    _run.log_scalar(f"mfvi.{name}.sigma.mean", s.mean(), current_step)
                    _run.log_scalar(
                        f"mfvi.{name}.sigma.median", s.median(), current_step
                    )
                    _run.log_scalar(f"mfvi.{name}.sigma.std", s.std(), current_step)
                    _run.log_scalar(
                        f"mfvi.{name}.rel_sigma.min", r_s.min(), current_step
                    )
                    _run.log_scalar(
                        f"mfvi.{name}.rel_sigma.max", r_s.max(), current_step
                    )
                    _run.log_scalar(
                        f"mfvi.{name}.rel_sigma.mean", r_s.mean(), current_step
                    )
                    _run.log_scalar(
                        f"mfvi.{name}.rel_sigma.median", r_s.median(), current_step
                    )
                    _run.log_scalar(
                        f"mfvi.{name}.rel_sigma.std", r_s.std(), current_step
                    )
            model.train(training)

        # iterate over samples (taken from the original code):
        def _n_samples_len(samples):
            n_samples = min(len(v) for _, v in samples.items())
            if not all((len(v) == n_samples) for _, v in samples.items()):
                warnings.warn(
                    "Not all samples have the same length. "
                    "Setting n_samples to the minimum."
                )
            return n_samples

        def sample_iter(samples):
            for i in range(_n_samples_len(samples)):
                yield {k: v[i] for k, v in samples.items()}

        # Prepare data loaders
        num_workers = (
            0 if isinstance(data.norm.train, t.utils.data.TensorDataset) else 2
        )
        dataloader = t.utils.data.DataLoader(
            data.norm.train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=num_workers,
        )
        dataloader_test = t.utils.data.DataLoader(
            data.norm.test,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )
        dataloader_ood = (
            t.utils.data.DataLoader(get_data(ood_data).norm.test, batch_size=batch_size)
            if ood_data is not None
            else None
        )

        # configure optimization
        batches_per_epoch = len(dataloader)
        last_step = epochs * batches_per_epoch - 1
        optimizer = t.optim.Adam(approximation_params, lr=lr)
        if schedule == "cosine":
            scheduler = t.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=last_step
            )
        else:
            scheduler = None

        _log_info("Model parameters:")
        for parameter_name, parameter in model.named_parameters():
            _log_info(f" - {parameter_name}: {parameter.shape}")

        _log_info("Start training")
        current_step = 0
        evaluate_and_store_metrics(current_step, n_samples)
        for epoch in range(epochs):
            _log_info(f"Epoch {epoch}")
            for x, y in dataloader:

                if current_step % 1000 == 0:
                    _run.log_scalar("progress", current_step / last_step, current_step)
                x, y = x.to(device("try_cuda")), y.to(device("try_cuda"))

                nlls, posterior_samples = sample_posterior(n_samples_training)
                prior_samples = sample_priors(n_samples_training)

                # sum up estimates for ELBO parts
                log_likelihood, log_prior, entropy = (
                    t.tensor(0.0, device=device("try_cuda")),
                    t.tensor(0.0, device=device("try_cuda")),
                    t.tensor(0.0, device=device("try_cuda")),
                )

                print(f"{current_step} step: train.entropy = {entropy.item()}")
                entropy += sum(
                    nll.sum() for nll in nlls.values()
                )  # sum over samples and then over variables
                print(f"{current_step} step: train.entropy = {entropy.item()}")
                print("nlls: ", nlls)
                # iterate and sum over samples:
                for sample in sample_iter({**posterior_samples, **prior_samples}):
                    overwrite_params(model, sample)

                    # preds = model(x); log_likelihood += preds.log_prob(y).sum() * x_train.shape[0]/x.shape[0]
                    log_likelihood += model.log_likelihood(x, y, x_train.shape[0])

                    if weights_posterior in ('mfvi', 'realnvp') or bias_posterior in ('mfvi', 'realnvp'):
                        log_prior += model.log_prior()

                elbo = (log_likelihood + log_prior + entropy) / n_samples_training
                loss_vi = -elbo

                _run.log_scalar(
                    "train.log_likelihood", log_likelihood.item(), current_step
                )
                _run.log_scalar("train.log_prior", log_prior.item(), current_step)
                _run.log_scalar("train.entropy", entropy.item(), current_step)
                _run.log_scalar("train.loss", loss_vi.item(), current_step)
                print(f"{current_step} step: train.entropy = {entropy.item()}")
                print(f"{current_step} step: train.loss = {loss_vi.item()}")

                optimizer.zero_grad()
                loss_vi.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()
                current_step += 1

            if epoch % metrics_skip == 0 and epoch > 0:
                evaluate_and_store_metrics(current_step, n_samples)

        _run.log_scalar("progress", 1.0, current_step)
        # save model
        state = model.state_dict()
        t.save(state, state_path)
        _log_info(f"Saved state to local path {str(state_path)}")
        # TODO possibly save samples in h5 file as they do in eval_bnn.py
        # final evaluation
        evaluate_and_store_metrics(current_step, n_samples, ood=True)

    # run
    ex.run_commandline()
    # log artifacts
    # TODO uncomment if models are to be uploaded to neptune/mongodb etc.
    ex.add_artifact(str(state_path))
    # stop neptune
    neptune_run.stop()


if __name__ == "__main__":
    main()
