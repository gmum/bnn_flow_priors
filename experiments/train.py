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
from torch.nn.functional import softplus
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
        # posterior, e.g. "mfvi", "realnvp"
        posterior = "mfvi"
        # weight prior, e.g., "gaussian", "laplace", "student-t"
        weight_prior = "gaussian"
        # bias prior, same as above
        bias_prior = "gaussian"
        # location parameter for the weight prior
        weight_loc = 0.0
        # scale parameter for the weight prior
        weight_scale = 2.0**0.5
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
        # total number of samples to be drawn from the posterior
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
        # whether to learn distributions (leaves only log likelihood loss)
        learn_distributions = True
        # whether to log sigma statistics
        log_mfvi = False
        # realnvp posterior settings
        realnvp_m = 256
        realnvp_num_layers = 8
        realnvp_max_input_size = 1024

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
        learn_distributions,
        n_samples,
        n_samples_training,
        ood_data,
        metrics_skip,
        posterior,
        log_mfvi,
        realnvp_m,
        realnvp_num_layers,
        realnvp_max_input_size,
        _run,
        _log,
    ):
        _log.info(f"Starting {neptune_run_id}")

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
                    _log.warning(f"key {k} not in model, ignoring")
                    del state_dict[k]
                elif model_sd[k].size() != state_dict[k].size():
                    _log.warning(
                        f"key {k} size mismatch, model={model_sd[k].size()}, loaded={state_dict[k].size()}"
                    )
                    state_dict[k] = model_sd[k]
            missing_keys = set(model_sd.keys()) - set(state_dict.keys())
            _log.warning(
                f"The following keys were not found in loaded state dict: {missing_keys}"
            )
            model_sd.update(state_dict)
            model.load_state_dict(model_sd)
            del state_dict
            del model_sd
        model.to(device("try_cuda"))

        _log.info(f"load_samples={load_samples}")
        _log.info(f"init_method={init_method}")

        # if save_samples:
        #     model_saver_fn = (lambda: exp_utils.HDF5ModelSaver(
        #         exp_utils.sneaky_artifact(_run, "samples.pt"), "w"))
        # else:
        #     @contextlib.contextmanager
        #     def model_saver_fn():
        #         yield None

        if posterior == "mfvi":
            # MFVI approx posterior:
            # locs   = {n: t.randn((p.shape), requires_grad=True, device=device(0)) for n, p in model.named_parameters() }
            locs = {
                n: p.clone().detach().requires_grad_(True)
                for n, p in model.named_parameters()
            }
            scales = {
                n: t.randn_like(p).requires_grad_(True)
                for n, p in model.named_parameters()
            }
            approximation_params = chain(locs.values(), scales.values())

            def sample_posterior(n_samples, only_pointwise_locs=False):
                qs = {
                    n: Normal(l, softplus(s) + 1e-8)
                    for (n, l), s in zip(locs.items(), scales.values())
                }
                if only_pointwise_locs:
                    # for point-wise training simply return repeated locs as samples:
                    samples = {
                        n: l.expand(t.Size([n_samples] + [-1 for _ in l.shape]))
                        for n, l in locs.items()
                    }
                else:  # for reparametrized gradients:
                    samples = {n: q.rsample(t.Size([n_samples])) for n, q in qs.items()}
                return qs, samples

        elif posterior == "realnvp":
            realnvps = {}

            def build_flow_on(full_name, params):
                d = params.numel()
                _log.info(f"Param {full_name} weights size: {d}")
                # (instantiate a flow for gs)
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
                realnvps[full_name] = RealNVP(
                    net_s, net_t, realnvp_num_layers, flow_prior
                )
                realnvps[full_name].to(device("try_cuda"))

            point_estimates = {}
            for module_name, module in model.named_modules():
                children = len(list(module.children()))
                _log.info(
                    f"Module {module_name} type: {type(module)} children: {children}"
                )
                build_flow = False
                if isinstance(module, layers.Conv2d):
                    weight_norm(module.weight_prior, name="p")
                    build_flow = True
                elif isinstance(module, layers.Linear):
                    weight_norm(module.weight_prior, name="p")
                    build_flow = True
                if build_flow:
                    # add vs to point estimates
                    v_full_name = f"{module_name}.weight_prior.p_v"
                    v_params = get_module_by_name(model, v_full_name)
                    point_estimates[v_full_name] = (
                        v_params.clone().detach().unsqueeze(0).requires_grad_(True)
                    )
                    _log.info(f"Param {v_full_name} weights size: {v_params.numel()}")
                    #
                    bias_module_name = f"{module_name}.bias_prior"
                    if get_module_by_name(model, bias_module_name) is not None:
                        bias_full_name = f"{module_name}.bias_prior.p"
                        bias_params = get_module_by_name(model, bias_full_name)
                        build_flow_on(bias_full_name, bias_params)
                    # point_estimates[bias_full_name] = bias_params.clone().detach().unsqueeze(0).requires_grad_(True)
                    # _log.info(f"Param {bias_full_name} weights size: {bias_params.numel()}")
                    # add gs to flow targets
                    g_full_name = f"{module_name}.weight_prior.p_g"
                    g_params = get_module_by_name(model, g_full_name)
                    build_flow_on(g_full_name, g_params)

            # all parameters unaccounted for
            for n, p in model.named_parameters():
                if n not in realnvps and n not in point_estimates:
                    point_estimates[n] = (
                        p.clone().detach().unsqueeze(0).requires_grad_(True)
                    )
            approximation_params = chain(
                (p for module in realnvps.values() for p in module.parameters()),
                point_estimates.values(),
            )

            def sample_posterior(n_samples):
                samples = {}
                nlls = {}
                for name, p in model.named_parameters():
                    if name in realnvps:
                        n_weights, nll = realnvps[name].sample(
                            n_samples, p.numel(), calculate_nll=True
                        )
                        n_weights = n_weights.reshape(n_samples, *p.size())
                        samples[name] = n_weights
                        nlls[name] = nll
                    else:
                        weights = point_estimates[name]
                        samples[name] = weights.repeat(
                            n_samples, *(1 for _ in range(weights.dim() - 1))
                        )
                return nlls, samples

        else:
            raise ValueError("Illegal 'posterior' value.")

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
                # assert new_value.shape == module._parameters[name].shape, f"sample_path={sample_path} shape={new_value.shape} current shape={module._parameters[name].shape}"
                module._parameters[name] = new_value
            return module

        def evaluate_and_store_metrics(current_step, n_samples, calib=True, ood=False):
            training = model.training
            model.eval()
            _, posterior_samples = sample_posterior(n_samples)
            prior_samples = sample_priors(n_samples)
            samples = {**posterior_samples, **prior_samples}
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
                _log.info(
                    f"[evaluate_and_store_metrics][step={current_step}] eval.{k}={v}"
                )
            if log_mfvi is True and posterior == "mfvi":
                for name in scales.keys():
                    l = locs[name]
                    s = softplus(scales[name]) + 1e-8
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

        # train
        current_step = 0
        evaluate_and_store_metrics(current_step, n_samples)
        for epoch in range(epochs):
            for x, y in dataloader:
                if current_step % 1000 == 0:
                    _run.log_scalar("progress", current_step / last_step, current_step)
                x, y = x.to(device("try_cuda")), y.to(device("try_cuda"))

                # sampling from approximate posterior
                if posterior == "mfvi":
                    qs, posterior_samples = sample_posterior(
                        n_samples_training, only_pointwise_locs=not learn_distributions
                    )
                elif posterior == "realnvp":
                    nlls, posterior_samples = sample_posterior(n_samples_training)
                else:
                    raise ValueError("Illegal 'posterior' value.")
                prior_samples = sample_priors(n_samples_training)

                # sum up estimates for ELBO parts
                log_likelihood, log_prior, entropy = (
                    t.tensor(0.0, device=device("try_cuda")),
                    t.tensor(0.0, device=device("try_cuda")),
                    t.tensor(0.0, device=device("try_cuda")),
                )
                for sample_i, sample in enumerate(
                    sample_iter({**posterior_samples, **prior_samples})
                ):
                    overwrite_params(
                        model, sample
                    )  # model.load_state_dict(sample, strict=True)

                    # likelihood:
                    log_likelihood += model.log_likelihood(
                        x, y, x_train.shape[0]
                    )  # preds = model(x); log_likelihood += preds.log_prob(y).sum() * x_train.shape[0]/x.shape[0]

                    if learn_distributions:
                        # priors:
                        log_prior += model.log_prior()
                        if posterior == "mfvi":
                            prior_samples = sample_priors(n_samples_training)
                            # entropy:
                            # entropy += sum( q.entropy().sum() for q in qs.values() ) # closed-form for Gaussian
                            entropy += sum(
                                -q.log_prob(s).sum()
                                for q, s in zip(qs.values(), posterior_samples.values())
                            )  # via samples
                        elif posterior == "realnvp":
                            entropy += sum(
                                nll.sum()
                                for nll, s in zip(
                                    nlls.values(), posterior_samples.values()
                                )
                            )
                        else:
                            raise ValueError("Illegal 'posterior' value.")

                elbo = log_likelihood + log_prior + entropy
                loss_vi = -elbo

                _run.log_scalar(
                    "train.log_likelihood", log_likelihood.item(), current_step
                )
                _run.log_scalar("train.log_prior", log_prior.item(), current_step)
                _run.log_scalar("train.entropy", entropy.item(), current_step)
                _run.log_scalar("train.loss", loss_vi.item(), current_step)

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
        _log.info(f"Saved state to local path {str(state_path)}")
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
