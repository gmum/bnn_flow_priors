import functools
import json
import logging
import os
from json import JSONDecodeError
from pathlib import Path

import neptune.new as neptune
import torch as t
from neptune.new.integrations.python_logger import NeptuneHandler
from neptune.new.integrations.sacred import NeptuneObserver
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

import reparameterized as r
from bnn_priors import exp_utils
from bnn_priors.data import Synthetic
from bnn_priors.exp_utils import evaluate_ood
from reparameterized import bnn_wrapper
from reparameterized import sampling


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
    ex = Experiment("bnn_training")
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
        depth = None
        # weight prior, e.g., "gaussian", "laplace", "student-t"
        weight_prior = "gaussian"
        # location parameter for the weight prior
        weight_loc = 0.0
        # scale parameter for the weight prior
        weight_scale = 2.0 ** 0.5
        # additional keyword arguments for the weight prior
        weight_prior_params = {}
        if not isinstance(weight_prior_params, dict):
            weight_prior_params = json.loads(weight_prior_params)
        # bias prior, same as above
        bias_prior = "gaussian"
        # location parameter for the bias prior
        bias_loc = 0.0
        # scale parameter for the bias prior
        bias_scale = 1.0
        # additional keyword arguments for the bias prior
        bias_prior_params = {}
        if not isinstance(bias_prior_params, dict):
            bias_prior_params = json.loads(bias_prior_params)
        # posterior settings
        # whether to learn distributions (None leaves only log likelihood loss)
        # choices: None, 'realnvp', 'mfvi'
        weight_posterior = None
        if isinstance(weight_posterior, str):
            try:
                weight_posterior = json.loads(weight_posterior)
            except JSONDecodeError:
                weight_posterior = weight_posterior
        # whether to learn distributions on biases (None leaves only log likelihood loss)
        bias_posterior = None
        if isinstance(bias_posterior, str):
            try:
                bias_posterior = json.loads(bias_posterior)
            except JSONDecodeError:
                bias_posterior = bias_posterior
        # initialization method for the network weights
        init_method = "he"
        # whether to use batch normalization
        batchnorm = True
        # batch size for the training
        batch_size = 128
        # total number of samples to be drawn from the posterior for evaluation
        n_samples = 100
        # total number of samples for training
        n_samples_training = 1
        # epochs
        epochs = 100
        # learning rate
        lr = 0.001
        # scheduler type to use
        schedule = None
        # weight of the KL term for ELBO (can be a schedule name for annealing)
        kl_weight = 1.0
        if not isinstance(kl_weight, float) and isinstance(kl_weight, str):
            kl_weight = json.loads(kl_weight)
        # weight of the prior loss term
        prior_weight = 1.0
        # normalize the KL loss by dividing by the number of weights, so that
        # KL losses are comparable between architectures
        kl_normalize = True
        # realnvp posterior settings
        realnvp_m = 128
        realnvp_num_layers = 4
        # whether to use rezero trick in normalizing flows or not
        rezero_trick = True
        # OOD dataset
        ood_data = None
        # number of epochs to skip between computing metrics
        metrics_skip = 10
        # device to use, "cpu", "cuda:0", "try_cuda"
        device = "cuda:0"

    # decorators
    device = ex.capture(exp_utils.device)
    get_model = ex.capture(exp_utils.get_model)

    logger.info(f'Device: {device("try_cuda")}')

    ###########################################################################

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
            weight_prior_params,
            bias_prior,
            bias_loc,
            bias_scale,
            bias_prior_params,
            weight_posterior,
            bias_posterior,
            init_method,
            batch_size,
            batchnorm,
            n_samples,
            n_samples_training,
            epochs,
            lr,
            schedule,
            kl_weight,
            prior_weight,
            realnvp_m,
            realnvp_num_layers,
            rezero_trick,
            ood_data,
            metrics_skip,
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

        _log_info("loading data")
        data = get_data()
        x_train = data.norm.train_X
        y_train = data.norm.train_y
        # x_test = data.norm.test_X
        # y_test = data.norm.test_y

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

        _log_info(f"Initalizing model: init_method={init_method}")
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
        model.to(device("try_cuda"))

        NAME2SAMPLER_CREATE = {
            None: sampling.create_delta_distribution_sampler,
            "gaussian": sampling.create_factorized_gaussian_sampler,
            "pointwise": sampling.create_delta_distribution_sampler,
            "realnvp": functools.partial(sampling.create_flow_sampler,
                                         device=device("try_cuda"),
                                         realnvp_m=realnvp_m,
                                         realnvp_num_layers=realnvp_num_layers,
                                         rezero_trick=rezero_trick),
            "mfvi": sampling.create_factorized_gaussian_sampler,
            "hypernet": functools.partial(sampling.create_bayesian_hypernet_sampler,
                                          device=device("try_cuda"),
                                          realnvp_m=realnvp_m,
                                          realnvp_num_layers=realnvp_num_layers,
                                          rezero_trick=rezero_trick),
        }

        _log_info("Register posteriors for the model")
        bnn = bnn_wrapper.BayesianNeuralNetwork(model)
        if isinstance(weight_posterior, str):
            bnn.set_posterior_samplers(
                create_sampler_func=NAME2SAMPLER_CREATE[weight_posterior],
                filter=lambda parameter_name: "weight" in parameter_name,
            )
        elif isinstance(weight_posterior, dict):
            # setting posterior type by parameter name
            for k, v in weight_posterior.items():
                bnn.set_posterior_samplers(
                    create_sampler_func=NAME2SAMPLER_CREATE[v],
                    filter=lambda parameter_name: k in parameter_name,
                )
        if isinstance(bias_posterior, str):
            bnn.set_posterior_samplers(
                create_sampler_func=NAME2SAMPLER_CREATE[bias_posterior],
                filter=lambda parameter_name: "bias" in parameter_name,
            )
        elif isinstance(bias_posterior, dict):
            # setting posterior type by parameter name
            for k, v in bias_posterior.items():
                bnn.set_posterior_samplers(
                    create_sampler_func=NAME2SAMPLER_CREATE[v],
                    filter=lambda parameter_name: k in parameter_name,
                )
        bnn_params, num_bnn_params = get_bnn_params(bnn)

        _log_info("Model parameters:")
        for parameter_name, parameter in model.named_parameters():
            _log_info(f"\t{parameter_name}: {parameter.shape}")

        _log_info(f"Model posteriors:")
        for parameter, sampler in bnn.parameters2sampler.items():
            _log_info(f"Posterior for {parameter}: {sampler}")

        _log_info(f"Model priors:")
        for name, buffer in model.named_buffers():
            _log_info(f"Prior for {name} = {buffer}")

        #######################################################################

        def sample_priors(n_s):
            prior_samples = {
                name: t.cat(n_s * [buffer[None, ...]])
                for name, buffer in model.named_buffers()
            }
            return prior_samples

        #######################################################################

        def evaluate_and_store_metrics(current_step, n_s, calib=True, ood=False):
            _log_info(f"Starting evaluate_and_store_metrics at step {current_step} with n_s {n_s}:")
            training = model.training
            model.eval()
            posterior_samples, _ = bnn.sample_posterior(n_s)
            prior_samples = sample_priors(n_s)
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
                    f"eval.{k}={v}"
                )
            # log stats
            for p_name, p_samples in posterior_samples.items():
                stds, means = t.std_mean(p_samples, dim=0, unbiased=False)
                std_to_mean = stds / means.abs()
                _run.log_scalar(f"samples.{p_name}.means.min", means.min(), current_step)
                _run.log_scalar(f"samples.{p_name}.means.max", means.max(), current_step)
                _run.log_scalar(f"samples.{p_name}.means.mean", means.mean(), current_step)
                _run.log_scalar(f"samples.{p_name}.means.median", means.median(), current_step)
                _run.log_scalar(f"samples.{p_name}.means.std", means.std(unbiased=False), current_step)
                _run.log_scalar(f"samples.{p_name}.stds.min", stds.min(), current_step)
                _run.log_scalar(f"samples.{p_name}.stds.max", stds.max(), current_step)
                _run.log_scalar(f"samples.{p_name}.stds.mean", stds.mean(), current_step)
                _run.log_scalar(f"samples.{p_name}.stds.median", stds.median(), current_step)
                _run.log_scalar(f"samples.{p_name}.stds.std", stds.std(unbiased=False), current_step)
                _run.log_scalar(f"samples.{p_name}.cv.min", std_to_mean.min(), current_step)
                _run.log_scalar(f"samples.{p_name}.cv.max", std_to_mean.max(), current_step)
                _run.log_scalar(f"samples.{p_name}.cv.mean", std_to_mean.mean(), current_step)
                _run.log_scalar(f"samples.{p_name}.cv.median", std_to_mean.median(), current_step)
                _run.log_scalar(f"samples.{p_name}.cv.std", std_to_mean.std(unbiased=False),
                                current_step)
            model.train(training)
            _log_info(f"Finished evaluate_and_store_metrics at step {current_step}.")

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
        # TODO optimizer as parameter?
        optimizer = t.optim.Adam([p for name, p in bnn.variational_params], lr=lr)
        if schedule == "cosine":
            scheduler = t.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=last_step
            )
        else:
            scheduler = None

        _log_info("Start training")
        current_step, last_step = 0, len(dataloader) * epochs
        evaluate_and_store_metrics(current_step, n_samples)
        for epoch in range(epochs):
            _log_info(f"Epoch {epoch}")
            for x, y in dataloader:
                x, y = x.to(device("try_cuda")), y.to(device("try_cuda"))
                if current_step % 1000 == 0:
                    _run.log_scalar("progress", current_step / last_step, current_step)
                posterior_samples, posterior_nlls = bnn.sample_posterior(
                    n_samples_training
                )
                prior_samples = sample_priors(n_samples_training)
                # sum up estimates for ELBO parts
                log_likelihood, log_prior, entropy = (
                    t.tensor(0.0, device=device("try_cuda")),
                    t.tensor(0.0, device=device("try_cuda")),
                    t.tensor(0.0, device=device("try_cuda")),
                )
                # entropy += posterior_nlls.sum()  # sum over over parameters and samples
                entropy += posterior_nlls.sum() / num_bnn_params
                # iterate and sum over samples:
                for sample in r.take_parameters_sample(
                        {**posterior_samples, **prior_samples}
                ):
                    r.load_state_dict(model, sample)
                    # preds = model(x); log_likelihood += preds.log_prob(y).sum() * x_train.shape[0]/x.shape[0]
                    # log_likelihood += model.log_likelihood(x, y, x_train.shape[0])
                    log_likelihood += model.log_likelihood_avg(x, y)
                    # log_prior += model.log_prior()
                    log_prior += calculate_log_prior(model, bnn_params) / num_bnn_params
                kl_weight_value = _calculate_kl_weight_value(
                    kl_weight, last_step, current_step
                )
                _run.log_scalar("train.kl_weight", kl_weight_value, current_step)
                kl = kl_weight_value * (prior_weight * log_prior + entropy)
                elbo = (log_likelihood + kl) / n_samples_training
                loss_vi = -elbo
                _run.log_scalar("train.log_likelihood", log_likelihood.item(), current_step)
                _run.log_scalar("train.log_prior", log_prior.item(), current_step)
                _run.log_scalar("train.entropy", entropy.item(), current_step)
                _run.log_scalar("train.weighted_kl", (kl / n_samples_training).item(), current_step)
                _run.log_scalar("train.loss", loss_vi.item(), current_step)
                optimizer.zero_grad()
                loss_vi.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
                current_step += 1
            if epoch % metrics_skip == 0 and epoch > 0:
                evaluate_and_store_metrics(current_step, n_samples)

        _run.log_scalar("progress", 1.0, current_step)
        #
        _log_info(f"Model priors:")
        for name, buffer in model.named_buffers():
            _log_info(f"Prior for {name} = {buffer}")
        # save model
        state = model.state_dict()
        t.save(state, state_path)
        _log_info(f"Saved state to local path {str(state_path)}")
        # final evaluation
        evaluate_and_store_metrics(current_step, n_samples, ood=True)

    # run
    ex.run_commandline()
    # log artifacts
    # TODO uncomment if models are to be uploaded to neptune/mongodb etc. - or add a flag?
    # ex.add_artifact(str(state_path))
    # stop neptune
    neptune_run.stop()


def get_by_name(module, name):
    names = name.split(sep='.')
    return functools.reduce(getattr, names, module)


def get_bnn_params(bnn_wrapper):
    model = bnn_wrapper._module
    bnn_params = [p_name for p_name in bnn_wrapper.parameters2sampler.keys()]
    num_bnn_params = 0
    for p_name in bnn_params:
        param = get_by_name(model, p_name)
        num_bnn_params += param.numel()
    return bnn_params, num_bnn_params


def calculate_log_prior(model, bnn_params):
    log_prior = 0.0
    for p_name in bnn_params:
        module_name = '.'.join(p_name.split('.')[:-1])
        module = get_by_name(model, module_name)
        log_prior += module.log_prob().sum()
    return log_prior


def _calculate_kl_weight_value(kl_weight, last_step, current_step):
    def linear_schedule(current, last, initial, final):
        return current / last * (final - initial) + initial

    if isinstance(kl_weight, dict):
        if kl_weight["type"] == "linear":
            kl_weight_value = linear_schedule(
                current_step,
                last_step,
                kl_weight["initial"],
                kl_weight["final"],
            )
        else:
            raise NotImplementedError()
    else:
        kl_weight_value = kl_weight
    return kl_weight_value


if __name__ == "__main__":
    main()
