import json
import logging
import uuid
import warnings
from pathlib import Path

import torch as t
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from torch.distributions import Normal
from torch.nn.functional import softplus

from bnn_priors import exp_utils
from bnn_priors.data import Synthetic
from bnn_priors.exp_utils import get_model

if t.cuda.is_available():
    t.backends.cudnn.benchmark = True

TMPDIR = "/tmp"

ex = Experiment("mfvi_training")
ex.captured_out_filter = apply_backspaces_and_linefeeds


# TODO extract config to a different run file
# config
@ex.config
def config():
    # the dataset to be trained on, e.g., "mnist", "cifar10", "UCI_boston"
    data = "mnist"
    # model to be used, e.g., "classificationdensenet", "classificationconvnet", "googleresnet"
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
    weight_loc = 0.
    # scale parameter for the weight prior
    weight_scale = 2. ** 0.5
    # location parameter for the bias prior
    bias_loc = 0.
    # scale parameter for the bias prior
    bias_scale = 1.
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
    # number of learning rate cycles for the Markov chain (see https://arxiv.org/abs/1902.03932)
    cycles = 3  # 60
    # number of epochs per cycle without added Langevin noise
    burnin = 0
    # number of epochs per cycle with added noise, but without sampling
    warmup = 45
    # number of epochs to skip between taking samples (1 means no skipping)
    skip = 1
    # number of epochs to skip between computing metrics
    metrics_skip = 10
    # number of first samples to discard when evaluating them
    skip_first = 50  # for evaluating accuracy et al at the end
    # temperature of the sampler
    temperature = 1.0
    # learning rate schedule during sampling
    sampling_decay = "cosine"
    # momentum for the sampler
    momentum = 0.994
    # update factor for the preconditioner
    precond_update = 1
    # learning rate
    lr = 0.001
    # epochs
    n_epochs = 100
    # initialization method for the network weights
    init_method = "he"
    # previous samples to be loaded to initialize the chain
    load_samples = None
    # batch size for the training
    batch_size = 128
    # whether to use Metropolis-Hastings rejection steps (works only with some integrators)
    reject_samples = False
    # whether to use batch normalization
    batchnorm = True
    # device to use, "cpu", "cuda:0", "try_cuda"
    device = "try_cuda"
    # whether the samples should be saved
    save_samples = True
    # whether a progressbar should be plotted to stdout during the training
    progressbar = True
    # a random unique ID for the run
    run_id = uuid.uuid4().hex


device = ex.capture(exp_utils.device)
get_model = ex.capture(exp_utils.get_model)

logging.info(f'Device: {device("try_cuda")}')


# dataset = bnn_priors_data.MNIST(device=device("try_cuda"), download=True)

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
        model = get_model(x_train=x_train, y_train=y_train, weight_prior=prior, weight_prior_params={})
        model.sample_all_priors()
        data = Synthetic(dataset=dataset, model=model, batch_size=batch_size, device=device())
        t.save(data, exp_utils.sneaky_artifact(_run, "synthetic_data.pt"))
        t.save(model, exp_utils.sneaky_artifact(_run, "true_model.pt"))
        return data
    else:
        return exp_utils.get_data(data, device())


@ex.capture
def evaluate_model(model, dataloader_test, samples):
    return exp_utils.evaluate_model(
        model=model, dataloader_test=dataloader_test, samples=samples,
        likelihood_eval=True, accuracy_eval=True,
        calibration_eval=False)


@ex.automain
def main(model, width, depth, weight_prior, weight_loc, weight_scale, bias_prior, bias_loc, bias_scale, \
         batchnorm, weight_prior_params, bias_prior_params, load_samples, init_method, batch_size, lr, \
         n_epochs, n_samples, n_samples_training, _run, _log):
    data = get_data()
    x_train = data.norm.train_X
    y_train = data.norm.train_y
    x_test = data.norm.test_X
    y_test = data.norm.test_y
    # model
    model = get_model(x_train, y_train, model,
                      width, depth,
                      weight_prior, weight_loc, weight_scale,
                      bias_prior, bias_loc, bias_scale,
                      batchnorm, weight_prior_params, bias_prior_params)
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
                _log.warning(f"key {k} size mismatch, model={model_sd[k].size()}, loaded={state_dict[k].size()}")
                state_dict[k] = model_sd[k]

        missing_keys = set(model_sd.keys()) - set(state_dict.keys())
        _log.warning(f"The following keys were not found in loaded state dict: {missing_keys}")
        model_sd.update(state_dict)
        model.load_state_dict(model_sd)
        del state_dict
        del model_sd
    model.to(device("try_cuda"))

    _log.info(f'{load_samples=}')
    _log.info(f'{init_method=}')

    # if save_samples:
    #     model_saver_fn = (lambda: exp_utils.HDF5ModelSaver(
    #         exp_utils.sneaky_artifact(_run, "samples.pt"), "w"))
    # else:
    #     @contextlib.contextmanager
    #     def model_saver_fn():
    #         yield None

    # MFVI approx posterior:
    # locs   = {n: t.randn((p.shape), requires_grad=True, device=device(0)) for n, p in model.named_parameters() }
    locs = {n: t.tensor((p.cpu().detach().numpy()), requires_grad=True, device=device(0)) for n, p in
            model.named_parameters()}
    scales = {n: t.randn((p.shape), requires_grad=True, device=device(0)) for n, p in model.named_parameters()}

    def sample_posterior(n_samples, only_pointwise_locs=False):
        qs = {n: Normal(l, softplus(s) + 1e-8) for (n, l), s in zip(locs.items(), scales.values())}

        if only_pointwise_locs:  # for point-wise training simply return repeated locs as samples:
            samples = {n: l.repeat(t.Size([n_samples] + [1 for _ in l.shape])) for n, l in locs.items()}
        else:  # for reparametrized gradients:
            samples = {name: q.rsample(t.Size([n_samples])) for name, q in qs.items()}

        return qs, samples

    approximation_params = list(locs.values()) + list(scales.values())

    def overwrite_params(module, samples, path="", skip_biases=False):
        for name, m in module._modules.items():
            overwrite_params(m, samples, path=path + "." + name, skip_biases=skip_biases)

        for name, p in module._parameters.items():
            if skip_biases and "bias" in name: continue
            sample_path = (path + "." + name)[1:]  # skip the leading dot
            new_value = samples[sample_path]
            # assert new_value.shape == module._parameters[name].shape, f"sample_path={sample_path} shape={new_value.shape} current shape={module._parameters[name].shape}"
            module._parameters[name] = new_value

        return module

    # iterate over samples (taken from the original code)

    def _n_samples_dict(samples):
        n_samples = min(len(v) for _, v in samples.items())

        if not all((len(v) == n_samples) for _, v in samples.items()):
            warnings.warn("Not all samples have the same length. "
                          "Setting n_samples to the minimum.")
        return n_samples

    def sample_iter(samples):
        for i in range(_n_samples_dict(samples)):
            yield dict((k, v[i]) for k, v in samples.items())

    num_workers = (0 if isinstance(data.norm.train, t.utils.data.TensorDataset) else 2)
    dataloader = t.utils.data.DataLoader(data.norm.train, batch_size=batch_size, shuffle=True, drop_last=False,
                                         num_workers=num_workers)
    dataloader_test = t.utils.data.DataLoader(data.norm.test, batch_size=batch_size, shuffle=False, drop_last=False,
                                              num_workers=num_workers)

    # evaluate before training
    # sample_epochs = n_samples * skip // cycles
    # n_samples = sample_epochs*cycles-skip_first
    n_samples = n_samples

    qs, posterior_samples = sample_posterior(n_samples)
    prior_samples = {name: buffer.repeat(n_samples) for name, buffer in model.named_buffers()}

    samples = {**posterior_samples, **prior_samples}
    evaluate_model(model, dataloader_test, samples)

    optimized_parameters = approximation_params
    optimizer = t.optim.Adam(optimized_parameters, lr=lr)

    # train
    current_step = 0
    max_step = n_epochs * len(dataloader)
    learn_distributions = True
    for epoch in range(n_epochs):
        _run.progress = current_step / max_step
        # print(f'Epoch {epoch}')
        for x, y in dataloader:
            x = x.to(device("try_cuda"))
            y = y.to(device("try_cuda"))
            # sampling from approximate posterior
            qs, posterior_samples = sample_posterior(n_samples_training, only_pointwise_locs=not learn_distributions)
            prior_samples = {name: buffer.repeat(n_samples_training) for name, buffer in model.named_buffers()}
            #
            log_likelihood, log_prior, entropy = t.tensor(0., device=device('try_cuda')), \
                                                 t.tensor(0., device=device('try_cuda')), \
                                                 t.tensor(0., device=device('try_cuda'))
            for sample_i, sample in enumerate(sample_iter({**posterior_samples, **prior_samples})):
                # model.load_state_dict(sample, strict=True)
                overwrite_params(model, sample)
                # likelihood:
                # preds = model(x)
                # log_likelihood += preds.log_prob(y).sum() * x_train.shape[0]/x.shape[0]
                log_likelihood += model.log_likelihood(x, y, x_train.shape[0])  # ???
                if learn_distributions:
                    # priors:
                    log_prior += model.log_prior()
                    # entropy:
                    # entropy += sum( q.entropy().sum() for q in qs.values() ) # closed-form for Gaussian
                    entropy += sum(
                        -q.log_prob(s).sum() for q, s in zip(qs.values(), posterior_samples.values()))  # via samples
            elbo = log_likelihood + log_prior + entropy
            _run.log_scalar('train.log_likelihood', log_likelihood.item(), current_step)
            _run.log_scalar('train.log_prior', log_prior.item(), current_step)
            _run.log_scalar('train.entropy', entropy.item(), current_step)
            loss_vi = -elbo
            _run.log_scalar('train.loss', loss_vi.item(), current_step)
            optimizer.zero_grad()
            loss_vi.backward()
            optimizer.step()
            current_step += 1
        if epoch % 10 == 0 and epoch > 0:
            qs, posterior_samples = sample_posterior(n_samples)
            prior_samples = {name: buffer.repeat(n_samples) for name, buffer in model.named_buffers()}
            results = evaluate_model(model, dataloader_test, {**posterior_samples, **prior_samples})
            for k, v in results.items():
                _run.log_scalar(f'eval.{k}', v, current_step)
    # TODO save as (sacred) artifact if possible?
    # save model
    # TODO add unique run names?
    state = model.state_dict()
    run_dir = Path.cwd() / 'runs'
    run_dir.mkdir(parents=True, exist_ok=True)
    state_path = run_dir / 'model.pth'
    t.save(state, state_path)
    _log.info(f'Saved state to local path {str(state_path)}')
