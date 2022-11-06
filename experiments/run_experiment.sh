#!/bin/bash
#SBATCH --job-name=bnn_flow
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2G


# singularity path - update if needed
source user.env
cd $PROJECT_PATH || exit
export PYTHONPATH=$PYTHONPATH:$PROJECT_PATH


# RealNVP CIFAR10 with or without biases for different LRs
for prior in gaussian; do
  for lr in 0.001 0.0001 0.0003 0.00001; do
    for bias_posterior in realnvp pointwise; do
      for add_change_log_det_J in True False; do
        for rezero_trick in True False; do
          srun --ntasks=1 --gpus=1 singularity exec $SINGULARITY_ARGS $SIF_PATH \
              python experiments/train.py with \
              data=cifar10_augmented model=googleresnet weight_prior=$prior weight_scale=1.41 bias_prior=$prior \
              n_samples=100 batch_size=128 lr=$lr epochs=50 \
              weight_normalization=True weights_posterior=realnvp bias_posterior=$bias_posterior \
              rezero_trick=$rezero_trick add_change_log_det_J=$add_change_log_det_J \
              ood_data=svhn save_samples=True &
        done
      done
    done
  done
done
wait

# RealNVP MNIST with or without biases for different LRs
#for prior in gaussian; do
#  for lr in 0.001 0.0001; do
#    for bias_posterior in realnvp pointwise; do
#      for entropy_weight in 1.0 0.1 0.01 0.001; do
#        srun --ntasks=1 --gpus=1 singularity exec $SINGULARITY_ARGS $SIF_PATH \
#            python experiments/train.py with \
#            data=mnist model=classificationconvnet weight_prior=$prior weight_scale=1.41 bias_prior=$prior \
#            n_samples=100 batch_size=128 lr=$lr epochs=100 \
#            weight_normalization=True weights_posterior=realnvp bias_posterior=$bias_posterior \
#            entropy_weight=$entropy_weight \
#            ood_data=fashion_mnist save_samples=True &
#      done
#    done
#  done
#done
#wait

# RealNVP MNIST sanity check - is our posterior $N(0, 1)$ without the cross entropy loss term?
#bias_posterior="pointwise"
#prior="gaussian"
#lr=0.001
#entropy_weight=1.0
#for ce_weight in 0.0 1.0; do
#  srun --ntasks=1 --gpus=1 singularity exec $SINGULARITY_ARGS $SIF_PATH \
#      python experiments/train.py with \
#      data=mnist model=classificationconvnet weight_prior=$prior weight_scale=1.41 bias_prior=$prior \
#      n_samples=100 batch_size=128 lr=$lr epochs=100 \
#      weight_normalization=True weights_posterior=realnvp bias_posterior=$bias_posterior \
#      ce_weight=$ce_weight entropy_weight=$entropy_weight \
#      ood_data=fashion_mnist save_samples=True &
#done
#wait


## PREVIOUS OUTDATED RUN SCRIPTS:

# RealNVP CIFAR with different priors
#for prior in gaussian student-t laplace; do
#for prior in gaussian; do
#  for lr in 0.01 0.001 0.0001; do
#  for lr in 0.001; do
#    srun --ntasks=1 --gpus=1 singularity exec $SINGULARITY_ARGS $SIF_PATH \
#        python experiments/train.py with \
#        posterior=realnvp \
#        data=cifar10_augmented model=googleresnet weight_prior=$prior weight_scale=1.41 bias_prior=$prior \
#        n_samples=100 batch_size=128 lr=$lr epochs=400 \
#        ood_data=svhn save_samples=True &
#  done
#done
#wait

# RealNVP MNIST with different priors - same for both weights and biases
#for prior in gaussian student-t laplace; do
#  for lr in 0.01 0.001 0.0001; do
#    srun --ntasks=1 --gpus=1 --mem-per-cpu=2G singularity exec $SINGULARITY_ARGS $SIF_PATH \
#        python experiments/train.py with \
#        posterior=realnvp \
#        data=mnist model=classificationconvnet weight_prior=$prior weight_scale=1.41 bias_prior=$prior \
#        n_samples=100 batch_size=128 lr=$lr epochs=100 \
#        ood_data=fashion_mnist save_samples=True &
#  done
#done
#wait

# CIFAR with different priors - same for both weights and biases
#for prior in gaussian student-t laplace; do
#  for lr in 0.01 0.001 0.0001; do
#    srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python experiments/train_mfvi.py with \
#      data=cifar10_augmented model=googleresnet batchnorm=True \
#      weight_prior=$prior weight_scale=1.41 bias_prior=$prior \
#      n_samples=300 batch_size=128 lr=$lr epochs=400 \
#      ood_data=svhn save_samples=True &
#  done
#done
#wait

# CIFAR with different priors - but always gaussian bias prior
#for prior in gaussian student-t laplace; do
#  for lr in 0.01 0.001 0.0001; do
#    srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python experiments/train_mfvi.py with \
#      data=cifar10_augmented model=googleresnet batchnorm=True \
#      weight_prior=$prior weight_scale=1.41 bias_prior=gaussian \
#      n_samples=300 batch_size=128 lr=$lr epochs=400 \
#      ood_data=svhn save_samples=True &
#  done
#done
#wait

# MNIST with different priors - same for both weights and biases
#for prior in gaussian student-t laplace; do
#  for lr in 0.1 0.01 0.001 0.0001; do
#    srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python experiments/train.py with \
#      data=mnist model=classificationconvnet weight_prior=$prior weight_scale=1.41 bias_prior=$prior  \
#      n_samples=300 batch_size=128 lr=$lr epochs=100 \
#      ood_data=fashion_mnist save_samples=True &
#  done
#done
#wait

# MNIST with different priors - but always gaussian bias prior
#for prior in gaussian student-t laplace; do
#  for lr in 0.1 0.01 0.001 0.0001; do
#    srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python experiments/train.py with \
#      data=mnist model=classificationconvnet weight_prior=$prior weight_scale=1.41 bias_prior=gaussian  \
#      n_samples=300 batch_size=128 lr=$lr epochs=100 \
#      ood_data=fashion_mnist save_samples=True &
#  done
#done
#wait

# MNIST 3 seeds for the best LRs
#for seed in 1 2 3; do
#  srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python experiments/train.py with \
#    data=mnist model=classificationconvnet weight_prior=gaussian weight_scale=1.41 bias_prior=gaussian  \
#    n_samples=300 batch_size=128 lr=0.001 epochs=100 \
#    ood_data=fashion_mnist save_samples=True &
#
#  srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python experiments/train.py with \
#    data=mnist model=classificationconvnet weight_prior=laplace weight_scale=1.41 bias_prior=laplace  \
#    n_samples=300 batch_size=128 lr=0.001 epochs=100 \
#    ood_data=fashion_mnist save_samples=True &
#
#  srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python experiments/train.py with \
#    data=mnist model=classificationconvnet weight_prior=laplace weight_scale=1.41 bias_prior=gaussian  \
#    n_samples=300 batch_size=128 lr=0.001 epochs=100 \
#    ood_data=fashion_mnist save_samples=True &
#
#  srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python experiments/train.py with \
#    data=mnist model=classificationconvnet weight_prior=student-t weight_scale=1.41 bias_prior=student-t  \
#    n_samples=300 batch_size=128 lr=0.001 epochs=100 \
#    ood_data=fashion_mnist save_samples=True &
#
#  srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python experiments/train.py with \
#    data=mnist model=classificationconvnet weight_prior=student-t weight_scale=1.41 bias_prior=gaussian  \
#    n_samples=300 batch_size=128 lr=0.001 epochs=100 \
#    ood_data=fashion_mnist save_samples=True &
#done
#wait

# MNIST with sigma stats logging
#for seed in 1; do
#  srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python experiments/train.py with \
#    data=mnist model=classificationconvnet weight_prior=gaussian weight_scale=1.41 bias_prior=gaussian  \
#    n_samples=300 batch_size=128 lr=0.001 epochs=100 \
#    ood_data=fashion_mnist save_samples=True log_mfvi=True &
#
#  srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python experiments/train.py with \
#    data=mnist model=classificationconvnet weight_prior=laplace weight_scale=1.41 bias_prior=laplace  \
#    n_samples=300 batch_size=128 lr=0.001 epochs=100 \
#    ood_data=fashion_mnist save_samples=True log_mfvi=True &
#
#  srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python experiments/train.py with \
#    data=mnist model=classificationconvnet weight_prior=laplace weight_scale=1.41 bias_prior=gaussian  \
#    n_samples=300 batch_size=128 lr=0.001 epochs=100 \
#    ood_data=fashion_mnist save_samples=True log_mfvi=True &
#
#  srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python experiments/train.py with \
#    data=mnist model=classificationconvnet weight_prior=student-t weight_scale=1.41 bias_prior=student-t  \
#    n_samples=300 batch_size=128 lr=0.001 epochs=100 \
#    ood_data=fashion_mnist save_samples=True log_mfvi=True &
#
#  srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python experiments/train.py with \
#    data=mnist model=classificationconvnet weight_prior=student-t weight_scale=1.41 bias_prior=gaussian  \
#    n_samples=300 batch_size=128 lr=0.001 epochs=100 \
#    ood_data=fashion_mnist save_samples=True log_mfvi=True &
#done
#wait