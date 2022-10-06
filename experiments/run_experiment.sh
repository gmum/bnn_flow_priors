#!/bin/bash
#SBATCH --job-name=bnn_flow
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=10G
#SBATCH --cpus-per-gpu=1

# singularity path - update if needed
source user.env
cd $PROJECT_PATH || exit
export PYTHONPATH=$PYTHONPATH:$PROJECT_PATH

# RealNVP posterior test
#for lr in 0.01 0.001 0.0001; do
for lr in 0.001; do
  srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python experiments/train.py with \
      posterior=realnvp \
      data=mnist model=classificationconvnet weight_prior=gaussian weight_scale=1.41 bias_prior=gaussian  \
      n_samples=1 batch_size=128 lr=$lr epochs=100 \
      ood_data=fashion_mnist save_samples=True &
  wait
done

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