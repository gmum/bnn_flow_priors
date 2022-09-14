#!/bin/bash
#SBATCH --job-name=bnn_flow
#SBATCH --qos=big
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=20G
#SBATCH --cpus-per-gpu=10

# singularity path - update if needed
source user.env
cd $PROJECT_PATH || exit
export PYTHONPATH=$PYTHONPATH:$PROJECT_PATH

# mfvi with different priors
#for prior in improper gaussian laplace student-t convcorrnormal; do
for prior in gaussian; do
  #  for lr in 0.1 0.01 0.001; do
  for lr in 0.1 0.01 0.001 0.0001; do
    # MNIST
    #    srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python experiments/train_mfvi.py with lr=$lr \
    #      weight_prior=$prior bias_prior=$prior epochs=100 data=mnist &
    # CIFAR
    #    srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python experiments/train_mfvi.py with lr=$lr \
    #      weight_prior=$prior bias_prior=$prior epochs=100 schedule=cosine data=cifar10 ood_data=svhn &
    srun --gres=gpu:1 singularity exec $SINGULARITY_ARGS $SIF_PATH python experiments/train_mfvi.py with \
      data=cifar10_augmented model=googleresnet batchnorm=True weight_prior=$prior weight_scale=1.41 \
      n_samples=300 batch_size=128 lr=$lr epochs=150 schedule=cosine save_samples=True \
      ood_data=svhn &
  done
done
wait
