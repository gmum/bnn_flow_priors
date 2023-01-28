#!/bin/bash
#SBATCH --job-name=bnn_flow
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2G

# singularity path - update if needed
source user.env
cd $PROJECT_PATH || exit
export PYTHONPATH=$PYTHONPATH:$PROJECT_PATH

# CIFAR10 smaller resnet without BN - pointwise
#for weight_prior in gaussian; do
#  for bias_prior in gaussian; do
#    for weight_posterior in pointwise; do
#      for bias_posterior in pointwise; do
#        for lr in 0.01 0.001 0.0001; do
#          srun --ntasks=1 --gpus=1 singularity exec $SINGULARITY_ARGS $SIF_PATH \
#            python experiments/train.py with \
#            data=cifar10_augmented model=googleresnet depth=8 batchnorm=False \
#            weight_prior=$weight_prior weight_scale=1.41 bias_prior=$bias_prior \
#            weight_posterior=$weight_posterior bias_posterior=$bias_posterior \
#            n_samples=100 batch_size=64 lr=$lr epochs=400 \
#            ood_data=svhn &
#        done
#      done
#    done
#  done
#done

#kl_weight="{'type':'linear','initial':0.0,'final':1.0}"
#kl_weight=0.01
#kl_weight=1.0
#kl_weight=0.0

# CIFAR10 smaller resnet without BN - RealNVP
#for weight_prior in gaussian; do
#  for bias_prior in gaussian; do
#    for weight_posterior in realnvp; do
##      for bias_posterior in realnvp pointwise; do
#      for bias_posterior in pointwise; do
##        for lr in 0.01 0.001 0.0001; do
#        for lr in 0.001; do
#          srun --ntasks=1 --gpus=1 singularity exec $SINGULARITY_ARGS $SIF_PATH \
#            python experiments/train.py with \
#            data=cifar10_augmented model=googleresnet depth=8 batchnorm=False \
#            weight_prior=$weight_prior weight_scale=1.41 bias_prior=$bias_prior \
#            weight_posterior=$weight_posterior bias_posterior=$bias_posterior \
#            kl_weight=0.0 \
#            realnvp_m=64 realnvp_num_layers=4 rezero_trick=False \
#            n_samples=100 batch_size=64 lr=$lr epochs=400 \
#            ood_data=svhn &
#        done
#      done
#    done
#  done
#done


# CIFAR10 smaller resnet without BN - Bayesian Hypernetworks
#for weight_prior in gaussian; do
#  for bias_prior in gaussian; do
#    for weight_posterior in hypernet; do
#      for bias_posterior in hypernet pointwise; do
#        for lr in 0.001; do
#          srun --ntasks=1 --gpus=1 singularity exec $SINGULARITY_ARGS $SIF_PATH \
#            python experiments/train.py with \
#            data=cifar10_augmented model=googleresnet depth=8 batchnorm=False \
#            weight_prior=$weight_prior weight_scale=1.41 bias_prior=$bias_prior \
#            weight_posterior=$weight_posterior bias_posterior=$bias_posterior \
#            n_samples=100 batch_size=64 lr=$lr epochs=400 \
#            ood_data=svhn &
#        done
#      done
#    done
#  done
#done

# MNIST
for weight_prior in gaussian; do
  for bias_prior in gaussian; do
    for weight_posterior in realnvp; do
      for bias_posterior in pointwise; do
        for lr in 0.001 0.0001 0.00001; do
          srun --ntasks=1 --gpus=1 singularity exec $SINGULARITY_ARGS $SIF_PATH \
            python experiments/train.py with \
            data=mnist model=classificationconvnet \
            weight_prior=$weight_prior weight_scale=1.41 bias_prior=$bias_prior \
            weight_posterior=$weight_posterior bias_posterior=$bias_posterior \
            kl_weight=0.0 \
            realnvp_m=64 realnvp_num_layers=4 rezero_trick=True \
            n_samples=100 batch_size=64 lr=$lr epochs=100 \
            ood_data=fashion_mnist &
        done
      done
    done
  done
done

wait