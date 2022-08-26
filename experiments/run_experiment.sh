#!/bin/bash
#SBATCH --job-name=bnn_flow
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=20G
#SBATCH --cpus-per-gpu=10

# singularity path - update if needed
source user.env
cd $PROJECT_PATH || exit
export PYTHONPATH=$PYTHONPATH:$PROJECT_PATH

# mfvi with different priors
for prior in laplace student-t convcorrnormal gaussian; do
  singularity exec $SINGULARITY_ARGS $SIF_PATH python experiments/train_mfvi.py with weight_prior=$prior bias_prior=$prior
done
