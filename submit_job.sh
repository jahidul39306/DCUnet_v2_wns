#!/bin/bash

#SBATCH --job-name=my_python_job
#SBATCH --output=my_job.out
#SBATCH --error=my_job.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:4
#SBATCH --time=6:00:00

# Run Python script in container
singularity exec --nv /ceph/container/pytorch/pytorch_26.02.sif python complex_train.py