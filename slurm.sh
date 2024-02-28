#!/bin/bash
#SBATCH -A dqy@v100
#SBATCH --job-name=Entity-Alignment-LLM-Experiments
#SBATCH --output=logs.out
#SBATCH --error=logs.err
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH -c 64
#SBATCH --time=19:59:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=billhappi@gmail.com

##source activate pytorch
# module load cuda/11.7.1 nccl/2.12.12-1-cuda cudnn/8.5.0.96-11.7-cuda gcc/8.5.0 openmpi/4.1.1-cuda intel-mkl/2020.4 magma/2.7.0-cuda sox/14.4.2 sparsehash/2.0.3 libjpeg-turbo/2.1.3
# module load pytorch-gpu/py3/2.0.0

sh ./job.sh
