#!/bin/bash

#SBATCH --job-name=behavior_cloning
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --mem=16g
#SBATCH -N 1
#SBATCH --mail-user=karangc2@illinois.edu

module load OpenCV/3.3.0-IGB-gcc-4.9.4-Python-3.6.1
module load Keras/2.0.8-IGB-gcc-4.9.4-Python-3.6.1
module load CUDA/8.0.61-IGB-gcc-4.9.4
module load Tensorflow-GPU/1.2.1-IGB-gcc-4.9.4-Python-3.6.1

python model.py
