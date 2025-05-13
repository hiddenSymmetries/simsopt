#!/bin/bash
#SBATCH --time=500
#SBATCH --nodes=1
#SBATCH --mem=48000
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --array=0-20

# Submit an array of 20 jobs. Each core is running 
# a different coil optimization problem with varying weights. 
export OMP_NUM_THREADS=1  # number of threads for OpenMP
export MKL_NUM_THREADS=1  # number of threads for Intel MKL
# IMPORTANT: Activate your conda environment 
# (e.g., 'conda activate simsopt_env') or any other required environment before running this script.
srun python initiation.py B2_Energy  # specify the force-related objective function to use
