#!/bin/bash
#SBATCH --time=600
#SBATCH --mail-type=END
#SBATCH --mail-user=aak572@nyu.edu
#SBATCH --nodes=1
#SBATCH --mem=48000
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --array=0-30
#SBATCH --partition=cs

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
srun /scratch/projects/kaptanoglulab/AK/run-simsopt.bash python initiation.py
