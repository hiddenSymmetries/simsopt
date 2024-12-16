#!/bin/bash
#SBATCH --time=300
#SBATCH --mail-type=END
#SBATCH --mail-user=aak572@nyu.edu
#SBATCH --nodes=1
#SBATCH --mem=100000
#SBATCH --ntasks-per-node=48
#SBATCH --cpus-per-task=1
#SBATCH --array=0-20
#SBATCH --partition=cs

srun /scratch/projects/kaptanoglulab/AK/run-simsopt.bash python initiation.py
