#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:30:00
#SBATCH --constraint=cpu
#SBATCH --qos=debug 
#SBATCH --account=m4680 # Change to your account number

module load python cray-hdf5/1.14.3.1 cray-netcdf/4.9.0.13
conda activate firm3d # Change to the name of your environment
srun -n 128 -c 1 python -u fusion_distribution.py
