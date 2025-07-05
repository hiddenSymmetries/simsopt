#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=0:60:00
#SBATCH --constraint=cpu
#SBATCH --qos=premium
#SBATCH --account=m4680 # Change to your account number

module load python cray-hdf5/1.14.3.1 cray-netcdf/4.9.0.13
conda activate firm3d # Change to the name of your environment
srun -n 128 -c 1 --chdir=fusion_distribution python -u fusion_distribution.py
srun -n 128 -c 1 --chdir=fusion_distribution_perturbed python -u fusion_distribution_perturbed.py
srun -n 128 -c 1 --chdir=passing_frequencies python -u passing_frequencies.py
srun -n 128 -c 1 --chdir=passing_map_perturbed_QA python -u passing_map_perturbed.py
srun -n 128 -c 1 --chdir=passing_map_perturbed_QH python -u passing_map_perturbed.py
srun -n 128 -c 1 --chdir=passing_map_unperturbed python -u passing_map.py
srun -n 128 -c 1 --chdir=plot_trajectory python -u plot_trajectory.py
srun -n 128 -c 1 --chdir=tracing_with_AE python -u tracing_with_AE.py
srun -n 128 -c 1 --chdir=trapped_frequencies python -u trapped_frequencies.py
srun -n 128 -c 1 --chdir=trapped_map python -u trapped_map.py
srun -n 128 -c 1 --chdir=trapped_map_QI python -u trapped_map_QI.py
srun -n 128 -c 1 --chdir=uniform_surf_distribution python -u uniform_surf_distribution.py
srun -n 128 -c 1 --chdir=uniform_vol_distribution python -u uniform_vol_distribution.py
