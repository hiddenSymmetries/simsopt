#!/bin/bash
#SBATCH -N 1
#SBATCH -A FUA36_OHARS
# SBATCH -p skl_fua_prod
#SBATCH --mem=18000 
#SBATCH --time 01:00:00
#SBATCH --job-name=single_stage_simsopt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rogerio.jorge@tecnico.ulisboa.pt
mpirun /marconi/home/userexternal/rjorge00/simsopt/examples/2_Intermediate/single_stage/single_stage.py > output.txt
