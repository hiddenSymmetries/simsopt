#!/bin/bash -l

# Standard output and error:
#SBATCH -o ./tjob_hybrid.out.%j
#SBATCH -e ./tjob_hybrid.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J f33
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#
#SBATCH --mail-type=ALL
#SBATCH --mail-user=issra.ali@ipp.mpg.de
#
# Wall clock Limit:
#SBATCH --time=00:30:00

# Load compiler and MPI modules with explicit version specifications,
# consistently with the versions used to build the executable.
# module purge
module load intel/2024.0 impi/2021.11 anaconda/3/2023.03 mkl/2024.0 netcdf-mpi/4.9.2 mpi4py/3.1.4 fftw-mpi/3.3.10 boost-mpi/1.83 hdf5-mpi/1.14.1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MKL_HOME}/lib/intel64:${HDF5_HOME}/lib:${NETCDF_HOME}/lib

source ~/envs/simsopt-e/bin/activate
# export PYTHONPATH=${PYTHONPATH}:~/.simsopt_venv/lib/python3.10/site-packages

# export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
# For pinning threads correctly:
export OMP_PLACES=threads

fname="/u/aliiss/projects/quasr2stell/quasr_vmec/input.0198998"

chmod +x /u/aliiss/projects/quasr2stell/script.py
chmod +x /u/aliiss/projects/quasr2stell/quasr_vmec/input.0198998
srun /u/aliiss/projects/quasr2stell/script.py --fname $fname > prog.out
# for filename in ./quasr_vmec/*;
# do
#     fname="$(basename ${filename})"
#     mkdir $fname
#     mv $filename ./$fname
#     cd $fname
#     srun /u/aliiss/projects/quasr2stell/script.py --fname $fname > prog.out
# done


 