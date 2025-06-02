#!/bin/bash

# Function to check last command's success
check_success() {
    if [ $? -ne 0 ]; then
        echo "Error: $1. Exiting."
        exit 1
    fi
}

# Load modules 
module load conda cray-hdf5/1.14.3.1 cray-netcdf/4.9.0.13

# Ensure Conda is available
type conda >/dev/null 2>&1 || { echo "conda command not found. Please install Anaconda/Miniconda first."; exit 1; }

echo "Adding conda-forge channel..."
conda config --add channels conda-forge
check_success "Failed to add conda-forge channel"

echo "Enter the name for the new conda environment (e.g., firm3d):"
read -p "Your input: " env_name
# Add validation for env_name if needed

echo "Creating conda environment: $env_name"
conda create -n "$env_name" python=3.9
check_success "Failed to create conda environment $env_name"

echo "Activating conda environment: $env_name"
source activate "$env_name" || conda activate "$env_name"
check_success "Failed to activate conda environment $env_name"

# Install Dependencies
echo "Installing FIRM3D dependencies..."
conda install -c conda-forge numpy "libblas=*=*mkl" --name "$env_name"
conda install matplotlib --name "$env_name"
env CC=cc MPICC=CC pip install mpi4py
check_success "Failed to install FIRM3D dependencies"

# FIRM3D Installation
cd simsopt || { echo "Error: firm3d directory not found. Exiting."; exit 1; }
env CC=cc CXX=CC pip install -e .
check_success "Failed to install FIRM3D"
cd ..

# BOOZ_XFORM Installation
cd booz_xform || { echo "Error: booz_xform directory not found. Exiting."; exit 1; }
env CC=cc CXX=CC pip install -e .
check_success "Failed to install BOOZ_XFORM"
cd ..

echo "Successfully installed FIRM3D into the conda environment '$env_name'"
echo "To activate, run: conda activate $env_name"
