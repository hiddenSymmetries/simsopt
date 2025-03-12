# firm3d

FIRM3D (Fast Ion Reduced Models in 3D) is a software suite for modeling of energetic particle dynamics in 3D magnetic fields. The guiding center equations of motion are integrated in magnetic fields represented in Boozer coordinates, including VMEC equilibria and Alfvén eigenmodes from AE3D or FAR3D. The core routines are based on [SIMSOPT](https://simsopt.readthedocs.io), but have been extended to include additional physics and diagnostics that are not typically required in the optimization context. This standalone framework enables more modular development of FIRM3D with minimal dependencies. 

- [Installation](#installation)
  - [macOS](#macos)
  - [Perlmutter](#perlmutter)
  - [Ginsburg](#ginsburg)
- [Magnetic field classes](#magnetic-field-classes)

## Installation

# macOS

All dependencies can be installed inside a conda environment using the provided install bash script located at `install/macOS/install_macos.sh`. 

# Perlmutter (NERSC)

module load conda cray-hdf5-parallel cray-netcdf-hdf5parallel openmpi
conda create --name simsopt
conda activate simsopt
conda install pip
export CI=True
env CC=$(which mpicc) CXX=$(which mpicxx) pip install -e . 

# Ginsburg

## Equilibrium magnetic field classes

The equilibrium magnetic field in Boozer coordinates can be represented using a simple [analytic model](#boozeranalytic), a [radial interpolant](#boozerradialinterpolant) of a `booz\_xform` equilibrium , or a [3D interpolant](#interpolatedboozerfield). 

# BoozerAnalytic

Computes a `BoozerMagneticField` based on a first-order expansion in
distance from the magnetic axis (Landreman & Sengupta, Journal of Plasma
Physics 2018). A possibility to include a QS-breaking perturbation is added, so the magnetic field strength is expressed as,

```math
B(s,\theta,\zeta) = B_0 \left(1 + \overline{\eta} \sqrt{2s\psi_0/\overline{B}}\cos(\theta - N \zeta)\right) + B_{0z}\cos(m\theta-n\zeta),
```

the covariant components of equilibrium field are,

```math
G(s) = G_0 + \sqrt{2s\psi_0/\overline{B}} G_1 \\

I(s) = I_0 + \sqrt{2s\psi_0/\overline{B}} I_1 \\

K(s,\theta,\zeta) = \sqrt{2s\psi_0/\overline{B}} K_1 \sin(\theta - N \zeta),
```

and the rotational transform is,

```math
\iota(s) = \iota_0.
```

While formally $I_0 = I_1 = G_1 = K_1 = 0$, these terms have been included
in order to test the guiding center equations at finite beta.

# BoozerRadialInterpolant

Given a :class:`Vmec` instance, performs a Boozer coordinate transformation using ``BOOZXFORM``. The magnetic field can be computed at any point in Boozer coordinates using radial spline interpolation (``scipy.interpolate.InterpolatedUnivariateSpline``) and an inverse Fourier transform in the two angles. 

# InterpolatedBoozerField