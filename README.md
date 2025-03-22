# firm3d

FIRM3D (Fast Ion Reduced Models in 3D) is a software suite for modeling of energetic particle dynamics in 3D magnetic fields. The guiding center equations of motion are integrated in magnetic fields represented in Boozer coordinates, including VMEC equilibria and Alfvén eigenmodes from AE3D or FAR3D. The core routines are based on [SIMSOPT](https://simsopt.readthedocs.io), but have been extended to include additional physics and diagnostics that are not typically required in the optimization context. This standalone framework enables more modular development of FIRM3D with minimal dependencies. 

- [Installation](#installation)
  - [macOS](#macos)
  - [Perlmutter](#perlmutter)
  - [Ginsburg](#ginsburg)
- [Magnetic field classes](#magnetic-field-classes)
  - [BoozerAnalytic](#boozeranalytic)
  - [BoozerRadialInterpolant](#boozerradialinterpolant)
  - [InterpolatedBoozerField](#interpolatedboozerfield)
- [Shear Alfvén wave field classes](#shear-alfvén-wave-field-classes)
  - [ShearAlfvenHarmonic](#shearalfvenharmonic)
  - [ShearAlfvenWavesSuperposition](#shearalfvenwavessuperposition)
- [Guiding center integration](#guiding-center-integration)
  - [Unperturbed guiding center integration](#unperturbed-guiding-center-integration)
  - [Perturbed guiding center integration](#perturbed-guiding-center-integration)
  - [Stopping criteria](#stopping-criteria)
  - [Trajectory saving](#trajectory-saving)
  - [Magnetic axis handling](#magnetic-axis-handling)
  - [Solvers and solver options](#solvers-and-solver-options)

## Installation

### macOS

All dependencies can be installed inside a conda environment using the provided install bash script located at `install/macOS/install_macos.sh`. 

### Perlmutter (NERSC)

The following commands will install 

```
module load conda cray-hdf5-parallel cray-netcdf-hdf5parallel openmpi
conda create --name firm3d
conda activate firm3d
conda install pip
export CI=True
cd firm3d
env CC=$(which mpicc) CXX=$(which mpicxx) pip install -e . 
```

## Equilibrium magnetic field classes

The equilibrium magnetic field in Boozer coordinates, an instance of ``BoozerMagneticField``, can be represented using a simple [analytic model](#boozeranalytic), a [radial interpolant](#boozerradialinterpolant) of a ``booz_xform`` equilibrium , or a [3D interpolant](#interpolatedboozerfield). 

### BoozerAnalytic

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


### BoozerRadialInterpolant

The magnetic field can be computed at any point in Boozer coordinates using radial spline interpolation (``scipy.interpolate.InterpolatedUnivariateSpline``) and an inverse Fourier transform in the two angles. While the Fourier representation is more accurate, typically [InterpolatedBoozerField](#interpolatedboozerfield) is used inside the tracing loop due to its efficiency. If given a `VMEC` output file, performs a Boozer coordinate transformation using ``booz_xform``. If given a ``booz_xform`` output file, the Boozer transformation must be performed with all surfaces on the VMEC half grid, and with `phip`, `chi`, `pres`, and `phi` saved in the file. 

### Preparing `booz_xform` equilibrium

As stated above, the `booz_xform` equilibrium must be performed with all surfaces on the VMEC half grid, and with `phip`, `chi`, `pres`, and `phi` saved in the file. This can be done using the [C++ implementation](https://github.com/hiddenSymmetries/booz_xform) with the phip_fix branch, by passing `flux=True` to `read_wout()`:

```
import booz_xform as bx

b = bx.Booz_xform()
b.read_wout(wout_filename,True)
b.mboz = mboz
b.nboz = nboz
b.run()
b.write_boozmn(boozmn_filename)
```
Equilibria produced with the [STELLOPT implementation](https://github.com/PrincetonUniversity/STELLOPT) can also be used.  

### InterpolatedBoozerField

This field takes an existing ``BoozerMagneticField`` instance, such as [BoozerRadialInterpolant](#boozerradialinterpolant), and interpolates it on a regular grid in $(s,\theta,\zeta)$. This resulting interpolant can then be evaluated very quickly inside the tracing loop. 

## Shear Alfvén wave field classes

Given an equilibrium field $\textbf{B}_0$, a shear Alfvén wave is modeled through the perturbed electrostatic potential, $\Phi$, and parameter $\alpha$ defining the perturbed vector potential $\delta \textbf{A} = \alpha \textbf{B}_0$. The perturbed electric and magnetic fields then satisfy:

```math
\delta \textbf{E} = -\nabla \Phi - \frac{\partial \alpha}{\partial t} \\
\delta \textbf{B} = \nabla \times \left(\alpha \textbf{B}_0 \right). 
```

The parameter $\alpha$ is related to $\Phi$ given ideal Ohm's law:
```math
\nabla_{||} \Phi = -B_0 \frac{\partial \alpha}{\partial t}.
```

In the ``ShearAlfvenWave`` classes, the equilibrium field is prescribed as a ``BoozerMagneticField`` class in addition to input parameters determining $\Phi$. 

### ShearAlfvenHarmonic

This class initializes a Shear Alfvén Wave with a scalar potential of the form:

```math
  \Phi(s, \theta, \zeta, t) = \hat{\Phi}(s) \sin(m \theta - n \zeta + \omega t + \text{phase}),
```

where $\hat{\Phi}(s)$ is a radial profile, $m$ is the poloidal mode number, $n$ is the toroidal mode number, $\omega$ is the frequency, and `phase` is the phase shift. The perturbed parallel vector potential parameter, $\alpha$, is then determined by ideal Ohm's law. This representation is used to describe SAWs propagating in an equilibrium magnetic field $\textbf{B}_0$. 

### ShearAlfvenWavesSuperposition

Class representing a superposition of multiple Shear Alfvén Waves (SAWs).

This class models the superposition of multiple Shear Alfvén waves, combining their scalar potentials $\Phi$, vector potential parameters $\alpha$, and their respective derivatives to represent a more complex wave structure in the equilibrium field $\textbf{B}_0$.

The superposition of waves is initialized with a base wave, which defines the reference equilibrium field $\textbf{B}_0$ for all subsequent waves added to the superposition. All added waves must have the same $\textbf{B}_0$ field.

See Paul et al., JPP (2023; 89(5):905890515. doi:10.1017/S0022377823001095) for more details.

## Unperturbed guiding center integration

Guiding center integration in Boozer coordinates is performed using equations of motion obtained from the Littlejohn Lagrangian,
```math
L(\psi,\theta,\zeta,\rho_{||})  = q\left(\left[\psi + I  \rho_{||}\right] \dot{\theta} + \left[- \chi + G \rho_{||} \right] \dot{\zeta} + \rho_{||} K \dot{\psi}\right)  - \frac{\rho_{||}^2 B_0^2q^2}{2m} - \mu B_0 ,
```
where $2\pi \psi$ is the toroidal flux, $2\pi \chi$ is the poloidal flux, $q$ is the charge, $m$ is the mass$, $\rho_{||} = q v_{||}/(m B)$ and the covariant form of the magnetic field is,
```math
\textbf{B} = G(\psi) \nabla \zeta + I(\psi) \nabla \theta + K(\psi,\theta,\zeta) \nabla \psi.
```
See R. White, Theory of Tokamak Plasmas, Sec. 3.2. 

The trajectory information is saved as $(s,\theta,\zeta,v_{||})$, where $s = \psi_0$ is the toroidal flux normalized to its value on the boundary, $2\pi\psi_0$

Various integrators, equations of motion, and solver options are detailed below.  

### Perturbed guiding center integration

The primary routine for unperturbed guiding center integration is ``trace_particles_boozer``. 

In the case of ``mode='gc_vac'`` we solve the guiding center equations under the vacuum assumption, i.e $G =$ const. and $I = 0$

```math
\dot s = -B_{,\theta} \frac{m\left(\frac{v_{||}^2}{B} + \mu \right)}{q \psi_0} \\

\dot \theta = B_{,s} m\frac{\frac{v_{||}^2}{B} + \mu}{q \psi_0} + \frac{\iota v_{||} B}{G} \\

\dot \zeta = \frac{v_{||}B}{G} \\

\dot v_{||} = -(\iota B_{,\theta} + B_{,\zeta})\frac{\mu B}{G}, \\
```

where $q$ is the charge, $m$ is the mass, and $v_\perp^2 = 2\mu B$.

In the case of ``mode='gc'`` we solve the general guiding center equations for an MHD equilibrium:

```math
  \dot s = (I B_{,\zeta} - G B_{,\theta})\frac{m\left(\frac{v_{||}^2}{B} + \mu\right)}{\iota D \psi_0} \\
  \dot \theta = \left((G B_{,\psi} - K B_{,\zeta}) m\left(\frac{v_{||}^2}{B} + \mu\right) - C v_{||} B\right)\frac{1}{\iota D} \\
  \dot \zeta = \left(F v_{||} B - (B_{,\psi} I - B_{,\theta} K) m\left(\frac{v_{||}^2}{B}+ \mu\right)\right) \frac{1}{\iota D} \\
  \dot v_{||} = (CB_{,\theta} - FB_{,\zeta})\frac{\mu B}{\iota D} \\
  C = - \frac{m v_{||} K_{,\zeta}}{B}  - q \iota + \frac{m v_{||}G'}{B} \\
  F = - \frac{m v_{||} K_{,\theta}}{B} + q + \frac{m v_{||}I'}{B} \\
  D = (F G - C I)/\iota 
```
where primes indicate differentiation wrt $\psi$. In the case ``mod='gc_noK'``, the above equations are used with $K=0$.

# Perturbed guiding center integration

In the case of ``mode='gc_vac'`` we solve the guiding center equations under the vacuum assumption, i.e. $G =$ const. and $I = 0$. 

```math
\dot s      = \left(-B_{,\theta} m \frac{\frac{v_{||}^2}{B} + \mu}{q} + \alpha_{,\theta}B v_{||} - \Phi_{,\theta}\right)\frac{1}{\psi_0} \\
\dot \theta = B_{,\psi} m \frac{\frac{v_{||}^2}{B} + \mu}{q} + (\iota - \alpha_{,\psi} G) \frac{v_{||}B}{G} + \Phi_{,\psi} \\
\dot \zeta  = \frac{v_{||}B}{G} \\
\dot v_{||} = -\frac{B}{Gm} \Bigg(m\mu \left(B_{,\zeta} + \alpha_{,\theta}B_{,\psi}G + B_{,\theta}(\iota - \alpha_{,\psi}G)\right) \\
              + q \left(\dot\alpha G + \alpha_{,\theta}G\Phi_{,\psi} + (\iota - \alpha_{,\psi}G)\Phi_{,\theta} + \Phi_{,\zeta}\right)\Bigg) \\
              + \frac{v_{||}}{B} (B_{,\theta}\Phi_{,\psi} - B_{,\psi} \Phi_{,\theta})
```
where $q$ is the charge, $m$ is the mass, and $v_\perp^2 = 2\mu B$.

In the case of ``mode='gc'`` we solve the general guiding center equations for an MHD equilibrium:
```math
   \dot{s} = \Bigg(-G \Phi_{,\theta}q + I\Phi_{,\zeta}q
                    + B qv_{||}(\alpha_{,\theta}G-\alpha_{,\zeta}I)
                    + (-B_{,\theta}G + B_{,\zeta}I)
                    \left(\frac{mv_{||}^2}{B} + m\mu\right)\Bigg)\frac{1}{D \psi_0} \\
   \dot{\theta} = \Bigg(G q \Phi_{,\psi}
                   + B q v_{||} (-\alpha_{,\psi} G - \alpha G_{,\psi} + \iota)
                   - G_{,\psi} m v_{||}^2 + B_{,\psi} G \left(\frac{mv_{||}^2}{B} + m\mu\right)\Bigg)\frac{1}{D} \\
   \dot{\zeta} =  \Bigg(-I (B_{,\psi} m \mu + \Phi_{,\psi} q) 
                    + B q v_{||} (1 + \alpha_{,\psi} I + \alpha I'(\psi))
                    + \frac{m v_{||}^2}{B} (B I'(\psi) - B_{,\psi} I)\Bigg)\frac{1}{D} \\
   \dot{v_{||}} = \Bigg(\frac{Bq}{m} \Big(-m \mu (B_{,\zeta}(1 + \alpha_{,\psi} I + \alpha I'(\psi)) 
                     + B_{,\psi} (\alpha_{,\theta} G - \alpha_{,\zeta} I) 
                     + B_{,\theta} (\iota - \alpha G'(\psi) - \alpha_{,\psi} G)) \\ 
                     - q \Big(\dot{\alpha} \left(G + I (\iota - \alpha G'(\psi)) + \alpha G I'(\psi)\right) 
                     + \left(\alpha_{,\theta} G - \alpha_{,\zeta} I\right) \Phi_{,\psi} \\
                     + \left(\iota - \alpha G_{,\psi} - \alpha_{,\psi} G\right) \Phi_{,\theta} 
                     + \left(1 + \alpha I'(\psi) + \alpha_{,\psi} I\right) \Phi_{,\zeta}\Big) \Big) \\ 
                     + \frac{q v_{||}}{B} \left((B_{,\theta} G - B_{,\zeta} I) \Phi_{,\psi}
                     + B_{,\psi} \left(I \Phi_{,\zeta} - G \Phi_{,\theta}\right)\right) \\
                     + v_{||} \big(m \mu \left(B_{,\theta} G'(\psi) - B_{,\zeta} I'(\psi)\right) \\
                     + q \left(\dot \alpha \left(G'(\psi) I - G I'(\psi)\right) \\
                     + G'(\psi) \Phi_{,\theta} - I'(\psi)\Phi_{,\zeta}\right)\big)\Bigg)\frac{1}{D} \\
  D = q (G + I(-\alpha G'(\psi) + \iota) + \alpha G I'(\psi)) 
                + \frac{m v_{\|}}{B} \left(-G'(\psi)I + G I'(\psi)\right)
```

## Stopping criteria

Guiding center integration is continued until the maximum integration time, `tmax`, is reached, or until one of the `StoppingCriteria` is hit. Stopping criteria include:
- `MaxToroidalFluxStoppingCriterion`: stop when trajectory reaches a maximum value of normalized toroidal flux (e.g., $s=1$ indicates the plasma boundary)
- `MinToroidalFluxStoppignCriterion`: stop when trajectory reaches a minimum value of normalized toroidal flux. Sometimes a point close to the axis, e.g. $s = 10^{-3}$, is chosen to avoid numerical issues associated with the coordinate singularity. 
- `ZetaStoppingCriterion`: stop when the toroidal angle reaches a given value (modulus $2\pi$).
- `VparStoppingCriterion`: stop when the parallel velocity reaches a given value. For example, can be used to terminate tracing when a particle mirrors. 
- `ToroidalTransitStoppingCriterion`: stop when the toroidal angle increases by an integer multiple of $2\pi$. Useful for resonance detection.
- `IterationStoppingCriterion`: stop when a number of iterations is reached. 
- `StepSizeStoppingCriterion`: stop when the step size gets too small. When using adaptive timestepping, can avoid particles getting "stuck" due to small step size.

## Trajectory saving

There are two ways the trajectory information can be saved: by recording "hits" of user-defined coordinate planes (e.g., Poincaré sections), or by recording uniform time intervals of the trajectory. The routines `trace_particles_boozer` or `trace_particles_boozer_perturbed` return this information in the tuple `(res_tys,res_hits)`. 

- If `forget_exact_path=False`, the parameter `dt_save` determines the time interval for trajectory saving. (Note that if this parameter is made too small, one may run into memory issues.) This trajectory information is returned in `res_tys`, which is a list (length = number of particles) of numpy arrays with shape `(nsave,5)`. Here `nsave` is the number of timesteps saved. Each row contains the time and the state, `[t, s, theta, zeta, v_par]`. If `forget_exact_path=True`, only the state at the initial and final time will be returned. 
- The "hits" are defined through the input lists `zetas`, `omegas`, `vpars`. If `vpars` is specified, the trajectory will be recorded when the parallel velocity hits a given value. For example, the Poincaré map for trapped particles is defined by recording the points with $v_{||} = 0$. If `zetas` is specified, the trajectory will be recorded when $\zeta - \omega t$ hits the values given in the `zetas` array, with the frequency $\omega$ given by the `omegas` array. The `zetas` and `omegas` lists must have same length. If `omegas` is not specified, it defaults to zeros. This feature is useful for defining the Poincaré map for passing particles (with or without a single-harmonic shear Alfvén wave). The hits are returned in `res_hits`, which is a list (length = number of particles) of numpy arrays with shape `(nhits,6)`, where `nhits` is the number of hits of a coordinate plane or stopping criteria. Each row or the array contains `[time] + [idx] + state`, where `idx` tells us which of the hit planes or stopping criteria was hit. If `idx>=0` and `idx<len(zetas)`, then the `zetas[idx]` plane was hit. If `idx>=len(zetas)`, then the `vpars[idx-len(zetas)]` plane was hit. If `idx<0`, then `stopping_criteria[int(-idx)-1]` was hit. The state vector is `[t, s, theta, zeta, v_par]`.

## Magnetic axis handling

The coordinate singularity at the magnetic axis can be handled in several ways using the keyword argument `axis` passed to 
`trace_particles_boozer` and `trace_particles_boozer_perturbed`.
- If `axis=0`, the trajectory will be integrated in standard Boozer coordinates $(s,\theta,\zeta)$. If this is used, it is recommended that one passes a `MinToroidalFluxStoppingCriterion` to prevent particles from passing to $s < 0$.
- If `axis=1`, the trajectory will be integrated in the pseudo-Cartesian coordinates $(\sqrt{s}\cos(\theta),\sqrt{s}\sin(\theta),\zeta)$, but all trajectory information will be saved in the standard Boozer coordinates $(s,\theta,\zeta)$. This option prevents particles from passing to $s < 0$. Because the equations of motion are mapped form $(s,\theta,\zeta)$ to $(\sqrt{s}\cos(\theta),\sqrt{s},\sin(\theta),\zeta)$, a division by $\sqrt{s}$ is performed. Thus this option may be ill-behaved near the axis. 
- If `axis=2`, the trajectory will be integrated in the pseudo-Cartesian coordinates $(s\cos(\theta),s\sin(\theta),\zeta)$, but all trajectory information will be saved in the standard Boozer coordinates $(s,\theta,\zeta)$. This option prevents particles from passing to $s < 0$. No division by $s$ is required to map to this coordinate system. This option is recommended if one would like to integrate near the magnetic axis. 

## Solvers and solver options

By default the Runge-Kutta Dormand-Prince 5(4) method implemented in [Boost](https://www.boost.org/doc/libs/1_54_0/boost/numeric/odeint/stepper/runge_kutta_dopri5.hpp) is used to integrate the ODEs. Adaptive time stepping is performed to satisfy the user-prescribed relative and absolute error tolerance parameters, `reltol` and `abstol`. 

If `solveSympl=True` in the `solver_options`, a symplectic solver is used with step size `dt`. The semi-implicit Euler scheme described in[Albert, C. G., et al. (2020). Symplectic integration with non-canonical quadrature for guiding-center orbits in magnetic confinement devices. Journal of computational physics, 403, 109065](https://doi.org/10.1016/j.jcp.2019.109065) is implemented. A root solve is performed to map from non-canonical to canonical variables, with tolerance given by `roottol`. If `predictor_step=True`, the initial guess for the next step is improved using first derivative information. 