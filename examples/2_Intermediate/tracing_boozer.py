#!/usr/bin/env python

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from simsopt.field import BoozerRadialInterpolant, InterpolatedBoozerField
from simsopt.field import trace_particles_boozer, MinToroidalFluxStoppingCriterion, \
    MaxToroidalFluxStoppingCriterion, ToroidalTransitStoppingCriterion, \
    compute_resonances
from simsopt.util.constants import PROTON_MASS, ELEMENTARY_CHARGE, ONE_EV
from simsopt.mhd import Vmec

ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']

filename = os.path.join(os.path.dirname(__file__), 'inputs', 'input.LandremanPaul2021_QH')
logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')

"""
Here we trace particles in the vacuum magnetic field in Boozer coordinates
from the QH equilibrium in Landreman & Paul (arXiv:2108.03711). We evaluate
energy conservation and compute resonant particle trajectories.
"""

# Compute VMEC equilibrium

vmec = Vmec(filename)

# Construct radial interpolant of magnetic field

order = 3
bri = BoozerRadialInterpolant(vmec, order, enforce_vacuum=True)

# Construct 3D interpolation

nfp = vmec.wout.nfp
degree = 3
srange = (0, 1, 15)
thetarange = (0, np.pi, 15)
zetarange = (0, 2*np.pi/nfp, 15)
field = InterpolatedBoozerField(bri, degree, srange, thetarange, zetarange, True, nfp=nfp, stellsym=True)

# Evaluate error in interpolation

print('Error in |B| interpolation', field.estimate_error_modB(1000), flush=True)

# Compute range of rotational transform

iota_min = np.min(vmec.wout.iotas[1::])
iota_max = np.max(vmec.wout.iotas[1::])

# Check energy conservation

thetas = np.linspace(0, 2*np.pi, 2)
s = np.linspace(0.05, 0.95, 2)
[s, thetas] = np.meshgrid(s, thetas)

nParticles = len(s.flatten())

stz_inits = np.zeros((nParticles, 3))
stz_inits[:, 0] = s.flatten()
stz_inits[:, 1] = thetas.flatten()
stz_inits[:, 2] = np.zeros_like(s.flatten())

field.set_points(stz_inits)
modB_inits = field.modB()

# Initialize vpar assuming mu = 0
Ekin = 100000*ONE_EV
mass = PROTON_MASS
vpar = np.sqrt(2*0.8*Ekin/mass)
vpar_inits = np.zeros((nParticles, 1))
vpar_inits[:, 0] = vpar
mu_inits = (Ekin - mass*0.5*vpar**2)/modB_inits  # m vperp^2 /(2B)

gc_tys, gc_zeta_hits = trace_particles_boozer(
    field, stz_inits, vpar_inits, tmax=1e-2, mass=mass, charge=ELEMENTARY_CHARGE,
    Ekin=Ekin, tol=1e-8, mode='gc_vac', stopping_criteria=[MaxToroidalFluxStoppingCriterion(0.99), MinToroidalFluxStoppingCriterion(0.01), ToroidalTransitStoppingCriterion(100, True)],
    forget_exact_path=False)

Nparticles = len(gc_tys)
for i in range(Nparticles):
    vpar = gc_tys[i][:, 4]
    points = np.zeros((len(gc_tys[i][:, 0]), 3))
    points[:, 0] = gc_tys[i][:, 1]
    points[:, 1] = gc_tys[i][:, 2]
    points[:, 2] = gc_tys[i][:, 3]
    field.set_points(points)
    modB = np.squeeze(field.modB())
    E = 0.5*mass*vpar**2 + mu_inits[i]*modB
    E = (E - Ekin)/Ekin
    print('Relative error in energy: ', np.max(np.abs(E)))

# Now initialize particles in theta and s and look for resonances

thetas = np.linspace(0, 2*np.pi, 50)
s = np.linspace(0.05, 0.95, 50)
[s, thetas] = np.meshgrid(s, thetas)

nParticles = len(s.flatten())

stz_inits = np.zeros((nParticles, 3))
stz_inits[:, 0] = s.flatten()
stz_inits[:, 1] = thetas.flatten()
stz_inits[:, 2] = np.zeros_like(s.flatten())

Ekin = 1e3*ONE_EV
mass = PROTON_MASS
vpar = np.sqrt(2*Ekin/mass)
vpar_inits = np.zeros((nParticles, 1))
vpar_inits[:, 0] = -vpar

gc_tys, gc_zeta_hits = trace_particles_boozer(
    field, stz_inits, vpar_inits, tmax=1e-2, mass=mass, charge=ELEMENTARY_CHARGE,
    Ekin=Ekin, zetas=[0], tol=1e-8, stopping_criteria=[MinToroidalFluxStoppingCriterion(0.01), MaxToroidalFluxStoppingCriterion(0.99), ToroidalTransitStoppingCriterion(30, True)],
    forget_exact_path=False)

resonances = compute_resonances(gc_tys, gc_zeta_hits, ma=None, delta=0.01)
resonances = np.array(resonances)
m = resonances[:, 5]
n = resonances[:, 6]
f = 1/resonances[:, 4]
m_min = min([0, np.min(m)])
m_max = max([0, np.max(m)])
ms = np.linspace(m_min, m_max, 10)

# Give resonances, evaluate resonant orbits

vpar_res = np.zeros((len(resonances), 1))
stz_res = np.zeros((len(resonances), 3))
for i in range(len(resonances)):
    vpar_res[i, 0] = resonances[i, 3]
    stz_res[i, :] = resonances[i, 0:3]

gc_tys, gc_zeta_hits = trace_particles_boozer(
    field, stz_res, vpar_res, tmax=1e-2, mass=mass, charge=ELEMENTARY_CHARGE,
    Ekin=Ekin, zetas=[0], tol=1e-8, stopping_criteria=[MinToroidalFluxStoppingCriterion(0.01), MaxToroidalFluxStoppingCriterion(0.99), ToroidalTransitStoppingCriterion(30, True)],
    forget_exact_path=False)

if not ci:
    plt.figure()
    plt.scatter(np.abs(m), np.abs(n))
    plt.plot(np.abs(ms), np.abs(ms/iota_min))
    plt.plot(np.abs(ms), np.abs(ms/iota_max))
    plt.ylim([0, 20])
    plt.xlabel('m (poloidal mode number)')
    plt.ylabel('n (toroidal mode number)')

    plt.figure()
    plt.scatter(np.abs(n), f/1e3)
    plt.xlabel('n (toroidal mode number)')
    plt.ylabel('frequency [kHz]')

    index = np.argmax(f)

    index_2pi = np.argmin(np.abs(gc_tys[index][:, 3]+2*np.pi))

    x = np.sqrt(gc_tys[index][0:index_2pi, 1])*np.cos(gc_tys[index][0:index_2pi, 2])
    y = np.sqrt(gc_tys[index][0:index_2pi, 1])*np.sin(gc_tys[index][0:index_2pi, 2])

    plt.figure()
    plt.plot(x, y, marker='*', linestyle='none')
    plt.xlabel(r'$x = \sqrt{s} \cos(\theta)$')
    plt.ylabel(r'$y = \sqrt{s} \sin(\theta)$')
    plt.title('First toroidal transit of highest frequency orbit')

    plt.show()
