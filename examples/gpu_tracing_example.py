#!/usr/bin/env python
import pandas as pd

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from math import sqrt
from booz_xform import Booz_xform

from simsopt.field import (BoozerRadialInterpolant, InterpolatedBoozerField, trace_particles_boozer,
                           MinToroidalFluxStoppingCriterion, MaxToroidalFluxStoppingCriterion,
                           ToroidalTransitStoppingCriterion, compute_resonances)

from simsopt.util.constants import (
        ALPHA_PARTICLE_MASS as MASS,
        FUSION_ALPHA_PARTICLE_ENERGY as ENERGY,
        ALPHA_PARTICLE_CHARGE as CHARGE
        )

filename = 'examples/boozmn_QH_boots.nc'

logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')

import simsoptpp as sopp

# Compute the pdf of birth rate in s
def s_density(s):
	return ((1-s**5)**2)*((1-s)**(-2/3))*np.exp(-19.94*(12*(1-s))**(-1/3))

# Rejection sample s
def sample_s():
	bound = 3e-4
	x = np.random.uniform()
	y = bound * np.random.uniform()

	while s_density(x) < y:
		assert s_density(x) <= bound
		x = np.random.uniform()
		y = bound * np.random.uniform()
	return x

# Sample theta, zeta for a given s via rejection sampling
def sample_tz(s, J_max, field):
	J = rand_J = 0
	while rand_J  >= J:
		theta = np.random.uniform(low=0, high=2*math.pi, size=1)
		zeta = np.random.uniform(low=0, high=2*math.pi, size=1)
		rand_J = np.random.uniform(low=0, high=J_max, size=1)

		loc = np.array([s, theta[0], zeta[0]]).reshape(1,3)
		field.set_points(loc)

		G = field.G()
		iota = field.iota()
		I = field.I()
		modB = field.modB()
		J = (G + iota*I)/(modB**2)
		J = J[0][0]
		assert J <= J_max
	return theta[0], zeta[0]

# Sample s,t,z 
def sample_stz(field, J_max):
	s = sample_s()
	theta, zeta = sample_tz(s, J_max, field)
	return np.array([s, theta, zeta])

# Compute VMEC equilibrium
t1 = time.time()
equil = Booz_xform()
equil.verbose = 0
equil.read_boozmn(filename)
nfp = equil.nfp
N=-4

order = 3
bri = BoozerRadialInterpolant(equil, order, no_K=True, N=N)

nfp = equil.nfp
degree = 3
srange = (0, 1, 15)
thetarange = (0, np.pi, 15)
zetarange = (0, 2*np.pi/nfp, 15)
field = InterpolatedBoozerField(bri, degree, srange, thetarange, zetarange, True, nfp=nfp, stellsym=True)

# Evaluate error in interpolation
print('Error in |B| interpolation', field.estimate_error_modB(1000), flush=True)


# Initialize vpar
vpar = np.sqrt(2*ENERGY/MASS)

# set up GPU interpolation grid
def gen_bfield_info(field, srange, trange, zrange):

	s_grid = np.linspace(srange[0], srange[1], srange[2])
	theta_grid = np.linspace(trange[0], trange[1], trange[2])
	zeta_grid = np.linspace(zrange[0], zrange[1], zrange[2])

	quad_pts = np.empty((srange[2]*trange[2]*zrange[2], 3))
	for i in range(srange[2]):
		for j in range(trange[2]):
			for k in range(zrange[2]):
				quad_pts[trange[2]*zrange[2]*i + zrange[2]*j + k, :] = [s_grid[i], theta_grid[j], zeta_grid[k]]


	field.set_points(quad_pts)
	G = field.G()
	iota = field.iota()
	I = field.I()
	modB = field.modB()
	J = (G + iota*I)/(modB**2)
	maxJ = np.max(J) # for rejection sampling

	psi0 = field.psi0

	# Build interpolation data
	modB_derivs = field.modB_derivs()

	quad_info = np.hstack((modB, modB_derivs, G, iota))
	quad_info = np.ascontiguousarray(quad_info)

	return quad_info, maxJ, psi0


# generate grid with 15 simsopt grid pts
n_grid_pts = 15
srange = (0, 1, 3*n_grid_pts+1)
trange = (0, np.pi, 3*n_grid_pts+1)
zrange = (0, 2*np.pi/nfp, 3*n_grid_pts+1)
quad_info, maxJ, psi0 = gen_bfield_info(field, srange, trange, zrange)

# set seed for consistency
np.random.seed(8)

# trace particles
nparticles = 25000

stz_inits = np.vstack([sample_stz(field, maxJ) for i in range(nparticles)])
vpar_inits = vpar * np.random.uniform(low=-1, high=1, size=nparticles)

print("tracing particles")

# trace on GPU
last_time = sopp.gpu_tracing(
	quad_pts=quad_info, 
	srange=srange,
	trange=trange,
	zrange=zrange, 
	stz_init=stz_inits,
	m=MASS, 
	q=CHARGE, 
	vtotal=sqrt(2*ENERGY/MASS),  
	vtang=vpar_inits, 
	tmax=1e-2, 
	tol=1e-9, 
	psi0=psi0, 
	nparticles=nparticles)

last_time = np.reshape(last_time, (nparticles, 7))


particle_data = pd.DataFrame({'s_start': stz_inits[:,0], 't_start': stz_inits[:,1], 'z_start':stz_inits[:,2], 'vpar_start':vpar_inits,
							  's_end': last_time[:,0], 't_end':last_time[:,1], 'z_end':last_time[:,2], 'vpar_end':last_time[:,3], 'last_time':last_time[:,4],
							  'steps_accepted':last_time[:,5], 'steps_attempted':last_time[:,6]})
particle_data.to_csv('particle_data.csv')


did_leave = [t < 1e-2 for t in particle_data['last_time']]
loss_frac = sum(did_leave) / len(did_leave)
print(f"Number of particles= {nparticles}")
print(f"Loss fraction: {loss_frac:.3f}")