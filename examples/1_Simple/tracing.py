#!/usr/bin/env python3
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.magneticfieldclasses import InterpolatedField, UniformInterpolationRule
from simsopt.geo.surfacexyztensorfourier import SurfaceRZFourier
from simsopt.geo.coilcollection import CoilCollection
from simsopt.field.tracing import trace_particles_starting_on_axis, SurfaceClassifier, \
    particles_to_vtk, compute_fieldlines, LevelsetStoppingCriterion
from simsopt.geo.curve import curves_to_vtk
from simsopt.util.zoo import get_ncsx_data
from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import logging
import sys
sys.path.append(os.path.join("..", "tests", "geo"))
logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')
logger.setLevel(1)

# check whether we're in CI, in that case we make the run a bit cheaper
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']
nfieldlines = 3 if ci else 80
tmax_fl = 10000 if ci else 40000
nparticles = 3 if ci else 40

comm = MPI.COMM_WORLD


"""
This examples demonstrate how to use SIMSOPT to compute Poincare plots and
guiding center trajectories of particles
"""


coils, currents, ma = get_ncsx_data(Nt_coils=25, Nt_ma=10)

# scale up the device to larger major radius and stronger B field
scale = 8.
for c in coils:
    c.set_dofs(scale*c.get_dofs())
ma.set_dofs(scale*ma.get_dofs())
currents = [32 * c for c in currents]

stellarator = CoilCollection(coils, currents, 3, True)
coils = stellarator.coils
currents = stellarator.currents
bs = BiotSavart(coils, currents)
print("Mean(|B|) on axis =", np.mean(np.linalg.norm(bs.set_points(ma.gamma()).B(), axis=1)))
print("Mean(Axis radius) =", np.mean(np.linalg.norm(ma.gamma(), axis=1)))
curves_to_vtk(coils + [ma], '/tmp/coils')

mpol = 5
ntor = 5
stellsym = True
nfp = 3
phis = np.linspace(0, 1, nfp*2*ntor+1, endpoint=False)
thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
s = SurfaceRZFourier(
    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
s.fit_to_curve(ma, scale*0.70, flip_theta=False)
s.to_vtk('/tmp/surface')
sc_fieldline = SurfaceClassifier(s, h=scale*0.1, p=2)
sc_fieldline.to_vtk('/tmp/levelset', h=scale*0.02)
s.fit_to_curve(ma, scale*0.20, flip_theta=False)
sc_particle = SurfaceClassifier(s, h=scale*0.1, p=2)


def trace_fieldlines(bfield, label):
    t1 = time.time()
    phis = [i*2*np.pi/(4*3) for i in range(4)]
    R0 = [ma.gamma()[0, 0] + i*0.015 for i in range(nfieldlines)]
    Z0 = [ma.gamma()[0, 2] for i in range(nfieldlines)]
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=1e-7, comm=comm,
        phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
    t2 = time.time()
    print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
    if comm.rank == 0:
        particles_to_vtk(fieldlines_tys, f'/tmp/fieldlines_{label}')
        for i in range(len(phis)):
            plt.figure()
            for j in range(len(fieldlines_phi_hits)):
                data_this_phi = fieldlines_phi_hits[j][np.where(fieldlines_phi_hits[j][:, 1] == i)[0], :]
                if data_this_phi.size == 0:
                    continue
                r = np.sqrt(data_this_phi[:, 2]**2+data_this_phi[:, 3]**2)
                plt.scatter(r, data_this_phi[:, 4], marker='o', s=0.2, linewidths=0)
            plt.savefig(f'/tmp/phi_{i}_{label}.png', dpi=600)
            plt.close()


def trace_particles(bfield, label, mode='gc_vac'):
    t1 = time.time()
    gc_tys, gc_phi_hits = trace_particles_starting_on_axis(
        ma.gamma(), bfield, nparticles, tmax=1e-4, seed=1, mass=4*1.67e-27, charge=2*1,
        Ekin=3.5*1e6, umin=-0.1, umax=+0.1, comm=comm,
        phis=[2*np.pi/6 for i in range(6)],
        stopping_criteria=[LevelsetStoppingCriterion(sc_particle.dist)], mode=mode)
    t2 = time.time()
    print(f"Time for particle tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in gc_tys])//nparticles}", flush=True)
    if comm.rank == 0:
        particles_to_vtk(gc_tys, f'/tmp/particles_{label}_{mode}')


# uncomment this to run tracing using the biot savart field (very slow!)
# trace_fieldlines(bs, 'bs')
# trace_particles(bs, 'bs', 'gc')

n = 16
rrange = (scale*1.0, scale*2.0, n)
phirange = (0, 2*np.pi, n*6)
zrange = (scale*-0.7, scale*0.7, n)
bsh = InterpolatedField(
    bs, UniformInterpolationRule(4),
    rrange, phirange, zrange, True
)
print('Error in B', bsh.estimate_error_B(1000), flush=True)
trace_fieldlines(bsh, 'bsh')
print('Error in AbsB', bsh.estimate_error_GradAbsB(1000), flush=True)
trace_particles(bsh, 'bsh', 'gc_vac')
trace_particles(bsh, 'bsh', 'full')
