#!/usr/bin/env python3
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.magneticfieldclasses import InterpolatedField, UniformInterpolationRule
from simsopt.geo.surfacexyztensorfourier import SurfaceRZFourier
from simsopt.geo.coilcollection import CoilCollection
from simsopt.field.tracing import trace_particles_starting_on_curve, SurfaceClassifier, \
    particles_to_vtk, LevelsetStoppingCriterion, plot_poincare_data
from simsopt.geo.curve import curves_to_vtk
from simsopt.util.zoo import get_ncsx_data
from simsopt.util.constants import PROTON_MASS, ELEMENTARY_CHARGE, ONE_EV
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.surfaceobjectives import boozer_surface_residual, ToroidalFlux, Area
import simsoptpp as sopp
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None

import numpy as np
import time
import os
import logging
import sys
if not sopp.with_boost():
    print("Please compile with boost to run this example.")
    sys.exit(0)
sys.path.append(os.path.join("..", "tests", "geo"))
logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')
logger.setLevel(1)

# check whether we're in CI, in that case we make the run a bit cheaper
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']
nparticles = 3 if ci else 20000
degree = 2 if ci else 4


"""
This examples demonstrate how to use SIMSOPT to compute guiding center
trajectories of particles
"""


coils, currents, ma = get_ncsx_data(Nt_coils=25, Nt_ma=20)

stellarator = CoilCollection(coils, currents, 3, True)
coils = stellarator.coils
currents = stellarator.currents
bs = BiotSavart(coils, currents)
print("Mean(|B|) on axis =", np.mean(np.linalg.norm(bs.set_points(ma.gamma()).B(), axis=1)), flush=True)
print("Mean(Axis radius) =", np.mean(np.linalg.norm(ma.gamma(), axis=1)), flush=True)

mpol = 7
ntor = 7
stellsym = True
nfp = 3

phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
s = SurfaceXYZTensorFourier(
    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
s.fit_to_curve(ma, 0.10, flip_theta=True)
iota = -0.4

ar = Area(s)
ar_target = ar.J()

boozer_surface = BoozerSurface(bs, s, ar, ar_target)
G0 = 2. * np.pi * np.sum(np.abs(bs.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))

res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(tol=1e-10, maxiter=300, constraint_weight=100., iota=iota, G=G0)
print(f"After LBFGS:   iota={res['iota']:.3f}, area={s.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}", flush=True)
res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-10, maxiter=100, constraint_weight=100., iota=res['iota'], G=res['G'], method='manual')
print(f"After Lev-Mar: iota={res['iota']:.3f}, area={s.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}", flush=True)

boozer_surface = BoozerSurface(bs, s, ar, 4*ar_target)
res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-10, maxiter=100, constraint_weight=100., iota=res['iota'], G=res['G'], method='manual')
print(f"After Lev-Mar: iota={res['iota']:.3f}, area={s.area():.3f}, ||residual||={np.linalg.norm(boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)):.3e}", flush=True)


phis = np.linspace(0, 1, 4*nfp*2*ntor+1, endpoint=False)
thetas = np.linspace(0, 1, 4*2*mpol+1, endpoint=False)
sfull = SurfaceXYZTensorFourier(
    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
sfull.set_dofs(s.get_dofs())
s = sfull
sc_particle = SurfaceClassifier(s, h=0.1, p=2)

if comm is None or comm.rank == 0:
    curves_to_vtk(coils + [ma], '/tmp/coils')
    print(s.gamma()[0, :, :], flush=True)
    s.to_vtk('/tmp/boozer')
    sc_particle.to_vtk('/tmp/levelset', h=0.01)

# s.fit_to_curve(ma, 0.20, flip_theta=False)

n = 25
rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
zs = s.gamma()[:, :, 2]

rrange = (np.min(rs), np.max(rs), n)
phirange = (0, 2*np.pi/3, n*2)
zrange = (0, np.max(zs), n//2)
bsh = InterpolatedField(
    bs, degree, rrange, phirange, zrange, True, nfp=3, stellsym=True
)

bsh.set_points(s.gamma().reshape((-1, 3)))
dBh = bsh.GradAbsB()
Bh = bsh.B()
bs.set_points(s.gamma().reshape((-1, 3)))
dB = bs.GradAbsB()
B = bs.B()
print(np.sort(np.abs(B-Bh).flatten()), flush=True)
print(np.sort(np.abs(dB-dBh).flatten()), flush=True)



def trace_particles(bfield, label, mode='gc_vac'):
    nphi = 3
    phis = [(i/nphi)*(2*np.pi) for i in range(nphi)]
    t1 = time.time()
    # _, gc_phi_hits_fwd = trace_particles_starting_on_curve(
    #     ma, bfield, nparticles, tmax=1e-2, seed=1, mass=PROTON_MASS, charge=ELEMENTARY_CHARGE,
    #     Ekin=5000*ONE_EV, umin=+0.05, umax=+0.05+1e-10, comm=comm,
    #     phis=phis, tol=1e-11,
    #     stopping_criteria=[LevelsetStoppingCriterion(sc_particle.dist)], mode=mode,
    #     forget_exact_path=True)
    # _, gc_phi_hits_bwd = trace_particles_starting_on_curve(
    #     ma, bfield, nparticles, tmax=1e-2, seed=1, mass=PROTON_MASS, charge=ELEMENTARY_CHARGE,
    #     Ekin=5000*ONE_EV, umin=-0.05, umax=-0.05+1e-10, comm=comm,
    #     phis=phis, tol=1e-11,
    #     stopping_criteria=[LevelsetStoppingCriterion(sc_particle.dist)], mode=mode,
    #     forget_exact_path=True)
    # np.save(f'phi_24_n_1000_many_u_{label}', gc_phi_hits_fwd + gc_phi_hits_bwd, allow_pickle=True)
    _, gc_phi_hits = trace_particles_starting_on_curve(
        ma, bfield, nparticles, tmax=1e-2, seed=1, mass=PROTON_MASS, charge=ELEMENTARY_CHARGE,
        Ekin=5000*ONE_EV, umin=-0.1, umax=+0.1, comm=comm,
        phis=phis, tol=1e-11,
        stopping_criteria=[LevelsetStoppingCriterion(sc_particle.dist)], mode=mode,
        forget_exact_path=True)

    _, gc_phi_hits_oos = trace_particles_starting_on_curve(
        ma, bfield, nparticles//10, tmax=1e-2, seed=2, mass=PROTON_MASS, charge=ELEMENTARY_CHARGE,
        Ekin=5000*ONE_EV, umin=-0.1, umax=+0.1, comm=comm,
        phis=phis, tol=1e-11,
        stopping_criteria=[LevelsetStoppingCriterion(sc_particle.dist)], mode=mode,
        forget_exact_path=True)
    t2 = time.time()
    np.save(f'phi_{nphi}_n_{nparticles}_many_u_{label}', gc_phi_hits, allow_pickle=True)
    np.save(f'phi_{nphi}_n_{nparticles}_many_u_{label}_oos', gc_phi_hits_oos, allow_pickle=True)

    print(f"Time for particle tracing={t2-t1:.3f}s.", flush=True)
    # print(f"Time for particle tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in gc_tys])//nparticles}", flush=True)
    return
    import sys; sys.exit()
    if comm is None or comm.rank == 0:
        # particles_to_vtk(gc_tys, f'/tmp/particles_{label}_{mode}')
        plot_poincare_data(gc_phi_hits, phis, f'/tmp/poincare_particle_{label}_loss.png', mark_lost=True)
        plot_poincare_data(gc_phi_hits, phis, f'/tmp/poincare_particle_{label}.png', mark_lost=False)


print('Error in B', bsh.estimate_error_B(1000), flush=True)
print('Error in AbsB', bsh.estimate_error_GradAbsB(1000), flush=True)
trace_particles(bsh, 'bsh', 'gc_vac')
# trace_particles(bs, 'bs', 'gc_vac')
# trace_particles(bsh, 'bsh', 'full')
