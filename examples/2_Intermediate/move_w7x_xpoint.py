from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_serial_solve
from simsopt.objectives import SimpleFixedPoint_RZ, SimpleIntegrator
from simsopt.configs import get_w7x_data
from simsopt.field import coils_via_symmetries, BiotSavart
import numpy as np
import logging 
from simsopt.geo import SurfaceRZFourier
mysurf = SurfaceRZFourier()

logging.basicConfig(level=logging.DEBUG)

NFP_W7X = 5

curves, currents, ma = get_w7x_data()

# we do not want to optimize the geometry of the coils. 
for curve in curves: 
    curve.fix_all()

#currents[1].fix_all()  # fix one current as field can be arbitrary scaled.

coils = coils_via_symmetries(curves, currents, NFP_W7X, True)

field = BiotSavart(coils)

simpleintegrator = SimpleIntegrator(field)

#confirm that the magnetic axis is a fixed point
ma_startpoint_xyz = ma.gamma()[0,:] 
ma_startpoint_RZ, phi = ma_startpoint_xyz[::2], ma_startpoint_xyz[1] # There really should be a simsopt utility for cylindrical-> cartesian transformations.
fullmap = simpleintegrator.integrate_in_phi(ma_startpoint_RZ, phi, 2*np.pi) # integrating just a single field period would suffice, but we do a full torus.

dist = np.linalg.norm(ma_startpoint_RZ - fullmap)
print(f"The magnetic axis maps to a distance of: {dist}")
print(f"This is not unexpected, the magnetic axis was fourier-decomposed with limited resolution." )

axis_target = SimpleFixedPoint_RZ(field, ma_startpoint_RZ, NFP_W7X, 1, phi_0=0)

# The standard configuration has an o-point of the outer island chain at R=6.234, Z=0.0, we are going to move this outwards. 
target_pos = np.array([6.20, 0.0])
opoint_target = SimpleFixedPoint_RZ(field, target_pos, NFP_W7X, 5, phi_0=0, tol=1e-7, max_step=500)

################## plot unoptimized field
fast_integrator = SimpleIntegrator(field, tol=1e-7)
poincare_startpoints = np.linspace(ma_startpoint_RZ, target_pos + np.array([0.05, 0.1]), 10)

prob = LeastSquaresProblem.from_tuples([(opoint_target.J, 0, 1)])

least_squares_serial_solve(prob, max_nfev=200, grad=False)

# make a simple slow-evaluating Poincare plot (using the biot-savart directly)
fast_integrator = SimpleIntegrator(field, tol=1e-7)
poincare_startpoints = np.linspace(ma_startpoint_RZ, target_pos + np.array([0.05, 0.1]), 10)
poincare_traj_iterators = [fast_integrator._return_fieldline_iterator(startpoint, 0.0, 2*np.pi/NFP_W7X) for startpoint in poincare_startpoints]

from matplotlib import pyplot as plt
plt.ion()
fig, ax = plt.subplots(1,1)
POINCARE_NUM = 1000
for traj in poincare_traj_iterators:
    ax.scatter(np.array(traj[:POINCARE_NUM])[:,0], np.array(traj[:POINCARE_NUM])[:,1], c='r', linewidths=0.0, s=1)






startcurves, startcurrents, startma = get_w7x_data()
startcoils = coils_via_symmetries(startcurves, startcurrents, NFP_W7X, True)
startfield = BiotSavart(startcoils)
startintegrator = SimpleIntegrator(startfield, tol=1e-7)
start_poincare_traj_iterators = [startintegrator._return_fieldline_iterator(startpoint, 0.0, 2*np.pi/NFP_W7X) for startpoint in poincare_startpoints]
for traj in start_poincare_traj_iterators:
    ax.scatter(np.array(traj[:POINCARE_NUM])[:,0], np.array(traj[:POINCARE_NUM])[:,1], c='g', linewidths=0.0, s=1)



ax.set_aspect('equal')

ax.axvline(target_pos[0], c='r', linestyle='--')
ax.axvline(6.234, c='g', linestyle='--')