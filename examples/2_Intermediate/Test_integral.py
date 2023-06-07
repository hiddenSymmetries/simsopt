#!/usr/bin/env python
r"""
In this example we solve a FOCUS like Stage II coil optimisation problem: the
goal is to find coils that generate a specific target normal field on a given
surface.  In this particular case we consider a vacuum field, so the target is
just zero.

The objective is given by

    J = (1/2) \int |B dot n|^2 ds
        + LENGTH_WEIGHT * (sum CurveLength)
        + DISTANCE_WEIGHT * MininumDistancePenalty(DISTANCE_THRESHOLD)
        + CURVATURE_WEIGHT * CurvaturePenalty(CURVATURE_THRESHOLD)
        + MSC_WEIGHT * MeanSquaredCurvaturePenalty(MSC_THRESHOLD)

if any of the weights are increased, or the thresholds are tightened, the coils
are more regular and better separated, but the target normal field may not be
achieved as well. This example demonstrates the adjustment of weights and
penalties via the use of the `Weight` class.

The target equilibrium is the QA configuration of arXiv:2108.03711.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from simsopt.objectives import Weight
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt.geo import curves_to_vtk, create_equally_spaced_curves, plot
from simsopt.field import BiotSavart, Coil
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import CurveLength, CurveCurveDistance, \
    MeanSquaredCurvature, LpCurveCurvature, CurveSurfaceDistance, \
    VacuumEnergy

# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2, multiply by 4 to get the total number of coils.)
ncoils = 4

# Major radius for the initial circular coils:
R0 = 1.0

# Minor radius for the initial circular coils:
R1 = 0.5

# Number of Fourier modes describing each Cartesian component of each coil:
order = 5
epsilon = 1e-2

# Weight on the curve lengths in the objective function. We use the `Weight`
# class here to later easily adjust the scalar value and rerun the optimization
# without having to rebuild the objective.
FLUX_WEIGHT = Weight(1e-6)

# Number of iterations to perform:
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']
MAXITER = 50 if ci else 400

# File for the desired boundary magnetic surface:
TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
filename = TEST_DIR / 'input.LandremanPaul2021_QA'

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

#######################################################
# End of input parameters.
#######################################################

# Initialize the boundary magnetic surface:
nphi = 32
ntheta = 32
s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)

# Create the initial coils:
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order)
base_currents = [Current(1e5) for i in range(ncoils)]
# Since the target field is zero, one possible solution is just to set all
# currents to 0. To avoid the minimizer finding that solution, we fix one
# of the currents:
base_currents[0].fix_all()

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape((-1, 3)))

print(coils[0].dofs_free_status)
print(coils[1].dofs_free_status)



#print("Coils degrees of Freedom")
#print(np.shape(coils[0].curve.get_dofs()))

curves = [c.curve for c in coils]
curves_to_vtk(curves, OUT_DIR + "curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)

# Define the individual terms objective function:
Jf = SquaredFlux(s, bs)
JE = VacuumEnergy(coils, bs, epsilon, ncoils)

# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:
""" JF = FLUX_WEIGHT * Jf \
    +  JE \
   """

JF = JE

print(JF.x)
# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize



def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}"
    outstr += f", JE={JE:.2e}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    return J, grad
#_______________________________________________________#
# PLOT THE COILS 
#_______________________________________________________#
#plt.figure
#plot(coils + [s], engine="matplotlib", close=True)

#________________________________________________________
# EXTRACT COILS PARAMETERS 
#________________________________________________________
curves = [c.curve for c in coils]
currents_coils  =  [c.current for c in coils]
current_density =  np.zeros(np.size(currents_coils))

#________________________________________________________
# COMPUTE CURVE LENGTH int 1*dl
#________________________________________________________
gamma_dash = curves[0].gammadash()
print(np.shape(gamma_dash))
print(curves[0].incremental_arclength())
print(gamma_dash)
print(np.linalg.norm(gamma_dash, axis=1))
longueur= np.mean(np.linalg.norm(gamma_dash, axis=1))
longueur2 = np.mean(curves[0].incremental_arclength())
longueur3 = CurveLength(curves[0]).J()
print(longueur-longueur2)
print(longueur3-longueur)



# current density in each coil 
for ii in range(np.size(curves)):
    longueur = CurveLength(curves[ii])
    current_density[ii] = currents_coils[ii].get_value()/longueur.J()
    outstr = f"L={longueur.J():.3e}"
    outstr += f", J={current_density[ii]:.1e}"
    print(outstr)

# _______________________________________________________#
# BELOW: FOURIER COEFFICIENTS 
# _______________________________________________________#

bst = BiotSavart(coils)
epsilon = 1e-2 # normal displacement of coil curve to remove singularities

ncurves = np.size(curves)
ratios  = np.zeros((3,75))
dEdOmega_ = np.zeros((ncurves,curves[0].num_dofs()))

for k in range(ncurves):
    frenet = curves[k].frenet_frame() # frenet frame
    current_density_vec = current_density[k]*frenet[0][:,:]     # tangent 
    inward_shifted = curves[k].gamma() - epsilon*frenet[1][:,:] # normal 
    bst.set_points(inward_shifted.reshape(-1,3)) # set points for B
    B = bst.B() # Evaluate B on inward displaced curve
    shape_gradient = np.cross(current_density_vec,B)
    dgamma_dcoeff = curves[k].dgamma_by_dcoeff() # dr/dOmega
    drdomega_pos = np.split(dgamma_dcoeff, 75)   # Collect pos. along curve
    dl = curves[k].incremental_arclength()
    # differentiate the energy wrt Fourier modes (dofs)
    for i in range(33):
        for j in range(75): # compute dot product along curve
            ratios[0][j] = drdomega_pos[j][0,0][i]
            ratios[1][j] = drdomega_pos[j][0,1][i]
            ratios[2][j] = drdomega_pos[j][0,2][i]    
            #print(np.shape(ratios))
            integrand = np.diag(np.dot(shape_gradient,ratios))
            #print(integrand)
            #print(np.shape(integrand))
        integrand_ = np.multiply(integrand,dl)
        dEdOmega_[k][i] = np.mean(integrand)
        #print(dEdomega1)

print(np.reshape(dEdOmega_,-1))
print(np.shape(np.reshape(dEdOmega_,-1)))
# dEdOmega = np.sum(dEdOmega_,axis=0)
# print(dEdOmega)



#______________________________________________________________
# ENERGY COMPUTATION FROM A.dl
#______________________________________________________________

Ei = np.zeros(ncurves)
for k in range(ncurves):
    frenet = curves[k].frenet_frame() # frenet frame
    dl = curves[k].incremental_arclength()
    #print("dl")
    #print(np.shape(dl))
    current_density_vec = current_density[k]*frenet[0][:,:]     # tangent 
    #print("current density")
    #print(np.shape(current_density_vec))
    inward_shifted = curves[k].gamma() - epsilon*frenet[1][:,:] # normal 
    bst.set_points(inward_shifted.reshape(-1,3)) # set points for A
    A = bst.A()
    #print(np.shape(A))
    integrand = np.diag(np.dot(A,np.transpose(current_density_vec)))
    Ei[k] = np.mean(np.multiply(integrand,dl))
E = np.sum(Ei)
print(E)