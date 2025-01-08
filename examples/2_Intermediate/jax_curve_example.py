import jax
import jax.numpy as jnp
from simsopt._core import Optimizable
from simsopt.geo import CurveRZFourier, JaxCurveXYZFourier, CurveXYZFourierSymmetries
import numpy as np
import matplotlib.pyplot as plt
"""
Tutorial using a JaxCurve in Simsopt.
"""


"""
First, lets define an instance of a JaxCurve.
"""
order = 1
nfp = 2
etabar = 0.7
nphi = 257
phi = np.linspace(0, 1 / nfp, nphi, endpoint=False)
stellsym = True
axis = CurveXYZFourierSymmetries(phi, order=order, nfp=nfp, stellsym=stellsym)

# now lets set the dofs.
axis.unfix_all()
axis.set('xc(0)',4)
# axis.set('xc(1)',0.2)
axis.set('ys(1)',0.1)
axis.set('zs(1)',0.2)
x = np.array(axis.x)


"""
The JaxCurve class inherits the simsopt Curve class, so that all Curve class 
attributes and methods can be used, such as the gamma() method.
"""
# the curve can be computed with the gamma function
axis.gamma()


"""
The attributes of the Curve class are not 'jax-friendly', in the sense that we
cannot differentiate them with respect to DOFs using autodiff. Instead, the methods
with the `_jax` suffix can be used for autodifferentation, such as gamma_jax().

gamma and gamma_jax can be used interchangeably (unless you want to use autodiff). The
main difference is that the DOF vector, x, must be passed to gamma_jax as an argument, 
whereas gamma uses the DOFS stored in the class.
"""
# x can be an np array or a jax numpy array
gamma = axis.gamma_jax(x) 

# plot the curve
ax = plt.figure().add_subplot(projection='3d')
ax.plot(gamma[:,0], gamma[:,1], gamma[:,2])
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.set_zlim(-5,5)
plt.show()

"""
Here are some more jax-friendly functions. These can be used for autodiff.
"""
# some more jax friendly functions
axis.incremental_arclength_jax(x)
axis.kappa_jax(x)
axis.kappadash_jax(x)
axis.torsion_jax(x)
axis.frenet_frame_jax(x)

# test the accuracy
print(f"{'method' : >27}", "|", "error")
print("","-"*51)
error = np.max(np.abs(axis.incremental_arclength_jax(x) - axis.incremental_arclength()))
print(f"{'incremental_arclength_jax' : >27}", "|", error)
error = np.max(np.abs(axis.kappa_jax(x) - axis.kappa()))
print(f"{'kappa_jax' : >27}", "|", error)
error = np.max(np.abs(axis.kappadash_jax(x) - axis.kappadash()))
print(f"{'kappadash_jax' : >27}", "|", error)
error = np.max(np.abs(axis.torsion_jax(x) - axis.torsion()))
print(f"{'torsion_jax' : >27}", "|", error)

frenet_frame_jax = axis.frenet_frame_jax(x)
frenet_frame = axis.frenet_frame()
for ii in range(3):
    error = np.max(np.abs(frenet_frame_jax[ii] - frenet_frame[ii]))
    print(f"{'frenet_frame_jax' : >27}", "|", error)


"""
The JaxCurve class has some derivatives built in that can be computed with jax, or via the hardcoded
functions. For most methods -- gamma, kappa, torsion, etc -- hard coded derivatives have been included in the class.
These derivative methods have not necessarilly been duplicated using jax. For example, 
we have dfrenet_frame_by_dcoeff but not dfrenet_frame_by_dcoeff_jax built in to the clas.
"""
# hard-coded derivative
axis.dgamma_by_dcoeff()

# or the built in jax derivative.
axis.dgamma_by_dcoeff_jax(x)

"""
The beauty of jax is that we can compute any derivative we want on the fly using the grad
and jacfwd operators.
"""
# incremental arclength 
dl_by_dcoeff_jax = jax.jacfwd(axis.incremental_arclength_jax)(x) # jacobian with jax
dl_by_dcoeff = axis.dincremental_arclength_by_dcoeff()
error = np.max(np.abs(dl_by_dcoeff_jax - dl_by_dcoeff))
print(f"{'dl_by_dcoeff_jax' : >27}", "|", error)

# curvature derivatives
dkappa_by_dcoeff_jax = jax.jacfwd(axis.kappa_jax)(x) # jacobian with jax
dkappa_by_dcoeff = axis.dkappa_by_dcoeff()
error = np.max(np.abs(dkappa_by_dcoeff_jax - dkappa_by_dcoeff))
print(f"{'dkappa_by_dcoeff_jax' : >27}", "|", error)

# kappadash derivatives
dkappadash_by_dcoeff_jax = jax.jacfwd(axis.kappadash_jax)(x) # jacobian with jax
dkappadash_by_dcoeff = axis.dkappadash_by_dcoeff()
error = np.max(np.abs(dkappadash_by_dcoeff_jax - dkappadash_by_dcoeff))
print(f"{'dkappadash_by_dcoeff_jax' : >27}", "|", error)

# torsion derivatives
dtorsion_by_dcoeff_jax = jax.jacfwd(axis.torsion_jax)(x) # jacobian with jax
dtorsion_by_dcoeff = axis.dtorsion_by_dcoeff()
error = np.max(np.abs(dtorsion_by_dcoeff_jax - dtorsion_by_dcoeff))
print(f"{'dtorsion_by_dcoeff_jax' : >27}", "|", error)

# frenet-frame derivatives
dfrenet_frame_by_dcoeff_jax = jax.jacfwd(axis.frenet_frame_jax)(x) # jacobian with jax
dfrenet_frame_by_dcoeff = axis.dfrenet_frame_by_dcoeff()
for ii in range(3):
    error = np.max(np.abs(dfrenet_frame_by_dcoeff_jax[ii] - dfrenet_frame_by_dcoeff[ii]))
    print(f"{'dfrenet_frame_by_dcoeff_jax' : >27}", "|", error)

"""
Jax can also be used to compute the derivatives of objective functions. However, the jax methods (i.e. gamma_jax)
must be used in place of their static counterparts (i.e. gamma). jax numpy operations should also be used instead
of numpy operations.
"""
# define the objective function using the jax methods of the curve
def func(x):
    (t,n,b) = axis.frenet_frame_jax(x)
    kappa = axis.kappa_jax(x)
    return jnp.sum( kappa * n[:, 0])

# differentiate with jax
func_grad_jax = jax.grad(func)

# compute the exact derivative for comparison
def func_grad_exact(x):
    kappa = axis.kappa() # (nphi,)
    (t,n,b) = axis.frenet_frame() # each is (nphi, 3)
    dkappa_by_dcoeff = axis.dkappa_by_dcoeff() # (nphi, ncoeff)
    (dt_by_dcoeff, dn_by_dcoeff, db_by_dcoeff) = axis.dfrenet_frame_by_dcoeff()  # each is (nphi, 3, ncoeff)
    g = np.sum(dkappa_by_dcoeff * n[:, 0].reshape((-1,1)) + kappa.reshape((-1,1)) * dn_by_dcoeff[:,0, :], axis=0)
    return g

# check the error
error = np.max(np.abs(func_grad_jax(x) - func_grad_exact(x)))
print(f"{'gradient' : >27}", "|", error)


"""
The jax functions, such as gamma_jax, always require the entire set
of curve coefficients as inputs. The JaxCurve does not play nicely
with the Optimizable framework of fixing dofs.

In other words, the jax functions should always take the full_x as
input,
    good: gamma_jax(axis.full_x)
    bad: gamma_jax(axis.x)

Derivatives with jax will also be taken with respect to full_x, 
rather than the unfixed degrees of freedom.
"""

# first lets change x a little and fix a DOF
axis.x = axis.x + 0.001
axis.fix('zs(1)')
x = jnp.array(axis.x)

# we will see a large error when using axis.x
error = np.max(np.abs(axis.gamma_jax(x) - axis.gamma()))
print(f"{'gamma_jax using x' : >27}", "|", error)

# but the error is gone if we use axis.full_x
x = jnp.array(axis.full_x)
error = np.max(np.abs(axis.gamma_jax(x) - axis.gamma()))
print(f"{'gamma_jax using full_x' : >27}", "|", error)

# the derivative will be w.r.t to the full_x
dkappa_by_dcoeff_jax = jax.jacfwd(axis.kappa_jax)(x) # jacobian with jax
print('Derivatives are w.r.t all', np.shape(dkappa_by_dcoeff_jax)[1], f'coeffs, not {len(axis.x)} the unfixed coeffs.')
