import os
import numpy as np
from scipy.optimize import minimize
from simsopt.geo import curves_to_vtk
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.objectives import QuadraticPenalty
from simsopt._core.derivative import Derivative
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (
    CurveLength, CurveCurveDistance,
    MeanSquaredCurvature, LpCurveCurvature, CurveCWSFourier
)
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

# Threshold and weight for the maximum length of each individual coil:
LENGTH_THRESHOLD = 6.5
LENGTH_WEIGHT = 0.1

# Threshold and weight for the coil-to-coil distance penalty in the objective function:
CC_THRESHOLD = 0.1
CC_WEIGHT = 1000

# Threshold and weight for the curvature penalty in the objective function:
CURVATURE_THRESHOLD = 6
CURVATURE_WEIGHT = 0.01

# Threshold and weight for the mean squared curvature penalty in the objective function:
MSC_THRESHOLD = 6
MSC_WEIGHT = 0.01

# SURFACE INPUT FILES FOR TESTING
wout = "/Users/rogeriojorge/local/vmec_equilibria/NCSX/li383_1.4m/wout_li383_1.4m.nc"

MAXITER = 10 
minor_radius_factor_cws = 1.9
ncoils = 3
order = 6  # order of dofs of cws curves
numquadpoints = 13 * order
ntheta = 50
nphi = 50

# CREATE SURFACES
s = SurfaceRZFourier.from_wout(wout, range="half period", ntheta=ntheta, nphi=nphi)
s_full = SurfaceRZFourier.from_wout(wout, range="full torus", ntheta=ntheta, nphi=int(nphi*2*s.nfp))
cws = SurfaceRZFourier.from_nphi_ntheta(nphi, ntheta, "half period", s.nfp)
cws_full = SurfaceRZFourier.from_nphi_ntheta(int(nphi*2*s.nfp), ntheta, "full torus", s.nfp)

R = s.get_rc(0, 0)
cws.set_dofs([R, s.get_zs(1, 0)*minor_radius_factor_cws, s.get_zs(1, 0)*minor_radius_factor_cws])
cws_full.set_dofs([R, s.get_zs(1, 0)*minor_radius_factor_cws, s.get_zs(1, 0)*minor_radius_factor_cws])


def create_cws_from_dofs(dofs_cws=cws.x, dofs_coils=None):
    base_curves = []
    for i in range(ncoils):
        curve_cws = CurveCWSFourier(
            mpol=cws.mpol,
            ntor=cws.ntor,
            idofs=dofs_cws,
            numquadpoints=numquadpoints,
            order=order,
            nfp=cws.nfp,
            stellsym=cws.stellsym,
        )
        angle = (i+0.5)*(2*np.pi)/((2)*s.nfp*ncoils)
        curve_dofs = np.zeros(len(curve_cws.get_dofs()),)
        curve_dofs[0] = 1
        curve_dofs[2*order+2] = 0
        curve_dofs[2*order+3] = angle
        curve_cws.set_dofs(curve_dofs)
        curve_cws.fix(0)
        curve_cws.fix(2*order+2)
        base_curves.append(curve_cws)
    base_currents = [Current(1)*1e5 for i in range(ncoils)]
    base_currents[0].fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
    curves = [c.curve for c in coils]
    bs = BiotSavart(coils)
    bs.set_points(s.gamma().reshape((-1, 3)))
    Jf = SquaredFlux(s, bs)
    Jls = [CurveLength(c) for c in base_curves]
    Jccdist = CurveCurveDistance(curves, CC_THRESHOLD)
    Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
    Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
    JF = (
        Jf
        + LENGTH_WEIGHT * sum(QuadraticPenalty(J, LENGTH_THRESHOLD) for J in Jls)
        + CC_WEIGHT * Jccdist
        + CURVATURE_WEIGHT * sum(Jcs)
        + MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD) for J in Jmscs)
    )
    if dofs_coils is not None:
        JF.x = dofs_coils
    return JF, bs, base_curves, curves, coils, Jf, Jls, Jccdist, Jcs, Jmscs


JF, bs, base_curves, curves, coils, Jf, Jls, Jccdist, Jcs, Jmscs = create_cws_from_dofs()

bs.set_points(s_full.gamma().reshape((-1, 3)))
curves_to_vtk(curves, OUT_DIR + "curves_init")
curves_to_vtk(base_curves, OUT_DIR + "base_curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((int(nphi*2*s_full.nfp), ntheta, 3)) * s_full.unitnormal(), axis=2)[:, :, None]}
s_full.to_vtk(OUT_DIR + "surf_init", extra_data=pointData)
cws_full.to_vtk(OUT_DIR + "cws_init")

bs.set_points(s.gamma().reshape((-1, 3)))

dofs_coils_length = len(JF.x)


def fun(dofs):
    JF, bs, base_curves, curves, coils, Jf, Jls, Jccdist, Jcs, Jmscs = create_cws_from_dofs(dofs_cws=dofs[dofs_coils_length:], dofs_coils=dofs[:dofs_coils_length])
    J = JF.J()
    coils_dJ = JF.dJ()
    ## Mixed term - derivative of squared flux with respect to the surface shape
    n = cws.normal()
    absn = np.linalg.norm(n, axis=2)
    B = bs.B().reshape((nphi, ntheta, 3))
    dB_by_dX = bs.dB_by_dX().reshape((nphi, ntheta, 3, 3))
    Bcoil = bs.B().reshape(n.shape)
    unitn = n * (1./absn)[:, :, None]
    Bcoil_n = np.sum(Bcoil*unitn, axis=2)
    mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
    B_n = Bcoil_n
    B_diff = Bcoil
    B_N = np.sum(Bcoil * n, axis=2)
    assert Jf.local
    dJdx = (B_n/mod_Bcoil**2)[:, :, None] * (np.sum(dB_by_dX*(n-B*(B_N/mod_Bcoil**2)[:, :, None])[:, :, None, :], axis=3))
    dJdN = (B_n/mod_Bcoil**2)[:, :, None] * B_diff - 0.5 * (B_N**2/absn**3/mod_Bcoil**2)[:, :, None] * n
    deriv = cws.dnormal_by_dcoeff_vjp(dJdN/(nphi*ntheta)) + cws.dgamma_by_dcoeff_vjp(dJdx/(nphi*ntheta))
    mixed_dJ = Derivative({cws: deriv})(cws)
    ## Put both gradients together
    grad_with_respect_to_coils = coils_dJ
    grad_with_respect_to_surface = mixed_dJ
    grad = np.concatenate((grad_with_respect_to_coils, grad_with_respect_to_surface))

    jf = Jf.J()
    BdotN = np.mean(np.abs(np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    outstr = f"J={J:.3e}, Jf={jf:.3e}, ⟨B·n⟩={BdotN:.1e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f}, ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}"
    print(outstr)
    return J, grad


dofs_coils = np.copy(JF.x)
dofs_cws = np.copy(cws.x)
dofs = np.append(dofs_coils, dofs_cws)

res = minimize(
    fun,
    dofs,
    jac=True,
    method="L-BFGS-B",
    options={"maxiter": MAXITER, "maxcor": 300},
    tol=1e-15,
)

bs.set_points(s_full.gamma().reshape((-1, 3)))
curves_to_vtk(curves, OUT_DIR + "curves_opt")
curves_to_vtk(base_curves, OUT_DIR + "base_curves_opt")
pointData = {"B_N": np.sum(bs.B().reshape((int(nphi*2*s_full.nfp), ntheta, 3)) * s_full.unitnormal(), axis=2)[:, :, None]}
s_full.to_vtk(OUT_DIR + "surf_opt", extra_data=pointData)
cws_full.to_vtk(OUT_DIR + "cws_opt")
bs.set_points(s.gamma().reshape((-1, 3)))