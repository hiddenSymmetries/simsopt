from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curve import Curve
from simsopt.geo.biotsavart import Current, Coil
from simsopt._core.graph_optimizable import Optimizable

curve = CurveXYZFourier(100, 1)
print(curve.gamma()[0:3, :])
current = Current(2.0)
coil = Coil(curve, current)
coil_x = coil.x
print(coil_x)
coil_x[1] = 1.
coil_x[-1] = 5
coil.x = coil_x
print(current.x)
print(coil.x)
print(curve.gamma()[0:3, :])


import sys
import os
sys.path.append(os.path.join("..", "..", "tests", "geo"))
from surface_test_helpers import get_ncsx_data
from simsopt.geo.coilcollection import coils_via_symmetries
from simsopt.geo.biotsavart import Current, BiotSavart
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
import numpy as np

curves, currents, ma = get_ncsx_data(Nt_coils=1)
currents = [Current(c) for c in currents]
coils = coils_via_symmetries(curves, currents, 3, True)

bs = BiotSavart(coils)
bs.x
num_expected_dofs = sum(len(c.get_dofs()) + 1 for c in curves)
print("num_expected_dofs", num_expected_dofs)
print("num actual dofs", len(bs.x))


class SquaredFlux(Optimizable):

    def __init__(self, surface, target, field):
        self.surface = surface
        self.target = target
        self.field = field
        Optimizable.__init__(self, x0=np.asarray([]), opts_in=[field])


    def J(self):
        xyz = self.surface.gamma()
        n = self.surface.normal()
        absn = np.linalg.norm(n, axis=2)
        unitn = n * (1./absn)[:,:,None]
        Bcoil = self.field.set_points(xyz.reshape((-1, 3))).B().reshape(xyz.shape)
        Bcoil_n = np.sum(Bcoil*unitn, axis=2)
        B_n = (Bcoil_n - self.target)
        return 0.5 * np.mean(B_n**2 * absn)

    def dJ(self):
        xyz = self.surface.gamma()
        n = self.surface.normal()
        absn = np.linalg.norm(n, axis=2)
        unitn = n * (1./absn)[:,:,None]
        Bcoil = self.field.set_points(xyz.reshape((-1, 3))).B().reshape(xyz.shape)
        Bcoil_n = np.sum(Bcoil*unitn, axis=2)
        B_n = (Bcoil_n - self.target)
        dJdB = (B_n[...,None] * unitn * absn[...,None])/absn.size
        dJdB = dJdB.reshape((-1, 3))
        return self.field.B_vjp_graph(dJdB)



surface = SurfaceRZFourier(nfp=3, stellsym=True, mpol=5, ntor=5, quadpoints_phi=63, quadpoints_theta=62)
surface.fit_to_curve(ma, 0.2)
target = np.zeros(surface.gamma().shape[:2])
J = SquaredFlux(surface, target, bs)
print(J.x)
J.J()
dJ = J.dJ()
print(dJ(curves[0]))
print(dJ.data)
dJvec = np.concatenate([dJ.data[d] for d in dJ.data])
x = J.x
h = np.random.uniform(size=x.shape)
dJh = np.sum(dJvec*h)
x0 = x.copy()
J0 = J.J()
for eps in [1e-2, 1e-3, 1e-4, 1e-5]:
    J.x = x + eps * h
    Jp = J.J()
    J.x = x - eps * h
    Jm = J.J()
    dJest = (Jp-Jm)/(2*eps)
    print(dJest, dJh, dJest-dJh)
    
import IPython; IPython.embed()
import sys; sys.exit()
