from simsopt._core.graph_optimizable import Optimizable
import numpy as np


class SquaredFlux(Optimizable):

    def __init__(self, surface, field, target=None):
        self.surface = surface
        self.target = target
        self.field = field
        xyz = self.surface.gamma()
        self.field.set_points(xyz.reshape((-1, 3)))
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field])

    def J(self):
        xyz = self.surface.gamma()
        n = self.surface.normal()
        absn = np.linalg.norm(n, axis=2)
        unitn = n * (1./absn)[:,:,None]
        Bcoil = self.field.B().reshape(xyz.shape)
        Bcoil_n = np.sum(Bcoil*unitn, axis=2)
        if self.target is not None:
            B_n = (Bcoil_n - self.target)
        else:
            B_n = Bcoil_n
        return 0.5 * np.mean(B_n**2 * absn)

    def dJ(self):
        n = self.surface.normal()
        absn = np.linalg.norm(n, axis=2)
        unitn = n * (1./absn)[:,:,None]
        Bcoil = self.field.B().reshape(n.shape)
        Bcoil_n = np.sum(Bcoil*unitn, axis=2)
        if self.target is not None:
            B_n = (Bcoil_n - self.target)
        else:
            B_n = Bcoil_n
        dJdB = (B_n[...,None] * unitn * absn[...,None])/absn.size
        dJdB = dJdB.reshape((-1, 3))
        return self.field.B_vjp(dJdB)


class FOCUSObjective(Optimizable):

    def __init__(self, Jflux, Jcls, alpha):
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[Jflux] + Jcls)
        self.Jflux = Jflux
        self.Jcls = Jcls
        self.alpha = alpha

    def J(self):
        self.vals = [self.Jflux.J()] + [Jcl.J() for Jcl in self.Jcls]
        return self.vals[0] + self.alpha * sum(self.vals[1:])

    def dJ(self):
        res = self.Jflux.dJ()
        for Jcl in self.Jcls:
            res += self.alpha * Jcl.dJ()
        return res

