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
        unitn = n * (1./absn)[:, :, None]
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

    def __init__(self, Jflux, Jcls=[], alpha=0., Jdist=[], beta=0.):
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[Jflux] + Jcls)
        self.Jflux = Jflux
        self.Jcls = Jcls
        self.alpha = alpha
        self.Jdist = Jdist
        self.beta = beta

    def J(self):
        res = self.Jflux.J()
        if self.alpha > 0:
            res += self.alpha * sum([J.J() for J in self.Jcls])
        if self.beta > 0:
            res += self.beta * self.Jdist.J()
        return  res

    def dJ(self):
        res = self.Jflux.dJ()
        if self.alpha > 0:
            for Jcl in self.Jcls:
                res += self.alpha * Jcl.dJ()
        if self.beta > 0:
            res += self.beta * self.Jdist.dJ()
        return res

