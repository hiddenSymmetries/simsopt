from simsopt._core.graph_optimizable import Optimizable
from .._core.derivative import derivative_dec
import numpy as np


class SquaredFlux(Optimizable):

    r"""
    Objective representing the quadratic flux of a field on a surface, that is

    .. math::
        \frac12 \int_{S} (\mathbf{B}\cdot \mathbf{n} - B_T)^2 ds

    where :math:`\mathbf{n}` is the surface unit normal vector and
    :math:`B_T` is an optional (zero by default) target value for the
    magnetic field.

    Args:
        surface: A :obj:`simsopt.geo.surface.Surface` object on which to compute the flux
        field: A :obj:`simsopt.field.magneticfield.MagneticField` for which to compute the flux.
        target: A ``nphi x ntheta`` numpy array containing target values for the flux. Here 
          ``nphi`` and ``ntheta`` correspond to the number of quadrature points on `surface` 
          in ``phi`` and ``theta`` direction.
    """

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

    @derivative_dec
    def dJ(self):
        n = self.surface.normal()
        absn = np.linalg.norm(n, axis=2)
        unitn = n * (1./absn)[:, :, None]
        Bcoil = self.field.B().reshape(n.shape)
        Bcoil_n = np.sum(Bcoil*unitn, axis=2)
        if self.target is not None:
            B_n = (Bcoil_n - self.target)
        else:
            B_n = Bcoil_n
        dJdB = (B_n[..., None] * unitn * absn[..., None])/absn.size
        dJdB = dJdB.reshape((-1, 3))
        return self.field.B_vjp(dJdB)


class CoilOptObjective(Optimizable):
    r"""
    Objective combining a single or a list of
    :obj:`simsopt.objectives.fluxobjective.SquaredFlux` with a list of
    curve objectives and a distance objective to form the basis of a
    classic Stage II optimization problem.     The objective functions are
    combined into a single scalar function using weights ``alpha`` and
    ``beta``.

    If a single :obj:`simsopt.objectives.fluxobjective.SquaredFlux`
    is given, then the objective is

    .. math::
        J = \mathrm{Jflux} + \alpha \sum_k \mathrm{Jcls}_k + \beta \mathrm{Jdist}.

    If a list of `n` :obj:`simsopt.objectives.fluxobjective.SquaredFlux` objects
    are given, then the objective is

    .. math::
        J = \frac1n \sum_{i=1}^n \mathrm{Jflux}_i + \alpha \sum_k \mathrm{Jcls}_k + \beta \mathrm{Jdist}.

    This latter case is useful for stochastic optimization.

    Args:
        Jfluxs: A single :obj:`simsopt.objectives.fluxobjective.SquaredFlux` or a list of them
        Jcls: Typically a list of
          :obj:`simsopt.geo.curveobjectives.CurveLength`, though any list of objectives
          that have a ``J()`` and ``dJ()`` function is fine.
        alpha: The scalar weight in front of the objectives in ``Jcls``.
        Jdist: Typically a
          :obj:`simsopt.geo.curveobjectives.MinimumDistance`, though any objective
          that has a ``J()`` and ``dJ()`` function is fine.
        beta: The scalar weight in front of the objective in ``Jdist``.
    """

    def __init__(self, Jfluxs, Jcls=[], alpha=0., Jdist=None, beta=0.):
        if isinstance(Jfluxs, SquaredFlux):
            Jfluxs = [Jfluxs]
        deps = Jfluxs + Jcls
        if Jdist is not None:
            deps.append(Jdist)
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=deps)
        self.Jfluxs = Jfluxs
        self.Jcls = Jcls
        self.alpha = alpha
        self.Jdist = Jdist
        self.beta = beta

    def J(self):
        res = sum(J.J() for J in self.Jfluxs)/len(self.Jfluxs)
        if self.alpha > 0:
            res += self.alpha * sum([J.J() for J in self.Jcls])
        if self.beta > 0 and self.Jdist is not None:
            res += self.beta * self.Jdist.J()
        return res

    @derivative_dec
    def dJ(self):
        res = self.Jfluxs[0].dJ(partials=True)
        for i in range(1, len(self.Jfluxs)):
            res += self.Jfluxs[i].dJ(partials=True)
        res *= 1./len(self.Jfluxs)
        if self.alpha > 0:
            for Jcl in self.Jcls:
                res += self.alpha * Jcl.dJ(partials=True)
        if self.beta > 0 and self.Jdist is not None:
            res += self.beta * self.Jdist.dJ(partials=True)
        return res
