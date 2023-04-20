import numpy as np
import scipy
# from monty.json import MSONable, MontyDecoder

from .._core.optimizable import Optimizable
from .._core.derivative import Derivative, derivative_dec
from .._core.json import GSONable

__all__ = ['MPIObjective', 'QuadraticPenalty', 'Weight', 'forward_backward']


def forward_backward(P, L, U, rhs, iterative_refinement=False):
    """
    Solve a linear system of the form (PLU)^T*adj = rhs for adj.


    Args:
        P: permutation matrix
        L: lower triangular matrix
        U: upper triangular matrix
        iterative_refinement: when true, applies iterative refinement which can improve
                              the accuracy of the computed solution when the matrix is
                              particularly ill-conditioned.
    """
    y = scipy.linalg.solve_triangular(U.T, rhs, lower=True)
    z = scipy.linalg.solve_triangular(L.T, y, lower=False)
    adj = P@z

    if iterative_refinement:
        yp = scipy.linalg.solve_triangular(U.T, rhs-(P@L@U).T@adj, lower=True)
        zp = scipy.linalg.solve_triangular(L.T, yp, lower=False)
        adj += P@zp

    return adj


def sum_across_comm(derivative, comm):
    r"""
    Compute the sum of :mod:`simsopt._core.derivative.Derivative` objects from
    several MPI ranks. This implementation is fairly basic and requires that
    the derivative dictionaries contain the same keys on all ranks.
    """
    newdict = {}
    for k in derivative.data.keys():
        data = derivative.data[k]
        alldata = sum(comm.allgather(data))
        if isinstance(alldata, float):
            alldata = np.asarray([alldata])
        newdict[k] = alldata
    return Derivative(newdict)


class MPIObjective(Optimizable):

    def __init__(self, objectives, comm, needs_splitting=False):
        r"""
        Compute the mean of a list of objectives in parallel using MPI.

        Args:
            objectives: A python list of objectives that provide ``.J()`` and ``.dJ()`` functions.
            comm: The MPI communicator to use.
            needs_splitting: if set to ``True``, then the list of objectives is
                             split into disjoint partitions and only one part is worked on per
                             mpi rank. If set to ``False``, then we assume that the user
                             constructed the list of ``objectives`` so that it only contains the
                             objectives relevant to that mpi rank.
        """

        if needs_splitting:
            from simsopt._core.util import parallel_loop_bounds
            startidx, endidx = parallel_loop_bounds(comm, len(objectives))
            objectives = objectives[startidx:endidx]
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=objectives)
        self.objectives = objectives
        self.comm = comm
        self.n = len(self.objectives) if comm is None else np.sum(self.comm.allgather(len(self.objectives)))

    def J(self):
        local_vals = [J.J() for J in self.objectives]
        global_vals = local_vals if self.comm is None else [i for o in self.comm.allgather(local_vals) for i in o]
        res = np.sum(global_vals)
        return res/self.n

    @derivative_dec
    def dJ(self):
        if len(self.objectives) == 0:
            raise NotImplementedError("`MPIObjective.dJ` currently requires that there is at least one objective per process.")
        local_derivs = sum([J.dJ(partials=True) for J in self.objectives])
        all_derivs = local_derivs if self.comm is None else sum_across_comm(local_derivs, self.comm)
        all_derivs *= 1./self.n
        return all_derivs


class QuadraticPenalty(Optimizable):

    def __init__(self, obj, cons=0., f="identity"):
        r"""
        A quadratic penalty function of the form :math:`0.5f(\text{obj}.J() - \text{cons})^2` for an underlying objective ``obj``
        and wrapping function ``f``. This can be used to implement a barrier penalty function for (in)equality
        constrained optimization problem. The wrapping function defaults to ``"identity"``.

        Args:
            obj: the underlying objective. It should provide a ``.J()`` and ``.dJ()`` function.
            cons: constant
            f: the function that wraps the difference :math:`obj-\text{cons}`.  The options are ``"min"``, ``"max"``, or ``"identity"``.
               which respectively return :math:`\min(\text{obj}-\text{cons}, 0)`, :math:`\max(\text{obj}-\text{cons}, 0)`, and :math:`\text{obj}-\text{cons}`.
        """
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[obj])
        self.obj = obj
        self.cons = cons
        self.f = f

    def J(self):
        val = self.obj.J()
        diff = float(val - self.cons)

        if self.f == 'max':
            return 0.5*np.maximum(diff, 0)**2
        elif self.f == 'min':
            return 0.5*np.minimum(diff, 0)**2
        elif self.f == 'identity':
            return 0.5*diff**2
        else:
            raise Exception('incorrect wrapping function f provided')

    @derivative_dec
    def dJ(self):
        val = self.obj.J()
        dval = self.obj.dJ(partials=True)
        diff = float(val - self.cons)

        if self.f == 'max':
            return np.maximum(diff, 0)*dval
        elif self.f == 'min':
            return np.minimum(diff, 0)*dval
        elif self.f == 'identity':
            return diff*dval
        else:
            raise Exception('incorrect wrapping function f provided')

    return_fn_map = {'J': J, 'dJ': dJ}


class Weight(object):

    def __init__(self, value):
        self.value = float(value)

    def __float__(self):
        return float(self.value)

    def __imul__(self, alpha):
        self.value *= alpha
        return self
