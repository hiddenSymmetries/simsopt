import numpy as np
from monty.json import MSONable, MontyDecoder

from .._core.optimizable import Optimizable
from .._core.derivative import Derivative, derivative_dec

__all__ = ['MPIObjective', 'QuadraticPenalty', 'Weight']


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

    def __init__(self, obj, threshold=0.):
        r"""
        A penalty function of the form :math:`\max(J - \text{threshold}, 0)^2` for an underlying objective ``J``.

        Args:
            obj: the underlying objective. It should provide a ``.J()`` and ``.dJ()`` function.
            threshold: the threshold past which values should be penalized.
        """
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[obj])
        self.obj = obj
        self.threshold = threshold

    def J(self):
        return 0.5*np.maximum(self.obj.J()-self.threshold, 0)**2

    @derivative_dec
    def dJ(self):
        val = self.obj.J()
        dval = self.obj.dJ(partials=True)
        return np.maximum(val-self.threshold, 0)*dval

    def as_dict(self) -> dict:
        d = {}
        d["@class"] = self.__class__.__name__
        d["@module"] = self.__class__.__module__
        d["obj"] = self.obj
        d["threshold"] = np.array(self.threshold)
        return d

    @classmethod
    def from_dict(cls, d):
        decoder = MontyDecoder()
        obj = decoder.process_decoded(d["obj"])
        threshold = decoder.process_decoded(d["threshold"])
        return cls(obj, threshold)

    return_fn_map = {'J': J, 'dJ': dJ}


class Weight(object):

    def __init__(self, value):
        self.value = float(value)

    def __float__(self):
        return float(self.value)

    def __imul__(self, alpha):
        self.value *= alpha
        return self
