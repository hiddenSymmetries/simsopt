# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""
This module provides JAX-backed Boozer-coordinate wrappers.
"""

import logging
from typing import Iterable, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import booz_xform_jax
except ImportError as e:
    booz_xform_jax = None
    logger.debug(str(e))

from .boozer import Quasisymmetry
from .._core.optimizable import Optimizable
from .._core.descriptor import Integer

__all__ = ["BoozerJax", "QuasisymmetryJax"]


class BoozerJax(Optimizable):
    """
    Compute Boozer coordinates for a VMEC equilibrium using booz_xform_jax.

    A ``BoozerJax`` instance follows the registry workflow of
    :class:`~simsopt.mhd.boozer.Boozer`: dependent objectives register the
    normalized toroidal-flux surfaces they need, and ``run()`` computes all
    requested surfaces in one Boozer transform.

    Args:
        equil: Equilibrium object with a ``run()`` method and ``wout`` data.
        mpol: Number of Boozer poloidal Fourier modes.
        ntor: Number of Boozer toroidal Fourier modes.
        verbose: Whether to print booz_xform_jax output.
    """

    mpol = Integer()
    ntor = Integer()

    def __init__(self, equil, mpol: int = 32, ntor: int = 32, verbose: bool = False) -> None:
        if booz_xform_jax is None:
            raise RuntimeError(
                "To use a BoozerJax object, the booz_xform_jax package "
                "must be installed."
            )

        self.equil = equil
        self.mpol = mpol
        self.ntor = ntor
        self.bx = booz_xform_jax.Booz_xform()
        self.bx.verbose = verbose
        self.s = set()
        self.need_to_run_code = True
        self._calls = 0
        self.mpi = getattr(equil, "mpi", None) if equil is not None else None

        if equil is not None:
            super().__init__(depends_on=[equil])
        else:
            super().__init__()

    def recompute_bell(self, parent=None):
        self.need_to_run_code = True

    def register(self, s: Union[float, Iterable[float]]) -> None:
        """
        Append surface labels to the registry.
        """
        try:
            ss = set(s)
        except TypeError:
            ss = {s}

        for new_s in ss:
            if new_s < 0 or new_s > 1:
                raise ValueError("Normalized toroidal flux values s must lie in the interval [0, 1]")
        logger.info("Adding entries to BoozerJax registry: {}".format(ss))
        self.s = self.s.union(ss)
        self.need_to_run_code = True

    def _prepare_surface_indices(self, s):
        wout = self.equil.wout
        ns = wout.ns
        s_full = np.linspace(0, 1, ns)
        ds = s_full[1] - s_full[0]
        s_half = s_full[1:] - 0.5 * ds

        compute_surfs = []
        s_to_index_all_surfs = dict()
        self.s_used = dict()
        for ss in s:
            index = np.argmin(np.abs(s_half - ss))
            compute_surfs.append(index)
            s_to_index_all_surfs[ss] = index
            self.s_used[ss] = s_half[index]

        compute_surfs = sorted(list(set(compute_surfs)))
        self.s_to_index = dict()
        for ss in s:
            self.s_to_index[ss] = compute_surfs.index(s_to_index_all_surfs[ss])
        return compute_surfs

    def _init_booz_xform_from_wout(self, compute_surfs):
        wout = self.equil.wout
        self.bx.asym = bool(wout.lasym)
        self.bx.nfp = wout.nfp
        self.bx.mpol = wout.mpol
        self.bx.ntor = wout.ntor
        self.bx.mnmax = wout.mnmax
        self.bx.xm = wout.xm
        self.bx.xn = wout.xn
        self.bx.mpol_nyq = int(wout.xm_nyq[-1])
        self.bx.ntor_nyq = int(wout.xn_nyq[-1] / wout.nfp)
        self.bx.mnmax_nyq = wout.mnmax_nyq
        self.bx.xm_nyq = wout.xm_nyq
        self.bx.xn_nyq = wout.xn_nyq

        if wout.lasym:
            rmns = wout.rmns
            zmnc = wout.zmnc
            lmnc = wout.lmnc
            bmns = wout.bmns
            bsubumns = wout.bsubumns
            bsubvmns = wout.bsubvmns
        else:
            arr = np.array([[]])
            rmns = arr
            zmnc = arr
            lmnc = arr
            bmns = arr
            bsubumns = arr
            bsubvmns = arr

        self.bx.init_from_vmec(
            wout.ns,
            wout.iotas,
            wout.rmnc,
            rmns,
            zmnc,
            wout.zmns,
            lmnc,
            wout.lmns,
            wout.bmnc,
            bmns,
            wout.bsubumnc,
            bsubumns,
            wout.bsubvmnc,
            bsubvmns,
        )
        self.bx.compute_surfs = compute_surfs
        self.bx.mboz = self.mpol
        self.bx.nboz = self.ntor

    def run(self):
        """
        Run booz_xform_jax on all registered surfaces.
        """
        if not self.need_to_run_code:
            logger.info("BoozerJax.run() called but no need to re-run Boozer transformation.")
            return
        if self.equil is None or not hasattr(self.equil, "wout"):
            raise ValueError("equil is not an equilibrium type supported by BoozerJax")

        s = sorted(list(self.s))
        logger.info("Preparing to run BoozerJax transformation. Registry:{}".format(s))
        self.equil.run()
        compute_surfs = self._prepare_surface_indices(s)
        self._init_booz_xform_from_wout(compute_surfs)

        logger.info("About to call booz_xform_jax.Booz_xform.run().")
        self.bx.run()
        self._calls += 1
        logger.info("Returned from calling booz_xform_jax.Booz_xform.run().")
        self.need_to_run_code = False


class QuasisymmetryJax(Quasisymmetry):
    """
    Quasisymmetry objective evaluated from a :class:`BoozerJax` spectrum.
    """
