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

__all__ = ["BoozerJax", "QuasisymmetryJax", "BoozerQuasisymmetryResidualJax"]


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

    def _populate_booz_xform_from_jax_output(self, out, compute_surfs):
        self.bx.xm_b = np.asarray(out["ixm_b"])
        self.bx.xn_b = np.asarray(out["ixn_b"])
        self.bx.mnboz = len(self.bx.xm_b)
        self.bx.bmnc_b = np.asarray(out["bmnc_b"]).T
        self.bx.bmns_b = np.asarray(out["bmns_b"]).T
        self.bx.rmnc_b = np.asarray(out["rmnc_b"]).T
        self.bx.rmns_b = np.asarray(out["rmns_b"]).T
        self.bx.zmnc_b = np.asarray(out["zmnc_b"]).T
        self.bx.zmns_b = np.asarray(out["zmns_b"]).T
        self.bx.numnc_b = -np.asarray(out["pmnc_b"]).T
        self.bx.numns_b = -np.asarray(out["pmns_b"]).T
        self.bx.gmnc_b = np.asarray(out["gmnc_b"]).T
        self.bx.gmns_b = np.asarray(out["gmns_b"]).T
        self.bx.Boozer_I = np.asarray(out["buco_b"])
        self.bx.Boozer_G = np.asarray(out["bvco_b"])
        self.bx.s_b = np.asarray(self.bx.s_in)[compute_surfs]
        self.bx._last_jax_output = out

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

        logger.info("About to call booz_xform_jax.Booz_xform.run_jax().")
        out = self.bx.run_jax()
        self._populate_booz_xform_from_jax_output(out, compute_surfs)
        self._calls += 1
        logger.info("Returned from calling booz_xform_jax.Booz_xform.run_jax().")
        self.need_to_run_code = False


class QuasisymmetryJax(Quasisymmetry):
    """
    Quasisymmetry objective evaluated from a :class:`BoozerJax` spectrum.
    """


class BoozerQuasisymmetryResidualJax(Optimizable):
    """
    Differentiable Boozer-spectrum quasisymmetry residual for VMEC-JAX states.

    This objective is intended for ``vmec_jax.FixedBoundaryExactOptimizer``.
    It keeps the Boozer residual modular, so it can be combined with other
    VMEC-JAX objective terms through ``VmecJaxLeastSquaresProblem``.

    Args:
        surfaces: Normalized toroidal flux surfaces.
        helicity_m: Desired poloidal helicity.
        helicity_n: Desired toroidal helicity divided by ``nfp``.
        mboz: Number of Boozer poloidal Fourier modes.
        nboz: Number of Boozer toroidal Fourier modes.
        normalization: Boozer-spectrum normalization, ``"B00"`` or
            ``"symmetric"``.
        weight: Residual weighting, matching :class:`Quasisymmetry`.
    """

    def __init__(
        self,
        surfaces: Union[float, Iterable[float]],
        helicity_m: int,
        helicity_n: int,
        mboz: int = 8,
        nboz: int = 8,
        normalization: str = "B00",
        weight: str = "even",
    ) -> None:
        if booz_xform_jax is None:
            raise RuntimeError(
                "To use a BoozerQuasisymmetryResidualJax object, the "
                "booz_xform_jax package must be installed."
            )
        try:
            self.surfaces = list(surfaces)
        except TypeError:
            self.surfaces = [surfaces]
        for surface in self.surfaces:
            if surface < 0 or surface > 1:
                raise ValueError("surfaces must lie in the interval [0, 1]")
        if helicity_m not in (0, 1):
            raise ValueError("m for quasisymmetry should be 0 or 1.")
        if normalization not in ("B00", "symmetric"):
            raise ValueError("normalization must be 'B00' or 'symmetric'")
        if weight not in ("even", "stellopt", "stellopt_ornl"):
            raise ValueError("Unrecognized value for weight in Quasisymmetry")

        self.helicity_m = helicity_m
        self.helicity_n = helicity_n
        self.mboz = mboz
        self.nboz = nboz
        self.normalization = normalization
        self.weight = weight
        super().__init__(depends_on=[])

    def surface_indices(self, static):
        """
        Return the VMEC-JAX half-grid indices and surfaces used.
        """
        import vmec_jax as vmec_jax_mod

        return vmec_jax_mod.surface_indices_from_static(static, self.surfaces)

    def residuals_from_state(self, static, indata, signgs=None, flux=None):
        """
        Return a JAX-compatible residual function of a solved VMEC state.
        """
        from booz_xform_jax.jax_api import booz_xform_jax_impl
        from booz_xform_jax.jax_api import prepare_booz_xform_constants_from_inputs
        import vmec_jax as vmec_jax_mod
        from vmec_jax._compat import jax, jnp

        if signgs is None:
            boundary = vmec_jax_mod.boundary_from_indata(indata, static.modes)
            state0 = vmec_jax_mod.initial_guess_from_boundary(
                static, boundary, indata, vmec_project=True
            )
            geom = vmec_jax_mod.eval_geom(state0, static)
            signgs = int(
                vmec_jax_mod.signgs_from_sqrtg(
                    np.asarray(geom.sqrtg), axis_index=1
                )
            )
        else:
            boundary = None
            state0 = None

        if flux is None:
            flux = vmec_jax_mod.flux_profiles_from_indata(
                indata, static.s, signgs=signgs
            )
        if state0 is None:
            boundary = vmec_jax_mod.boundary_from_indata(indata, static.modes)
            state0 = vmec_jax_mod.initial_guess_from_boundary(
                static, boundary, indata, vmec_project=True
            )

        initial_inputs = vmec_jax_mod.booz_xform_inputs_from_state(
            state=state0,
            static=static,
            indata=indata,
            signgs=signgs,
            flux=flux,
        )
        constants, grids = prepare_booz_xform_constants_from_inputs(
            inputs=initial_inputs,
            mboz=self.mboz,
            nboz=self.nboz,
            asym=bool(static.cfg.lasym),
        )
        surface_indices, surfaces_used = self.surface_indices(static)
        surface_indices = jnp.asarray(surface_indices, dtype=jnp.int32)
        surfaces_used = jnp.asarray(surfaces_used, dtype=jnp.float64)

        xm_b = np.asarray(grids.xm_b, dtype=int)
        xn_b = np.asarray(grids.xn_b, dtype=int) / int(static.cfg.nfp)
        if self.helicity_n == 0:
            symmetric = xn_b == 0
        elif self.helicity_m == 0:
            symmetric = xm_b == 0
        else:
            symmetric = xm_b * self.helicity_n + xn_b * self.helicity_m == 0
        nonsymmetric_indices = jnp.asarray(
            np.nonzero(np.logical_not(symmetric))[0], dtype=jnp.int32
        )
        symmetric_indices = jnp.asarray(np.nonzero(symmetric)[0], dtype=jnp.int32)
        booz_fn = jax.jit(booz_xform_jax_impl, static_argnames=("constants",))

        def qs_residuals_from_state(state):
            inputs = vmec_jax_mod.booz_xform_inputs_from_state(
                state=state,
                static=static,
                indata=indata,
                signgs=signgs,
                flux=flux,
            )
            out = booz_fn(
                rmnc=inputs.rmnc,
                zmns=inputs.zmns,
                lmns=inputs.lmns,
                bmnc=inputs.bmnc,
                bsubumnc=inputs.bsubumnc,
                bsubvmnc=inputs.bsubvmnc,
                iota=inputs.iota,
                xm=inputs.xm,
                xn=inputs.xn,
                xm_nyq=inputs.xm_nyq,
                xn_nyq=inputs.xn_nyq,
                constants=constants,
                grids=grids,
                bmns=inputs.bmns,
                bsubumns=inputs.bsubumns,
                bsubvmns=inputs.bsubvmns,
                surface_indices=surface_indices,
            )
            bmnc_b = out["bmnc_b"]
            if self.normalization == "B00":
                bnorm = bmnc_b[:, 0:1]
            else:
                symmetric_b = jnp.take(bmnc_b, symmetric_indices, axis=1)
                bnorm = jnp.sqrt(jnp.sum(symmetric_b * symmetric_b, axis=1))
                bnorm = bnorm[:, None]
            bnorm = jnp.where(jnp.abs(bnorm) > 0.0, bnorm, 1.0)
            nonsymmetric_b = jnp.take(bmnc_b / bnorm, nonsymmetric_indices, axis=1)
            if self.weight == "stellopt":
                nonsymmetric_b = nonsymmetric_b / (surfaces_used[:, None] ** 2)
            elif self.weight == "stellopt_ornl":
                nonsymmetric_b = jnp.sqrt(jnp.sum(nonsymmetric_b ** 2, axis=1))
            return jnp.ravel(nonsymmetric_b)

        def qs_total_from_state(state):
            residuals = qs_residuals_from_state(state)
            return jnp.sum(residuals * residuals)

        qs_residuals_from_state._n_non_qs = 0
        qs_residuals_from_state._qs_total_from_state = qs_total_from_state
        return qs_residuals_from_state
