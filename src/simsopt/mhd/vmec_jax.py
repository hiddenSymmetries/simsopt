# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""
This module provides a Simsopt wrapper for the vmec_jax equilibrium code.
"""

import logging
import os
import os.path
from dataclasses import fields, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import netcdf_file
from scipy.integrate import quad

logger = logging.getLogger(__name__)

try:
    import vmec_jax as vmec_jax_mod
except ImportError as e:
    vmec_jax_mod = None
    logger.debug(str(e))

from .._core.optimizable import Optimizable
from .._core.util import Struct, ObjectiveFailure
from ..geo.surface import Surface
from ..geo.surfacerzfourier import SurfaceRZFourier

__all__ = [
    "VmecJax",
    "AspectRatioJax",
    "B_cartesian_jax",
    "B_cartesian_jax_tangent_columns",
    "VmecJaxLeastSquaresProblem",
    "make_vmec_jax_residuals_from_terms",
]


_INDATA_DEFAULTS = {
    "NFP": 1,
    "LASYM": False,
    "LFREEB": False,
    "MPOL": 0,
    "NTOR": 0,
    "PHIEDGE": 1.0,
    "CURTOR": 0.0,
    "PRES_SCALE": 1.0,
    "GAMMA": 0.0,
    "NCURR": 0,
    "DELT": 0.9,
}


def _coerce_indata_value(value):
    if isinstance(value, list):
        return np.asarray(value)
    return value


def _store_indata_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _profile_type(indata, name, default):
    value = indata.get(name, default)
    if isinstance(value, bytes):
        value = value.decode()
    return str(value).strip().lower()


def _set_indata_array(indata, name, values):
    indata.scalars[name.upper()] = np.asarray(values).tolist()


class _VmecJaxInData:
    """
    Attribute-style view of a vmec_jax InData object.
    """

    def __init__(self, indata):
        object.__setattr__(self, "_indata", indata)

    @property
    def raw(self):
        return self._indata

    @property
    def indexed(self):
        return self._indata.indexed

    @property
    def scalars(self):
        return self._indata.scalars

    def get(self, name, default=None):
        return self._indata.get(name, default)

    def get_bool(self, name, default=False):
        return self._indata.get_bool(name, default)

    def get_int(self, name, default=0):
        return self._indata.get_int(name, default)

    def get_float(self, name, default=0.0):
        return self._indata.get_float(name, default)

    def __getattr__(self, name):
        key = name.upper()
        if key in self._indata.scalars:
            return _coerce_indata_value(self._indata.scalars[key])
        if key in _INDATA_DEFAULTS:
            return _INDATA_DEFAULTS[key]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self._indata.scalars[name.upper()] = _store_indata_value(value)


class AspectRatioJax(Optimizable):
    """
    Aspect-ratio objective for VMEC-JAX workflows.

    Args:
        vmec: Optional equilibrium object. If supplied, :meth:`J` evaluates the
            aspect ratio through the usual SIMSOPT-style zero-argument method.
    """

    def __init__(self, vmec=None):
        self.vmec = vmec
        super().__init__(depends_on=[] if vmec is None else [vmec])

    def J(self):
        """
        Return the aspect ratio from the associated equilibrium.
        """
        if self.vmec is None:
            raise RuntimeError("AspectRatioJax.J requires a VMEC object.")
        return self.vmec.aspect()

    def value_from_state(self, static):
        """
        Return a JAX-compatible aspect-ratio function of a solved VMEC state.
        """
        _require_vmec_jax()
        from vmec_jax.wout import equilibrium_aspect_ratio_from_state

        def aspect_from_state(state):
            return equilibrium_aspect_ratio_from_state(state=state, static=static)

        aspect_from_state._n_non_qs = 1
        return aspect_from_state


def _require_vmec_jax():
    if vmec_jax_mod is None:
        raise RuntimeError(
            "VmecJax requires the vmec_jax package. Install vmec_jax to use "
            "the JAX-backed VMEC wrapper."
        )


class VmecJaxLeastSquaresProblem:
    """
    JAX-state least-squares objective for ``FixedBoundaryExactOptimizer``.

    This class mirrors the tuple-based construction of
    :class:`simsopt.objectives.LeastSquaresProblem`, but its functions accept a
    solved VMEC-JAX state instead of reading zero-argument SIMSOPT objects.

    Args:
        goals: Target values.
        weights: Least-squares weights.
        funcs_in: JAX-compatible functions of a solved VMEC state.
    """

    def __init__(self, goals, weights, funcs_in):
        _require_vmec_jax()
        if np.isscalar(goals):
            goals = [goals]
        if np.isscalar(weights):
            weights = [weights]
        self.goals = tuple(goals)
        self.weights = tuple(weights)
        self.funcs_in = tuple(funcs_in)
        if len(self.funcs_in) == 0:
            raise ValueError("at least one VMEC-JAX objective term is required")
        if not (
            len(self.goals) == len(self.weights) == len(self.funcs_in)
        ):
            raise ValueError("goals, weights, and funcs_in must have the same length")
        if np.any(np.asarray(self.weights) < 0):
            raise ValueError("Weight cannot be negative")
        self.residuals_from_state = self._make_residuals_from_state()

    @classmethod
    def from_tuples(cls, tuples):
        """
        Construct a VMEC-JAX least-squares problem from ``(func, goal, weight)``.
        """
        tuples = tuple(tuples)
        if len(tuples) == 0:
            raise ValueError("at least one VMEC-JAX objective term is required")
        funcs_in, goals, weights = zip(*tuples)
        return cls(goals, weights, funcs_in)

    def _term_residual(self, state, func, goal, weight):
        from vmec_jax._compat import jnp

        value = jnp.ravel(jnp.asarray(func(state), dtype=jnp.float64))
        goal = jnp.asarray(goal, dtype=jnp.float64)
        if goal.ndim > 0:
            goal = jnp.ravel(goal)
        return jnp.sqrt(jnp.asarray(weight, dtype=jnp.float64)) * (value - goal)

    def _make_residuals_from_state(self):
        from vmec_jax._compat import jnp

        def residuals_from_state(state):
            parts = [
                self._term_residual(state, func, goal, weight)
                for func, goal, weight in zip(
                    self.funcs_in, self.goals, self.weights
                )
            ]
            return jnp.concatenate(parts)

        residuals_from_state._n_non_qs = sum(
            int(getattr(func, "_n_non_qs", 0)) for func in self.funcs_in
        )
        if any(hasattr(func, "_qs_total_from_state") for func in self.funcs_in):
            residuals_from_state._qs_total_from_state = self._qs_total_from_state
        residuals_from_state._state_cotangent_operator_from_packed = (
            self._state_cotangent_operator_from_packed
        )
        return residuals_from_state

    def objective_from_state(self, state):
        """
        Return the least-squares objective for a solved VMEC state.
        """
        from vmec_jax._compat import jnp

        residuals = self.residuals_from_state(state)
        return jnp.vdot(residuals, residuals)

    def _qs_total_from_state(self, state):
        total = 0.0
        for func, weight in zip(self.funcs_in, self.weights):
            qs_total = getattr(func, "_qs_total_from_state", None)
            if qs_total is not None:
                total = total + float(weight) * qs_total(state)
        return total

    def _state_cotangent_operator_from_packed(self, packed_state, layout):
        from vmec_jax._compat import jax, jnp
        from vmec_jax.state import unpack_state

        packed_state = jnp.asarray(packed_state, dtype=jnp.float64)
        blocks = []
        offset = 0

        for func, goal, weight in zip(self.funcs_in, self.goals, self.weights):
            def _term_from_packed(packed, func=func, goal=goal, weight=weight):
                state = unpack_state(packed, layout)
                return self._term_residual(state, func, goal, weight)

            term_value, term_vjp = jax.vjp(_term_from_packed, packed_state)
            size = int(term_value.size)
            blocks.append((slice(offset, offset + size), term_vjp))
            offset += size

        def _apply(residual_cotangent):
            residual_cotangent = jnp.asarray(
                residual_cotangent, dtype=jnp.float64
            ).reshape(-1)
            total = jnp.zeros_like(packed_state)
            for selector, vjp_fun in blocks:
                cot = residual_cotangent[selector]
                total = total + jax.lax.cond(
                    jnp.any(cot != 0.0),
                    lambda cot_block: vjp_fun(cot_block)[0],
                    lambda cot_block: jnp.zeros_like(packed_state),
                    cot,
                )
            return total

        return _apply


def make_vmec_jax_residuals_from_terms(
    terms,
    n_non_qs=None,
    qs_total_from_state=None,
):
    """
    Combine VMEC-JAX residual terms into a single residual function.

    Each entry in ``terms`` must be a JAX-compatible callable accepting a
    solved VMEC-JAX state and returning a residual array. The returned callable
    is suitable for ``vmec_jax.FixedBoundaryExactOptimizer``.
    """
    terms = tuple(terms)
    if len(terms) == 0:
        raise ValueError("at least one VMEC-JAX residual term is required")
    if len(terms) == 1:
        residuals_from_state = terms[0]
        if n_non_qs is not None:
            residuals_from_state._n_non_qs = int(n_non_qs)
        if qs_total_from_state is not None:
            residuals_from_state._qs_total_from_state = qs_total_from_state
        return residuals_from_state
    if n_non_qs is None:
        n_non_qs = sum(int(getattr(term, "_n_non_qs", 0)) for term in terms)
    problem = VmecJaxLeastSquaresProblem(
        np.zeros(len(terms)),
        np.ones(len(terms)),
        terms,
    )
    problem.residuals_from_state._n_non_qs = int(n_non_qs)
    if qs_total_from_state is not None:
        problem.residuals_from_state._qs_total_from_state = qs_total_from_state
    return problem.residuals_from_state


def _filename_kind(filename):
    basename = os.path.basename(str(filename))
    if basename[:5] == "input":
        return "input"
    if basename[:4] == "wout":
        return "wout"
    raise ValueError("Invalid filename")


def _surface_from_indata(indata, ntheta, nphi, range_surface):
    nfp = int(indata.get_int("NFP", _INDATA_DEFAULTS["NFP"]))
    lasym = bool(indata.get_bool("LASYM", _INDATA_DEFAULTS["LASYM"]))
    mpol = int(indata.get_int("MPOL", _INDATA_DEFAULTS["MPOL"]))
    ntor = int(indata.get_int("NTOR", _INDATA_DEFAULTS["NTOR"]))

    surf = SurfaceRZFourier.from_nphi_ntheta(
        nfp=nfp,
        stellsym=not lasym,
        mpol=mpol,
        ntor=ntor,
        ntheta=ntheta,
        nphi=nphi,
        range=range_surface,
    )

    rbc = indata.indexed.get("RBC", {})
    zbs = indata.indexed.get("ZBS", {})
    rbs = indata.indexed.get("RBS", {})
    zbc = indata.indexed.get("ZBC", {})

    for m in range(mpol + 1):
        for n in range(-ntor, ntor + 1):
            surf.rc[m, n + ntor] = float(rbc.get((n, m), 0.0))
            surf.zs[m, n + ntor] = float(zbs.get((n, m), 0.0))
            if lasym:
                surf.rs[m, n + ntor] = float(rbs.get((n, m), 0.0))
                surf.zc[m, n + ntor] = float(zbc.get((n, m), 0.0))

    surf.local_full_x = surf.get_dofs()
    return surf


def _grid_points_for_b_cartesian(vmec, quadpoints_phi, quadpoints_theta, range, nphi, ntheta):
    if nphi is None and quadpoints_phi is None:
        phi1D = vmec.boundary.quadpoints_phi
    elif quadpoints_phi is None:
        phi1D = Surface.get_phi_quadpoints(range=range, nphi=nphi, nfp=vmec.wout.nfp)
    else:
        phi1D = quadpoints_phi

    if ntheta is None and quadpoints_theta is None:
        theta1D = vmec.boundary.quadpoints_theta
    elif quadpoints_theta is None:
        theta1D = Surface.get_theta_quadpoints(ntheta=ntheta)
    else:
        theta1D = quadpoints_theta

    return np.asarray(phi1D), np.asarray(theta1D)


def _wout_config(wout, nphi, ntheta):
    from vmec_jax.config import VMECConfig

    return VMECConfig(
        mpol=int(wout.mpol),
        ntor=int(wout.ntor),
        ns=int(wout.ns),
        nfp=int(wout.nfp),
        lasym=bool(wout.lasym),
        lthreed=bool(int(wout.ntor) > 0),
        lconm1=True,
        ntheta=int(ntheta),
        nzeta=int(nphi),
    )


def B_cartesian_jax(vmec, quadpoints_phi=None, quadpoints_theta=None,
                    range=Surface.RANGE_FULL_TORUS, nphi=None, ntheta=None,
                    use_wout_bsup=None):
    r"""
    Compute Cartesian magnetic field components on a ``VmecJax`` boundary.

    This function mirrors :func:`simsopt.mhd.vmec_diagnostics.B_cartesian`,
    but evaluates the field using ``vmec_jax.b_cartesian_from_state``. The
    return value is the same ``(Bx, By, Bz)`` tuple of arrays with shape
    ``(nphi, ntheta)``.
    """
    _require_vmec_jax()
    if not isinstance(vmec, VmecJax):
        vmec = VmecJax(vmec)
    if not hasattr(vmec_jax_mod, "b_cartesian_from_state"):
        raise RuntimeError(
            "B_cartesian_jax requires a vmec_jax version with "
            "b_cartesian_from_state."
        )

    vmec.run()
    if vmec.wout.lasym:
        raise RuntimeError("B_cartesian_jax presently only works for stellarator symmetry")

    phi1D, theta1D = _grid_points_for_b_cartesian(
        vmec, quadpoints_phi, quadpoints_theta, range, nphi, ntheta
    )
    from vmec_jax.grids import AngleGrid
    from vmec_jax.static import build_static
    from vmec_jax.wout import state_from_wout

    nfp = int(vmec.wout.nfp)
    grid = AngleGrid(
        theta=theta1D * (2 * np.pi),
        zeta=phi1D * (2 * np.pi * nfp),
        nfp=nfp,
    )

    if use_wout_bsup is None:
        use_wout_bsup = not vmec.runnable

    if vmec.runnable and vmec._run is not None and not use_wout_bsup:
        cfg = replace(vmec._run.cfg, ntheta=len(theta1D), nzeta=len(phi1D))
        static = build_static(cfg, grid=grid)
        B = vmec_jax_mod.b_cartesian_from_state(
            vmec._run.state,
            static,
            indata=vmec._run.indata,
            signgs=vmec._run.signgs,
        )
    else:
        cfg = _wout_config(vmec._wout_jax, len(phi1D), len(theta1D))
        static = build_static(cfg, grid=grid)
        state = state_from_wout(vmec._wout_jax)
        B = vmec_jax_mod.b_cartesian_from_state(
            state,
            static,
            wout=vmec._wout_jax,
            use_wout_bsup=True,
        )

    B = np.transpose(np.asarray(B), (1, 0, 2))
    return B[:, :, 0], B[:, :, 1], B[:, :, 2]


def _grid_points_for_exact_optimizer(exact_optimizer, quadpoints_phi, quadpoints_theta,
                                     range, nphi, ntheta):
    nfp = int(exact_optimizer._static.cfg.nfp)
    if quadpoints_phi is None:
        if nphi is None:
            nphi = int(exact_optimizer._static.cfg.nzeta)
        phi1D = Surface.get_phi_quadpoints(range=range, nphi=nphi, nfp=nfp)
    else:
        phi1D = quadpoints_phi

    if quadpoints_theta is None:
        if ntheta is None:
            ntheta = int(exact_optimizer._static.cfg.ntheta)
        theta1D = Surface.get_theta_quadpoints(ntheta=ntheta)
    else:
        theta1D = quadpoints_theta

    return np.asarray(phi1D), np.asarray(theta1D)


def B_cartesian_jax_tangent_columns(
    exact_optimizer,
    params,
    quadpoints_phi=None,
    quadpoints_theta=None,
    range=Surface.RANGE_FULL_TORUS,
    nphi=None,
    ntheta=None,
):
    r"""
    Return the VMEC-JAX boundary field and exact tangent columns.

    ``exact_optimizer`` should be a ``vmec_jax.FixedBoundaryExactOptimizer``.
    The returned field has shape ``(nphi, ntheta, 3)`` and the tangent array
    has shape ``(nphi, ntheta, 3, nparams)``. The tangent columns use the same
    accepted-point tape replay and frozen-axis initial-state convention as the
    optimizer's exact Jacobian.
    """
    _require_vmec_jax()
    if not hasattr(exact_optimizer, "_static"):
        raise TypeError(
            "B_cartesian_jax_tangent_columns requires a vmec_jax "
            "FixedBoundaryExactOptimizer-like object."
        )
    if not hasattr(vmec_jax_mod, "b_cartesian_from_state"):
        raise RuntimeError(
            "B_cartesian_jax_tangent_columns requires a vmec_jax version with "
            "b_cartesian_from_state."
        )

    from vmec_jax._compat import jnp
    from vmec_jax.grids import AngleGrid
    from vmec_jax.static import build_static

    params = jnp.asarray(np.asarray(params, dtype=float), dtype=jnp.float64)
    phi1D, theta1D = _grid_points_for_exact_optimizer(
        exact_optimizer, quadpoints_phi, quadpoints_theta, range, nphi, ntheta
    )
    nfp = int(exact_optimizer._static.cfg.nfp)
    grid = AngleGrid(
        theta=theta1D * (2 * np.pi),
        zeta=phi1D * (2 * np.pi * nfp),
        nfp=nfp,
    )
    cfg = replace(exact_optimizer._static.cfg, ntheta=len(theta1D), nzeta=len(phi1D))
    field_static = build_static(cfg, grid=grid)

    if not hasattr(exact_optimizer, "b_cartesian_tangent_columns_fun"):
        raise RuntimeError(
            "B_cartesian_jax_tangent_columns requires a vmec_jax version "
            "with FixedBoundaryExactOptimizer.b_cartesian_tangent_columns_fun."
        )

    B, tangents = exact_optimizer.b_cartesian_tangent_columns_fun(
        params,
        field_static,
    )
    B = np.transpose(np.asarray(B), (1, 0, 2))
    tangents = np.transpose(np.asarray(tangents), (1, 0, 2, 3))
    return B, tangents


def _set_sparse_coeff(coeffs, n, m, value):
    value = float(value)
    if value != 0.0:
        coeffs[(int(n), int(m))] = value


def _copy_wout_data_to_struct(wout_data):
    wout = Struct()
    if is_dataclass(wout_data):
        names = [field.name for field in fields(wout_data)]
    else:
        names = [name for name in dir(wout_data) if not name.startswith("_")]

    for name in names:
        value = getattr(wout_data, name)
        if isinstance(value, np.ndarray):
            value = value.T.copy() if value.ndim == 2 else value.copy()
        wout.__setattr__(name, value)

    if hasattr(wout, "volume_p"):
        wout.volume = wout.volume_p
    if hasattr(wout, "lasym"):
        wout.lasym = bool(wout.lasym)
    if not hasattr(wout, "ier_flag"):
        wout.ier_flag = 0
    return wout


class VmecJax(Optimizable):
    r"""
    This class represents the JAX implementation of VMEC.

    The interface intentionally follows :class:`~simsopt.mhd.vmec.Vmec`.
    A ``VmecJax`` object can be initialized from either a VMEC
    ``input.<extension>`` file or a ``wout_<extension>.nc`` output file.
    When initialized from a ``wout`` file, all data in that file is available
    in memory but the equilibrium cannot be re-run.

    The wrapper owns the same scalar degrees of freedom as ``Vmec``:
    ``phiedge``, ``curtor``, and ``pres_scale``. The boundary degrees of
    freedom are owned by the boundary surface.

    Args:
        filename: VMEC ``input`` or ``wout`` file. If ``None``, the default
          Simsopt VMEC input file is used.
        mpi: Accepted for compatibility with ``Vmec``. The initial
          implementation runs vmec_jax locally and does not partition work
          across MPI groups.
        keep_all_files: Whether to retain generated input and wout files.
        verbose: Whether to print vmec_jax solver progress.
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        mpi=None,
        keep_all_files: bool = False,
        verbose: bool = True,
        ntheta=50,
        nphi=50,
        range_surface="full torus",
    ):
        _require_vmec_jax()

        if filename is None:
            filename = os.path.join(os.path.dirname(__file__), "input.default")
            logger.info(f"Initializing a VmecJax object from defaults in {filename}")

        filename = str(filename)
        kind = _filename_kind(filename)
        self.verbose = verbose
        self.mpi = mpi
        self.keep_all_files = keep_all_files
        self.files_to_delete = []
        self.iter = -1
        self.wout = Struct()
        self._wout_jax = None
        self._run = None
        self._pressure_profile = None
        self._current_profile = None
        self._iota_profile = None
        self.n_pressure = 10
        self.n_current = 10
        self.n_iota = 10

        if kind == "input":
            logger.info(f"Initializing a VmecJax object from input file: {filename}")
            self.input_file = filename
            self.runnable = True
            self.indata = _VmecJaxInData(vmec_jax_mod.read_indata(filename))
            self._boundary = _surface_from_indata(self.indata.raw, ntheta, nphi, range_surface)
            self.free_boundary = bool(self.indata.lfreeb)
            self.need_to_run_code = True
        else:
            logger.info(f"Initializing a VmecJax object from wout file: {filename}")
            self.runnable = False
            self.output_file = filename
            self.indata = None
            self._boundary = SurfaceRZFourier.from_wout(
                filename, nphi=nphi, ntheta=ntheta, range=range_surface
            )
            self.free_boundary = False
            self.load_wout()

        x0 = self.get_dofs()
        fixed = np.full(len(x0), True)
        names = ["phiedge", "curtor", "pres_scale"]
        super().__init__(
            x0=x0,
            fixed=fixed,
            names=names,
            depends_on=[self._boundary],
            external_dof_setter=VmecJax.set_dofs,
        )

        if not self.runnable:
            self.need_to_run_code = False

    @property
    def boundary(self):
        return self._boundary

    @boundary.setter
    def boundary(self, boundary):
        if boundary is not self._boundary:
            logging.debug("Replacing surface in boundary setter")
            self.remove_parent(self._boundary)
            self._boundary = boundary
            self.append_parent(boundary)
            self.need_to_run_code = True

    @property
    def pressure_profile(self):
        return self._pressure_profile

    @pressure_profile.setter
    def pressure_profile(self, pressure_profile):
        if pressure_profile is not self._pressure_profile:
            logging.debug("Replacing pressure_profile in setter")
            if self._pressure_profile is not None:
                self.remove_parent(self._pressure_profile)
            self._pressure_profile = pressure_profile
            if pressure_profile is not None:
                self.append_parent(pressure_profile)
                self.need_to_run_code = True

    @property
    def current_profile(self):
        return self._current_profile

    @current_profile.setter
    def current_profile(self, current_profile):
        if current_profile is not self._current_profile:
            logging.debug("Replacing current_profile in setter")
            if self._current_profile is not None:
                self.remove_parent(self._current_profile)
            self._current_profile = current_profile
            if current_profile is not None:
                self.append_parent(current_profile)
                self.need_to_run_code = True

    @property
    def iota_profile(self):
        return self._iota_profile

    @iota_profile.setter
    def iota_profile(self, iota_profile):
        if iota_profile is not self._iota_profile:
            logging.debug("Replacing iota_profile in setter")
            if self._iota_profile is not None:
                self.remove_parent(self._iota_profile)
            self._iota_profile = iota_profile
            if iota_profile is not None:
                self.append_parent(iota_profile)
                self.need_to_run_code = True

    def get_dofs(self):
        if not self.runnable:
            return np.array([1.0, 0.0, 1.0])
        return np.array([self.indata.phiedge, self.indata.curtor, self.indata.pres_scale])

    def set_dofs(self, x):
        if self.runnable:
            self.need_to_run_code = True
            self.indata.phiedge = x[0]
            self.indata.curtor = x[1]
            self.indata.pres_scale = x[2]

    def recompute_bell(self, parent=None):
        self.need_to_run_code = True

    def set_profile(self, longname, shortname, letter):
        """
        Set a pressure, current, or iota profile from a Simsopt profile object.
        """
        profile = self.__getattribute__(longname + "_profile")
        if profile is None:
            return

        n = self.__getattribute__("n_" + longname)
        vmec_profile_type = _profile_type(self.indata.raw, "p" + shortname + "_type", "power_series")
        if vmec_profile_type.startswith("power_series"):
            nodes, _ = np.polynomial.legendre.leggauss(n)
            x = nodes * 0.5 + 0.5
            y = profile(x)
            poly = np.polynomial.polynomial.Polynomial.fit(x, y, n - 1, domain=[0, 1]).convert().coef
            logger.debug(
                "Setting vmec_jax " + longname + f" profile using power series. x: {x} y: {y} poly: {poly}"
            )
            _set_indata_array(self.indata.raw, "a" + letter, poly)

        elif (
            vmec_profile_type.startswith("cubic_spline")
            or vmec_profile_type.startswith("akima_spline")
            or vmec_profile_type.startswith("line_segment")
        ):
            x = np.linspace(0, 1, n)
            y = profile(x)
            logger.debug("Setting vmec_jax " + longname + f" profile using splines. x: {x} y: {y}")
            _set_indata_array(self.indata.raw, "a" + letter + "_aux_s", x)
            _set_indata_array(self.indata.raw, "a" + letter + "_aux_f", y)

        else:
            raise RuntimeError(
                "To use a simsopt Profile class with vmec_jax, vmec profile type must be "
                "power_series, cubic_spline, akima_spline, or line_segment. For current "
                "profiles, _i or _ip can be appended."
            )

    def set_indata(self):
        """
        Transfer Simsopt surface data to the vmec_jax namelist data.
        """
        if not self.runnable:
            raise RuntimeError("Cannot access indata for a VmecJax object that was initialized from a wout file.")

        boundary_RZFourier = self.boundary.to_RZFourier()
        indata = self.indata.raw
        indata.scalars["NFP"] = int(boundary_RZFourier.nfp)
        indata.scalars["LASYM"] = not bool(boundary_RZFourier.stellsym)
        indata.scalars["MPOL"] = int(boundary_RZFourier.mpol)
        indata.scalars["NTOR"] = int(boundary_RZFourier.ntor)

        ntor = int(boundary_RZFourier.ntor)
        mpol = int(boundary_RZFourier.mpol)
        lasym = not bool(boundary_RZFourier.stellsym)
        indata.indexed["RBC"] = {}
        indata.indexed["ZBS"] = {}
        if lasym:
            indata.indexed["RBS"] = {}
            indata.indexed["ZBC"] = {}
        else:
            indata.indexed.pop("RBS", None)
            indata.indexed.pop("ZBC", None)

        for m in range(mpol + 1):
            for n in range(-ntor, ntor + 1):
                _set_sparse_coeff(indata.indexed["RBC"], n, m, boundary_RZFourier.get_rc(m, n))
                _set_sparse_coeff(indata.indexed["ZBS"], n, m, boundary_RZFourier.get_zs(m, n))
                if lasym:
                    _set_sparse_coeff(indata.indexed["RBS"], n, m, boundary_RZFourier.get_rs(m, n))
                    _set_sparse_coeff(indata.indexed["ZBC"], n, m, boundary_RZFourier.get_zc(m, n))

        self.set_profile("pressure", "mass", "m")
        self.set_profile("current", "curr", "c")
        self.set_profile("iota", "iota", "i")
        if self.pressure_profile is not None:
            self.indata.pres_scale = 1.0
        if self.current_profile is not None:
            current_type = _profile_type(
                self.indata.raw, "pcurr_type", _profile_type(self.indata.raw, "pc_type", "power_series")
            )
            if current_type in [
                "power_series",
                "gauss_trunc",
                "two_power",
                "cubic_spline_ip",
                "akima_spline_ip",
            ]:
                integral, _ = quad(self.current_profile, 0, 1)
                self.indata.curtor = integral
            else:
                self.indata.curtor = self.current_profile(1.0)

        return boundary_RZFourier

    def get_input(self):
        """
        Generate a VMEC input file as a string.
        """
        self.set_indata()
        tmp = Path(os.getcwd()) / f".simsopt_vmec_jax_input_{os.getpid()}"
        try:
            vmec_jax_mod.write_indata(tmp, self.indata.raw)
            lines = tmp.read_text().splitlines()
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass
        header = [
            "&INDATA",
            "! This file created by simsopt on " + datetime.now().strftime("%B %d %Y, %H:%M:%S"),
        ]
        return "\n".join(header + lines[1:]) + "\n"

    def write_input(self, filename):
        """
        Write a VMEC input file.
        """
        input_namelist = self.get_input()
        if filename is not None:
            with open(filename, "w") as f:
                f.write(input_namelist)

    def run(self):
        """
        Run vmec_jax, if ``need_to_run_code`` is ``True``.
        """
        if not self.need_to_run_code:
            logger.info("run() called but no need to re-run vmec_jax.")
            return
        if not self.runnable:
            raise RuntimeError("Cannot run a VmecJax object that was initialized from a wout file.")

        logger.info("Preparing to run vmec_jax.")
        self.iter += 1
        base_filename = self.input_file + "_jax_{:06d}".format(self.iter)
        input_file = os.path.join(os.getcwd(), os.path.basename(base_filename))
        self.output_file = os.path.join(
            os.getcwd(),
            os.path.basename(base_filename).replace("input.", "wout_") + ".nc",
        )
        self.write_input(input_file)

        try:
            self._run = vmec_jax_mod.run_fixed_boundary(input_file, verbose=self.verbose)
            self._wout_jax = vmec_jax_mod.write_wout_from_fixed_boundary_run(
                self.output_file, self._run
            )
        except Exception as e:
            raise ObjectiveFailure(f"vmec_jax did not converge or failed: {e}") from e

        self.load_wout()

        if not self.keep_all_files and self.iter > 0:
            self.files_to_delete += [input_file, self.output_file]
        for filename in self.files_to_delete:
            try:
                os.remove(filename)
            except FileNotFoundError:
                logger.debug(f"Tried to delete the file {filename} but it was not found")
        self.files_to_delete = []
        self.need_to_run_code = False

    def load_wout(self):
        """
        Read the most recent ``wout`` file and cache vmec_jax data too.
        """
        ierr = 0
        logger.info(f"Attempting to read file {self.output_file}")
        if self._wout_jax is None:
            self._wout_jax = vmec_jax_mod.load_wout(self.output_file)

        try:
            with netcdf_file(self.output_file, mmap=False) as f:
                self.wout = Struct()
                for key, val in f.variables.items():
                    val2 = val[()]
                    val3 = val2.T if len(val2.shape) == 2 else val2
                    self.wout.__setattr__(key, val3)

                if hasattr(self.wout, "ier_flag") and self.wout.ier_flag != 0:
                    logger.info("vmec_jax did not succeed!")
                    raise ObjectiveFailure("vmec_jax did not succeed")

                if "lasym__logical__" in f.variables:
                    self.wout.lasym = f.variables["lasym__logical__"][()]
                elif hasattr(self._wout_jax, "lasym"):
                    self.wout.lasym = bool(self._wout_jax.lasym)
                if hasattr(self.wout, "volume_p"):
                    self.wout.volume = self.wout.volume_p
        except FileNotFoundError:
            self.wout = _copy_wout_data_to_struct(self._wout_jax)

        self.s_full_grid = np.linspace(0, 1, self.wout.ns)
        self.ds = self.s_full_grid[1] - self.s_full_grid[0]
        self.s_half_grid = self.s_full_grid[1:] - 0.5 * self.ds
        return ierr

    def B_cartesian(self, quadpoints_phi=None, quadpoints_theta=None,
                    range=Surface.RANGE_FULL_TORUS, nphi=None, ntheta=None,
                    use_wout_bsup=None):
        """
        Compute Cartesian magnetic field components on the boundary.
        """
        return B_cartesian_jax(
            self,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
            range=range,
            nphi=nphi,
            ntheta=ntheta,
            use_wout_bsup=use_wout_bsup,
        )

    def update_mpi(self, new_mpi):
        """
        Replace the stored MPI partition for API compatibility.
        """
        self.mpi = new_mpi

    def aspect(self):
        """
        Return the plasma aspect ratio.
        """
        self.run()
        return self.wout.aspect

    def volume(self):
        """
        Return the volume inside the VMEC last closed flux surface.
        """
        self.run()
        return self.wout.volume

    def iota_axis(self):
        """
        Return the rotational transform on axis.
        """
        self.run()
        return self.wout.iotaf[0]

    def iota_edge(self):
        """
        Return the rotational transform at the boundary.
        """
        self.run()
        return self.wout.iotaf[-1]

    def mean_iota(self):
        """
        Return the mean rotational transform.
        """
        self.run()
        return np.mean(self.wout.iotas[1:])

    def mean_shear(self):
        """
        Return an average magnetic shear.
        """
        self.run()
        poly = np.polynomial.Polynomial.fit(self.s_half_grid, self.wout.iotas[1:], deg=1)
        return poly.deriv()(0)

    def get_max_mn(self):
        """
        Return the largest boundary mode implied by the input data.
        """
        if not self.runnable:
            return (int(self.wout.mpol), int(self.wout.ntor))

        max_m = int(self.indata.mpol)
        max_n = int(self.indata.ntor)
        for key in ("RBC", "RBS", "ZBC", "ZBS"):
            for n, m in self.indata.indexed.get(key, {}):
                max_m = max(max_m, abs(int(m)))
                max_n = max(max_n, abs(int(n)))
        return (max_m, max_n)

    def __repr__(self):
        """
        Print the object in an informative way.
        """
        if self.runnable:
            return f"{self.name} (nfp={self.indata.nfp} mpol={self.indata.mpol} ntor={self.indata.ntor})"
        return f"{self.name} (nfp={self.wout.nfp} mpol={self.wout.mpol} ntor={self.wout.ntor})"

    def external_current(self):
        """
        Return the total electric current associated with external currents.
        """
        self.run()
        bvco = self.wout.bvco[-1] * 1.5 - self.wout.bvco[-2] * 0.5
        mu0 = 4 * np.pi * (1.0e-7)
        return 2 * np.pi * bvco / mu0

    def vacuum_well(self):
        """
        Compute the vacuum magnetic well.
        """
        self.run()
        dVds = 4 * np.pi * np.pi * np.abs(self.wout.gmnc[0, 1:])
        dVds_s0 = 1.5 * dVds[0] - 0.5 * dVds[1]
        dVds_s1 = 1.5 * dVds[-1] - 0.5 * dVds[-2]
        return (dVds_s0 - dVds_s1) / dVds_s0

    return_fn_map = {
        "aspect": aspect,
        "volume": volume,
        "iota_axis": iota_axis,
        "iota_edge": iota_edge,
        "mean_iota": mean_iota,
        "mean_shear": mean_shear,
        "vacuum_well": vacuum_well,
    }
