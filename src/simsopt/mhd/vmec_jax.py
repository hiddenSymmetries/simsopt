# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""
JAX-backed VMEC wrapper for SIMSOPT workflows.

This module provides a minimal interface that mirrors the parts of
`simsopt.mhd.vmec.Vmec` used by the QH/Q A optimization examples, but
implemented on top of `vmec_jax` with implicit differentiation.
"""

from __future__ import annotations

from dataclasses import replace
import re
from typing import Sequence
from types import SimpleNamespace

import numpy as np

try:
    import vmec_jax as vj
    from vmec_jax._compat import jax, jnp, has_jax
    from vmec_jax.boundary import boundary_from_input_convention, boundary_input_from_indata
except Exception as exc:  # pragma: no cover - optional dependency
    vj = None
    jax = None
    jnp = None
    has_jax = None
    boundary_from_input_convention = None
    boundary_input_from_indata = None
    _import_error = exc
else:
    _import_error = None


__all__ = ["VmecJax", "JaxBoundary"]


def _clone_state(state):
    """Return a JAX-friendly copy of a VMEC-JAX state object."""
    return vj.VMECState(
        layout=state.layout,
        Rcos=jnp.asarray(state.Rcos),
        Rsin=jnp.asarray(state.Rsin),
        Zcos=jnp.asarray(state.Zcos),
        Zsin=jnp.asarray(state.Zsin),
        Lcos=jnp.asarray(state.Lcos),
        Lsin=jnp.asarray(state.Lsin),
    )


def _require_vmec_jax():
    """Validate that vmec_jax and JAX are available."""
    if vj is None or has_jax is None or not has_jax():
        raise ImportError(
            "VmecJax requires vmec_jax and JAX. "
            "Install vmec_jax (and jax/jaxlib) before importing VmecJax."
        ) from _import_error


def _format_name(kind: str, m: int, n: int) -> str:
    """Format a boundary coefficient name in SurfaceRZFourier style."""
    return f"{kind}({m},{n})"


def _parse_name(name: str) -> tuple[str, int, int]:
    """Parse a SurfaceRZFourier-style coefficient name into components."""
    m = re.match(r"([a-zA-Z]+)\(([-\d]+),([-\d]+)\)", name.replace(" ", ""))
    if not m:
        raise ValueError(f"Unrecognized boundary dof name: {name}")
    return m.group(1).lower(), int(m.group(2)), int(m.group(3))


def _as_list_like(value):
    """Normalize scalar or array-like VMEC input values into a Python list."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return list(np.asarray(value).reshape(-1).tolist())
    if isinstance(value, (int, float, np.integer, np.floating)):
        return [value]
    try:
        return list(value)
    except Exception:
        return None


def _last_input_value(indata, *, array_key: str, scalar_key: str, default, cast):
    """Return the last staged VMEC input value, falling back to the scalar key."""
    values = _as_list_like(indata.get(array_key, None))
    if values:
        try:
            return cast(values[-1])
        except Exception:
            pass
    getter_name = "get_int" if cast is int else "get_float"
    getter = getattr(indata, getter_name, None)
    if callable(getter):
        return cast(getter(scalar_key, default))
    return cast(indata.get(scalar_key, default))


def _surface_rzfourier_mode_is_valid(kind: str, m: int, n: int, *, lasym: bool) -> bool:
    """Mirror the SurfaceRZFourier coefficient set used by SIMSOPT.

    VMEC-JAX stores all four Fourier coefficient families on the boundary
    object, but SIMSOPT's SurfaceRZFourier only exposes the structurally
    valid subset:
    - no negative-n modes for m=0,
    - no sine coefficients at (m=0, n=0),
    - `lasym=False` surfaces expose only rc/zs.
    """
    kind = str(kind).lower()
    if m == 0 and n < 0:
        return False
    if (m == 0) and (n == 0) and kind in ("rs", "zs"):
        return False
    if not bool(lasym) and kind in ("rs", "zc"):
        return False
    return kind in ("rc", "rs", "zc", "zs")


def _surface_rzfourier_spec_sort_key(spec, *, lasym: bool):
    """Sort coefficients in the same order used by SurfaceRZFourier."""
    kind_order = ("rc", "rs", "zc", "zs") if bool(lasym) else ("rc", "zs")
    kind_rank = kind_order.index(str(spec.kind).lower())
    m = int(spec.m)
    n = int(spec.n)
    if m == 0:
        n_key = n
    else:
        n_key = n + 100000
    return (kind_rank, m, n_key)


def _vmec_wrout_nyquist_cos_coeffs_jax(*, f, modes, trig):
    """Return wrout-style Nyquist cosine coefficients using JAX arrays.

    This mirrors ``vmec_jax.wout._vmec_wrout_nyquist_cos_coeffs`` closely but
    avoids NumPy conversions so it can be used inside traced optimization
    loops.
    """
    f = jnp.asarray(f)
    if f.ndim != 3:
        raise ValueError(f"Expected f with shape (ns, ntheta, nzeta), got {f.shape}")

    m = jnp.asarray(modes.m, dtype=jnp.int32)
    n = jnp.asarray(modes.n, dtype=jnp.int32)
    if int(m.shape[0]) == 0:
        return jnp.zeros((int(f.shape[0]), 0), dtype=f.dtype)

    nt2 = int(trig.ntheta2)
    if int(f.shape[1]) < nt2:
        raise ValueError("Input theta grid is smaller than VMEC ntheta2")
    f = f[:, :nt2, :]

    cosmui = jnp.asarray(trig.cosmui, dtype=f.dtype)[:nt2, :]
    sinmui = jnp.asarray(trig.sinmui, dtype=f.dtype)[:nt2, :]
    cosnv = jnp.asarray(trig.cosnv, dtype=f.dtype)
    sinnv = jnp.asarray(trig.sinnv, dtype=f.dtype)

    mnyq = int(cosmui.shape[1] - 1)
    if mnyq > 0:
        cosmui = cosmui.at[:, mnyq].multiply(jnp.asarray(0.5, dtype=f.dtype))
    nnyq = int(cosnv.shape[1] - 1)
    if nnyq > 0:
        cosnv = cosnv.at[:, nnyq].multiply(jnp.asarray(0.5, dtype=f.dtype))

    f_theta_cos = jnp.einsum("sik,im->smk", f, cosmui, optimize=True)
    f_theta_sin = jnp.einsum("sik,im->smk", f, sinmui, optimize=True)
    cos_zeta = jnp.einsum("smk,kn->smn", f_theta_cos, cosnv, optimize=True)
    sin_zeta = jnp.einsum("smk,kn->smn", f_theta_sin, sinnv, optimize=True)

    n_abs = jnp.abs(n)
    sgn = jnp.where(n < 0, -1.0, 1.0).astype(f.dtype)
    coeff = cos_zeta[:, m, n_abs] + sgn[None, :] * sin_zeta[:, m, n_abs]

    mscale = jnp.asarray(trig.mscale, dtype=f.dtype)
    nscale = jnp.asarray(trig.nscale, dtype=f.dtype)
    dmult = mscale[m] * nscale[n_abs] * jnp.asarray(0.5 / float(getattr(trig, "r0scale", 1.0)) ** 2, dtype=f.dtype)
    dmult = jnp.where((m == 0) | (n == 0), 2.0 * dmult, dmult)
    return coeff * dmult[None, :]


class JaxBoundary:
    """Boundary DOF wrapper mirroring the SurfaceRZFourier interface used by examples."""

    def __init__(
        self,
        boundary,
        boundary_input,
        modes,
        *,
        lasym: bool,
        include: Sequence[str] = ("rc", "zs", "rs", "zc"),
    ) -> None:
        """Wrap vmec_jax boundary coefficients with a SurfaceRZFourier-like API."""
        _require_vmec_jax()
        self._boundary0 = boundary
        self._boundary_input0 = boundary_input
        self._modes = modes
        self._lasym = bool(lasym)
        specs = vj.optimization.boundary_param_specs(
            boundary_input,
            modes,
            max_mode=None,
            min_coeff=0.0,
            include=include,
            fix=(),
            include_axis=True,
        )
        self._specs = [
            spec
            for spec in specs
            if _surface_rzfourier_mode_is_valid(spec.kind, int(spec.m), int(spec.n), lasym=bool(lasym))
        ]
        self._specs.sort(key=lambda spec: _surface_rzfourier_spec_sort_key(spec, lasym=bool(lasym)))
        self._names = [_format_name(s.kind, s.m, s.n) for s in self._specs]
        self._base = np.array([self._get_coeff(boundary_input, s) for s in self._specs], dtype=float)
        self._x = self._base.copy()
        self._free = np.ones_like(self._x, dtype=bool)

    def _get_coeff(self, boundary, spec):
        """Read one Fourier coefficient from a vmec_jax boundary object."""
        if spec.kind == "rc":
            return float(boundary.R_cos[spec.index])
        if spec.kind == "rs":
            return float(boundary.R_sin[spec.index])
        if spec.kind == "zc":
            return float(boundary.Z_cos[spec.index])
        if spec.kind == "zs":
            return float(boundary.Z_sin[spec.index])
        raise ValueError(f"Unknown boundary coeff kind: {spec.kind}")

    @property
    def dof_names(self) -> list[str]:
        """Return the ordered coefficient names exposed to SIMSOPT."""
        return list(self._names)

    @property
    def x(self) -> np.ndarray:
        """Return the full coefficient vector in input convention."""
        return self._x.copy()

    @property
    def free(self) -> np.ndarray:
        """Return a boolean mask marking active optimization variables."""
        return self._free.copy()

    def fix_all(self) -> None:
        """Mark every boundary coefficient as fixed."""
        self._free[:] = False

    def unfix_all(self) -> None:
        """Mark every boundary coefficient as free."""
        self._free[:] = True

    def fix(self, name: str) -> None:
        """Fix a single coefficient by name."""
        idx = self._names.index(name)
        self._free[idx] = False

    def unfix(self, name: str) -> None:
        """Unfix a single coefficient by name."""
        idx = self._names.index(name)
        self._free[idx] = True

    def fixed_range(
        self,
        *,
        mmin: int,
        mmax: int,
        nmin: int,
        nmax: int,
        fixed: bool = True,
    ) -> None:
        """Apply a free/fixed setting over a rectangular Fourier mode window."""
        for i, name in enumerate(self._names):
            kind, m, n = _parse_name(name)
            if mmin <= m <= mmax and nmin <= n <= nmax:
                self._free[i] = not fixed

    def set_full_params(self, x_full: np.ndarray) -> None:
        """Replace the complete coefficient vector."""
        if x_full.shape[0] != self._x.shape[0]:
            raise ValueError("x_full has wrong length")
        self._x[:] = np.asarray(x_full, dtype=float)

    def get_free_params(self) -> np.ndarray:
        """Return only the currently free parameters."""
        return self._x[self._free]

    def set_free_params(self, x_free: np.ndarray) -> None:
        """Update the active optimization variables in-place."""
        if x_free.shape[0] != int(np.sum(self._free)):
            raise ValueError("x_free has wrong length")
        self._x[self._free] = np.asarray(x_free, dtype=float)

    def expand_free(self, x_free: jnp.ndarray) -> jnp.ndarray:
        """Expand free parameters into the full coefficient vector."""
        x_full = jnp.asarray(self._x)
        if int(np.sum(self._free)) == 0:
            return x_full
        free_idx = jnp.asarray(np.nonzero(self._free)[0], dtype=jnp.int32)
        return x_full.at[free_idx].set(jnp.asarray(x_free))

    def apply_params(self, x_full: jnp.ndarray):
        """Apply a full parameter vector and return vmec_jax solver coefficients."""
        delta = jnp.asarray(x_full) - jnp.asarray(self._base)
        boundary_input = vj.optimization.apply_boundary_params(self._boundary_input0, self._specs, delta)
        return boundary_from_input_convention(
            boundary_input,
            self._modes,
            lasym=self._lasym,
            apply_m1_constraint=False,
        )


class _InDataProxy:
    """Proxy exposing the small subset of VMEC indata used by SIMSOPT."""

    def __init__(self, parent, cfg, indata):
        """Create a mutable view that keeps the wrapper config in sync."""
        self._parent = parent
        self._cfg = cfg
        self._indata = indata
        self._mpol = int(cfg.mpol)
        self._ntor = int(cfg.ntor)

    @property
    def mpol(self) -> int:
        """Return the active poloidal truncation used by the wrapper."""
        return self._mpol

    @mpol.setter
    def mpol(self, val: int) -> None:
        """Update the active poloidal truncation and rebuild dependent state."""
        self._mpol = int(val)
        self._parent._update_cfg(mpol=self._mpol, ntor=self._ntor)

    @property
    def ntor(self) -> int:
        """Return the active toroidal truncation used by the wrapper."""
        return self._ntor

    @ntor.setter
    def ntor(self, val: int) -> None:
        """Update the active toroidal truncation and rebuild dependent state."""
        self._ntor = int(val)
        self._parent._update_cfg(mpol=self._mpol, ntor=self._ntor)

    def get_int(self, key: str, default=None):
        """Delegate integer lookups to the underlying indata object."""
        return self._indata.get_int(key, default)

    def get_float(self, key: str, default=None):
        """Delegate float lookups to the underlying indata object."""
        return self._indata.get_float(key, default)

    def get_bool(self, key: str, default=None):
        """Delegate boolean lookups to the underlying indata object."""
        return self._indata.get_bool(key, default)

    def get(self, key: str, default=None):
        """Delegate generic lookups to the underlying indata object."""
        return self._indata.get(key, default)


class VmecJax:
    """
    JAX-backed fixed-boundary VMEC wrapper.

    This class exposes a minimal subset of the Fortran VMEC interface used
    in the SIMSOPT examples. It is intended for JAX autodiff workflows.
    """

    def __init__(self, filename: str, *, verbose: bool = False, warm_start_iters: int = 0) -> None:
        """Load a VMEC input file and initialize the JAX-backed wrapper."""
        _require_vmec_jax()
        try:
            cache_helper = getattr(getattr(vj, "driver", None), "_maybe_enable_compilation_cache", None)
            if callable(cache_helper):
                cache_helper()
        except Exception:
            pass
        self.filename = filename
        self.verbose = bool(verbose)
        self._warm_start_iters = max(0, int(warm_start_iters))
        self._load_config()

    def _load_config(self) -> None:
        """Load the VMEC input file and initialize wrapper defaults."""
        cfg, indata = vj.load_config(self.filename)
        self._cfg = cfg
        self._indata_raw = indata
        self.indata = _InDataProxy(self, cfg, indata)
        self._static = None
        self._boundary = None
        self._context = None
        self._phipf = None
        self._chipf = None
        self._lamscale = None
        self._pressure = None
        self._signgs = None
        self._static_dirty = True
        self._context_dirty = True
        self._context_seed_guess = None
        self._solver = "vmec2000"
        self._max_iter = _last_input_value(
            self._indata_raw,
            array_key="NITER_ARRAY",
            scalar_key="NITER",
            default=50,
            cast=int,
        )
        self._grad_tol = _last_input_value(
            self._indata_raw,
            array_key="FTOL_ARRAY",
            scalar_key="FTOL",
            default=1e-13,
            cast=float,
        )
        self._step_size = 5e-3
        self._step_size_override = None
        self._history_size = 8
        self._jacobian_penalty = 1e3
        self._preconditioner_override = None
        self._precond_exponent = 1.0
        self._precond_radial_alpha_override = None
        self._implicit_cg_max_iter = 60
        self._implicit_cg_tol = 1e-10
        self._implicit_damping = 1e-5
        self._implicit_converge_tol = None
        self._implicit_zero_unconverged = False
        self._residual_adjoint_mode = "auto"
        self._residual_tangent_mode = "opaque"
        self._stateless_evaluations = False
        self._reset_caches()

    def _reset_caches(self, *, reset_warm_start: bool = False) -> None:
        """Clear cached solve/wout results and optionally reset the warm-start seed."""
        self._cached_x = None
        self._cached_state = None
        self._cached_wout = None
        self._cached_run = None
        if reset_warm_start and self._context is not None and self._context_seed_guess is not None:
            self._context.st_guess = _clone_state(self._context_seed_guess)

    def _invalidate_runtime(self) -> None:
        """Invalidate stateful runtime caches that depend on static setup."""
        self._context_dirty = True
        self._context_seed_guess = None
        self._reset_caches()

    def _update_cfg(self, *, mpol: int, ntor: int) -> None:
        """Change the active resolution and invalidate dependent state."""
        self._cfg = replace(self._cfg, mpol=int(mpol), ntor=int(ntor))
        self._static_dirty = True
        self._invalidate_runtime()

    def _ensure_static(self) -> None:
        """Build static VMEC-JAX structures and the boundary wrapper lazily."""
        if not self._static_dirty and self._static is not None and self._boundary is not None:
            return
        old_boundary = self._boundary
        self._static = vj.build_static(self._cfg)
        # Keep the wrapper boundary in the external/input convention. vmec_jax's
        # initial-guess and residual paths apply the VMEC m=1 constraint
        # internally; pre-constraining here mixes the m=1, n=+/-1 pair twice and
        # corrupts the low-order geometry before the solve starts.
        boundary_input0 = boundary_input_from_indata(self._indata_raw, self._static.modes)
        boundary0 = vj.boundary_from_indata(self._indata_raw, self._static.modes, apply_m1_constraint=False)
        self._boundary = JaxBoundary(boundary0, boundary_input0, self._static.modes, lasym=bool(self._cfg.lasym))
        if old_boundary is not None:
            old_names = {name: i for i, name in enumerate(old_boundary.dof_names)}
            x_full = self._boundary.x
            free = self._boundary.free
            for i, name in enumerate(self._boundary.dof_names):
                j = old_names.get(name)
                if j is None:
                    continue
                x_full[i] = old_boundary.x[j]
                free[i] = old_boundary.free[j]
            self._boundary.set_full_params(x_full)
            self._boundary._free[:] = free
        self._static_dirty = False
        self._invalidate_runtime()

    def _ensure_context(self) -> None:
        """Build the warm-start context and profile data for subsequent solves."""
        self._ensure_static()
        if not self._context_dirty and self._context is not None:
            return

        boundary0 = self._boundary._boundary0
        st_guess = vj.initial_guess_from_boundary(self._static, boundary0, self._indata_raw, vmec_project=True)
        geom = vj.eval_geom(st_guess, self._static)
        signgs = vj.signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1)
        if self._warm_start_iters > 0:
            try:
                from vmec_jax.solve import solve_fixed_boundary_lbfgs_vmec_residual

                res = solve_fixed_boundary_lbfgs_vmec_residual(
                    st_guess,
                    self._static,
                    indata=self._indata_raw,
                    signgs=signgs,
                    max_iter=int(self._warm_start_iters),
                    verbose=False,
                )
                st_guess = res.state
                geom = vj.eval_geom(st_guess, self._static)
                signgs = vj.signgs_from_sqrtg(np.asarray(geom.sqrtg), axis_index=1)
            except Exception:
                pass

        flux = vj.flux_profiles_from_indata(self._indata_raw, self._static.s, signgs=signgs)
        profiles = vj.eval_profiles(self._indata_raw, self._static.s)
        pressure = jnp.asarray(profiles.get("pressure", jnp.zeros_like(self._static.s)))
        self._context = SimpleNamespace(
            st_guess=st_guess,
            signgs=signgs,
            flux=flux,
            pressure=pressure,
            booz_inputs=None,
        )
        self._context_seed_guess = _clone_state(st_guess)
        self._phipf = jnp.asarray(flux.phipf)
        self._chipf = jnp.asarray(flux.chipf)
        self._lamscale = jnp.asarray(flux.lamscale)
        self._pressure = jnp.asarray(pressure)
        self._signgs = int(signgs)
        self._context_dirty = False

    def set_solver_options(
        self,
        *,
        max_iter: int | None = None,
        grad_tol: float | None = None,
        solver: str | None = None,
        warm_start_iters: int | None = None,
        step_size: float | None = None,
        history_size: int | None = None,
        jacobian_penalty: float | None = None,
        preconditioner: str | None = None,
        precond_exponent: float | None = None,
        precond_radial_alpha: float | None = None,
        implicit_cg_max_iter: int | None = None,
        implicit_cg_tol: float | None = None,
        implicit_damping: float | None = None,
        implicit_converge_tol: float | None = None,
        implicit_zero_unconverged: bool | None = None,
        residual_adjoint_mode: str | None = None,
        residual_tangent_mode: str | None = None,
        stateless_evaluations: bool | None = None,
    ) -> None:
        """Update VMEC-JAX solver controls used by this wrapper."""
        if max_iter is not None:
            max_iter = int(max_iter)
            if max_iter != self._max_iter:
                self._max_iter = max_iter
                self._reset_caches(reset_warm_start=True)
        if grad_tol is not None:
            grad_tol = float(grad_tol)
            if grad_tol != self._grad_tol:
                self._grad_tol = grad_tol
                self._reset_caches(reset_warm_start=True)
        if solver is not None:
            solver = str(solver).strip().lower()
            if solver != self._solver:
                self._solver = solver
                self._reset_caches(reset_warm_start=True)
        if warm_start_iters is not None:
            warm_start_iters = int(warm_start_iters)
            if warm_start_iters != self._warm_start_iters:
                self._warm_start_iters = warm_start_iters
                self._invalidate_runtime()
        if step_size is not None:
            step_size = float(step_size)
            if step_size != self._step_size or step_size != self._step_size_override:
                self._step_size = step_size
                self._step_size_override = step_size
                self._reset_caches(reset_warm_start=True)
        if history_size is not None:
            history_size = int(history_size)
            if history_size != self._history_size:
                self._history_size = history_size
                self._reset_caches(reset_warm_start=True)
        if jacobian_penalty is not None:
            jacobian_penalty = float(jacobian_penalty)
            if jacobian_penalty != self._jacobian_penalty:
                self._jacobian_penalty = jacobian_penalty
                self._reset_caches(reset_warm_start=True)
        if preconditioner is not None:
            preconditioner = str(preconditioner).strip().lower()
            if preconditioner != self._preconditioner_override:
                self._preconditioner_override = preconditioner
                self._reset_caches(reset_warm_start=True)
        if precond_exponent is not None:
            precond_exponent = float(precond_exponent)
            if precond_exponent != self._precond_exponent:
                self._precond_exponent = precond_exponent
                self._reset_caches(reset_warm_start=True)
        if precond_radial_alpha is not None:
            precond_radial_alpha = float(precond_radial_alpha)
            if precond_radial_alpha != self._precond_radial_alpha_override:
                self._precond_radial_alpha_override = precond_radial_alpha
                self._reset_caches(reset_warm_start=True)
        if implicit_cg_max_iter is not None:
            implicit_cg_max_iter = int(implicit_cg_max_iter)
            if implicit_cg_max_iter != self._implicit_cg_max_iter:
                self._implicit_cg_max_iter = implicit_cg_max_iter
                self._reset_caches(reset_warm_start=True)
        if implicit_cg_tol is not None:
            implicit_cg_tol = float(implicit_cg_tol)
            if implicit_cg_tol != self._implicit_cg_tol:
                self._implicit_cg_tol = implicit_cg_tol
                self._reset_caches(reset_warm_start=True)
        if implicit_damping is not None:
            implicit_damping = float(implicit_damping)
            if implicit_damping != self._implicit_damping:
                self._implicit_damping = implicit_damping
                self._reset_caches(reset_warm_start=True)
        if implicit_converge_tol is not None:
            implicit_converge_tol = float(implicit_converge_tol)
            if implicit_converge_tol != self._implicit_converge_tol:
                self._implicit_converge_tol = implicit_converge_tol
                self._reset_caches(reset_warm_start=True)
        if implicit_zero_unconverged is not None:
            implicit_zero_unconverged = bool(implicit_zero_unconverged)
            if implicit_zero_unconverged != self._implicit_zero_unconverged:
                self._implicit_zero_unconverged = implicit_zero_unconverged
                self._reset_caches(reset_warm_start=True)
        if residual_adjoint_mode is not None:
            residual_adjoint_mode = str(residual_adjoint_mode).strip().lower()
            if residual_adjoint_mode != self._residual_adjoint_mode:
                self._residual_adjoint_mode = residual_adjoint_mode
                self._reset_caches(reset_warm_start=True)
        if residual_tangent_mode is not None:
            residual_tangent_mode = str(residual_tangent_mode).strip().lower()
            if residual_tangent_mode != self._residual_tangent_mode:
                self._residual_tangent_mode = residual_tangent_mode
                self._reset_caches(reset_warm_start=True)
        if stateless_evaluations is not None:
            stateless_evaluations = bool(stateless_evaluations)
            if stateless_evaluations != self._stateless_evaluations:
                self._stateless_evaluations = stateless_evaluations
                self._reset_caches(reset_warm_start=True)

    def use_residual_autodiff_defaults(
        self,
        *,
        outer_method: str | None = None,
        residual_adjoint_mode: str = "lineax",
        stateless_evaluations: bool = False,
    ) -> None:
        """Apply wrapper defaults that worked best for the fixed-resolution JAX examples."""
        method = None if outer_method is None else str(outer_method).strip().lower()
        tangent_method = "linearize" if method in ("gauss_newton", "trust_region", "levenberg_marquardt", "scipy") else "opaque"
        self.set_solver_options(
            solver="vmec2000",
            residual_adjoint_mode=residual_adjoint_mode,
            residual_tangent_mode=tangent_method,
            stateless_evaluations=stateless_evaluations,
        )

    def _x_cache_key(self, x_free) -> tuple[float, ...]:
        """Return a hashable cache key for concrete parameter vectors."""
        try:
            arr = np.asarray(x_free, dtype=float).reshape(-1)
        except Exception:
            return None
        return tuple(arr.tolist())

    def _solve_state(self, x_free: jnp.ndarray):
        """Solve the fixed-boundary equilibrium and return the VMEC-JAX state."""
        from vmec_jax.implicit import (
            ImplicitFixedBoundaryOptions,
            solve_fixed_boundary_state_implicit,
            solve_fixed_boundary_state_implicit_vmec_residual,
        )
        from vmec_jax.solve import solve_fixed_boundary_residual_iter
        self._ensure_context()
        x_key = self._x_cache_key(x_free)
        use_stateful_cache = x_key is not None and not self._stateless_evaluations
        if use_stateful_cache and self._cached_x == x_key and self._cached_state is not None:
            return self._cached_state

        seed_state = self._context.st_guess
        if self._stateless_evaluations and self._context_seed_guess is not None:
            seed_state = _clone_state(self._context_seed_guess)

        x_full = self.boundary.expand_free(x_free)
        boundary = self.boundary.apply_params(x_full)

        # Use edge boundary coefficients for fixed-boundary enforcement.
        edge_Rcos = jnp.asarray(boundary.R_cos)
        edge_Rsin = jnp.asarray(boundary.R_sin)
        edge_Zcos = jnp.asarray(boundary.Z_cos)
        edge_Zsin = jnp.asarray(boundary.Z_sin)

        implicit_opts = ImplicitFixedBoundaryOptions(
            cg_max_iter=int(self._implicit_cg_max_iter),
            cg_tol=float(self._implicit_cg_tol),
            damping=float(self._implicit_damping),
            residual_adjoint_mode=str(self._residual_adjoint_mode),
            residual_tangent_mode=str(self._residual_tangent_mode),
        )
        state0_host = vj.VMECState(
            layout=seed_state.layout,
            Rcos=np.asarray(seed_state.Rcos),
            Rsin=np.asarray(seed_state.Rsin),
            Zcos=np.asarray(seed_state.Zcos),
            Zsin=np.asarray(seed_state.Zsin),
            Lcos=np.asarray(seed_state.Lcos),
            Lsin=np.asarray(seed_state.Lsin),
        )
        residual_step_size = (
            float(self._step_size_override)
            if self._step_size_override is not None
            else float(self._indata_raw.get_float("DELT", 1.0))
        )
        solver = self._solver
        use_residual_implicit_wrapper = (
            solver in ("residual", "vmec2000")
            and (
                str(self._residual_adjoint_mode) != "auto"
                or str(self._residual_tangent_mode) != "opaque"
            )
        )

        def _solve(solver: str):
            if solver in ("residual", "vmec2000"):
                if x_key is None or use_residual_implicit_wrapper:
                    return solve_fixed_boundary_state_implicit_vmec_residual(
                        seed_state,
                        self._static,
                        indata=self._indata_raw,
                        signgs=self._signgs,
                        state0_host=state0_host,
                        max_iter=int(self._max_iter),
                        step_size=residual_step_size,
                        ftol=float(self._grad_tol),
                        implicit=implicit_opts,
                        edge_Rcos=edge_Rcos,
                        edge_Rsin=edge_Rsin,
                        edge_Zcos=edge_Zcos,
                        edge_Zsin=edge_Zsin,
                    )
                st0 = vj.initial_guess_from_boundary(self._static, boundary, self._indata_raw, vmec_project=True)
                geom0 = vj.eval_geom(st0, self._static)
                signgs0 = vj.signgs_from_sqrtg(np.asarray(geom0.sqrtg), axis_index=1)
                res = solve_fixed_boundary_residual_iter(
                    st0,
                    self._static,
                    indata=self._indata_raw,
                    signgs=int(signgs0),
                    ftol=float(self._grad_tol),
                    max_iter=int(self._max_iter),
                    step_size=residual_step_size,
                    vmec2000_control=True,
                    reference_mode=False,
                    backtracking=True,
                    limit_dt_from_force=True,
                    limit_update_rms=True,
                    verbose=False,
                    verbose_vmec2000_table=False,
                    jit_forces="auto",
                    use_scan=False,
                )
                return res.state
            preconditioner = self._preconditioner_override
            if preconditioner is None:
                preconditioner = "mode_diag+radial_tridi" if solver == "lbfgs" else "none"
            precond_radial_alpha = self._precond_radial_alpha_override
            if precond_radial_alpha is None:
                precond_radial_alpha = 0.5 if solver == "lbfgs" else 0.0
            return solve_fixed_boundary_state_implicit(
                seed_state,
                self._static,
                phipf=self._phipf,
                chipf=self._chipf,
                signgs=self._signgs,
                lamscale=self._lamscale,
                pressure=self._pressure,
                gamma=float(self._indata_raw.get_float("GAMMA", 0.0)),
                jacobian_penalty=float(self._jacobian_penalty),
                solver=solver,
                max_iter=int(self._max_iter),
                step_size=float(self._step_size),
                history_size=int(self._history_size),
                grad_tol=float(self._grad_tol),
                preconditioner=preconditioner,
                precond_exponent=float(self._precond_exponent),
                precond_radial_alpha=precond_radial_alpha,
                implicit=implicit_opts,
                implicit_converge_tol=(
                    float(self._implicit_converge_tol)
                    if self._implicit_converge_tol is not None
                    else float(self._grad_tol)
                ),
                implicit_zero_unconverged=bool(self._implicit_zero_unconverged),
                edge_Rcos=edge_Rcos,
                edge_Rsin=edge_Rsin,
                edge_Zcos=edge_Zcos,
                edge_Zsin=edge_Zsin,
            )

        if solver == "initial":
            return vj.initial_guess_from_boundary(self._static, boundary, self._indata_raw, vmec_project=False)
        try:
            state = _solve(solver)
        except ValueError as exc:
            msg = str(exc).lower()
            if "invalid jacobian" in msg and solver != "gd":
                state = _solve("gd")
            else:
                raise
        # Reuse the last converged state as the next solve's starting point for
        # ordinary concrete evaluations. Avoid mutating the warm-start state
        # during traced/JAX-transformed solves, which would make subsequent
        # objective calls history-dependent and can leak traced values into the
        # Python wrapper state.
        if use_stateful_cache:
            self._context.st_guess = state
        if use_stateful_cache:
            self._cached_x = x_key
            self._cached_state = state
            self._cached_wout = None
            self._cached_run = None
        return state

    def get_run(self, x_free):
        """Return a :class:`vmec_jax.driver.FixedBoundaryRun` for the current point."""
        state = self._solve_state(x_free)
        if self._cached_run is not None and self._cached_state is state:
            return self._cached_run
        self._ensure_context()
        run = vj.FixedBoundaryRun(
            cfg=self._cfg,
            indata=self._indata_raw,
            static=self.get_static(),
            state=state,
            result=None,
            flux=self._context.flux,
            profiles={},
            signgs=self._signgs,
        )
        if self._x_cache_key(x_free) is not None:
            self._cached_run = run
        return run

    def get_wout(self, x_free):
        """Return a VMEC-style ``wout`` object for the current point."""
        run = self.get_run(x_free)
        if self._cached_wout is not None and self._cached_run is run:
            return self._cached_wout
        wout = vj.wout_from_fixed_boundary_run(run, include_fsq=False, fast_bcovar=True)
        if self._x_cache_key(x_free) is not None:
            self._cached_wout = wout
        return wout

    def get_wout_from_state(self, state):
        """Return a VMEC-style ``wout`` object for an already solved state."""
        self._ensure_context()
        run = vj.FixedBoundaryRun(
            cfg=self._cfg,
            indata=self._indata_raw,
            static=self.get_static(),
            state=state,
            result=None,
            flux=self._context.flux,
            profiles={},
            signgs=self._signgs,
        )
        return vj.wout_from_fixed_boundary_run(run, include_fsq=False, fast_bcovar=True)

    def quasisymmetry_diagnostics_from_state_jax(self, state):
        """Build the VMEC-only QS channels directly from a solved state.

        This helper mirrors the subset of ``wout`` synthesis needed by
        :class:`simsopt.mhd.QuasisymmetryRatioResidual`, but keeps the data on
        the JAX side instead of materializing a Python ``wout`` object. The
        returned namespace contains the Nyquist Fourier coefficients and 1D
        profiles needed to evaluate the VMEC-only quasisymmetry residual.
        """
        self._ensure_context()

        from vmec_jax.booz_input import _filter_bsubuv_jxbforce_parity_jax
        from vmec_jax.driver import _final_flux_profiles_from_state
        from vmec_jax.energy import _iotaf_from_iotas
        from vmec_jax.integrals import cumrect_s_halfmesh
        from vmec_jax.modes import nyquist_mode_table_from_grid
        from vmec_jax.vmec_bcovar import vmec_bcovar_half_mesh_from_wout
        from vmec_jax.vmec_lforbal import currents_from_bcovar
        from vmec_jax.vmec_tomnsp import vmec_trig_tables

        static = self.get_static()
        context = self.get_context()
        flux, prof = _final_flux_profiles_from_state(
            indata=self._indata_raw,
            static_in=static,
            state=state,
            signgs=self._signgs,
            flux_local=context.flux,
            prof_local={"pressure": context.pressure},
            pressure_local=context.pressure,
        )

        s_full = jnp.asarray(static.s)
        pres = jnp.asarray(prof.get("pressure", context.pressure))
        if int(pres.shape[0]) > 0:
            pres = pres.at[0].set(0.0)

        iotas = prof.get("iota", None)
        if iotas is None:
            phips = jnp.asarray(flux.phips)
            chipf = jnp.asarray(flux.chipf)
            chips = jnp.concatenate([chipf[:1], 0.5 * (chipf[1:] + chipf[:-1])], axis=0)
            safe_phips = jnp.where(phips != 0.0, phips, 1.0)
            iotas = jnp.where(phips != 0.0, chips / safe_phips, 0.0)
        iotas = jnp.asarray(iotas)
        if int(iotas.shape[0]) > 0:
            iotas = iotas.at[0].set(0.0)
        iotaf = jnp.asarray(prof.get("iotaf", _iotaf_from_iotas(iotas, lrfp=bool(self._indata_raw.get_bool("LRFP", False)))))

        wout_like = SimpleNamespace(
            phipf=jnp.asarray(flux.phipf),
            chipf=jnp.asarray(flux.chipf),
            phips=jnp.asarray(flux.phips),
            iotas=iotas,
            iotaf=iotaf,
            nfp=int(self._cfg.nfp),
            mpol=int(self._cfg.mpol),
            ntor=int(self._cfg.ntor),
            lasym=bool(self._cfg.lasym),
            signgs=int(self._signgs),
            ncurr=int(self._indata_raw.get_int("NCURR", 0)),
            lcurrent=bool(int(self._indata_raw.get_int("NCURR", 0)) == 1),
            flux_is_internal=True,
        )

        nyq_modes = nyquist_mode_table_from_grid(
            mpol=int(self._cfg.mpol),
            ntor=int(self._cfg.ntor),
            ntheta=int(self._cfg.ntheta),
            nzeta=int(self._cfg.nzeta),
        )
        mmax_nyq = int(np.max(np.asarray(nyq_modes.m))) if int(nyq_modes.K) > 0 else 0
        nmax_nyq = int(np.max(np.abs(np.asarray(nyq_modes.n)))) if int(nyq_modes.K) > 0 else 0
        trig = vmec_trig_tables(
            ntheta=int(self._cfg.ntheta),
            nzeta=int(self._cfg.nzeta),
            nfp=int(self._cfg.nfp),
            mmax=max(int(self._cfg.mpol) - 1, mmax_nyq),
            nmax=max(int(self._cfg.ntor), nmax_nyq),
            lasym=bool(self._cfg.lasym),
            dtype=jnp.asarray(state.Rcos).dtype,
        )

        bc = vmec_bcovar_half_mesh_from_wout(
            state=state,
            static=static,
            wout=wout_like,
            pres=pres,
            use_wout_bsup=False,
            use_wout_bsub_for_lambda=False,
            use_wout_bmag_for_bsq=False,
            use_vmec_synthesis=True,
            trig=trig,
        )
        buco, bvco, _, _ = currents_from_bcovar(bc=bc, trig=trig, wout=wout_like, s=s_full)

        bsq = jnp.asarray(bc.bsq)
        # Keep |B| differentiable where the pressure-corrected magnitude lands
        # exactly on zero. This only changes exact-zero values by machine tiny,
        # but avoids undefined reverse-mode sensitivities in the QS diagnostic.
        bmod_sq = jnp.maximum(
            2.0 * (bsq - pres[:, None, None]),
            jnp.asarray(jnp.finfo(bsq.dtype).tiny, dtype=bsq.dtype),
        )
        bmod = jnp.sqrt(bmod_sq)
        bsubu = jnp.asarray(bc.bsubu)
        bsubv = jnp.asarray(bc.bsubv)
        if not bool(self._cfg.lasym):
            pshalf = jnp.sqrt(jnp.maximum(0.5 * (s_full[1:] + s_full[:-1]), 0.0))
            pshalf = jnp.concatenate([pshalf[:1], pshalf], axis=0)
            if int(pshalf.shape[0]) > 1:
                pshalf = pshalf.at[0].set(pshalf[1])
            pshalf = pshalf[:, None, None]
            bsubu, bsubv = _filter_bsubuv_jxbforce_parity_jax(
                bsubu_even=bsubu,
                bsubu_odd=pshalf * bsubu,
                bsubv_even=bsubv,
                bsubv_odd=pshalf * bsubv,
                trig=trig,
                mmax_force=max(int(self._cfg.mpol) - 1, 0),
                nmax_force=int(self._cfg.ntor),
                s=s_full,
            )

        gmnc = _vmec_wrout_nyquist_cos_coeffs_jax(f=jnp.asarray(bc.jac.sqrtg), modes=nyq_modes, trig=trig)
        bmnc = _vmec_wrout_nyquist_cos_coeffs_jax(f=bmod, modes=nyq_modes, trig=trig)
        bsubumnc = _vmec_wrout_nyquist_cos_coeffs_jax(f=bsubu, modes=nyq_modes, trig=trig)
        bsubvmnc = _vmec_wrout_nyquist_cos_coeffs_jax(f=bsubv, modes=nyq_modes, trig=trig)
        bsupumnc = _vmec_wrout_nyquist_cos_coeffs_jax(f=jnp.asarray(bc.bsupu), modes=nyq_modes, trig=trig)
        bsupvmnc = _vmec_wrout_nyquist_cos_coeffs_jax(f=jnp.asarray(bc.bsupv), modes=nyq_modes, trig=trig)

        if not bool(self._cfg.lasym):
            mask_bsub = (jnp.asarray(nyq_modes.m) >= int(self._cfg.mpol)) | (
                jnp.abs(jnp.asarray(nyq_modes.n)) > int(self._cfg.ntor)
            )
            bsubumnc = jnp.where(mask_bsub[None, :], 0.0, jnp.asarray(bsubumnc))
            bsubvmnc = jnp.where(mask_bsub[None, :], 0.0, jnp.asarray(bsubvmnc))

        phi = cumrect_s_halfmesh(jnp.asarray(flux.phipf) * float(2.0 * np.pi * self._signgs), s_full)
        return SimpleNamespace(
            lasym=bool(self._cfg.lasym),
            nfp=int(self._cfg.nfp),
            iotas=iotas,
            buco=jnp.asarray(buco),
            bvco=jnp.asarray(bvco),
            gmnc=jnp.asarray(gmnc),
            bmnc=jnp.asarray(bmnc),
            bsubumnc=jnp.asarray(bsubumnc),
            bsubvmnc=jnp.asarray(bsubvmnc),
            bsupumnc=jnp.asarray(bsupumnc),
            bsupvmnc=jnp.asarray(bsupvmnc),
            xm_nyq=jnp.asarray(nyq_modes.m, dtype=jnp.float64),
            xn_nyq=jnp.asarray(nyq_modes.n * int(self._cfg.nfp), dtype=jnp.float64),
            phi=jnp.asarray(phi),
        )

    def aspect_jax(self, x_free: jnp.ndarray):
        """Return the fast boundary-only aspect-ratio surrogate."""
        self._ensure_static()
        boundary = self.boundary.apply_params(self.boundary.expand_free(x_free))
        return vj.boundary_aspect_ratio_from_static(boundary, self._static)

    def aspect_equilibrium_from_state_jax(self, state):
        """Return the equilibrium aspect ratio for an already-solved state."""
        return vj.equilibrium_aspect_ratio_from_state(state=state, static=self.get_static())

    def aspect_equilibrium_jax(self, x_free: jnp.ndarray):
        """Return the equilibrium aspect ratio directly from the solved state."""
        state = self._solve_state(x_free)
        return self.aspect_equilibrium_from_state_jax(state)

    def aspect_equilibrium(self, x_free=None):
        """Return the equilibrium aspect ratio derived from the solved ``wout``."""
        if x_free is None:
            x_free = self.boundary.get_free_params()
        return float(np.asarray(self.aspect_equilibrium_jax(jnp.asarray(x_free))))

    def aspect(self, x_free=None):
        """Return the plasma aspect ratio, matching :class:`simsopt.mhd.Vmec`."""
        return self.aspect_equilibrium(x_free)

    def mean_iota_from_state_jax(self, state):
        """Return the mean rotational transform, matching :class:`simsopt.mhd.Vmec`."""
        _chips, iotas, _iotaf = vj.equilibrium_iota_profiles_from_state(
            state=state,
            static=self.get_static(),
            indata=self._indata_raw,
            signgs=self._signgs,
        )
        iotas = jnp.asarray(iotas)
        if int(iotas.shape[0]) <= 1:
            return jnp.asarray(0.0, dtype=iotas.dtype)
        return jnp.mean(iotas[1:])

    def mean_iota_jax(self, x_free: jnp.ndarray):
        """Return the mean rotational transform directly from a solved state."""
        state = self._solve_state(x_free)
        return self.mean_iota_from_state_jax(state)

    def mean_iota(self, x_free=None):
        """Return the mean rotational transform in the same style as :class:`simsopt.mhd.Vmec`."""
        if x_free is None:
            x_free = self.boundary.get_free_params()
        return float(np.asarray(self.mean_iota_jax(jnp.asarray(x_free))))

    def get_static(self):
        """Expose the cached vmec_jax static data structure."""
        self._ensure_static()
        return self._static

    def get_context(self):
        """Expose the cached warm-start/profile context."""
        self._ensure_context()
        return self._context

    @property
    def boundary(self):
        """Return the mutable boundary wrapper for this VMEC-JAX instance."""
        self._ensure_static()
        return self._boundary
