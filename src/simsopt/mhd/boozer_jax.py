# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""
JAX-backed Boozer transform and quasisymmetry objective.
"""

from __future__ import annotations

import logging
from typing import Iterable, Union

try:
    import vmec_jax as vj
    from vmec_jax._compat import jax, jnp, has_jax
    from vmec_jax.booz_input import BoozXformInputs
    import numpy as np
except Exception as exc:  # pragma: no cover - optional dependency
    vj = None
    jax = None
    jnp = None
    BoozXformInputs = None
    np = None
    has_jax = None
    _import_error = exc
else:
    _import_error = None

try:
    from booz_xform_jax.jax_api import (
        booz_xform_from_inputs,
        prepare_booz_xform_constants_from_inputs,
    )
except Exception as exc:  # pragma: no cover - optional dependency
    booz_xform_from_inputs = None
    prepare_booz_xform_constants_from_inputs = None
    _booz_import_error = exc
else:
    _booz_import_error = None


logger = logging.getLogger(__name__)

__all__ = ["BoozerJax", "QuasisymmetryJax"]


def _require_jax():
    if vj is None or has_jax is None or not has_jax():
        raise ImportError(
            "BoozerJax requires vmec_jax and JAX. "
            "Install vmec_jax (and jax/jaxlib) before using BoozerJax."
        ) from _import_error
    if booz_xform_from_inputs is None:
        raise ImportError(
            "BoozerJax requires booz_xform_jax. "
            "Install booz_xform_jax before using BoozerJax."
        ) from _booz_import_error


class BoozerJax:
    def __init__(
        self,
        vmec,
        mpol: int = 32,
        ntor: int = 32,
        verbose: bool = False,
        *,
        jit: bool = True,
        source: str = "auto",
    ) -> None:
        _require_jax()
        self.vmec = vmec
        self.mpol = int(mpol)
        self.ntor = int(ntor)
        self.verbose = bool(verbose)
        self.jit = bool(jit)
        self.source = str(source).strip().lower()
        self.s = []
        self.s_to_index = {}
        self.s_used = {}
        self._compute_surfs = None
        self._compute_surfs_jax = None
        self._constants = None
        self._grids = None
        self._constants_sig = None
        self._booz_fn = None
        self._inputs_fn = None
        self._inputs_sig = None
        self._cached_x = None
        self._cached_out = None

    def register(self, s: Union[float, Iterable[float]]) -> None:
        try:
            ss = list(s)  # type: ignore[arg-type]
        except Exception:
            ss = [s]
        self.s = list(sorted(set(ss)))
        self._compute_surfs = None
        self._compute_surfs_jax = None
        self._constants = None
        self._grids = None
        self._constants_sig = None
        self._booz_fn = None
        self._inputs_fn = None
        self._inputs_sig = None
        self._cached_x = None
        self._cached_out = None

    def _x_cache_key(self, x_free) -> tuple[float, ...]:
        try:
            arr = np.asarray(x_free, dtype=float).reshape(-1)
        except Exception:
            return None
        return tuple(arr.tolist())

    def _build_booz_fn(self):
        def _run(inputs):
            return booz_xform_from_inputs(
                inputs=inputs,
                constants=self._constants,
                grids=self._grids,
                surface_indices=self._compute_surfs_jax,
                jit=False,
            )

        if self.jit:
            self._booz_fn = jax.jit(_run)
        else:
            self._booz_fn = _run

    def _build_inputs_fn(self, static, indata):
        from vmec_jax.modes import nyquist_mode_table_from_grid
        from vmec_jax.vmec_tomnsp import vmec_trig_tables

        nyq_modes = nyquist_mode_table_from_grid(
            mpol=int(static.cfg.mpol),
            ntor=int(static.cfg.ntor),
            ntheta=int(static.cfg.ntheta),
            nzeta=int(static.cfg.nzeta),
        )
        nyq_m = np.asarray(nyq_modes.m)
        nyq_n = np.asarray(nyq_modes.n)
        mmax = int(np.max(nyq_m)) if nyq_m.size else 0
        nmax = int(np.max(np.abs(nyq_n))) if nyq_n.size else 0
        trig = vmec_trig_tables(
            ntheta=int(static.cfg.ntheta),
            nzeta=int(static.cfg.nzeta),
            nfp=int(static.cfg.nfp),
            mmax=mmax,
            nmax=nmax,
            lasym=bool(static.cfg.lasym),
            dtype=jnp.float64,
            cache=True,
        )

        def _run(state):
            return vj.optimization.booz_xform_inputs_from_state(
                state=state,
                static=static,
                indata=indata,
                signgs=self.vmec._signgs,
                trig=trig,
            )

        if self.jit:
            self._inputs_fn = jax.jit(_run)
        else:
            self._inputs_fn = _run

    def _inputs_from_wout(self, wout):
        kwargs = {}
        if bool(getattr(wout, "lasym", False)):
            kwargs["bmns"] = jnp.asarray(getattr(wout, "bmns"))
            kwargs["bsubumns"] = jnp.asarray(getattr(wout, "bsubumns"))
            kwargs["bsubvmns"] = jnp.asarray(getattr(wout, "bsubvmns"))
        return BoozXformInputs(
            rmnc=jnp.asarray(getattr(wout, "rmnc")),
            zmns=jnp.asarray(getattr(wout, "zmns")),
            lmns=jnp.asarray(getattr(wout, "lmns")),
            bmnc=jnp.asarray(getattr(wout, "bmnc")),
            bsubumnc=jnp.asarray(getattr(wout, "bsubumnc")),
            bsubvmnc=jnp.asarray(getattr(wout, "bsubvmnc")),
            iota=jnp.asarray(getattr(wout, "iotas")),
            xm=jnp.asarray(getattr(wout, "xm"), dtype=jnp.int32),
            xn=jnp.asarray(getattr(wout, "xn"), dtype=jnp.int32),
            xm_nyq=jnp.asarray(getattr(wout, "xm_nyq"), dtype=jnp.int32),
            xn_nyq=jnp.asarray(getattr(wout, "xn_nyq"), dtype=jnp.int32),
            nfp=int(getattr(wout, "nfp")),
            **kwargs,
        )

    def run_jax(self, x_free: jnp.ndarray):
        x_key = self._x_cache_key(x_free)
        if x_key is not None and self._cached_x == x_key and self._cached_out is not None:
            return self._cached_out
        static = self.vmec.get_static()
        indata = self.vmec._indata_raw
        use_wout = self.source in ("auto", "wout") and x_key is not None and hasattr(self.vmec, "get_wout")
        if use_wout:
            try:
                inputs = self._inputs_from_wout(self.vmec.get_wout(x_free))
            except Exception:
                use_wout = False
        if not use_wout:
            state = self.vmec._solve_state(x_free)
            inputs_sig = (
                int(static.cfg.mpol),
                int(static.cfg.ntor),
                int(static.cfg.ntheta),
                int(static.cfg.nzeta),
                int(static.cfg.nfp),
                bool(static.cfg.lasym),
                int(self.vmec._signgs),
            )
            if self._inputs_fn is None or self._inputs_sig != inputs_sig:
                self._build_inputs_fn(static, indata)
                self._inputs_sig = inputs_sig
            inputs = self._inputs_fn(state)

        # Determine surface indices on the half grid.
        if self._compute_surfs is None:
            compute_surfs, s_used = vj.optimization.surface_indices_from_static(static, self.s)
            self._compute_surfs = compute_surfs
            self._compute_surfs_jax = jnp.asarray(compute_surfs, dtype=jnp.int32)
            self.s_used = {s: float(s_used[i]) for i, s in enumerate(self.s)}
            self.s_to_index = {s: int(i) for i, s in enumerate(self.s)}

        constants_sig = (int(self.mpol), int(self.ntor), bool(static.cfg.lasym))
        if self._constants is None or self._constants_sig != constants_sig:
            self._constants, self._grids = prepare_booz_xform_constants_from_inputs(
                inputs=inputs,
                mboz=int(self.mpol),
                nboz=int(self.ntor),
                asym=bool(static.cfg.lasym),
            )
            self._constants_sig = constants_sig
            self._booz_fn = None

        if self._booz_fn is None:
            self._build_booz_fn()
        out = self._booz_fn(inputs)
        if x_key is not None:
            self._cached_x = x_key
            self._cached_out = out
        return out


class QuasisymmetryJax:
    def __init__(
        self,
        boozer: BoozerJax,
        s: Union[float, Iterable[float]],
        helicity_m: int,
        helicity_n: int,
        normalization: str = "B00",
        weight: str = "even",
    ) -> None:
        _require_jax()
        self.boozer = boozer
        self.helicity_m = int(helicity_m)
        self.helicity_n = int(helicity_n)
        self.normalization = normalization
        self.weight = weight

        try:
            iter(s)  # type: ignore[arg-type]
        except Exception:
            s = [s]
        self.s = list(s)
        self._s_array = None
        self._s_index = None
        self._s_used = None
        self._sym_idx = None
        self._nonsym_idx = None
        self.boozer.register(s)

    def prepare(self, x_free: jnp.ndarray) -> None:
        """Populate mode and surface indices for JIT-safe quasisymmetry evaluation."""
        out = self.boozer.run_jax(x_free)
        self._ensure_indices(out)

    def _ensure_indices(self, out) -> None:
        if self._s_array is None:
            bmnc_b = jnp.asarray(out["bmnc_b"])
            self._s_array = jnp.asarray(self.s, dtype=bmnc_b.dtype)
            self._s_index = jnp.asarray([self.boozer.s_to_index[s] for s in self.s], dtype=jnp.int32)
            self._s_used = jnp.asarray([self.boozer.s_used[s] for s in self.s], dtype=bmnc_b.dtype)

        if self._nonsym_idx is not None:
            return

        xm_b = np.asarray(out["ixm_b"])
        xn_b = np.asarray(out["ixn_b"]) / float(out["nfp_b"])

        if self.helicity_m not in (0, 1):
            raise ValueError("m for quasisymmetry should be 0 or 1.")

        if self.helicity_n == 0:
            symmetric = (xn_b == 0)
        elif self.helicity_m == 0:
            symmetric = (xm_b == 0)
        else:
            symmetric = (xm_b * self.helicity_n + xn_b * self.helicity_m == 0)
        nonsymmetric = ~symmetric

        self._sym_idx = jnp.asarray(np.nonzero(symmetric)[0], dtype=jnp.int32)
        self._nonsym_idx = jnp.asarray(np.nonzero(nonsymmetric)[0], dtype=jnp.int32)

    def J(self, x_free: jnp.ndarray) -> jnp.ndarray:
        out = self.boozer.run_jax(x_free)

        bmnc_b = jnp.asarray(out["bmnc_b"])  # (ns_sel, mnboz)
        if self._nonsym_idx is None:
            if isinstance(out["bmnc_b"], jax.core.Tracer):
                raise RuntimeError(
                    "QuasisymmetryJax needs precomputed mode indices for JIT. "
                    "Call qs.prepare(x_free) before jitting."
                )
            self._ensure_indices(out)

        bmnc = bmnc_b[self._s_index, :]

        if self.normalization == "B00":
            bnorm = bmnc[:, 0]
        elif self.normalization == "symmetric":
            bmnc_sym = jnp.take(bmnc, self._sym_idx, axis=1)
            bnorm = jnp.sqrt(jnp.sum(jnp.square(bmnc_sym), axis=1))
        else:
            raise ValueError("Unrecognized value for normalization in Quasisymmetry")

        bmnc = bmnc / bnorm[:, None]

        if self.weight == "even":
            out = jnp.take(bmnc, self._nonsym_idx, axis=1)
            return out.reshape(-1)
        if self.weight == "stellopt":
            rad_sigma = jnp.asarray(self._s_used * self._s_used, dtype=bmnc.dtype)
            out = jnp.take(bmnc / rad_sigma[:, None], self._nonsym_idx, axis=1)
            return out.reshape(-1)
        if self.weight == "stellopt_ornl":
            out = jnp.linalg.norm(jnp.take(bmnc, self._nonsym_idx, axis=1), axis=1)
            return out.reshape(-1)
        raise ValueError("Unrecognized value for weight in Quasisymmetry")
