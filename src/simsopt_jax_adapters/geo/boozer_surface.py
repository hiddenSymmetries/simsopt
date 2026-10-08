"""Native ``BoozerSurface`` with its solves in JAX.

:class:`BoozerSurfaceJAX` takes what native ``BoozerSurface`` takes, with a
:class:`~simsopt_jax_adapters.field.BiotSavartJAX` field, and has native's
``run_code``, solvers, defaults and result dictionaries, so native objectives
such as ``Iotas``, ``MajorRadius``, ``NonQuasiSymmetricRatio`` and
``BoozerResidual`` use it unchanged (with a ``ToroidalFlux`` label, use
:class:`BoozerResidualJAX` for ``BoozerResidual``)::

    import numpy as np
    from simsopt.configs import get_data
    from simsopt.geo import Iotas, SurfaceXYZTensorFourier, Volume
    from simsopt_jax_adapters.field import BiotSavartJAX
    from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX

    base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
    surface = SurfaceXYZTensorFourier(
        mpol=6, ntor=6, stellsym=True, nfp=nfp,
        quadpoints_phi=np.linspace(0, 1 / nfp, 13, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, 13, endpoint=False),
    )
    surface.fit_to_curve(ma, 0.1, flip_theta=True)
    volume = Volume(surface)
    boozer_surface = BoozerSurfaceJAX(BiotSavartJAX(bs.coils), surface, volume, volume.J())
    G0 = 2 * np.pi * sum(abs(c.current.get_value()) for c in bs.coils) * 2e-7
    res = boozer_surface.run_code(-0.406, G=G0)  # BoozerExact Newton, as natively
    iotas = Iotas(boozer_surface)
    print(res["success"], res["iter"], iotas.J(), iotas.dJ()[:3])

Each solve evaluates the formulations of :mod:`simsopt_jax.core.boozer_problem`
for the current surface, coils, label, target and weight: the Newton-type
loops of :mod:`simsopt_jax.core.boozer_solvers` run on the device in one
program, BFGS/L-BFGS-B and ``least_squares`` are SciPy's (as natively) over
the jitted value and gradient. Results are native's: NumPy arrays, the
solution in ``surface``, host ``PLU`` factors and a ``vjp`` that returns the
coil ``Derivative``. Like native ``BoozerSurface``, an instance mutates its
surface and ``res`` and belongs to one thread.

Differences from native: the label must be ``Volume``, ``Area``,
``AspectRatio`` or ``ToroidalFlux`` on the surface or a surface sharing its
DOFs; fixed surface DOFs are solved for (native raises); the exact ``vjp``
also handles surfaces without stellarator symmetry (native raises); the
BoozerLS ``vjp`` is the derivative of the whole penalty, including a
``ToroidalFlux`` label's coil dependence and ``G``'s dependence on the
currents, which native's ``boozer_surface_dlsqgrad_dcoils_vjp`` drops (a
native bug: with a ``ToroidalFlux`` label, native's BoozerLS coil gradients
are not the derivatives of the solved surface; with ``Volume``, ``Area`` or
``AspectRatio`` labels it agrees with native when ``G`` is optimized or the
currents are fixed, while with free currents and ``G=None`` it also carries
``dG/dI``, which native drops); :class:`BoozerResidualJAX` replaces native
``BoozerResidual``, whose explicit coil derivative misses the same terms
(a deliberate correction of a native bug). Native ``BoozerResidual`` stays
correct on ``BoozerSurfaceJAX`` for coil-independent ``Volume``, ``Area`` and
``AspectRatio`` labels with ``G`` optimized or the currents fixed; the field's evaluation
points are left as they were; ``options`` is copied, not filled in.
``minimize_boozer_exact_constraints_newton`` is not provided. The penalty
Newton evaluates ``d2B/dXdX``: on large grids call
``simsopt_jax.backend.set_backend("jax", device=...)`` first for its point
chunks. Iterates agree with native to round-off, which BFGS and undamped
Newton walks can magnify. The default ``newton_tol`` (native's: ``1e-13``
for BoozerExact, ``1e-11`` for BoozerLS) is kept; ``1e-13`` sits at float64's
round-off floor of the residual, so iteration counts and success can differ
from native by one step there: pass a looser tolerance such as ``1e-12``
when reproducibility matters. This module imports the field adapters, so it
is not re-exported from :mod:`simsopt_jax_adapters.geo`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial
from types import MappingProxyType
from typing import cast

import jax
import numpy as np
from scipy.linalg import lu
from scipy.optimize import OptimizeResult, least_squares, minimize

from simsopt._core.derivative import Derivative, derivative_dec
from simsopt._core.optimizable import Optimizable
from simsopt.geo.surfaceobjectives import Area, AspectRatio, ToroidalFlux, Volume
from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.objectives.utilities import forward_backward
from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.core.boozer_problem import (
    BoozerProblem,
    boozer_penalty_constraints,
    boozer_penalty_residual,
)
from simsopt_jax.core.boozer_solvers import (
    boozer_exact_newton,
    boozer_exact_residual_coil_vjp,
    boozer_penalty_coil_vjp,
    boozer_penalty_gauss_newton,
    boozer_penalty_newton,
    boozer_residual_objective,
)
from simsopt_jax.runtime.host_boundary import host_tree
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

from .boozer_problem import boozer_exact_residual_mask, boozer_exact_residual_rows, boozer_problem
from .surface_specs import surface_spec_from_surface

__all__ = ["BoozerResidualJAX", "BoozerSurfaceJAX"]

_DEFAULT_OPTIONS = MappingProxyType(
    {
        "exact": MappingProxyType({"verbose": True, "newton_tol": 1e-13, "newton_maxiter": 40}),
        "ls": MappingProxyType(
            {
                "verbose": True,
                "bfgs_tol": 1e-10,
                "newton_tol": 1e-11,
                "newton_maxiter": 40,
                "bfgs_maxiter": 1500,
                "limited_memory": False,
                "weight_inv_modB": True,
            }
        ),
    }
)


# SciPy's unannotated ``jac='2-point'`` default makes type checkers take ``jac`` for a string.
_least_squares = cast(Callable[..., OptimizeResult], least_squares)


def _place(values, problem: BoozerProblem) -> jax.Array:
    """``values`` as float64 on the problem's device."""
    return explicit_device_array(np.asarray(values, dtype=np.float64), dtype=np.float64, reference=problem.target_label)


def _solution(x: np.ndarray, optimize_G: bool):
    """Native's layout of ``x``: the surface DOFs, ``iota`` and ``G`` (``None``
    unless optimized)."""
    if optimize_G:
        return x[:-2], x[-2], x[-1]
    return x[:-1], x[-1], None


class BoozerSurfaceJAX(Optimizable):
    """Native ``BoozerSurface(biotsavart, surface, label, targetlabel,
    constraint_weight, options)`` with a ``BiotSavartJAX`` field.

    ``constraint_weight`` selects BoozerLS (truthy) or BoozerExact for
    :meth:`run_code`; the solvers, their arguments, defaults, ``options`` and
    ``res`` keys are native's. Every solve reads the current ``surface``,
    coils, ``label``, ``targetlabel`` and options.
    """

    res: dict

    def __init__(
        self,
        biotsavart: BiotSavartJAX,
        surface: SurfaceXYZFourier | SurfaceXYZTensorFourier,
        label: Volume | Area | AspectRatio | ToroidalFlux,
        targetlabel: float,
        constraint_weight: float | None = None,
        options: dict | None = None,
    ):
        super().__init__(depends_on=[biotsavart])
        if not isinstance(surface, (SurfaceXYZTensorFourier, SurfaceXYZFourier)):
            raise Exception("The input surface must be a SurfaceXYZTensorFourier or SurfaceXYZFourier.")
        self.biotsavart = biotsavart
        self.surface = surface
        self.label = label
        self.targetlabel = targetlabel
        self.constraint_weight = constraint_weight
        self.boozer_type = "ls" if constraint_weight else "exact"
        self.need_to_run_code = True
        self.options = {**_DEFAULT_OPTIONS[self.boozer_type], **(options or {})}

    def recompute_bell(self, parent=None):
        self.need_to_run_code = True

    def run_code(self, iota, G=None):
        """Native ``run_code``: BoozerExact Newton, or BFGS then the penalty
        Newton for BoozerLS, with the options' tolerances and caps."""
        if not self.need_to_run_code:
            return

        # As natively: with G from the currents, coil gradients need fixed currents.
        if G is None:
            assert np.all([c.current.dofs.all_fixed() for c in self.biotsavart.coils])

        if self.boozer_type == "exact":
            return self.solve_residual_equation_exactly_newton(
                iota=iota,
                G=G,
                tol=self.options["newton_tol"],
                maxiter=self.options["newton_maxiter"],
                verbose=self.options["verbose"],
            )

        assert self.constraint_weight is not None
        res = self.minimize_boozer_penalty_constraints_LBFGS(
            constraint_weight=self.constraint_weight,
            iota=iota,
            G=G,
            tol=self.options["bfgs_tol"],
            maxiter=self.options["bfgs_maxiter"],
            verbose=self.options["verbose"],
            limited_memory=self.options["limited_memory"],
            weight_inv_modB=self.options["weight_inv_modB"],
        )
        self.need_to_run_code = True
        return self.minimize_boozer_penalty_constraints_newton(
            constraint_weight=self.constraint_weight,
            iota=res["iota"],
            G=res["G"],
            verbose=self.options["verbose"],
            tol=self.options["newton_tol"],
            maxiter=self.options["newton_maxiter"],
            weight_inv_modB=self.options["weight_inv_modB"],
        )

    def _problem(self, constraint_weight: float | None = None) -> BoozerProblem:
        return boozer_problem(
            self.biotsavart, self.surface, self.label, self.targetlabel, constraint_weight
        )

    def _decision_vector(self, iota, G) -> np.ndarray:
        """Native's ``x``: the surface DOFs, ``iota`` and, if given, ``G``."""
        return np.concatenate((self.surface.get_dofs(), [iota] if G is None else [iota, G]))

    def _commit(self, x: np.ndarray, optimize_G: bool):
        """``x``'s DOFs into the surface; returns ``iota`` and ``G`` (``None``
        unless optimized)."""
        surface_dofs, iota, G = _solution(x, optimize_G)
        self.surface.set_dofs(surface_dofs)
        return iota, G

    def _store(self, res: dict, iota, G, *, with_surface: bool) -> dict:
        """Native's ending of a stored solve: ``G`` (over ``res``'s ``None``
        placeholder), ``s`` if ``with_surface`` and ``iota`` into ``res``,
        which becomes ``self.res``."""
        res["G"] = G
        if with_surface:
            res["s"] = self.surface
        res["iota"] = iota
        self.res = res
        self.need_to_run_code = False
        return res

    def minimize_boozer_penalty_constraints_LBFGS(
        self,
        tol=1e-3,
        maxiter=1000,
        constraint_weight=1.0,
        iota=0.0,
        G=None,
        limited_memory=True,
        weight_inv_modB=True,
        verbose=False,
    ):
        """Native's: SciPy BFGS (or L-BFGS-B) on the penalty."""
        if not self.need_to_run_code:
            return self.res
        optimize_G = G is not None
        problem = self._problem(constraint_weight)

        def penalty(x):
            return host_tree(
                boozer_penalty_constraints(
                    problem,
                    _place(x, problem),
                    derivatives=1,
                    optimize_G=optimize_G,
                    weight_inv_modB=weight_inv_modB,
                )
            )

        method = "L-BFGS-B" if limited_memory else "BFGS"
        options = {"maxiter": maxiter, "gtol": tol}
        if limited_memory:
            options["maxcor"] = 200
            options["ftol"] = tol
        res = minimize(penalty, self._decision_vector(iota, G), jac=True, method=method, options=options)

        resdict = {
            "fun": res.fun,
            "gradient": res.jac,
            "iter": res.nit,
            "info": res,
            "success": res.success,
            "G": None,
            "weight_inv_modB": weight_inv_modB,
            "type": "ls",
        }
        iota, G = self._commit(res.x, optimize_G)
        resdict = self._store(resdict, iota, G, with_surface=True)
        if verbose:
            print(
                f"{method} solve - {resdict['success']}  iter={resdict['iter']}, "
                f"iota={resdict['iota']:.16f}, "
                f"||grad||_inf = {np.linalg.norm(resdict['gradient'], ord=np.inf):.3e}",
                flush=True,
            )
        return resdict

    def minimize_boozer_penalty_constraints_newton(
        self,
        tol=1e-12,
        maxiter=10,
        constraint_weight=1.0,
        iota=0.0,
        G=None,
        stab=0.0,
        weight_inv_modB=True,
        verbose=False,
    ):
        """Native's: Newton on the penalty with its analytic Hessian."""
        if not self.need_to_run_code:
            return self.res
        optimize_G = G is not None
        problem = self._problem(constraint_weight)
        result = host_tree(
            boozer_penalty_newton(
                problem,
                _place(self._decision_vector(iota, G), problem),
                _place(tol, problem),
                _place(maxiter, problem),
                _place(stab, problem),
                optimize_G=optimize_G,
                weight_inv_modB=weight_inv_modB,
            )
        )
        # Native evaluates every iterate on the surface before factorising its
        # Hessian, so the surface holds the last iterate also when that raises.
        iota, G = self._commit(result.x, optimize_G)
        if result.singular:
            raise np.linalg.LinAlgError("Singular matrix")

        res = {
            "residual": result.gradient,
            "jacobian": result.gradient,
            "hessian": result.hessian,
            "iter": int(result.iterations),
            "success": result.norm <= tol,
            "G": None,
            "PLU": lu(result.hessian),
            "vjp": partial(_penalty_coil_vjp, weight_inv_modB=weight_inv_modB, constraint_weight=constraint_weight),
            "type": "ls",
            "weight_inv_modB": weight_inv_modB,
        }
        res = self._store(res, iota, G, with_surface=False)
        if verbose:
            print(
                f"NEWTON solve - {res['success']}  iter={res['iter']}, iota={res['iota']:.16f}, "
                f"||grad||_inf = {np.linalg.norm(res['jacobian'], ord=np.inf):.3e}",
                flush=True,
            )
        return res

    def minimize_boozer_penalty_constraints_ls(
        self,
        tol=1e-12,
        maxiter=10,
        constraint_weight=1.0,
        iota=0.0,
        G=None,
        method="lm",
        weight_inv_modB=True,
    ):
        """Native's: SciPy ``least_squares(method=method)`` on the penalty's
        residuals, or for ``method='manual'`` native's damped Gauss-Newton
        (whose result, as natively, is returned but not stored in ``res``)."""
        if not self.need_to_run_code:
            return self.res
        optimize_G = G is not None
        problem = self._problem(constraint_weight)
        x = self._decision_vector(iota, G)
        if method == "manual":
            result = host_tree(
                boozer_penalty_gauss_newton(
                    problem,
                    _place(x, problem),
                    _place(tol, problem),
                    _place(maxiter, problem),
                    optimize_G=optimize_G,
                    weight_inv_modB=weight_inv_modB,
                )
            )
            # As natively, the surface holds the last iterate also when its step raises.
            iota, G = self._commit(result.x, optimize_G)
            if result.singular:
                raise np.linalg.LinAlgError("Singular matrix")
            resdict = {
                "residual": result.residual,
                "gradient": result.gradient,
                "jacobian": result.normal_matrix,
                "success": result.norm <= tol,
            }
            if optimize_G:
                resdict["G"] = G
            resdict["s"] = self.surface
            resdict["iota"] = iota
            return resdict

        def residuals(x, derivatives):
            return boozer_penalty_residual(
                problem,
                _place(x, problem),
                derivatives=derivatives,
                optimize_G=optimize_G,
                weight_inv_modB=weight_inv_modB,
            )[derivatives]

        res = _least_squares(
            lambda x: host_tree(residuals(x, 0)),
            x,
            jac=lambda x: host_tree(residuals(x, 1)),
            method=method,
            ftol=tol,
            xtol=tol,
            gtol=tol,
            x_scale=1.0,
            max_nfev=maxiter,
        )
        resdict = {
            "info": res,
            "residual": res.fun,
            "gradient": res.grad,
            "jacobian": res.jac,
            "success": res.status > 0,
            "G": None,
        }
        iota, G = self._commit(res.x, optimize_G)
        return self._store(resdict, iota, G, with_surface=True)

    def solve_residual_equation_exactly_newton(self, tol=1e-10, maxiter=10, iota=0.0, G=None, verbose=False):
        """Native's BoozerExact Newton on ``get_stellsym_mask()``'s residuals,
        the label and, without stellarator symmetry, ``z(0, 0)``."""
        if not self.need_to_run_code:
            return self.res
        mask = boozer_exact_residual_mask(self.surface)
        problem = self._problem()
        result = host_tree(
            boozer_exact_newton(
                problem,
                _place(self._decision_vector(iota, G), problem),
                boozer_exact_residual_rows(self.surface, problem.target_label),
                _place(tol, problem),
                _place(maxiter, problem),
                G_from_currents=G is None,
            )
        )
        surface_dofs, iota, G = _solution(result.x, optimize_G=True)
        # Native moves the surface at every step, also when it then raises.
        if result.iterations > 0:
            self.surface.set_dofs(surface_dofs)
        if result.singular:
            raise np.linalg.LinAlgError("Singular matrix")

        res = {
            "residual": result.residual,
            "jacobian": result.jacobian,
            "iter": int(result.iterations),
            "success": result.norm <= tol,
            "G": G,
            "s": self.surface,
            "iota": iota,
            "PLU": lu(result.jacobian),
            "mask": mask,
            "type": "exact",
            "vjp": _exact_coil_vjp,
        }
        if verbose:
            print(
                f"NEWTON solve - {res['success']}  iter={res['iter']}, iota={res['iota']:.16f}, "
                f"||residual||_inf = {np.linalg.norm(res['residual'], ord=np.inf):.3e}",
                flush=True,
            )
        self.res = res
        self.need_to_run_code = False
        return res


def _coil_derivative(booz_surf: BoozerSurfaceJAX, cotangents) -> Derivative:
    return booz_surf.biotsavart.coil_cotangents_to_derivative(
        cotangents.field_inputs(), cotangents.coil_index_lists()
    )


class BoozerResidualJAX(Optimizable):
    """Native ``BoozerResidual(boozer_surface, bs)`` on a BoozerLS
    :class:`BoozerSurfaceJAX`, with ``bs`` a ``BiotSavartJAX`` of the surface's
    coils: ``J = 0.5 |r|^2 / len(r) + 0.5 w (label - target)^2`` on a private
    ``SurfaceXYZTensorFourier`` copy of the solved surface (its quadrature,
    ``w`` the surface's ``constraint_weight`` at construction), re-solving
    first when the surface needs it, as natively.

    ``dJ`` is the derivative of ``J`` through the solve: every explicit coil
    dependence of ``J`` minus the adjoint term of ``res['vjp']``. Native
    ``BoozerResidual`` takes the explicit part through the field only, so it
    is not the derivative with a ``ToroidalFlux`` label, nor with free
    currents when ``G`` is not optimized (a native bug); for ``Volume``,
    ``Area`` and ``AspectRatio`` labels with ``G`` optimized or the currents
    fixed, native ``BoozerResidual`` works on a ``BoozerSurfaceJAX`` and agrees
    with this class. Evaluating the objective does not set the field's
    evaluation points, but a ``ToroidalFlux`` label sharing the field resets
    them through its own callbacks when the surface is re-solved.
    Shallow copies register with the same solved surface and field, with an
    independent private surface and empty objective caches.
    """

    def __init__(self, boozer_surface: BoozerSurfaceJAX, bs: BiotSavartJAX):
        Optimizable.__init__(self, depends_on=[boozer_surface])
        in_surface = boozer_surface.surface
        self.boozer_surface = boozer_surface
        surface = SurfaceXYZTensorFourier(
            mpol=in_surface.mpol,
            ntor=in_surface.ntor,
            stellsym=in_surface.stellsym,
            nfp=in_surface.nfp,
            quadpoints_phi=in_surface.quadpoints_phi,
            quadpoints_theta=in_surface.quadpoints_theta,
        )
        surface.set_dofs(in_surface.get_dofs())
        self.constraint_weight = boozer_surface.constraint_weight
        self.in_surface = in_surface
        self.surface = surface
        self.biotsavart = bs
        self.recompute_bell()

    def __copy__(self):
        copied = type(self)(self.boozer_surface, self.biotsavart)
        copied.constraint_weight = self.constraint_weight
        return copied

    def J(self):
        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            self.compute()
        return self._dJ

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def compute(self):
        booz_surf = self.boozer_surface
        if booz_surf.need_to_run_code:
            res = booz_surf.res
            booz_surf.run_code(res["iota"], G=res["G"])
        self.surface.set_dofs(self.in_surface.get_dofs())

        res = booz_surf.res
        iota, G, weight_inv_modB = res["iota"], res["G"], res["weight_inv_modB"]
        # The residual on the private copy, the label on its own surface (native's split).
        problem = replace(
            boozer_problem(self.biotsavart, self.in_surface, booz_surf.label, booz_surf.targetlabel, self.constraint_weight),
            surface=surface_spec_from_surface(self.surface),
        )
        x = np.concatenate((self.surface.get_dofs(), [iota] if G is None else [iota, G]))
        value, dJ_dx, dJ_dcoils = boozer_residual_objective(
            problem, _place(x, problem), optimize_G=G is not None, weight_inv_modB=weight_inv_modB
        )
        self._J = host_tree(value)[()]

        P, L, U = res["PLU"]
        adj = forward_backward(P, L, U, host_tree(dJ_dx))
        explicit = self.biotsavart.coil_cotangents_to_derivative(
            dJ_dcoils.field_inputs(), dJ_dcoils.coil_index_lists()
        )
        self._dJ = explicit - res["vjp"](adj, booz_surf, iota, G)


def _exact_coil_vjp(lm, booz_surf: BoozerSurfaceJAX, iota, G) -> Derivative:
    """Native ``boozer_surface_dexactresidual_dcoils_dcurrents_vjp`` at the
    surface's current DOFs and on its ``res['mask']`` rows."""
    assert G is not None
    problem = booz_surf._problem()
    x = booz_surf._decision_vector(iota, G)
    rows = boozer_exact_residual_rows(booz_surf.surface, problem.target_label)
    return _coil_derivative(
        booz_surf, boozer_exact_residual_coil_vjp(problem, _place(x, problem), rows, _place(lm, problem))
    )


def _penalty_coil_vjp(
    lm, booz_surf: BoozerSurfaceJAX, iota, G, weight_inv_modB=True, *, constraint_weight: float
) -> Derivative:
    """The coil term of the BoozerLS adjoint at the surface's current DOFs,
    for the penalty with the solve's ``constraint_weight``: native
    ``boozer_surface_dlsqgrad_dcoils_vjp``'s signature, with the label's and
    ``G``'s coil dependence that native drops (see the module docstring)."""
    problem = booz_surf._problem(constraint_weight)
    return _coil_derivative(
        booz_surf,
        boozer_penalty_coil_vjp(
            problem,
            _place(booz_surf._decision_vector(iota, G), problem),
            _place(lm, problem),
            optimize_G=G is not None,
            weight_inv_modB=weight_inv_modB,
        ),
    )
