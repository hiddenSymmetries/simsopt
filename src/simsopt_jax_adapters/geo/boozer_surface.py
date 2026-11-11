"""JAX Boozer surface solvers and residual objectives for native simsopt workflows.

Provides :class:`JaxBoozerSurface` and :class:`JaxBoozerResidual` with
native-style result dictionaries and coil derivatives.
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
from scipy.optimize import LbfgsInvHessProduct, OptimizeResult, least_squares, minimize

from simsopt._core.derivative import Derivative, derivative_dec
from simsopt._core.optimizable import Optimizable
from simsopt.geo.boozersurface import BoozerSurface, _keep_iterate
from simsopt.geo.surfaceobjectives import Area, AspectRatio, ToroidalFlux, Volume
from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.objectives.utilities import forward_backward
from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.backend.runtime import host_tree
from simsopt_jax.core.boozer_problem import BoozerProblem, boozer_penalty_residual
from simsopt_jax.core.boozer_solvers import (
    _penalty_value_and_grad,
    boozer_exact_newton,
    boozer_exact_residual_coil_vjp,
    boozer_penalty_bfgs,
    boozer_penalty_coil_vjp,
    boozer_penalty_gauss_newton,
    boozer_penalty_lbfgsb,
    boozer_penalty_newton,
    boozer_residual_objective,
)
from simsopt_jax.core.quasi_newton import (
    BFGS_MESSAGES,
    LBFGSB_MESSAGES,
    LbfgsbCorrections,
    QuasiNewtonResult,
)
from simsopt_jax_adapters.field.biotsavart_backend import JaxBiotSavart

from .boozer_problem import boozer_exact_residual_mask, boozer_exact_residual_rows, boozer_problem
from .surface_specs import surface_spec_from_surface

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
                "host_bfgs": False,
            }
        ),
    }
)


# SciPy's unannotated ``jac='2-point'`` default makes type checkers take ``jac`` for a string.
_least_squares = cast(Callable[..., OptimizeResult], least_squares)


def _place(values, problem: BoozerProblem) -> jax.Array:
    """``values`` as float64 on the problem's device."""
    return explicit_device_array(np.asarray(values, dtype=np.float64), dtype=np.float64, reference=problem.targetlabel)


def _optimize_result(result: QuasiNewtonResult) -> OptimizeResult:
    """Convert a device solve result to a host SciPy OptimizeResult."""
    result = host_tree(result)
    if isinstance(result.hess_inv, LbfgsbCorrections):
        count = int(result.hess_inv.count)
        hess_inv = LbfgsInvHessProduct(result.hess_inv.s[:count], result.hess_inv.y[:count])
        messages = LBFGSB_MESSAGES
        order = ("fun", "jac", "nfev", "njev", "nit", "status", "message", "x", "success", "hess_inv")
    else:
        hess_inv, messages = result.hess_inv, BFGS_MESSAGES
        order = ("fun", "jac", "hess_inv", "nfev", "njev", "status", "success", "message", "x", "nit")
    status = int(result.status)
    fields = {
        "fun": float(result.fun),
        "jac": result.jac,
        "hess_inv": hess_inv,
        "nfev": int(result.nfev),
        "njev": int(result.njev),
        "status": status,
        "success": status == 0,
        "message": messages[int(result.message)],
        "x": result.x,
        "nit": int(result.iterations),
    }
    return OptimizeResult({key: fields[key] for key in order})


def _solution(x: np.ndarray, optimize_G: bool):
    """Unpack surface DOFs, iota, and G from x; return None for fixed G."""
    if optimize_G:
        return x[:-2], x[-2], x[-1]
    return x[:-1], x[-1], None


class JaxBoozerSurface(Optimizable):
    r"""Solve for a magnetic surface parametrized by Boozer angles.

    BoozerLS minimizes the mean half-squared Boozer residual with weighted
    label and z(0, 0) penalties. BoozerExact solves the unweighted masked
    residual and label equations, plus z(0, 0) without stellarator symmetry.
    Solves mutate the surface and ``res``; use each instance in one thread.
    Results contain NumPy arrays, host PLU factors, and a coil ``Derivative`` VJP
    including label and current-derived G dependence. Field evaluation points
    are preserved. ``minimize_boozer_exact_constraints_newton`` is unavailable.
    Iteration counts and success can differ from native near round-off limits;
    use a looser tolerance such as ``1e-12`` when reproducibility matters.
    A singular Newton step raises ``numpy.linalg.LinAlgError`` and leaves the
    surface at the last iterate.

    Args:
        biotsavart (JaxBiotSavart): Field whose coil DOFs are this solver's parents.
        surface (SurfaceXYZFourier | SurfaceXYZTensorFourier): Mutable native surface;
            geometry has shape (nphi, ntheta, 3); exact solves require the tensor class.
        label (Volume | Area | AspectRatio | ToroidalFlux): One of these four classes,
            on the surface or a surface sharing its DOFs.
        targetlabel (float): Target label in cubic meters (volume), square meters
            (area), webers (toroidal flux), or dimensionless (aspect ratio).
        constraint_weight (float | None): Native penalty coefficient for the squared
            label and z constraints; None for an exact problem. Truthy selects BoozerLS;
            otherwise run_code uses BoozerExact. Default None.
        options (dict | None): Copied solver options overriding defaults for the
            selected formulation: ``verbose`` True, ``newton_tol`` 1e-13 for
            BoozerExact or 1e-11 for BoozerLS, ``newton_maxiter`` 40,
            ``bfgs_tol`` 1e-10, ``bfgs_maxiter`` 1500, ``limited_memory`` False,
            ``weight_inv_modB`` True, and ``host_bfgs`` False. Setting ``host_bfgs``
            True runs SciPy BFGS or L-BFGS-B on the host. Default None uses defaults.

    With ``G=None`` in :meth:`run_code`, coil currents must be fixed.
    """

    res: dict

    def __init__(
        self,
        biotsavart: JaxBiotSavart,
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
        """Mark the surface solve as needing to run again after a parent changes.

        Args:
            parent (Optimizable | None): Parent notifying the solver of changed DOFs;
                unused.
        """
        self.need_to_run_code = True

    # Native's, so native objectives get the label row their adjoints expect: shape
    # (surface.local_full_dof_size,), in the label's units per meter.
    dlabel_dsurface = BoozerSurface.dlabel_dsurface

    def run_code(self, iota, G=None):
        """Solve with BoozerExact Newton or BoozerLS BFGS followed by penalty Newton.

        Args:
            iota (float): Initial dimensionless rotational transform.
            G (float | None): Initial G in tesla meters; None derives G from fixed
                currents (fixed in penalty solves, optimized in exact solves).

        Returns:
            dict | None: Solve result stored in res after updating the surface;
                None if no solve is needed.
        """
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
            host_bfgs=self.options["host_bfgs"],
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
        """Return the surface DOFs, iota, and optional G as a decision vector."""
        return np.concatenate((self.surface.get_dofs(), [iota] if G is None else [iota, G]))

    def _commit(self, x: np.ndarray, optimize_G: bool):
        """Update surface DOFs from x and return iota and G, or None for fixed G."""
        surface_dofs, iota, G = _solution(x, optimize_G)
        self.surface.set_dofs(surface_dofs)
        return iota, G

    def _store(self, res: dict, iota, G, *, with_surface: bool) -> dict:
        """Store the result, iota, G, and optional surface in self.res."""
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
        host_bfgs=False,
    ):
        """Minimize the penalty with BFGS or L-BFGS-B, on the device by default.

        Args:
            tol (float): Gradient tolerance (SciPy gtol), also relative objective
                tolerance (ftol) for L-BFGS-B; default 1e-3.
            maxiter (int | float): Maximum iterations; default 1000.
            constraint_weight (float): Penalty coefficient for squared label and z
                constraints; default 1.0.
            iota (float): Initial dimensionless rotational transform; default 0.0.
            G (float | None): Initial G in tesla meters; None derives a fixed value
                from currents. Default None.
            limited_memory (bool): Use L-BFGS-B when true, otherwise BFGS; default True.
            weight_inv_modB (bool): Divide each point's Boozer residual by the field
                magnitude in teslas; default True.
            verbose (bool): Print the native-style solver summary; default False.
            host_bfgs (bool): Run SciPy's solver on the host over the jitted value and
                gradient, as natively; default False.

        Returns:
            dict: Cached or stored result with value, gradient, iteration count,
                success, iota, G, and SciPy OptimizeResult in info; updates DOFs.
                A failed solve with a larger or non-finite final penalty restores the
                starting point; info retains the solver's result.
        """
        if not self.need_to_run_code:
            return self.res
        optimize_G = G is not None
        problem = self._problem(constraint_weight)
        x = self._decision_vector(iota, G)

        method = "L-BFGS-B" if limited_memory else "BFGS"
        options = {"maxiter": maxiter, "gtol": tol}
        if limited_memory:
            options["maxcor"] = 200
            options["ftol"] = tol

        value_and_grad = _penalty_value_and_grad(problem, optimize_G, weight_inv_modB)

        def penalty(x):
            return host_tree(value_and_grad(_place(x, problem)))

        if host_bfgs:
            res = minimize(penalty, x, jac=True, method=method, options=options)
        else:
            flags = dict(optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)
            start, tol, maxiter = (_place(value, problem) for value in (x, tol, maxiter))
            if limited_memory:
                result = boozer_penalty_lbfgsb(problem, start, tol, tol, maxiter, maxcor=options["maxcor"], **flags)
            else:
                result = boozer_penalty_bfgs(problem, start, tol, maxiter, **flags)
            res = _optimize_result(result)
        # As natively, a failed solve that ended at a larger or non-finite value restores its start.
        xsol, fsol, gsol = res.x, res.fun, res.jac
        if not res.success:
            f0, g0 = penalty(x)
            if not _keep_iterate(fsol, f0):
                xsol, fsol, gsol = x, f0, g0

        resdict = {
            "fun": fsol,
            "gradient": gsol,
            "iter": res.nit,
            "info": res,
            "success": res.success,
            "G": None,
            "weight_inv_modB": weight_inv_modB,
            "type": "ls",
        }
        iota, G = self._commit(xsol, optimize_G)
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
        divergence_factor=None,
    ):
        """Minimize the penalty by Newton iteration with its analytic Hessian.

        For large grids, configure backend point chunks with
        ``simsopt_jax.backend.set_backend("jax", device=...)`` first.

        Args:
            tol (float): Euclidean penalty-gradient norm threshold; default 1e-12.
                Success is True when the final iterate's norm is at most tol,
                including when the last permitted step reaches maxiter.
            maxiter (int | float): Iteration cap; default 10. Fractional caps allow steps
                while the integer iteration count is below the cap.
            constraint_weight (float): Penalty coefficient for squared label and z
                constraints; default 1.0.
            iota (float): Initial dimensionless rotational transform; default 0.0.
            G (float | None): Initial G in tesla meters; None derives a fixed value
                from currents. Default None.
            stab (float): Diagonal step-Hessian shift, in native units; default 0.0.
            weight_inv_modB (bool): Divide each point's Boozer residual by the field
                magnitude in teslas; default True.
            verbose (bool): Print the native-style solver summary; default False.
            divergence_factor (float | None): Stop, unsuccessfully, once the Euclidean
                gradient norm exceeds this dimensionless multiple of its starting
                value; default None disables the check.

        Returns:
            dict: Cached or stored result with derivatives, success, LU factors, and
                coil VJP; updates the surface, including before singular-step errors.
                A larger or non-finite final gradient norm restores the starting point
                and its derivatives and marks the solve unsuccessful.
        """
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
                None if divergence_factor is None else _place(divergence_factor, problem),
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
        """Minimize penalty residuals with SciPy least squares or manual Gauss-Newton.

        Args:
            tol (float): SciPy ftol, xtol and gtol, or the Euclidean penalty-gradient
                norm threshold for manual Gauss-Newton; default 1e-12.
            maxiter (int | float): SciPy residual-evaluation cap (max_nfev), or manual
                Gauss-Newton step cap; default 10.
            constraint_weight (float): Penalty coefficient for squared label and z
                constraints; default 1.0.
            iota (float): Initial dimensionless rotational transform; default 0.0.
            G (float | None): Initial G in tesla meters; None derives a fixed value
                from currents. Default None.
            method (str): SciPy least_squares method, or manual for damped Gauss-Newton;
                default "lm".
            weight_inv_modB (bool): Divide each point's Boozer residual by the field
                magnitude in teslas; default True.

        Returns:
            dict: Result after updating the surface; SciPy results are cached in res,
                manual results are returned without storing. Failed SciPy solves
                (status <= 0) with a larger or non-finite final residual norm restore
                the starting point and residuals. Manual solves restore them for a
                larger or non-finite final penalty-gradient norm.
        """
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
            )

        res = _least_squares(
            lambda x: host_tree(residuals(x, 0)[0]),
            x,
            jac=lambda x: host_tree(residuals(x, 1)[1]),
            method=method,
            ftol=tol,
            xtol=tol,
            gtol=tol,
            x_scale=1.0,
            max_nfev=maxiter,
        )
        # As natively, a failed solve that ended at a larger or non-finite residual restores its start.
        xsol, rsol, gsol, Jsol = res.x, res.fun, res.grad, res.jac
        if res.status <= 0:
            r0, J0 = host_tree(residuals(x, 1))
            if not _keep_iterate(np.linalg.norm(rsol), np.linalg.norm(r0)):
                xsol, rsol, gsol, Jsol = x, r0, J0.T @ r0, J0
        resdict = {
            "info": res,
            "residual": rsol,
            "gradient": gsol,
            "jacobian": Jsol,
            "success": res.status > 0,
            "G": None,
        }
        iota, G = self._commit(xsol, optimize_G)
        return self._store(resdict, iota, G, with_surface=True)

    def solve_residual_equation_exactly_newton(self, tol=1e-10, maxiter=10, iota=0.0, G=None, verbose=False):
        """Solve the masked, unweighted BoozerExact system by Newton iteration.

        Args:
            tol (float): Threshold for the Euclidean norm of the full Newton system
                vector: Boozer residual entries selected by the native exact mask,
                followed by ``label - targetlabel`` and, without stellarator symmetry,
                ``z(0, 0)``. Default 1e-10; the norm combines residual and constraint
                units. The stopping norm and success follow
                :func:`simsopt_jax.core.boozer_solvers.boozer_exact_newton`.
            maxiter (int | float): Iteration cap; default 10. Fractional caps allow steps
                while the integer iteration count is below the cap.
            iota (float): Initial dimensionless rotational transform; default 0.0.
            G (float | None): Initial G in tesla meters; None derives the starting
                value from currents, then optimizes it. Default None.
            verbose (bool): Print the native-style solver summary; default False.

        Returns:
            dict: Cached or stored result with residual, Jacobian, mask, LU factors,
                coil VJP, and final-iterate success; updates the surface after steps.
                A larger or non-finite final norm restores the starting point and its
                residual and Jacobian and marks the solve unsuccessful.
        """
        if not self.need_to_run_code:
            return self.res
        mask = boozer_exact_residual_mask(self.surface)
        problem = self._problem()
        result = host_tree(
            boozer_exact_newton(
                problem,
                _place(self._decision_vector(iota, G), problem),
                boozer_exact_residual_rows(self.surface, problem.targetlabel),
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
            "weight_inv_modB": False,
            "vjp": _exact_coil_vjp,
        }
        if verbose:
            print(
                f"NEWTON solve - {res['success']}  iter={res['iter']}, iota={res['iota']:.16f}, "
                f"||residual||_inf = {np.linalg.norm(res['residual'], ord=np.inf):.3e}",
                flush=True,
            )
        return self._store(res, iota, G, with_surface=True)


def _coil_derivative(field: JaxBiotSavart, cotangents) -> Derivative:
    return field.coil_cotangents_to_derivative(cotangents.field_inputs(), cotangents.coil_index_lists())


class JaxBoozerResidual(Optimizable):
    r"""Evaluate the Boozer residual penalty of a solved JAX Boozer surface.

    The objective is half the mean squared Boozer residual plus half the
    weighted squared label error, using the solve's inverse-field weighting.
    The constraint weight is captured at construction (zero for BoozerExact);
    there is no z penalty. Geometry is copied with the same class, quadrature,
    and tensor clamping. The surface is re-solved when needed, and ``dJ`` is the
    total coil derivative including label and current-derived G dependence.

    Args:
        boozer_surface (JaxBoozerSurface): Mutable BoozerLS or BoozerExact solver; the
            objective owns a separate copy of its solved geometry.
        bs (JaxBiotSavart): Field with the solver's coils; evaluating this objective
            leaves its evaluation points unchanged except for shared label callbacks
            during re-solves.

    Raises:
        TypeError: If the solver's surface is not exactly a ``SurfaceXYZFourier`` or
            ``SurfaceXYZTensorFourier`` (subclasses may override the geometry).

    Use this class for ``ToroidalFlux`` labels on BoozerLS solvers.
    """

    def __init__(self, boozer_surface: JaxBoozerSurface, bs: JaxBiotSavart):
        Optimizable.__init__(self, depends_on=[boozer_surface])
        in_surface = boozer_surface.surface
        self.boozer_surface = boozer_surface
        # The copy keeps the solved surface's class and clamped dimensions, so its DOFs give the same geometry.
        if type(in_surface) is SurfaceXYZTensorFourier:
            representation = {"clamped_dims": list(in_surface.clamped_dims)}
        elif type(in_surface) is SurfaceXYZFourier:
            representation = {}
        else:
            raise TypeError(
                "JaxBoozerResidual supports SurfaceXYZFourier and SurfaceXYZTensorFourier surfaces, "
                f"got {type(in_surface).__name__}."
            )
        surface = type(in_surface)(
            mpol=in_surface.mpol,
            ntor=in_surface.ntor,
            stellsym=in_surface.stellsym,
            nfp=in_surface.nfp,
            quadpoints_phi=in_surface.quadpoints_phi,
            quadpoints_theta=in_surface.quadpoints_theta,
            **representation,
        )
        surface.set_dofs(in_surface.get_dofs())
        # BoozerExact surfaces have no label penalty (constraint_weight is None), as natively.
        self.constraint_weight = 0.0 if boozer_surface.constraint_weight is None else boozer_surface.constraint_weight
        self.in_surface = in_surface
        self.surface = surface
        self.biotsavart = bs
        self.recompute_bell()

    def __copy__(self):
        copied = type(self)(self.boozer_surface, self.biotsavart)
        copied.constraint_weight = self.constraint_weight
        return copied

    def J(self):
        """Return the cached objective value, computing it when needed."""
        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        """Return the total coil derivative through the surface solve."""
        if self._dJ is None:
            self.compute()
        return self._dJ

    def recompute_bell(self, parent=None):
        """Clear the cached value and derivative after a parent changes.

        Args:
            parent (Optimizable | None): Parent notifying a DOF change; unused.
        """
        self._J = None
        self._dJ = None

    def compute(self):
        """Cache the objective and total coil derivative, re-solving if needed."""
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
        x = booz_surf._decision_vector(iota, G)
        value, dJ_dx, dJ_dcoils = boozer_residual_objective(
            problem, _place(x, problem), optimize_G=G is not None, weight_inv_modB=weight_inv_modB
        )
        self._J = host_tree(value)[()]

        P, L, U = res["PLU"]
        adj = forward_backward(P, L, U, host_tree(dJ_dx))
        explicit = _coil_derivative(self.biotsavart, dJ_dcoils)
        self._dJ = explicit - res["vjp"](adj, booz_surf, iota, G)


def _exact_coil_vjp(lm, booz_surf: JaxBoozerSurface, iota, G) -> Derivative:
    """Return the exact-system coil VJP at current surface DOFs and masked rows."""
    assert G is not None
    problem = booz_surf._problem()
    x = booz_surf._decision_vector(iota, G)
    rows = boozer_exact_residual_rows(booz_surf.surface, problem.targetlabel)
    return _coil_derivative(
        booz_surf.biotsavart, boozer_exact_residual_coil_vjp(problem, _place(x, problem), rows, _place(lm, problem))
    )


def _penalty_coil_vjp(
    lm, booz_surf: JaxBoozerSurface, iota, G, weight_inv_modB=True, *, constraint_weight: float
) -> Derivative:
    """Return the penalty-gradient coil VJP, including label and G dependence."""
    problem = booz_surf._problem(constraint_weight)
    return _coil_derivative(
        booz_surf.biotsavart,
        boozer_penalty_coil_vjp(
            problem,
            _place(booz_surf._decision_vector(iota, G), problem),
            _place(lm, problem),
            optimize_G=G is not None,
            weight_inv_modB=weight_inv_modB,
        ),
    )
