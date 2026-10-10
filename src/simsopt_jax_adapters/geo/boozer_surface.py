"""Native ``BoozerSurface`` with its solves in JAX.

:class:`JaxBoozerSurface` takes what native ``BoozerSurface`` takes, with a
:class:`~simsopt_jax_adapters.field.JaxBiotSavart` field, and has native's
``run_code``, solvers, defaults and result dictionaries, so native objectives
such as ``Iotas``, ``MajorRadius``, ``NonQuasiSymmetricRatio`` and
``BoozerResidual`` use it unchanged (with a ``ToroidalFlux`` label, use
:class:`JaxBoozerResidual` for ``BoozerResidual``)::

    import numpy as np
    from simsopt.configs import get_data
    from simsopt.geo import Iotas, SurfaceXYZTensorFourier, Volume
    from simsopt_jax_adapters.field import JaxBiotSavart
    from simsopt_jax_adapters.geo.boozer_surface import JaxBoozerSurface

    base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
    surface = SurfaceXYZTensorFourier(
        mpol=6, ntor=6, stellsym=True, nfp=nfp,
        quadpoints_phi=np.linspace(0, 1 / nfp, 13, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, 13, endpoint=False),
    )
    surface.fit_to_curve(ma, 0.1, flip_theta=True)
    volume = Volume(surface)
    boozer_surface = JaxBoozerSurface(JaxBiotSavart(bs.coils), surface, volume, volume.J())
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
``dG/dI``, which native drops); :class:`JaxBoozerResidual` replaces native
``BoozerResidual``, whose explicit coil derivative misses the same terms
(a deliberate correction of a native bug). Native ``BoozerResidual`` stays
correct on ``JaxBoozerSurface`` for coil-independent ``Volume``, ``Area`` and
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
from simsopt_jax_adapters.field.biotsavart_backend import JaxBiotSavart

from .boozer_problem import boozer_exact_residual_mask, boozer_exact_residual_rows, boozer_problem
from .surface_specs import surface_spec_from_surface

__all__ = ["JaxBoozerResidual", "JaxBoozerSurface"]

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
    return explicit_device_array(np.asarray(values, dtype=np.float64), dtype=np.float64, reference=problem.targetlabel)


def _solution(x: np.ndarray, optimize_G: bool):
    """Native's layout of ``x``: the surface DOFs, ``iota`` and ``G`` (``None``
    unless optimized)."""
    if optimize_G:
        return x[:-2], x[-2], x[-1]
    return x[:-1], x[-1], None


class JaxBoozerSurface(Optimizable):
    r"""JAX version of :class:`simsopt.geo.BoozerSurface`: a magnetic surface of a
    :class:`~simsopt_jax_adapters.field.JaxBiotSavart` field parametrized by Boozer angles.

    At each surface quadrature point the Boozer residual is

    .. math::

        \mathbf r = w \left(G \mathbf B - \|\mathbf B\|^2
            (\mathbf x_\varphi + \iota \mathbf x_\theta)\right),

    with :math:`\mathbf x_\varphi`, :math:`\mathbf x_\theta` the surface tangents (angles in turns),
    :math:`\mathbf B` the field at :math:`\mathbf x`, and :math:`w = 1/\|\mathbf B\|` when
    ``weight_inv_modB`` is set, else :math:`w = 1`. The unknowns are the surface DOFs, :math:`\iota`
    and :math:`G`; when ``G`` is not optimized it is fixed at :math:`\mu_0 \sum_k |I_k|`.

    A truthy ``constraint_weight`` :math:`w_c` selects BoozerLS [2], which minimizes the penalty

    .. math::

        J(x) = \frac{1}{2 n_r} \|\mathbf r(x)\|^2 + \frac{w_c}{2} (l(x) - l_0)^2
            + \frac{w_c}{2} z(\varphi=0, \theta=0)^2,

    with :math:`n_r = 3 n_\varphi n_\theta` residual entries, label :math:`l` and target :math:`l_0`.
    Otherwise :meth:`run_code` uses BoozerExact [1], which solves by Newton's method

    .. math::

        \mathbf r_M(x) = 0, \qquad l(x) = l_0, \qquad z(\varphi=0, \theta=0) = 0,

    with :math:`w = 1`, :math:`\mathbf r_M` the residual entries of native's stellarator-symmetry
    mask, and the :math:`z` equation only without stellarator symmetry.

    Stopping rules follow native and are described on the solver methods below.
    A singular Newton step raises :class:`numpy.linalg.LinAlgError`, leaving the
    surface at the last iterate.

    References:
        [1] A. Giuliani, F. Wechsung, G. Stadler, A. Cerfon, M. Landreman, "Direct computation of
        magnetic surfaces in Boozer coordinates and coil optimization for quasisymmetry",
        J. Plasma Phys. 88(4), 905880401 (2022), doi:10.1017/S0022377822000563.

        [2] A. Giuliani, F. Wechsung, A. Cerfon, M. Landreman, G. Stadler, "Direct stellarator coil
        optimization for nested magnetic surfaces with precise quasi-symmetry", Phys. Plasmas
        30(4) (2023).

    Args:
        biotsavart (JaxBiotSavart): Field whose coil DOFs are this solver's parents.
        surface (SurfaceXYZFourier | SurfaceXYZTensorFourier): Mutable native surface;
            geometry has shape (nphi, ntheta, 3); exact solves require the tensor class.
        label (Volume | Area | AspectRatio | ToroidalFlux): Constraint label on the
            surface or a surface sharing its DOFs.
        targetlabel (float): Target label in cubic meters (volume), square meters
            (area), webers (toroidal flux), or dimensionless (aspect ratio).
        constraint_weight (float | None): Native penalty coefficient for the squared
            label and z constraints; None for an exact problem. Truthy selects BoozerLS;
            otherwise run_code uses BoozerExact. Default None.
        options (dict | None): Native solver options overriding the defaults for the
            selected formulation (``verbose`` True, ``newton_tol`` 1e-13 for BoozerExact
            or 1e-11 for BoozerLS, ``newton_maxiter`` 40, ``bfgs_tol`` 1e-10, ``bfgs_maxiter`` 1500,
            ``limited_memory`` False, ``weight_inv_modB`` True); default None uses the defaults.

    The label must be one of the four classes above; with ``G=None`` in :meth:`run_code` the coil
    currents must be fixed, as natively. ``minimize_boozer_exact_constraints_newton`` is not provided.
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

    def run_code(self, iota, G=None):
        """Native ``run_code``: BoozerExact Newton, or BFGS then the penalty
        Newton for BoozerLS, with the options' tolerances and caps.

        Args:
            iota (float): Initial dimensionless rotational transform.
            G (float | None): Initial G in tesla meters; None derives G from currents (fixed
                during penalty solves, optimized during exact solves).

        Returns:
            dict | None: Native solve result stored in res, after updating the surface; None
                if no solve is needed. BoozerLS runs BFGS then Newton, while BoozerExact
                runs exact Newton.
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
        """Native's: SciPy BFGS (or L-BFGS-B) on the penalty.

        Args:
            tol (float): SciPy gradient stopping tolerance (gtol), also relative objective
                stopping tolerance (ftol) for L-BFGS-B; default 1e-3.
            maxiter (int | float): Maximum SciPy iterations; default 1000.
            constraint_weight (float): Native penalty coefficient for the squared label and
                z constraints; default 1.0.
            iota (float): Initial dimensionless rotational transform; default 0.0.
            G (float | None): Initial G in tesla meters; default None derives G from currents.
                Penalty solves then keep G fixed; exact solves optimize G.
            limited_memory (bool): Use L-BFGS-B when true, otherwise BFGS; default True.
            weight_inv_modB (bool): Divide each point's Boozer residual by the field
                magnitude in teslas; default True.
            verbose (bool): Print the native-style solver summary; default False.

        Returns:
            dict: Cached or newly stored native-style result, including value, gradient,
                iteration count, success, iota and G; updates the surface DOFs.
        """
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
        """Native's: Newton on the penalty with its analytic Hessian.

        Args:
            tol (float): Euclidean penalty-gradient norm threshold; default 1e-12.
                Success is True when the final iterate's norm is at most tol,
                including when the last permitted step reaches maxiter.
            maxiter (int | float): Newton step cap; default 10. Positive fractional caps admit
                steps while the integer iteration count is below the cap.
            constraint_weight (float): Native penalty coefficient for the squared label and
                z constraints; default 1.0.
            iota (float): Initial dimensionless rotational transform; default 0.0.
            G (float | None): Initial G in tesla meters; default None derives G from currents.
                Penalty solves then keep G fixed; exact solves optimize G.
            stab (float): Diagonal Hessian shift for step solves, in native units; default 0.0.
            weight_inv_modB (bool): Divide each point's Boozer residual by the field
                magnitude in teslas; default True.
            verbose (bool): Print the native-style solver summary; default False.

        Returns:
            dict: Cached or newly stored native-style result with final derivatives,
                success, LU factors and coil VJP; updates the surface, including the last
                iterate before a singular-step error.
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
        (whose result, as natively, is returned but not stored in ``res``).

        Args:
            tol (float): SciPy ftol, xtol and gtol, or the Euclidean penalty-gradient
                norm threshold for manual Gauss-Newton; default 1e-12.
            maxiter (int | float): Maximum SciPy residual evaluations (max_nfev), or manual
                Gauss-Newton steps; default 10.
            constraint_weight (float): Native penalty coefficient for the squared label and
                z constraints; default 1.0.
            iota (float): Initial dimensionless rotational transform; default 0.0.
            G (float | None): Initial G in tesla meters; default None derives G from currents.
                Penalty solves then keep G fixed; exact solves optimize G.
            method (str): SciPy least_squares method, or manual for damped Gauss-Newton;
                default "lm".
            weight_inv_modB (bool): Divide each point's Boozer residual by the field
                magnitude in teslas; default True.

        Returns:
            dict: Native-style least-squares result after updating the surface. SciPy
                results are cached in res; the manual result is returned without storing it,
                as natively.
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
        the label and, without stellarator symmetry, ``z(0, 0)``.

        Args:
            tol (float): Threshold for the Euclidean norm of the full Newton system
                vector: Boozer residual entries selected by the native exact mask,
                followed by ``label - targetlabel`` and, without stellarator symmetry,
                ``z(0, 0)``. Default 1e-10; the norm combines residual and constraint
                units. The stopping norm and success follow
                :func:`simsopt_jax.core.boozer_solvers.boozer_exact_newton`.
            maxiter (int | float): Newton step cap; default 10. Positive fractional caps admit
                steps while the integer iteration count is below the cap.
            iota (float): Initial dimensionless rotational transform; default 0.0.
            G (float | None): Initial G in tesla meters; default None derives G from currents.
                Penalty solves then keep G fixed; exact solves optimize G.
            verbose (bool): Print the native-style solver summary; default False.

        Returns:
            dict: Cached or newly stored native-style exact solve result, including
                residual, Jacobian, mask, LU factors, coil VJP and success; updates the
                surface after completed steps.
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


def _coil_derivative(booz_surf: JaxBoozerSurface, cotangents) -> Derivative:
    return booz_surf.biotsavart.coil_cotangents_to_derivative(
        cotangents.field_inputs(), cotangents.coil_index_lists()
    )


class JaxBoozerResidual(Optimizable):
    r"""JAX version of :class:`simsopt.geo.BoozerResidual`: the Boozer residual penalty of a solved
    BoozerLS :class:`JaxBoozerSurface`,

    .. math::

        J = \frac{1}{2 n_r} \|\mathbf r\|^2 + \frac{w_c}{2} (l - l_0)^2,

    where :math:`\mathbf r` is the Boozer residual of :class:`JaxBoozerSurface` (with the solve's
    ``weight_inv_modB``) at the solved :math:`\iota` and :math:`G` on a
    ``SurfaceXYZTensorFourier`` copy of the solved surface with the same quadrature points,
    :math:`n_r = 3 n_\varphi n_\theta`, :math:`l` the solver's label with target :math:`l_0`, and
    :math:`w_c` the solver's ``constraint_weight`` at construction. Unlike the solver's penalty, it
    has no :math:`z(\varphi=0, \theta=0)` term. The surface is re-solved first when it needs to be.

    ``dJ`` is the total coil derivative through the solve, the explicit coil derivative of
    :math:`J` minus ``res["vjp"]`` of the adjoint. It includes the coil dependence of a
    ``ToroidalFlux`` label and, when ``G`` is not optimized, of :math:`G`, which native
    ``BoozerResidual`` drops; native ``BoozerResidual`` agrees with this class only for
    ``Volume``, ``Area`` and ``AspectRatio`` labels with ``G`` optimized or the currents fixed.

    References:
        A. Giuliani, F. Wechsung, G. Stadler, A. Cerfon, M. Landreman, "Direct computation of
        magnetic surfaces in Boozer coordinates and coil optimization for quasisymmetry",
        J. Plasma Phys. 88(4), 905880401 (2022), doi:10.1017/S0022377822000563.

    Args:
        boozer_surface (JaxBoozerSurface): Mutable BoozerLS solver; the objective owns a
            separate copy of its solved geometry.
        bs (JaxBiotSavart): Field with the solver's coils; evaluating this objective
            leaves its evaluation points unchanged except for shared label callbacks
            during re-solves.

    The solver must be BoozerLS (a truthy ``constraint_weight``); with a ``ToroidalFlux`` label use
    this class rather than native ``BoozerResidual``.
    """

    def __init__(self, boozer_surface: JaxBoozerSurface, bs: JaxBiotSavart):
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
        """Return the cached objective value, computing it when needed.


        Returns:
            float: Cached scalar objective, recomputed through the surface solve when
                invalidated, in native BoozerResidual penalty units.
        """
        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        """Return the total coil derivative through the surface solve.


        Returns:
            numpy.ndarray | Derivative: Total coil derivative including the surface-solve
                adjoint. By default the decorator projects to shape (nfree,) in the
                objective's free DOF order; partials=True returns Derivative.
        """
        if self._dJ is None:
            self.compute()
        return self._dJ

    def recompute_bell(self, parent=None):
        """Clear the cached value and derivative after a parent changes.

        Args:
            parent (Optimizable | None): Parent notifying this objective of changed DOFs;
                unused.
        """
        self._J = None
        self._dJ = None

    def compute(self):
        """Compute and cache the objective and its direct-minus-adjoint coil derivative,
        re-solving the Boozer surface when required."""
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


def _exact_coil_vjp(lm, booz_surf: JaxBoozerSurface, iota, G) -> Derivative:
    """Native ``boozer_surface_dexactresidual_dcoils_dcurrents_vjp`` at the
    surface's current DOFs and on its ``res['mask']`` rows."""
    assert G is not None
    problem = booz_surf._problem()
    x = booz_surf._decision_vector(iota, G)
    rows = boozer_exact_residual_rows(booz_surf.surface, problem.targetlabel)
    return _coil_derivative(
        booz_surf, boozer_exact_residual_coil_vjp(problem, _place(x, problem), rows, _place(lm, problem))
    )


def _penalty_coil_vjp(
    lm, booz_surf: JaxBoozerSurface, iota, G, weight_inv_modB=True, *, constraint_weight: float
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
