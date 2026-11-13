import importlib.util

import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve

from .._core.optimizable import Optimizable
from .surfacerzfourier import SurfaceRZFourier
from .surfaceobjectives import ToroidalFlux

# torux is imported only when a surface is computed: importing it caps the threads of
# the BLAS libraries in the process (see TORUX_THREADS in torux's documentation).
torux_available = importlib.util.find_spec("torux") is not None

__all__ = ['ToruxSurface']


class ToruxSurface(Optimizable):
    r"""
    The ToruxSurface class computes a flux surface of a BiotSavart magnetic field with the solver
    torux [1], as a :obj:`~simsopt.geo.SurfaceRZFourier` in straight-field-line angles.  It has the
    structure of :obj:`~simsopt.geo.BoozerSurface`, and can take its place in the objectives that
    need only the surface and the rotational transform, :obj:`~simsopt.geo.Iotas` and
    :obj:`~simsopt.geo.MajorRadius`, derivatives with respect to the coils included.

    The surface is

        .. math::

            R(\theta, \phi) = \sum_{m,n} r_{c,m,n} \cos(m\theta - n_{\text{fp}} n \phi)
                + r_{s,m,n} \sin(m\theta - n_{\text{fp}} n \phi), \quad
            Z(\theta, \phi) = \sum_{m,n} z_{c,m,n} \cos(m\theta - n_{\text{fp}} n \phi)
                + z_{s,m,n} \sin(m\theta - n_{\text{fp}} n \phi),

    with :math:`r_{s,m,n} = z_{c,m,n} = 0` for a stellarator symmetric surface, where :math:`\phi`
    is the cylindrical angle and :math:`\theta` the poloidal angle in which field lines are straight
    on the surface. Its unknowns are the coefficients of ``surface`` and the rotational transform
    :math:`\iota`. A field line on the surface obeys

        .. math::

            R_\phi + \iota R_\theta = R B_R / B_\phi, \quad
            Z_\phi + \iota Z_\theta = R B_Z / B_\phi,

    and torux solves the Galerkin projection of these two equations together with the toroidal
    flux constraint :math:`\Psi = \Psi_0` by Newton's method, at a resolution it chooses. Without
    stellarator symmetry, the origin of :math:`\theta` is free and one projected equation is implied
    by the others up to the truncation, since the flux of :math:`\mathbf B` out of any closed
    surface is zero: a phase condition fixes the origin to that of torux's own surface, and the
    implied equation is eliminated with the multiplier torux carries for it, so that there are as
    many equations as unknowns. At equal ``mpol`` and ``ntor``, the surface has about two thirds of
    the degrees of freedom of a :obj:`~simsopt.geo.SurfaceXYZTensorFourier` with the same symmetry.

    The unknowns of torux's equations are ``[surface.get_dofs(), iota]`` in this order, with or
    without stellarator symmetry, so the adjoint machinery of :obj:`~simsopt.geo.BoozerSurface`
    applies unchanged: ``res['PLU']`` is the LU factorization of the Jacobian of these equations and
    ``res['vjp']`` their vector-Jacobian product with respect to the coils. :math:`G` is not an
    unknown, and ``res['G']`` is None. torux solves on one field period, so the derivatives with
    respect to the coils are exact for coils whose copies share their degrees of freedom, as
    :obj:`~simsopt.field.coils_via_symmetries` builds them.

    Where a flux surface exists, it is the surface a :obj:`~simsopt.geo.QfmSurface` approximates. A
    QFM surface minimizes the quadratic flux over surfaces whose parameterization is free. Here the
    angles are fixed, which removes that nonuniqueness and leaves a square system for Newton's
    method. Where no flux surface exists, these equations have no solution.

    The angles are not Boozer angles. Objectives that need Boozer angles,
    :obj:`~simsopt.geo.NonQuasiSymmetricRatio` and :obj:`~simsopt.geo.BoozerResidual`, do not
    accept a ToruxSurface.

    torux is an optional dependency (``pip install torux``, Python 3.12 or newer): version 0.1.1 or
    newer for stellarator symmetric surfaces, 0.1.3 or newer without stellarator symmetry.

    *[1]: P. Bogdan, torux, https://github.com/philippbogdan/torux*
    """

    def __init__(self, biotsavart, surface, label, targetlabel, options=None):
        """
        Args:
            biotsavart (:obj:`~simsopt.field.BiotSavart`): BiotSavart object.
            surface (:obj:`~simsopt.geo.SurfaceRZFourier`): the surface, with or without stellarator
                symmetry. Its ``mpol`` and ``ntor`` fix the resolution of the solution, and its quadrature
                points are kept.
            label (:obj:`~simsopt.geo.ToroidalFlux`): the toroidal flux through the surface, the only
                label torux's equations take.
            targetlabel (float): the target value of the toroidal flux, in Wb.
            options (dict, Optional): A dictionary of solver options. If a keyword is not specified, then a
                default value is used. Possible keywords are:

                - `verbose` (bool): display convergence information. Defaults to True.
                - `newton_tol` (float): tolerance on the largest residual of torux's equations at the
                  resolution of ``surface``. Defaults to 1e-13.
                - `newton_maxiter` (int): maximum number of Newton iterations at the resolution of
                  ``surface``. Defaults to 10.
                - `verify` (bool): let torux check the surface with its independent evaluator. Defaults to True.
                - `budget_s` (float): time budget of torux, in seconds. Defaults to None, torux's own.
        """
        if not torux_available:
            raise RuntimeError("ToruxSurface requires torux, which can be installed with 'pip install torux' "
                               "(Python 3.12 or newer).")
        super().__init__(depends_on=[biotsavart])

        if not isinstance(surface, SurfaceRZFourier):
            raise ValueError("The input surface must be a SurfaceRZFourier.")
        if not isinstance(label, ToroidalFlux):
            raise ValueError("The label must be a ToroidalFlux: torux's equations constrain the toroidal flux.")

        self.biotsavart = biotsavart
        self.surface = surface
        self.label = label
        self.targetlabel = targetlabel
        self.need_to_run_code = True

        if options is None:
            options = {}
        options.setdefault('verbose', True)
        options.setdefault('newton_tol', 1e-13)
        options.setdefault('newton_maxiter', 10)
        options.setdefault('verify', True)
        options.setdefault('budget_s', None)
        self.options = options

    def recompute_bell(self, parent=None):
        self.need_to_run_code = True

    def run_code(self, iota=None, G=None):
        """
        Compute the flux surface with torux, then solve torux's equations at the resolution of
        ``surface`` by Newton's method, starting from torux's solution. When ``surface`` has
        torux's own resolution, the solution is copied coefficient for coefficient and Newton's
        method only checks it.

        Args:
            iota (float, Optional): not used, since torux needs no initial guess. Accepted so that
                the objectives that call :obj:`~simsopt.geo.BoozerSurface.run_code` can call this one.
            G (float, Optional): not used, for the same reason.

        Returns:
            dict: A dictionary containing the results. The dictionary contains the following keys:

                - `"success"`: True if torux found the surface (and verified it, unless the option `verify`
                  is False) and Newton's method converged
                - `"status"`: torux's outcome, e.g. ``'surface'`` or ``'undetermined'``
                - `"iota"`: the rotational transform on the surface
                - `"G"`: None
                - `"s"`: the surface object
                - `"residual"`: the residual of torux's equations, shape ``(n+1,)`` for ``n`` surface dofs
                - `"jacobian"`: its Jacobian with respect to ``[surface dofs, iota]``, shape ``(n+1, n+1)``
                - `"PLU"`: the LU decomposition of the Jacobian
                - `"vjp"`: the vector-Jacobian product of the residual with respect to the coils
                - `"iter"`: the number of Newton iterations at the resolution of ``surface``
                - `"torux_mpol_ntor"`: the resolution torux chose
                - `"answer"`: torux's answer, with its evidence
                - `"type"`: 'torux'
        """
        if not self.need_to_run_code:
            return self.res

        import torux

        s = self.surface
        answer = torux.solve(self.biotsavart, flux=self.targetlabel, budget_s=self.options['budget_s'],
                             verify=self.options['verify'])
        t = answer.surface
        res = {"success": False, "status": answer.status, "iota": np.nan, "G": None, "s": s,
               "answer": answer, "type": "torux"}
        if answer.status != "surface":
            self.res = res
            self.need_to_run_code = False
            if self.options['verbose']:
                print(f"torux - {answer.status}", flush=True)
            return res
        if t.nfp != s.nfp:
            raise RuntimeError(f"torux found a surface with {t.nfp} field periods, and the surface has nfp={s.nfp}.")
        if s.stellsym and (hasattr(t, "rs") or t.phi0 != 0):
            raise RuntimeError("torux found a surface without stellarator symmetry about phi = 0: a ToruxSurface of "
                               "it needs a SurfaceRZFourier with stellsym=False.")
        if not s.stellsym and not hasattr(answer, "system_general"):
            raise RuntimeError("A ToruxSurface without stellarator symmetry needs torux 0.1.3 or newer "
                               "('pip install -U torux').")

        # torux's solution at the resolution of the surface, in the series of the cylindrical angle
        mpol, ntor = s.mpol, s.ntor
        m, n = min(mpol, t.M), min(ntor, t.N)
        if s.stellsym:
            names, coefs = ("rc", "zs"), (t.rc, t.zs)
        else:
            names, coefs = ("rc", "rs", "zc", "zs"), _series_in_phi(t)
        for name, c in zip(names, coefs):
            a = np.zeros((mpol + 1, 2 * ntor + 1))
            a[:m + 1, ntor - n:ntor + n + 1] = np.asarray(c)[:m + 1, t.N - n:t.N + n + 1]
            getattr(s, name)[:] = a
        s.set_dofs(s.get_dofs())
        x = np.concatenate((s.get_dofs(), [t.iota]))

        def system(iota):
            if s.stellsym:
                return answer.system(s.rc, s.zs, iota)
            return answer.system_general(s.rc, s.rs, s.zc, s.zs, iota)

        i = 0
        r, J = system(x[-1])
        norm = np.linalg.norm(r, ord=np.inf)
        while i < self.options['newton_maxiter'] and norm > self.options['newton_tol']:
            lu_piv = lu_factor(J)
            dx = lu_solve(lu_piv, r)
            dx += lu_solve(lu_piv, r - J @ dx)
            x = x - dx
            s.set_dofs(x[:-1])
            r, J = system(x[-1])
            norm = np.linalg.norm(r, ord=np.inf)
            i += 1

        P, L, U = lu(J)
        res.update({"success": bool(norm <= self.options['newton_tol']), "iota": x[-1], "residual": r,
                    "jacobian": J, "PLU": (P, L, U), "vjp": toruxsurface_dresidual_dcoils_vjp, "iter": i,
                    "torux_mpol_ntor": (t.M, t.N)})
        self.res = res
        self.need_to_run_code = False

        if self.options['verbose']:
            print(f"torux - {answer.status}, NEWTON at mpol={mpol}, ntor={ntor} - {res['success']}  iter={i}, "
                  f"iota={res['iota']:.16f}, ||residual||_inf = {norm:.3e}", flush=True)
        return res


def _series_in_phi(t):
    r"""
    The coefficients ``rc, rs, zc, zs`` of torux's surface ``t`` in the series of the cylindrical
    angle, :math:`\cos(m\theta - n_{\text{fp}} n \phi)` and :math:`\sin(m\theta - n_{\text{fp}} n \phi)`:
    torux writes its series in :math:`\phi - \phi_0`, with :math:`\phi_0` the plane of a stellarator
    symmetric surface, and ``rs`` and ``zc`` are zero for such a surface.
    """
    rc, zs = np.asarray(t.rc, dtype=float), np.asarray(t.zs, dtype=float)
    rs = np.asarray(t.rs, dtype=float) if hasattr(t, "rs") else np.zeros_like(rc)
    zc = np.asarray(t.zc, dtype=float) if hasattr(t, "zc") else np.zeros_like(rc)
    if t.phi0 == 0:
        return rc, rs, zc, zs
    # cos(a + d) = cos(a) cos(d) - sin(a) sin(d) and sin(a + d) = sin(a) cos(d) + cos(a) sin(d), d = n nfp phi0
    d = np.arange(-t.N, t.N + 1)[None, :] * t.nfp * t.phi0
    cd, sd = np.cos(d), np.sin(d)
    return rc * cd + rs * sd, rs * cd - rc * sd, zc * cd + zs * sd, zs * cd - zc * sd


def toruxsurface_dresidual_dcoils_vjp(lm, torux_surface, iota, G):
    r"""
    The vector-Jacobian product :math:`\lambda^T \partial \mathbf F / \partial \text{coils}` of the
    residual :math:`\mathbf F` of torux's equations (the Galerkin residuals, the flux constraint
    and, without stellarator symmetry, the phase condition) with respect to the coil degrees of
    freedom and currents, at fixed surface coefficients and rotational transform.

    torux holds each coil :math:`k` as a Fourier series :math:`\mathbf c_k` of order :math:`K`,
    whose values at the quadrature points of the curve are :math:`\Phi \mathbf c_k = \gamma_k`.
    The product is carried to the curve by :math:`\partial \mathbf F / \partial \gamma_k = \Phi^{+T}
    \partial \mathbf F / \partial \mathbf c_k`, exact for curves that are such Fourier series.

    Args:
        lm (array, shape ``(n+1,)``): adjoint variable, one entry per equation.
        torux_surface (:obj:`ToruxSurface`): the surface, after :obj:`ToruxSurface.run_code`.
        iota (float): not used, the rotational transform is in ``torux_surface.res``.
        G (None): not used.

    Returns:
        :obj:`~simsopt._core.derivative.Derivative`: the product, with respect to the coils of
        ``torux_surface.biotsavart``.
    """
    res = torux_surface.res
    s = torux_surface.surface
    if s.stellsym:
        v = res['answer'].vjp(s.rc, s.zs, res['iota'], lm)
    else:
        v = res['answer'].vjp_general(s.rc, s.rs, s.zc, s.zs, res['iota'], lm)
    coef, current, d_coef, d_current = v['coef'], v['current'], v['d_coef'], v['d_current']
    coils = torux_surface.biotsavart.coils
    if coef.shape[0] != len(coils):
        raise RuntimeError(f"torux holds {coef.shape[0]} coils, the BiotSavart {len(coils)}.")
    nk = coef.shape[2]
    out = None
    for k, coil in enumerate(coils):
        t = coil.curve.quadpoints
        Phi = np.ones((t.size, nk))
        for j in range(1, (nk + 1) // 2):
            Phi[:, 2 * j - 1] = np.sin(2 * np.pi * j * t)
            Phi[:, 2 * j] = np.cos(2 * np.pi * j * t)
        gamma = coil.curve.gamma()
        if t.size < nk or np.max(np.abs(Phi @ coef[k].T - gamma)) > 1e-12 * np.max(np.abs(gamma)) \
                or not np.isclose(current[k], coil.current.get_value(), rtol=1e-14):
            raise RuntimeError(f"Coil {k} is not the Fourier series of order {(nk - 1) // 2} that torux holds "
                               f"at its {t.size} quadrature points.")
        v_gamma = np.linalg.pinv(Phi).T @ d_coef[k].T
        d = coil.vjp(v_gamma, np.zeros_like(gamma), np.array([d_current[k]]))
        out = d if out is None else out + d
    return out
