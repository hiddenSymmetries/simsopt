import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve
from scipy.sparse import csr_matrix

from .._core.optimizable import Optimizable
from .surfacerzfourier import SurfaceRZFourier

__all__ = ['PestSurface']


def pest_surface_residual(surface, iota, biotsavart, derivatives=1):
    r"""
    Compute the residual :math:`(E_R, E_Z)` of the field-line equations of a :obj:`PestSurface` at its
    collocation points, flattened to ``[E_R, E_Z]``, along with its Jacobian with respect to
    ``[surface.get_dofs(), iota]`` and its derivative with respect to the Cartesian components of
    :math:`\mathbf B` at the points. Returns the tuple ``(res, dres, dres_dB)``, or ``(res, dres_dB)`` if
    ``derivatives`` is 0.

    The collocation points are the points :math:`\theta_i = 2\pi i/(2 m_{\text{pol}} + 1)`,
    :math:`\phi_j = 2\pi j/((2 n_{\text{tor}} + 1) n_{\text{fp}})` of one field period. With stellarator
    symmetry, :math:`E_R` is odd and :math:`E_Z` even under :math:`(\theta, \phi) \to (-\theta, -\phi)`, so
    only half of the grid is used: the point :math:`(i, j) = (m, n)` for each cosine mode :math:`(m, n)` of
    the surface. Without it, the whole grid is used. In both cases the first point is
    :math:`\theta = \phi = 0`.
    """
    s = surface
    nfp, mpol, ntor = s.nfp, s.mpol, s.ntor
    if s.stellsym:
        ncos = ntor + 1 + mpol * (2 * ntor + 1)
        i, j = s.m[:ncos], s.n[:ncos]
    else:
        i, j = (a.ravel() for a in np.meshgrid(np.arange(2 * mpol + 1), np.arange(2 * ntor + 1), indexing='ij'))
    theta = 2 * np.pi * i / (2 * mpol + 1)
    phi = 2 * np.pi * j / ((2 * ntor + 1) * nfp)
    npoints = phi.size

    # each dof is the coefficient of cos or sin of m theta - nfp n phi in R or in Z, and sin(a) = cos(a - pi/2)
    names = s.local_full_dof_names
    is_R = np.array([name[0] == 'r' for name in names])
    is_sin = np.array([name[1] == 's' for name in names])
    angle = theta[:, None] * s.m - phi[:, None] * nfp * s.n - np.pi / 2 * is_sin
    basis = np.cos(angle)
    # the derivative of each basis function with respect to its angle
    dbasis = -np.sin(angle)
    del angle

    x = s.get_dofs()
    R, Z = basis @ (x * is_R), basis @ (x * ~is_R)
    # d/dphi + iota d/dtheta multiplies the derivative of each basis function by iota m - nfp n
    k = iota * s.m - nfp * s.n

    points = np.stack((R * np.cos(phi), R * np.sin(phi), Z), axis=1)
    biotsavart.set_points(points)
    B = biotsavart.B()
    # the unit vectors e_R, e_phi, e_Z at the points, as the rows of e[p]
    zero, one = np.zeros_like(phi), np.ones_like(phi)
    e = np.stack((np.stack((np.cos(phi), np.sin(phi), zero), axis=1),
                  np.stack((-np.sin(phi), np.cos(phi), zero), axis=1),
                  np.stack((zero, zero, one), axis=1)), axis=1)
    # the cylindrical components of B, and grad[p, a, b] the derivative of B_b along e_a, with
    # dB[p, i, j] the derivative of B_j along x_i
    Bcyl = np.einsum('pai,pi->pa', e, B)
    Bphi = Bcyl[:, 1]
    if derivatives:
        grad = np.einsum('pai,pij,pbj->pab', e, biotsavart.dB_by_dX(), e)
        dres = np.empty((2 * npoints, x.size + 1))

    res = np.empty(2 * npoints)
    dres_dB = np.empty((2 * npoints, 3))
    for rows, a, own in ((slice(0, npoints), 0, is_R), (slice(npoints, None), 2, ~is_R)):
        # E_a = (d/dphi + iota d/dtheta) X_a - R B_a / B_phi, with X_a = R or Z
        Ba = Bcyl[:, a]
        res[rows] = dbasis @ (k * x * own) - R * Ba / Bphi
        dres_dB[rows] = -(R / Bphi)[:, None] * e[:, a] + (R * Ba / Bphi**2)[:, None] * e[:, 1]
        if derivatives:
            # the derivatives of R B_a / B_phi with respect to R and Z
            dV_dR = Ba / Bphi + R * (grad[:, 0, a] * Bphi - Ba * grad[:, 0, 1]) / Bphi**2
            dV_dZ = R * (grad[:, 2, a] * Bphi - Ba * grad[:, 2, 1]) / Bphi**2
            dres[rows, :-1] = k * own * dbasis
            dres[rows, :-1] -= np.where(is_R, dV_dR[:, None], dV_dZ[:, None]) * basis
            dres[rows, -1] = dbasis @ (s.m * x * own)
    if derivatives:
        return res, dres, dres_dB
    return res, dres_dB


def pest_surface_dcoils_dcurrents_vjp(lm, pest_surface, iota, G):
    """
    Vector-Jacobian product of the equations of a :obj:`PestSurface` with the vector ``lm``, with respect to
    the coil and current dofs, at fixed surface dofs and ``iota``. ``G`` is not used. Returns a ``Derivative``.
    """
    surface, biotsavart = pest_surface.surface, pest_surface.biotsavart
    _, dres_dB = pest_surface_residual(surface, iota, biotsavart, derivatives=0)
    # the equations are the residual with the label constraint in place of its first entry, then without
    # stellarator symmetry the condition on the origin of theta, which does not depend on the coils
    npoints = dres_dB.shape[0] // 2
    lm_res = lm[:2 * npoints].copy()
    lm_label, lm_res[0] = lm_res[0], 0.
    lm_times_dres_dB = lm_res[:npoints, None] * dres_dB[:npoints] + lm_res[npoints:, None] * dres_dB[npoints:]
    lm_times_dres_dcoils = biotsavart.B_vjp(lm_times_dres_dB)
    lm_times_dlabel_dcoils = lm_label * pest_surface.label.dJ(partials=True)(biotsavart, as_derivative=True)
    return lm_times_dres_dcoils + lm_times_dlabel_dcoils


class PestSurface(Optimizable):
    r"""
    The PestSurface class computes a flux surface of a BiotSavart magnetic field as a
    :obj:`~simsopt.geo.SurfaceRZFourier`, in which :math:`\phi` is the cylindrical angle and :math:`\theta`
    the poloidal angle in which field lines are straight (PEST angles). It has the interface of
    :obj:`~simsopt.geo.BoozerSurface`, so :obj:`~simsopt.geo.Iotas` and :obj:`~simsopt.geo.MajorRadius`
    accept it, with their derivatives with respect to the coils.

    On a flux surface :math:`(R, Z)(\theta, \phi)`, a field line advances :math:`\theta` by the rotational
    transform :math:`\iota` for each radian of :math:`\phi`, so that

        .. math::

            E_R = \frac{\partial R}{\partial \phi} + \iota \frac{\partial R}{\partial \theta}
                - \frac{R B_R}{B_\phi} = 0, \qquad
            E_Z = \frac{\partial Z}{\partial \phi} + \iota \frac{\partial Z}{\partial \theta}
                - \frac{R B_Z}{B_\phi} = 0.

    The unknowns are the surface dofs and :math:`\iota`. The equations are :math:`E_R` and :math:`E_Z` at
    collocation points, as for the BoozerExact approach of :obj:`~simsopt.geo.BoozerSurface`, and the label
    constraint :math:`l = l_0`. The points are the :math:`(2 m_{\text{pol}} + 1) \times (2 n_{\text{tor}} + 1)`
    grid of one field period, or half of it with stellarator symmetry (see ``pest_surface_residual``), and
    they are chosen by the class, not taken from the quadrature points of ``surface``. :math:`E_R` at
    :math:`\theta = \phi = 0` is replaced by the label constraint: with stellarator symmetry it vanishes for
    any surface. Without stellarator symmetry, it is implied by the other equations, since the flux of
    :math:`\mathbf B` through a closed surface vanishes,

        .. math::

            \int_0^{2\pi} \int_0^{2\pi} B_\phi \left( \frac{\partial R}{\partial \theta} E_Z
                - \frac{\partial Z}{\partial \theta} E_R \right) d\theta ~d\phi
            = -\int_0^{2\pi} \int_0^{2\pi} \mathbf B \cdot \mathbf n ~d\theta ~d\phi = 0,

    and one more equation fixes the origin of :math:`\theta`:
    :math:`Z(0, 0) = \frac{1}{2\pi} \int_0^{2\pi} Z(\theta, 0) ~d\theta`, that is, the ``zc`` with
    :math:`m \ge 1` sum to zero. The system is square in both cases.

    It is solved by Newton's method from the initial guess given by ``surface`` and by the ``iota`` passed
    to :obj:`run_code`, which has to be close enough to the solution. On the NCSX coils of
    :obj:`~simsopt.configs.get_data`, for example, a torus of minor radius 0.1 m about the magnetic axis
    (as :obj:`~simsopt.geo.PeriodicFieldLine` computes it), made with ``surface.fit_to_curve``, is close
    enough at ``mpol = ntor = 6``, and the solution there, copied with ``surface.copy(mpol=..., ntor=...)``,
    is close enough at a higher resolution. In an optimization, each solve starts from the last.

    As for :obj:`~simsopt.geo.BoozerSurface`, ``res['success']`` means that Newton's method converged, not
    that the surface is a flux surface: inside an island chain or beyond the last closed flux surface, a torus
    that the field crosses can also solve the equations. On a flux surface, the largest
    :math:`|\mathbf B \cdot \mathbf n|/|\mathbf B|` on the surface falls with ``mpol`` and ``ntor``.

    ``res['PLU']`` is the LU factorization of the Jacobian with respect to ``[surface.get_dofs(), iota]``
    and ``res['vjp']`` the vector-Jacobian product with respect to the coils, as for
    :obj:`~simsopt.geo.BoozerSurface`. :math:`G` is not an unknown, and ``res['G']`` is None. The angles are
    not Boozer angles, so :obj:`~simsopt.geo.NonQuasiSymmetricRatio` and :obj:`~simsopt.geo.BoozerResidual`
    do not apply to a PestSurface.

    The algorithm comes from torux [1].

    *[1]: P. Bogdan, torux, https://github.com/philippbogdan/torux*
    """

    def __init__(self, biotsavart, surface, label, targetlabel, options=None):
        """
        Args:
            biotsavart (:obj:`~simsopt.field.BiotSavart`): BiotSavart object.
            surface (:obj:`~simsopt.geo.SurfaceRZFourier`): the initial guess, with or without stellarator
                symmetry. Its ``mpol`` and ``ntor`` are the resolution of the solution.
            label (:obj:`~simsopt._core.optimizable.Optimizable`): A method that computes a flux surface
                label for the surface, such as :obj:`~simsopt.geo.ToroidalFlux` or
                :obj:`~simsopt.geo.Volume`.
            targetlabel (float): The target value of the label on the surface.
            options (dict, Optional): A dictionary of solver options. If a keyword is not specified, then a
                default value is used. Possible keywords are:

                - `verbose` (bool): display convergence information. Defaults to True.
                - `newton_tol` (float): tolerance of Newton's method. Defaults to 1e-13.
                - `newton_maxiter` (int): maximum number of iterations of Newton's method. Defaults to 40.
        """
        super().__init__(depends_on=[biotsavart])

        if not isinstance(surface, SurfaceRZFourier):
            raise ValueError("The input surface must be a SurfaceRZFourier.")

        self.biotsavart = biotsavart
        self.surface = surface
        self.label = label
        self.targetlabel = targetlabel
        self.need_to_run_code = True

        if options is None:
            options = {}
        options.setdefault('verbose', True)
        options.setdefault('newton_tol', 1e-13)
        options.setdefault('newton_maxiter', 40)
        self.options = options

    def recompute_bell(self, parent=None):
        self.need_to_run_code = True

    def run_code(self, iota, G=None):
        """
        Run Newton's method from the current surface and ``iota``.

        Args:
            iota (float): Guess for the rotational transform on the surface.
            G (float, Optional): not used, since :math:`G` is not an unknown. Accepted so that the objectives
                that call :obj:`~simsopt.geo.BoozerSurface.run_code` can call this one.

        Returns:
            dict: the results, as described in :obj:`solve_residual_equation_exactly_newton`.
        """
        if not self.need_to_run_code:
            return self.res
        return self.solve_residual_equation_exactly_newton(tol=self.options['newton_tol'],
                                                           maxiter=self.options['newton_maxiter'], iota=iota,
                                                           verbose=self.options['verbose'])

    def equations(self, iota):
        """
        The equations of the surface at the current dofs of ``surface`` and ``iota``, and their Jacobian with
        respect to ``[surface.get_dofs(), iota]``: the residual of ``pest_surface_residual`` with the label
        constraint in place of its first entry, :math:`E_R` at :math:`\\theta = \\phi = 0`, and without
        stellarator symmetry the condition on the origin of :math:`\\theta` after it.
        """
        s = self.surface
        b, J, _ = pest_surface_residual(s, iota, self.biotsavart)
        b[0] = self.label.J() - self.targetlabel
        J[0] = np.append(self.label.dJ(partials=True)(s), 0.)
        if not s.stellsym:
            phase = np.array([name.startswith('zc') for name in s.local_full_dof_names]) & (s.m > 0)
            b = np.append(b, phase @ s.get_dofs())
            J = np.vstack((J, np.append(phase, 0.)))
        return b, J

    def solve_residual_equation_exactly_newton(self, tol=1e-13, maxiter=40, iota=0., G=None, verbose=False):
        """
        Solve the equations by Newton's method, starting from the current ``surface`` and ``iota``.

        Args:
            tol (float, Optional): tolerance on the largest entry of the residual. Defaults to 1e-13.
            maxiter (int, Optional): maximum number of iterations. Defaults to 40.
            iota (float, Optional): Guess for the rotational transform on the surface. Defaults to 0.
            G (float, Optional): not used, as in :obj:`run_code`.
            verbose (bool, Optional): print convergence information. Defaults to False.

        Returns:
            dict: A dictionary containing the results. The dictionary contains the following keys:

                - `"success"`: True if Newton's method converged
                - `"iter"`: the number of iterations
                - `"iota"`: the rotational transform on the surface
                - `"G"`: None
                - `"s"`: the surface object
                - `"residual"`: the residual of the equations at the last iterate
                - `"PLU"`: the LU factorization of the Jacobian at the last iterate
                - `"vjp"`: the vector-Jacobian product of the equations with respect to the coils
        """
        if not self.need_to_run_code:
            return self.res
        # the factorization of the last solve is not kept in memory during this one
        self.res = None
        s = self.surface
        x = np.concatenate((s.get_dofs(), [iota]))
        b, J = self.equations(iota)
        norm = np.linalg.norm(b, ord=np.inf)
        i = 0
        # Newton's method also stops if it diverges so far that the Jacobian overflows
        while i < maxiter and norm > tol and np.all(np.isfinite(J)):
            factors = lu_factor(J)
            dx = lu_solve(factors, b)
            # one step of iterative refinement: solve again for the residual of the solve, which recovers
            # the digits that the first solve loses to the conditioning of J
            dx += lu_solve(factors, b - J @ dx)
            # free this Jacobian and its factors before the next is assembled
            del J, factors
            x -= dx
            s.set_dofs(x[:-1])
            iota = x[-1]
            i += 1
            b, J = self.equations(iota)
            norm = np.linalg.norm(b, ord=np.inf)

        p, L, U = lu(J, p_indices=True, check_finite=False)
        # J = L[p] @ U, that is P @ L @ U with the sparse permutation matrix P below
        P = csr_matrix((np.ones(p.size), (np.arange(p.size), p)))
        res = {
            "success": norm <= tol, "iter": i, "iota": iota, "G": None, "s": s,
            "residual": b, "PLU": (P, L, U), "vjp": pest_surface_dcoils_dcurrents_vjp,
        }
        if verbose:
            print(f"NEWTON solve - {res['success']}  iter={i}, iota={iota:.16f}, "
                  f"||residual||_inf = {norm:.3e}", flush=True)
        self.res = res
        self.need_to_run_code = False
        return res
