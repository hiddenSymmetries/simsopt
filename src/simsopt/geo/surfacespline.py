import warnings

import matplotlib.pyplot as plt
import numpy as np
import simsoptpp as sopp
from scipy.interpolate import CloughTocher2DInterpolator, griddata
from scipy.optimize import bisect, fsolve, newton

from .._core import Optimizable
from .._core.optimizable import DOFs
from ..geo import SurfaceRZFourier

# from ..mhd import Vmec
from ..util.mpi import MpiPartition
from ..util.spline_helpers import (
    b_p,
    b_p_deriv,
    b_p_deriv2,
    chord_length_knots,
    double_reflection_rmf,
    uniform_knots,
)
from .curve import Curve
from .surface import Surface

mpi = MpiPartition()

__all__ = ["CrossSectionFixedZeta", "PseudoAxis", "SurfaceBSpline"]


class CrossSectionFixedZeta(Optimizable):
    r"""
    Toroidal control point cross section, in local polar coordinates.
    Initializes to a polygon of control points approximating a circle.

    Parameters
    ----------
    zeta_index : int
        Index of the cross section, as tracked by the parent `PseudoAxisSurface`.
    n_ctrl_pts : int
        Number of control points per cross section (includes endpoints for
        z-symmetric cross sections).
    z_sym : bool
        Whether the cross section is up-down symmetric.
    equispaced : bool
        Whether to fix the polar-angle degrees of freedom of the control
        vectors.
    default_r : float
        Default radius.
    nurbs : bool
        Whether to use NURBS.
    """

    def __init__(
        self,
        zeta_index,
        n_ctrl_pts=7,
        z_sym=False,
        equispaced=False,
        default_r=0.3,
        nurbs=False,
    ):
        self.zeta_index = zeta_index
        self.n_ctrl_pts = n_ctrl_pts
        self.z_sym = z_sym
        self.nurbs = nurbs
        if z_sym:
            n_pts = (n_ctrl_pts // 2) + 1
            max_angle = np.pi
            # default behaviour: circular
            r_ctrl = default_r * np.ones(n_pts)
            theta_ctrl = np.arange(0, n_pts, 1) * 2 * np.pi / n_ctrl_pts
            w_ctrl = 0.5 * np.ones(n_pts)
            cs_dofs = np.concatenate([r_ctrl, theta_ctrl, w_ctrl])

            assert len(cs_dofs) == 3 * ((n_ctrl_pts // 2) + 1)
            if not n_ctrl_pts % 2:
                assert theta_ctrl[-1] == np.pi, f"theta_ctrl: {theta_ctrl}"
        else:
            n_pts = n_ctrl_pts
            max_angle = 2 * np.pi
            # default behaviour: circular
            r_ctrl = default_r * np.ones(n_pts)
            theta_ctrl = np.linspace(0, max_angle, n_pts + 1)[:-1]
            w_ctrl = 0.5 * np.ones(n_pts)
            cs_dofs = np.concatenate([r_ctrl, theta_ctrl, w_ctrl])

            assert len(cs_dofs) == 3 * n_ctrl_pts

        self.r_ctrl = cs_dofs[:n_pts]
        self.theta_ctrl = cs_dofs[n_pts : 2 * n_pts :]
        self.w_ctrl = cs_dofs[2 * n_pts :]
        self.n_pts = n_pts

        # naming dofs
        names = self._name_dofs(n_pts)

        if equispaced:
            dofs = DOFs(
                cs_dofs,
                names,
                [True] * (n_pts) + [False] * (n_pts) + [nurbs] * (n_pts),
                [0] * (n_pts)
                + (
                    np.linspace(0, max_angle, n_pts + 1)[:-1]
                    - 0.5 * np.linspace(0, max_angle, n_pts + 1)[1]
                ).tolist()
                + [0] * (n_pts),
                [1] * (n_pts)
                + (
                    np.linspace(0, max_angle, n_pts + 1)[1:]
                    - 0.5 * np.linspace(0, max_angle, n_pts + 1)[1]
                ).tolist()
                + [1] * (n_pts),
            )
            super().__init__(
                dofs=dofs,
                external_dof_setter=CrossSectionFixedZeta.set_dofs_impl,
            )
        else:
            dofs = DOFs(
                cs_dofs,
                names,
                # [True]*(n_pts) + [True]*(n_pts) + [nurbs]*(n_pts-1) + [False], # fixing one of the angles, weights in the cross section
                [True] * (n_pts)
                + [False]
                + [True] * (n_pts - 1)
                + [nurbs] * (n_pts - 1)
                + [
                    False
                ],  # fixing one of the angles, weights in the cross section
                [0] * (n_pts)
                + (
                    np.linspace(0, max_angle, n_pts + 1)[:-1]
                    - 0.5 * np.linspace(0, max_angle, n_pts + 1)[1]
                ).tolist()
                + [0] * (n_pts),
                [1] * (n_pts)
                + (
                    np.linspace(0, max_angle, n_pts + 1)[1:]
                    - 0.5 * np.linspace(0, max_angle, n_pts + 1)[1]
                ).tolist()
                + [1] * (n_pts),
            )
            super().__init__(
                dofs=dofs,
                external_dof_setter=CrossSectionFixedZeta.set_dofs_impl,
            )
            if z_sym:
                assert self.get("theta_0") == 0, (
                    f"theta_0 = {self.get('theta_0')}"
                )
                dofs.fix("theta_0")
                if not n_ctrl_pts % 2:
                    assert self.get(f"theta_{n_pts - 1}") == np.pi, (
                        f"theta_0 = {self.get(f'theta_{n_pts - 1}')}"
                    )
                    dofs.fix(f"theta_{n_pts - 1}")
                    new_bounds = np.linspace(0, max_angle, n_pts - 1)
                    for k in range(1, n_pts - 1):
                        dofs.update_bounds(
                            f"theta_{k}", (new_bounds[k - 1], new_bounds[k])
                        )

    def set_dofs_impl(self, v):
        """
        Set the shape coefficients from a 1D list/array, same layout as
        _name_dofs: [r_ctrl, theta_ctrl, w_ctrl], each n_pts long.

        Without this, r_ctrl/theta_ctrl/w_ctrl are just numpy views taken
        once at construction time (see __init__) -- they go stale silently
        whenever the underlying DOFs array gets *reassigned* rather than
        mutated in place, which is exactly what DOFs.full_x's setter does
        (self._x = new_array). Ordinary self.x = ... (DOFs.free_x's setter,
        self._x[self._free] = ...) mutates in place and never triggers
        this, so the bug is invisible unless something sets full_x -- which
        MPIFiniteDifference does, once, to broadcast fixed dofs across MPI
        ranks. The result: under real MPI-parallel finite differences,
        perturbing any cross-section dof was silently a no-op on a
        persistent SurfaceBSpline (PseudoAxis was never affected -- it
        already has this same kind of callback wired up).
        """
        n_pts = self.n_pts
        n = 3 * n_pts
        if len(v) != n:
            raise ValueError(
                f"Input vector should have {n} elements but instead has {len(v)}"
            )
        self.r_ctrl = v[:n_pts]
        self.theta_ctrl = v[n_pts : 2 * n_pts]
        self.w_ctrl = v[2 * n_pts :]

    def _name_dofs(self, n_pts):
        namelist = []
        for i in range(n_pts):
            namelist.append(f"r_{i}")
        for i in range(n_pts):
            namelist.append(f"theta_{i}")
        for i in range(n_pts):
            namelist.append(f"w_{i}")
        return namelist

    def get_r_ctrl_full(self):
        if self.z_sym:
            if self.n_ctrl_pts % 2 == 1:
                full = np.concatenate((self.r_ctrl, self.r_ctrl[:0:-1]))
            else:
                full = np.concatenate((self.r_ctrl, self.r_ctrl[-2:0:-1]))
        else:
            full = self.r_ctrl
        return full

    def get_theta_ctrl_full(self):
        if self.z_sym:
            if self.n_ctrl_pts % 2 == 1:
                full = np.concatenate(
                    (self.theta_ctrl, 2 * np.pi - self.theta_ctrl[:0:-1])
                )
            else:
                full = np.concatenate(
                    (self.theta_ctrl, 2 * np.pi - self.theta_ctrl[-2:0:-1])
                )
        else:
            full = self.theta_ctrl
        return full

    def get_w_ctrl_full(self):
        if self.z_sym:
            if self.n_ctrl_pts % 2 == 1:
                full = np.concatenate((self.w_ctrl, self.w_ctrl[:0:-1]))
            else:
                full = np.concatenate((self.w_ctrl, self.w_ctrl[-2:0:-1]))
        else:
            full = self.w_ctrl
        return full

    def flipped(self):
        if self.z_sym:
            return self
        else:
            r_flipped = np.insert(self.r_ctrl[:0:-1], 0, self.r_ctrl[0])
            ws_flipped = np.insert(self.w_ctrl[:0:-1], 0, self.w_ctrl[0])
            theta_flipped = 2 * np.pi - np.insert(
                self.theta_ctrl[:0:-1], 0, self.theta_ctrl[0]
            )
            dofs_flipped = np.concatenate(
                (r_flipped, theta_flipped, ws_flipped)
            )
            fixed_list = [
                self.is_fixed(key) for key in self.local_full_dof_names
            ]
            self.unfix_all()
            flipped_cs = CrossSectionFixedZeta(
                zeta_index=self.zeta_index,
                n_ctrl_pts=self.n_ctrl_pts,
                z_sym=False,
                nurbs=self.nurbs,
            )
            flipped_cs.unfix_all()
            flipped_cs.x = dofs_flipped
            # fixed_list holds bools, not dof names/indices -- fix() takes a
            # Key (str name or int index), and passing a bool straight
            # through silently does numpy boolean-mask indexing on the
            # underlying _free array instead (True fixes every dof at once,
            # False is a no-op), not "fix the dof at this position".
            for is_fixed, name in zip(fixed_list, self.local_full_dof_names):
                if is_fixed:
                    flipped_cs.fix(name)
                    self.fix(name)
            return flipped_cs


class PseudoAxis(sopp.Curve, Curve):
    r"""
    Pseudo-axis around which to build a spline surface. The number of dofs
    includes BOTH endpoints. Control points are vectors described in
    cylindrical coordinates.

    Parameters
    ----------
    n_ctrl_pts : int
        Number of control points.
    nfp : int
        Number of field periods.
    stellsym : bool
        Whether the axis is stellarator symmetric.
    axis_angles_fixed : bool
        Whether to freeze the dofs for the polar angle of control points.
    quadpoints : int or array-like
        Number of uniformly-spaced default quadpoints, or an explicit array
        of quadpoints (fractions in [0, 1)), matching CurveXYZFourier's
        convention -- no static default-quadpoints helper exists for Curve
        the way Surface has one.
    knot_parametrization : 'chord' or 'uniform'
        Whether the internal NURBS knot vector is spaced by actual
        chord length between control points ('chord', the default) or
        assumed evenly spaced regardless of where the control points
        are ('uniform', the original behavior). See
        "Chord-length parametrization.md" for the math and references.
    """

    def __init__(
        self,
        n_ctrl_pts=2,
        p=3,
        nfp=2,
        stellsym=True,
        axis_angles_fixed=False,
        quadpoints=61,
        knot_parametrization="chord",
    ):
        if knot_parametrization not in ("chord", "uniform"):
            raise ValueError(
                "knot_parametrization must be 'chord' or 'uniform', "
                f"got {knot_parametrization!r}"
            )
        self.n_ctrl_pts = n_ctrl_pts
        self.stellsym = stellsym
        self.nfp = nfp
        self.p = p
        self.knot_parametrization = knot_parametrization

        if isinstance(quadpoints, int):
            quadpoints = list(np.linspace(0, 1, quadpoints, endpoint=False))
        elif isinstance(quadpoints, np.ndarray):
            quadpoints = list(quadpoints)

        if not stellsym:
            max_angle = 2 * np.pi / nfp
        else:
            max_angle = np.pi / nfp

        # default behaviour: circular axis
        r_ctrl = np.ones(n_ctrl_pts)
        z_ctrl = np.zeros(n_ctrl_pts)
        zeta_ctrl = np.linspace(0, max_angle, n_ctrl_pts)
        axisdofs = np.concatenate([r_ctrl, z_ctrl, zeta_ctrl])

        self.r_ctrl = axisdofs[:n_ctrl_pts]
        self.z_ctrl = axisdofs[n_ctrl_pts : 2 * n_ctrl_pts]
        self.zeta_ctrl = axisdofs[2 * n_ctrl_pts :]

        # naming dofs
        names = self._name_dofs()
        dofs = DOFs(
            axisdofs,
            names,
            [True] * len(r_ctrl)
            + [True] * len(z_ctrl)
            + [not axis_angles_fixed] * len(zeta_ctrl),
            [0.3] * len(r_ctrl)
            + [-1] * len(z_ctrl)
            + (
                np.linspace(0, max_angle, len(zeta_ctrl))
                - (0.5 * max_angle) / (len(zeta_ctrl) - 1)
            ).tolist(),
            [2.5] * len(r_ctrl)
            + [1] * len(z_ctrl)
            + (
                np.linspace(0, max_angle, len(zeta_ctrl))
                + (0.5 * max_angle) / (len(zeta_ctrl) - 1)
            ).tolist(),
        )

        dofs.set("r_axis_0", 1)
        dofs.set("z_axis_0", 0)
        dofs.set("zeta_axis_0", 0)
        # dofs.fix('r_axis_0')
        dofs.fix("z_axis_0")
        dofs.fix("zeta_axis_0")
        if stellsym:
            dofs.set(f"z_axis_{n_ctrl_pts - 1}", 0)
            dofs.set(f"zeta_axis_{n_ctrl_pts - 1}", np.pi / self.nfp)
            dofs.fix(f"z_axis_{n_ctrl_pts - 1}")
            dofs.fix(f"zeta_axis_{n_ctrl_pts - 1}")

        # _control_net_and_knots cache -- see that method's docstring.
        self._centroids_im = None
        self._knots_a = None

        sopp.Curve.__init__(self, quadpoints)
        Curve.__init__(
            self, dofs=dofs, external_dof_setter=PseudoAxis.set_dofs_impl
        )

    def num_dofs(self):
        return len(self.full_x)

    def get_dofs(self):
        return self.full_x

    def _get_control_points_xyz(self):

        if self.stellsym:
            r_ctrl_1fp = np.append(self.r_ctrl, self.r_ctrl[-2::-1])[:-1]
            z_ctrl_1fp = np.append(self.z_ctrl, -self.z_ctrl[-2::-1])[:-1]
            zeta_ctrl_1fp = np.append(
                self.zeta_ctrl, (2 * np.pi / self.nfp) - self.zeta_ctrl[-2::-1]
            )[:-1]

        r_ctrl = np.tile(r_ctrl_1fp, self.nfp)
        z_ctrl = np.tile(z_ctrl_1fp, self.nfp)
        zeta_ctrl = np.concatenate(
            [zeta_ctrl_1fp + n * 2 * np.pi / self.nfp for n in range(self.nfp)]
        )
        x_ctrl = r_ctrl * np.cos(zeta_ctrl)
        y_ctrl = r_ctrl * np.sin(zeta_ctrl)

        xyz_list = np.vstack((x_ctrl, y_ctrl, z_ctrl)).T  # [:-1]
        return xyz_list

    def _control_net_and_knots(self):
        """
        Build (or return the cached) periodic-wrapped control net and knot
        vector shared by `gamma_impl`/`fsolve_centroid_axis_from_zetas`-style
        callers. Both are pure functions of the free dofs (r_ctrl, z_ctrl,
        zeta_ctrl) -- independent of any evaluation point -- so they're
        cached here and only rebuilt when the dofs have actually changed
        (self.new_x), rather than on every call (this used to be rebuilt
        from scratch on every single gamma/root-find evaluation, dominating
        runtime -- profiled at ~1200 rebuilds for one to_RZFourier() call).

        The knot vector gets special-cased: 'uniform' knots depend only on
        the control point count and p (fixed at construction), never on
        where the dofs actually put the points, so they're computed once,
        ever, and never invalidated by new_x. 'chord' knots do depend on
        point positions and are invalidated by new_x same as the net
        itself.
        """
        p = self.p
        recompute_net = self.new_x or self._centroids_im is None
        if self.knot_parametrization == "chord":
            recompute_knots = recompute_net or self._knots_a is None
        else:
            recompute_knots = self._knots_a is None

        if recompute_net:
            if self.stellsym:
                r_ctrl_1fp = np.append(self.r_ctrl, self.r_ctrl[-2::-1])[:-1]
                z_ctrl_1fp = np.append(self.z_ctrl, -self.z_ctrl[-2::-1])[:-1]
                zeta_ctrl_1fp = np.append(
                    self.zeta_ctrl,
                    (2 * np.pi / self.nfp) - self.zeta_ctrl[-2::-1],
                )[:-1]

            r_ctrl = np.tile(r_ctrl_1fp, self.nfp)
            z_ctrl = np.tile(z_ctrl_1fp, self.nfp)
            zeta_ctrl = np.concatenate(
                [
                    zeta_ctrl_1fp + n * 2 * np.pi / self.nfp
                    for n in range(self.nfp)
                ]
            )
            x_ctrl = r_ctrl * np.cos(zeta_ctrl)
            y_ctrl = r_ctrl * np.sin(zeta_ctrl)

            centroids_im = np.vstack((x_ctrl, y_ctrl, z_ctrl)).T

            if recompute_knots:
                if self.knot_parametrization == "chord":
                    knots_a = chord_length_knots(centroids_im, p)
                else:
                    knots_a = uniform_knots(centroids_im.shape[0] - 1, p)
                self._knots_a = knots_a

            # Periodic wraparound: tile p points from each end onto the
            # opposite side. Reflection symmetry doesn't depend on p's parity
            # here -- the mirror is already baked into r_ctrl/z_ctrl/zeta_ctrl
            # above (stellsym reflect-and-tile), not into how u maps to zeta,
            # so it survives regardless of where the knots fall. See
            # "Chord-length knots break stellarator symmetry.md" in the
            # Obsidian vault.
            centroids_im = np.concatenate(
                [centroids_im[-p:], centroids_im, centroids_im[:p]], axis=0
            )
            self._centroids_im = centroids_im

        if self.new_x:
            self.new_x = False

        return self._centroids_im, self._knots_a

    def _solve_v(self, phi, trimmed_ctrl_pts_im, knots_a):
        """
        Newton-solve for the NURBS parameter v(phi) such that the axis's
        toroidal angle atan2(Y(v), X(v)) matches the target phi (mod 2pi).
        v enters transcendentally through atan2 of a B-spline curve, so
        there's no closed form -- gamma_impl and the analytic derivatives
        in _gamma_and_derivs below all go through this same solve.
        """
        p = self.p

        def _xy_and_derivs(v):
            # wrap into the valid periodic domain (rather than clip) so scipy's
            # own, otherwise-unconstrained iterate always gets a well-defined
            # basis evaluation
            v_wrapped = v % (2 * np.pi)
            basis_a, dbasis_a = b_p_deriv(knots_a, p, v_wrapped)
            X = np.einsum("i,ti->t", trimmed_ctrl_pts_im[:, 0], basis_a)
            Y = np.einsum("i,ti->t", trimmed_ctrl_pts_im[:, 1], basis_a)
            dX = np.einsum("i,ti->t", trimmed_ctrl_pts_im[:, 0], dbasis_a)
            dY = np.einsum("i,ti->t", trimmed_ctrl_pts_im[:, 1], dbasis_a)
            return X, Y, dX, dY

        def func(v):
            X, Y, *_ = _xy_and_derivs(v)
            zeta_cur = np.arctan2(Y, X) % (2 * np.pi)
            return ((zeta_cur - phi + np.pi) % (2 * np.pi)) - np.pi

        def fprime(v):
            X, Y, dX, dY = _xy_and_derivs(v)
            return (X * dY - Y * dX) / (X**2 + Y**2)

        v_sol = newton(
            func, x0=phi.copy(), fprime=fprime, tol=1e-12, maxiter=50
        )
        return v_sol % (2 * np.pi)

    def _gamma_and_derivs(self, quadpoints, max_deriv=2):
        """
        Shared core for gamma/gammadash/gammadashdash, evaluated at
        arbitrary quadpoints (t, a fraction in [0,1)) rather than only
        self.quadpoints -- mirrors gamma_impl's own explicit-quadpoints
        convention (needed by e.g. _axis_rz's custom zeta grids), since the
        base Curve interface's gammadash_impl/gammadashdash_impl only ever
        get called with self.quadpoints.

        Implicit differentiation of the Newton-solved v(t): the axis is a
        plain B-spline X(v), Y(v), Z(v) in NURBS parameter v, with v(t)
        implicitly pinned by the toroidal-angle-matching constraint
        F(v, t) = atan2(Y(v), X(v)) - 2*pi*t = 0 (_solve_v's Newton solve).
        Differentiating F(v(t), t) = 0 w.r.t. t:
            dv/dt   = -F_t / F_v = 2*pi / G(v),   G(v) := dF/dv (= fprime)
            d2v/dt2 = -4*pi^2 * G'(v) / G(v)^3
        (product/quotient rule; G'(v) needs X'', Y'', i.e. the
        second-derivative basis from b_p_deriv2). Then, since
        gamma(t) = Gamma(v(t)), the chain rule gives
            dGamma/dt   = Gamma'(v) * v'(t)
            d2Gamma/dt2 = Gamma''(v) * v'(t)^2 + Gamma'(v) * v''(t)
        Verified against finite differences of gamma_impl.

        Returns (X, Y, Z) and, if max_deriv >= 1, also
        (dXdt, dYdt, dZdt), and if max_deriv >= 2, also
        (d2Xdt2, d2Ydt2, d2Zdt2).
        """
        trimmed_ctrl_pts_im, knots_a = self._control_net_and_knots()
        p = self.p
        phi = np.asarray(quadpoints) * 2 * np.pi
        v_sol = self._solve_v(phi, trimmed_ctrl_pts_im, knots_a)

        if max_deriv >= 2:
            basis_a, dbasis_a, d2basis_a = b_p_deriv2(knots_a, p, v_sol)
        elif max_deriv == 1:
            basis_a, dbasis_a = b_p_deriv(knots_a, p, v_sol)
        else:
            basis_a = b_p(knots_a, p, v_sol)

        X = np.einsum("i,ti->t", trimmed_ctrl_pts_im[:, 0], basis_a)
        Y = np.einsum("i,ti->t", trimmed_ctrl_pts_im[:, 1], basis_a)
        Z = np.einsum("i,ti->t", trimmed_ctrl_pts_im[:, 2], basis_a)
        out = (X, Y, Z)
        if max_deriv == 0:
            return out

        dX = np.einsum("i,ti->t", trimmed_ctrl_pts_im[:, 0], dbasis_a)
        dY = np.einsum("i,ti->t", trimmed_ctrl_pts_im[:, 1], dbasis_a)
        dZ = np.einsum("i,ti->t", trimmed_ctrl_pts_im[:, 2], dbasis_a)

        # G = F_v = fprime(v_sol) from _solve_v, recomputed here since
        # _solve_v doesn't hand it back.
        G = (X * dY - Y * dX) / (X**2 + Y**2)
        dvdt = 2 * np.pi / G
        dXdt = dX * dvdt
        dYdt = dY * dvdt
        dZdt = dZ * dvdt
        out = out + (dXdt, dYdt, dZdt)
        if max_deriv == 1:
            return out

        d2X = np.einsum("i,ti->t", trimmed_ctrl_pts_im[:, 0], d2basis_a)
        d2Y = np.einsum("i,ti->t", trimmed_ctrl_pts_im[:, 1], d2basis_a)
        d2Z = np.einsum("i,ti->t", trimmed_ctrl_pts_im[:, 2], d2basis_a)

        # G'(v): N = X*dY - Y*dX -> N' = X*d2Y - Y*d2X (the dX*dY, dY*dX
        # cross terms cancel); D = X^2+Y^2 -> D' = 2*(X*dX + Y*dY); G=N/D.
        N = X * dY - Y * dX
        D = X**2 + Y**2
        Np = X * d2Y - Y * d2X
        Dp = 2 * (X * dX + Y * dY)
        Gp = (Np * D - N * Dp) / D**2

        d2vdt2 = -4 * np.pi**2 * Gp / G**3
        d2Xdt2 = d2X * dvdt**2 + dX * d2vdt2
        d2Ydt2 = d2Y * dvdt**2 + dY * d2vdt2
        d2Zdt2 = d2Z * dvdt**2 + dZ * d2vdt2
        out = out + (d2Xdt2, d2Ydt2, d2Zdt2)
        return out

    def is_toroidally_monotonic(self, n_check=2000):
        """
        Check whether the axis's toroidal angle atan2(Y(v), X(v))
        increases monotonically with the NURBS parameter v -- the
        assumption gamma_impl's Newton solve (_solve_v) relies on to find
        a unique point at a given toroidal angle. Large dof perturbations
        (especially with few control points) can produce an axis whose
        projection onto the XY-plane briefly winds backward; there,
        the toroidal-angle-matching problem has multiple solutions or
        none nearby, and _solve_v either silently converges to the wrong
        branch or raises scipy's generic "failed to converge" warning --
        neither of which explains what's actually wrong. Checking this
        before trusting an axis shape is much cheaper and clearer than
        debugging a failed Newton solve after the fact.

        Returns True if atan2(Y(v), X(v)) is monotonically increasing
        over a dense sample of v in [0, 2*pi) -- this is a diagnostic,
        not part of gamma_impl's own hot path, so it isn't called there.
        """
        trimmed_ctrl_pts_im, knots_a = self._control_net_and_knots()
        p = self.p
        v = np.linspace(0, 2 * np.pi, n_check, endpoint=False)
        basis_a, dbasis_a = b_p_deriv(knots_a, p, v)
        X = np.einsum("i,ti->t", trimmed_ctrl_pts_im[:, 0], basis_a)
        Y = np.einsum("i,ti->t", trimmed_ctrl_pts_im[:, 1], basis_a)
        dX = np.einsum("i,ti->t", trimmed_ctrl_pts_im[:, 0], dbasis_a)
        dY = np.einsum("i,ti->t", trimmed_ctrl_pts_im[:, 1], dbasis_a)
        # sign of dphi/dv == sign of (X*dY - Y*dX) -- the shared
        # denominator X^2+Y^2 in the actual fprime is always positive.
        return bool(np.all(X * dY - Y * dX > 0))

    def gamma_impl(self, data, quadpoints):
        X, Y, Z = self._gamma_and_derivs(quadpoints, max_deriv=0)
        data[:, 0] = X
        data[:, 1] = Y
        data[:, 2] = Z

    def gammadash_impl(self, data):
        _, _, _, dXdt, dYdt, dZdt = self._gamma_and_derivs(
            self.quadpoints, max_deriv=1
        )
        data[:, 0] = dXdt
        data[:, 1] = dYdt
        data[:, 2] = dZdt

    def gammadashdash_impl(self, data):
        (_, _, _, _, _, _, d2Xdt2, d2Ydt2, d2Zdt2) = self._gamma_and_derivs(
            self.quadpoints, max_deriv=2
        )
        data[:, 0] = d2Xdt2
        data[:, 1] = d2Ydt2
        data[:, 2] = d2Zdt2

    def bishop_frame(self, quadpoints=None, n_prop=2000):
        r"""
        Rotation-minimizing (Bishop) frame of the axis, via the discrete
        double-reflection method (Wang, Juttler, Zheng, Liu, ACM TOG
        27(1), 2008), forced to close up exactly around the full torus.

        Unlike the Frenet frame, this frame does NOT respect the axis's
        own stellarator-symmetric reflection within a field period, and
        no correction can make it -- verified directly (see "Bishop frame
        stellarator symmetry.md" in the Obsidian vault for the full
        derivation): the Frenet frame, being built purely from local
        derivatives at each point, automatically inherits any exact
        symmetry of the curve, but the Bishop/RMF normal accumulates a
        path-dependent twist ("holonomy") as it's parallel-transported
        along the curve, and this holonomy has no reason to vanish or
        respect a discrete reflection symmetry. The correction that would
        be needed to force reflection-symmetry is provably impossible (it
        would have to be an odd function equal to a nonzero constant,
        which no function can be) -- this is a genuine geometric
        invariant of the axis shape, not a bug or a fixable choice of
        initial normal.

        What CAN be forced exactly is the field-period-to-field-period
        ROTATIONAL periodicity: a linear-in-t twist correction added on
        top of the raw propagated frame satisfies the periodicity
        relation exactly (unlike the reflection case), so the corrected
        frame tiles perfectly across all `nfp` field periods with no
        seam. Construction: propagate the raw frame over one field
        period, measure the (constant) rotational holonomy defect
        between its two ends, subtract a linear-in-t correction sized to
        cancel it exactly, then tile the corrected one-period frame by
        the nfp-fold rotation to cover the full axis.

        quadpoints : array-like of t (fraction in [0,1)), or None for
            self.quadpoints.
        n_prop : number of points used for the internal double-reflection
            propagation over one field period -- an accuracy knob,
            independent of how many points are actually being requested.

        Returns (T, N, B), each (n, 3): T is the unit tangent, N is the
        rotation-minimizing normal (up to the field-period tiling
        correction above), B = T x N.
        """
        if quadpoints is None:
            quadpoints = self.quadpoints
        quadpoints = np.asarray(quadpoints)
        nfp = self.nfp

        # dense propagation grid over one field period -- avoid the exact
        # t=0 point, which sits on the periodic wrap seam where the
        # pre-existing Newton solve in _solve_v has a floating-point-level
        # quirk (see gammadashdash_impl's validation).
        t_dense = np.linspace(1e-8, 1.0 / nfp - 1e-8, n_prop)
        X, Y, Z, dX, dY, dZ = self._gamma_and_derivs(t_dense, max_deriv=1)
        gamma_dense = np.vstack([X, Y, Z]).T
        gammadash_dense = np.vstack([dX, dY, dZ]).T

        # seed the propagation with R-hat (the radial direction in the R-Z
        # plane at the first point's own toroidal angle), projected to be
        # orthogonal to the tangent there -- rather than a bare hardcoded
        # axis, which would only coincide with R-hat by the accident of
        # t_dense[0] sitting at zeta=0.
        x0, y0 = gamma_dense[0, 0], gamma_dense[0, 1]
        normal0 = np.array([x0, y0, 0.0])
        normal0 /= np.linalg.norm(normal0)
        t0 = gammadash_dense[0]
        normal0 = normal0 - (normal0 @ t0) * t0 / (t0 @ t0)
        normal0 /= np.linalg.norm(normal0)
        _, N_dense, B_dense = double_reflection_rmf(
            gamma_dense, gammadash_dense, normal0
        )

        def _Rz(ang):
            c, s = np.cos(ang), np.sin(ang)
            return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

        Rzp = _Rz(2 * np.pi / nfp)

        # constant rotational holonomy defect over one field period: how
        # far the propagated N at the far end differs from Rz(2pi/nfp)
        # applied to N at the near end.
        N_pred_end = Rzp @ N_dense[0]
        B_pred_end = Rzp @ B_dense[0]
        cos_defect = N_dense[-1] @ N_pred_end
        sin_defect = N_dense[-1] @ B_pred_end
        rot_defect = np.arctan2(sin_defect, cos_defect)

        # linear-in-t correction: beta(t) = -rot_defect*nfp*t satisfies
        # beta(t) - beta(t + 1/nfp) = rot_defect exactly, which is exactly
        # what's needed for the corrected frame to tile without a seam.
        beta = -rot_defect * nfp * t_dense
        cB, sB = np.cos(beta), np.sin(beta)
        N_corr = cB[:, None] * N_dense + sB[:, None] * B_dense

        # evaluate at the actually-requested quadpoints: reduce into one
        # field period, interpolate the corrected propagated normal there
        # (T is cheap and exact, so it's recomputed directly rather than
        # interpolated), then tile by the rigid nfp-fold rotation and
        # re-orthonormalize against the exact tangent.
        k = np.floor(quadpoints * nfp).astype(int)
        t_local = quadpoints - k / nfp

        N_local = np.empty((len(quadpoints), 3))
        for i in range(3):
            N_local[:, i] = np.interp(t_local, t_dense, N_corr[:, i])

        N_rot = np.empty_like(N_local)
        for kk in np.unique(k):
            mask = k == kk
            Rk = _Rz(kk * 2 * np.pi / nfp)
            N_rot[mask] = (Rk @ N_local[mask].T).T

        X, Y, Z, dX, dY, dZ = self._gamma_and_derivs(quadpoints, max_deriv=1)
        gammadash = np.vstack([dX, dY, dZ]).T
        T = gammadash / np.linalg.norm(gammadash, axis=1, keepdims=True)

        N = N_rot - np.sum(N_rot * T, axis=1, keepdims=True) * T
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        B = np.cross(T, N)

        return T, N, B

    def set_dofs_impl(self, v):
        """
        Set the shape coefficients from a 1D list/array, same layout as
        get_dofs(): [r_ctrl, z_ctrl, zeta_ctrl], each n_ctrl_pts long.
        """
        # n_ctrl_pts, not num_dofs() -- num_dofs() reads self.full_x, which needs
        # self._unique_dof_opts, not set up yet the first time this runs (called
        # synchronously from inside Optimizable.__init__ itself, since dofs= is
        # supplied together with external_dof_setter=)
        n_ctrl = self.n_ctrl_pts
        n = 3 * n_ctrl
        if len(v) != n:
            raise ValueError(
                "Input vector should have "
                + str(n)
                + " elements but instead has "
                + str(len(v))
            )

        index = 0
        self.r_ctrl = v[index : index + n_ctrl]
        index += n_ctrl

        self.z_ctrl = v[index : index + n_ctrl]
        index += n_ctrl

        self.zeta_ctrl = v[index : index + n_ctrl]
        self.invalidate_cache()

    def _name_dofs(self):
        name_list = [
            f"{j}_axis_{i}"
            for j in ("r", "z", "zeta")
            for i in range(self.n_ctrl_pts)
        ]
        return name_list


class SurfaceBSpline(sopp.Surface, Surface):
    r"""
    B-spline surface, as described in Ali et al. (manuscript in progress).
    The main benefits of this representation are threefold: easy to
    box-bound to a space of diverse but feasible stellarator shapes, local
    control, and can be constrained to be unique.

    Notes
    -----
    `SurfaceBSpline` is a composite of two other classes: `PseudoAxis`, a
    B-spline curve intended to be constrained to lie in the interior of the
    control points, and the control point cross sections
    (`CrossSectionFixedZeta` or `CrossSectionFixedZetaCartesian`), which
    define the control points in local cartesian or polar coordinates in
    planes of constant toroidal angle, respectively.
    """

    def __init__(
        self,
        axis_points=4,
        points_per_cs=7,
        cs_equispaced=True,
        rays_equispaced=False,
        axis_angles_fixed=False,
        cs_global_angle_free=False,
        n_cs=2,
        nfp=2,
        M=12,
        N=12,
        p_u=2,
        p_v=2,
        default_r=0.5,
        stellsym=True,
        cs_basis="polar",
        nurbs=False,
        dofs=None,
        quadpoints_phi=None,
        quadpoints_theta=None,
        knot_parametrization="chord",
        use_bishop_frame=False,
    ):
        """
        Parameters
        ----------
        axis_points : int
            Number of points for the axis spline.
        points_per_cs : int
            Number of points per cross section.
        n_cs : int
            Number of toroidal cross sections per half field period.
        knot_parametrization : 'chord' or 'uniform'
            Whether the internal NURBS knot vectors (both u and v, and the
            pseudo-axis's own) are spaced by actual chord length between
            control points ('chord', the default) or assumed evenly spaced
            regardless of where the control points are ('uniform', the
            original behavior). See "Chord-length parametrization.md" for
            the math and references.
        use_bishop_frame : bool
            If False (the default), cross sections are placed in the
            fixed-zeta poloidal plane (independent of how the axis bends),
            matching the original behavior. If True, each cross section's
            local (r, theta) is instead rotated into the axis's own Bishop
            (rotation-minimizing) frame at that cross section's toroidal
            angle -- see PseudoAxis.bishop_frame -- letting the axis's own
            bending/twisting contribute to the boundary shape rather than
            always sitting flat in the lab-frame R-Z plane.
        """
        if stellsym:
            max_angle = np.pi / nfp
        else:
            raise NotImplementedError
            max_angle = 2 * np.pi

        if knot_parametrization not in ("chord", "uniform"):
            raise ValueError(
                "knot_parametrization must be 'chord' or 'uniform', "
                f"got {knot_parametrization!r}"
            )

        self.axis_points = axis_points
        self.points_per_cs = points_per_cs
        self.cs_equispaced = cs_equispaced
        self.rays_equispaced = rays_equispaced
        self.axis_angles_fixed = axis_angles_fixed
        self.cs_global_angle_free = cs_global_angle_free
        self.n_cs = n_cs
        self.nfp = nfp
        self.M = M
        self.N = N
        self.p_u = p_u
        self.p_v = p_v
        self.default_r = default_r
        self.stellsym = stellsym
        self.cs_basis = cs_basis
        self.nurbs = nurbs
        self.knot_parametrization = knot_parametrization
        self.use_bishop_frame = use_bishop_frame

        if dofs is None:
            # create equidistant points in zeta
            cs_zeta = np.linspace(0, max_angle, n_cs)
            # all angles zero
            cs_angles = np.zeros(n_cs)

            self.cs_zeta = cs_zeta
            self.cs_angles = cs_angles

            cs_dofs = np.array([None] * n_cs)

            if stellsym & np.all(cs_dofs != None):  # noqa: E711 -- elementwise vs. numpy array, "is not" checks object identity instead and breaks this
                assert len(cs_dofs[0]) == 2 * ((points_per_cs // 2) + 1)
                assert len(cs_dofs[-1]) == 2 * ((points_per_cs // 2) + 1)

            dofs = np.append(self.cs_zeta, self.cs_angles)

            names = [f"cs_zeta{i}" for i in range(n_cs)] + [
                f"cs_angle{i}" for i in range(n_cs)
            ]
            dofs = DOFs(
                dofs,
                names,
                [not cs_equispaced] * n_cs + [cs_global_angle_free] * n_cs,
                [(n - 1) * max_angle / (n_cs - 2) for n in range(n_cs)]
                + [-2 * np.pi / points_per_cs] * n_cs,
                [(n) * max_angle / (n_cs - 2) for n in range(n_cs)]
                + [2 * np.pi / points_per_cs] * n_cs,
            )

        dofs.fix("cs_angle0")
        dofs.fix(f"cs_angle{n_cs - 1}")
        dofs.fix("cs_zeta0")
        dofs.fix(f"cs_zeta{n_cs - 1}")

        self.axis = PseudoAxis(
            n_ctrl_pts=axis_points,
            nfp=nfp,
            stellsym=stellsym,
            axis_angles_fixed=axis_angles_fixed,
            knot_parametrization=knot_parametrization,
        )

        self.cs_list = []

        for i in range(n_cs):
            if cs_basis == "polar":
                cross_section = CrossSectionFixedZeta(
                    zeta_index=i,
                    n_ctrl_pts=points_per_cs,
                    equispaced=rays_equispaced,
                    default_r=default_r,
                    z_sym=((i == 0) or (i == n_cs - 1)),
                    nurbs=nurbs,
                )
            elif cs_basis == "cartesian":
                raise NotImplementedError("see dev branch")
            self.cs_list.append(cross_section)

        if quadpoints_theta is None:
            quadpoints_theta = Surface.get_theta_quadpoints()
        if quadpoints_phi is None:
            quadpoints_phi = Surface.get_phi_quadpoints(nfp=nfp)
        sopp.Surface.__init__(self, quadpoints_phi, quadpoints_theta)

        # _control_net_and_knots cache -- see that method's docstring.
        self._control_points_jim = None
        self._w_list_jim = None
        self._knots_u = None
        self._knots_v = None

        # dofs actually live on self.axis/self.cs_list (see get_dofs/set_dofs_impl below);
        # depends_on keeps them real Optimizable ancestors so surf.x/dof_names/bounds/fix
        # all work across the composite structure, same as before this class was a Surface
        Surface.__init__(
            self,
            dofs=dofs,
            external_dof_setter=SurfaceBSpline.set_dofs_impl,
            depends_on=[self.axis] + self.cs_list,
        )

    def num_dofs(self):
        return len(self.full_x)

    def get_dofs(self):
        return self.full_x

    def set_dofs_impl(self, v):
        # gamma_lin/gamma_impl read self.axis/self.cs_list live (via
        # _control_net_and_knots), not any state cached here, so there's nothing
        # to unpack from v -- the real dof values already landed on self.axis/
        # self.cs_list through the normal Optimizable .x/.full_x setters that
        # triggered this callback. All that's needed is telling the C++ side its
        # cached gamma/area/volume/etc are stale.
        self.invalidate_cache()

    def recompute_bell(self, parent=None):
        # Surface (unlike Curve) doesn't wire this to invalidate_cache() itself --
        # harmless for every other Surface subclass since none of them use
        # depends_on, but we do (self.axis, self.cs_list), and dof changes on a
        # dependency only reach us via set_recompute_flag -> recompute_bell, not
        # via set_dofs_impl. Mirrors Curve.recompute_bell.
        self.invalidate_cache()

    def get_cs_zeta_angle(self):
        zeta_list = np.array(
            [self.get(f"cs_zeta{i}") for i in range(self.n_cs)]
        )
        cs_angle_list = np.array(
            [self.get(f"cs_angle{i}") for i in range(self.n_cs)]
        )
        return zeta_list, cs_angle_list

    def _axis_rz(self, zeta):
        """
        Evaluate the pseudo-axis's (R, Z) at physical toroidal angle(s)
        zeta, via `PseudoAxis.gamma_impl`.

        Parameters
        ----------
        zeta : array-like
            Toroidal angle(s) in radians. Converted internally to quadpoint
            fractions in [0, 1), per the Curve/Surface convention.

        Returns
        -------
        r_axis, z_axis : ndarray
            Axis (R, Z) at the given zeta.
        """
        data = np.zeros((len(zeta), 3))
        self.axis.gamma_impl(data, np.asarray(zeta) / (2 * np.pi))
        r_axis = np.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2)
        z_axis = data[:, 2]
        return r_axis, z_axis

    def _axis_local_basis(self, zeta):
        """
        Return the axis position and the local 2D basis (e1, e2) that
        cross-section (r, theta) offsets are placed in at each zeta --
        either the fixed-zeta poloidal-plane basis (-R_hat, Z_hat, the
        original behavior), or, if self.use_bishop_frame, the axis's own
        Bishop (rotation-minimizing) frame (-N, -B).

        The sign flip on both N and B (rather than using them directly)
        is so a planar/circular axis gives IDENTICAL cross sections
        either way: there, Bishop's (N, B) reduce to (+R_hat, -Z_hat)
        (Bishop coincides with Frenet up to the initial-normal choice,
        and PseudoAxis.bishop_frame seeds N(0) along +R_hat -- see its
        docstring), so (-N, -B) = (-R_hat, +Z_hat), exactly matching the
        non-Bishop basis below. Toggling use_bishop_frame therefore
        doesn't introduce a spurious flip for the simplest axis shape.

        Returns axis_pos, e1, e2 -- each (len(zeta), 3).
        """
        zeta = np.asarray(zeta)
        if self.use_bishop_frame:
            data = np.zeros((len(zeta), 3))
            self.axis.gamma_impl(data, zeta / (2 * np.pi))
            _, N, B = self.axis.bishop_frame(zeta / (2 * np.pi))
            return data, -N, -B
        else:
            r_axis, z_axis = self._axis_rz(zeta)
            axis_pos = np.stack(
                [r_axis * np.cos(zeta), r_axis * np.sin(zeta), z_axis],
                axis=1,
            )
            e1 = np.stack(
                [-np.cos(zeta), -np.sin(zeta), np.zeros_like(zeta)], axis=1
            )
            e2 = np.tile(np.array([0.0, 0.0, 1.0]), (len(zeta), 1))
            return axis_pos, e1, e2

    def _get_control_points_xyz(self, return_w=False):
        """
        Return, for each cross section over the entire device domain, the
        (X, Y, Z) control points.

        Parameters
        ----------
        return_w : bool
            If True, also return the NURBS weights for each control point.

        Returns
        -------
        point_list : list of ndarray
            (X, Y, Z) control points for each cross section.
        w_list : list of ndarray, optional
            NURBS weights for each control point. Only returned if
            `return_w`.
        """
        point_list = []
        w_list = []

        cs_zeta, cs_angles = self.get_cs_zeta_angle()

        cs_zeta_1fp = np.append(
            cs_zeta, (2 * np.pi / self.nfp) - cs_zeta[-2:0:-1]
        )
        cs_zeta_full = np.concatenate(
            [cs_zeta_1fp + n * (2 * np.pi / self.nfp) for n in range(self.nfp)]
        )

        axis_pos, e1, e2 = self._axis_local_basis(cs_zeta_full)

        cs_list_1fp = [
            cs if (i // self.n_cs) == 0 else cs.flipped()
            for i, cs in enumerate(self.cs_list + self.cs_list[-2:0:-1])
        ]  #!!!!
        cs_list_full = np.tile(cs_list_1fp, self.nfp)

        cs_angle_1fp = [
            angle if (i // self.n_cs) == 0 else -angle
            for i, angle in enumerate(np.append(cs_angles, cs_angles[-2:0:-1]))
        ]
        cs_angle_full = np.tile(cs_angle_1fp, self.nfp)

        for i, cs in enumerate(cs_list_full):
            # point by point in each cross section
            cs_r_ctrl_full = cs.get_r_ctrl_full()
            cs_theta_ctrl_full = cs.get_theta_ctrl_full()
            cs_w_ctrl_full = cs.get_w_ctrl_full()
            cs_pointlist = []
            for j, r_cs in enumerate(cs_r_ctrl_full):
                theta = cs_theta_ctrl_full[j]
                offset = r_cs * np.cos(theta + cs_angle_full[i]) * e1[i] + (
                    r_cs * np.sin(theta + cs_angle_full[i]) * e2[i]
                )
                new_point = axis_pos[i] + offset
                cs_pointlist.append(new_point)
            point_list.append(np.array(cs_pointlist))
            w_list.append(cs_w_ctrl_full)

        if return_w:
            return point_list, w_list
        else:
            return point_list

    def get_xyz_centroids(self):
        """
        Return the centroid (X, Y, Z) of each cross section over the entire
        device domain.

        Returns
        -------
        point_list : list of ndarray
            Centroid (X, Y, Z) for each cross section.
        """
        cs_zeta, cs_angles = self.get_cs_zeta_angle()
        point_list = []
        # cross section by cross section
        cs_zeta_1fp = np.append(
            cs_zeta, (2 * np.pi / self.nfp) - cs_zeta[-2:0:-1]
        )
        cs_zeta_full = np.concatenate(
            [cs_zeta_1fp + n * (2 * np.pi / self.nfp) for n in range(self.nfp)]
        )

        axis_pos, e1, e2 = self._axis_local_basis(cs_zeta_full)

        cs_list_1fp = [
            cs if (i // self.n_cs) == 0 else cs.flipped()
            for i, cs in enumerate(self.cs_list + self.cs_list[-2:0:-1])
        ]  #!!!!
        cs_list_full = np.tile(cs_list_1fp, self.nfp)

        cs_angle_1fp = [
            angle if (i // self.n_cs) == 0 else -angle
            for i, angle in enumerate(np.append(cs_angles, cs_angles[-2:0:-1]))
        ]
        cs_angle_full = np.tile(cs_angle_1fp, self.nfp)

        for i, cs in enumerate(cs_list_full):
            # point by point in each cross section
            cs_r_ctrl_full = cs.get_r_ctrl_full()
            cs_theta_ctrl_full = cs.get_theta_ctrl_full()
            point_sum = np.zeros(3)
            for j, r_cs in enumerate(cs_r_ctrl_full):
                theta = cs_theta_ctrl_full[j]
                offset = r_cs * np.cos(theta + cs_angle_full[i]) * e1[i] + (
                    r_cs * np.sin(theta + cs_angle_full[i]) * e2[i]
                )
                point_sum += axis_pos[i] + offset
            new_point = point_sum / len(cs_r_ctrl_full)
            point_list.append(new_point)
        return point_list

    def get_rtz_full_device(self):
        """
        Return (r, theta, zeta) for each control point over the entire
        device domain.

        Returns
        -------
        point_list : list of ndarray
            (r, theta, zeta) for each control point, per cross section.
        """
        point_list = []

        cs_zeta, cs_angles = self.get_cs_zeta_angle()

        cs_zeta_1fp = np.append(
            cs_zeta, (2 * np.pi / self.nfp) - cs_zeta[-2:0:-1]
        )
        cs_zeta_full = np.concatenate(
            [cs_zeta_1fp + n * (2 * np.pi / self.nfp) for n in range(self.nfp)]
        )

        cs_list_1fp = [
            cs if (i // self.n_cs) == 0 else cs.flipped()
            for i, cs in enumerate(self.cs_list + self.cs_list[-2:0:-1])
        ]  #!!!!
        cs_list_full = np.tile(cs_list_1fp, self.nfp)

        cs_angle_1fp = [
            angle if (i // self.n_cs) == 0 else -angle
            for i, angle in enumerate(np.append(cs_angles, cs_angles[-2:0:-1]))
        ]
        cs_angle_full = np.tile(cs_angle_1fp, self.nfp)

        for i, cs in enumerate(cs_list_full):
            # point by point in each cross section
            zeta = cs_zeta_full[i]
            cs_pointlist = []
            cs_r_ctrl_full = cs.get_r_ctrl_full()
            cs_theta_ctrl_full = cs.get_theta_ctrl_full()
            for j, r_cs in enumerate(cs_r_ctrl_full):
                point_r = r_cs
                point_theta = cs_theta_ctrl_full[j] + cs_angle_full[i]
                point_zeta = zeta
                new_point = np.array([point_r, point_theta, point_zeta]).T
                cs_pointlist.append(new_point)
            point_list.append(np.array(cs_pointlist))
        return point_list

    def _control_net_and_knots(self):
        """
        Build (or return the cached) periodic-wrapped control net, NURBS
        weights, and knot vectors shared by `surf_callable` and `gamma_lin`.

        All four are pure functions of the free dofs (self.axis, self.cs_list)
        -- independent of any (u, v) evaluation point -- so they're cached
        here and only rebuilt when the dofs have actually changed (self.new_x),
        rather than on every call. This used to rebuild from scratch on every
        single gamma/root-find evaluation (profiled at ~1200 rebuilds, each
        re-walking every cross section including CrossSectionFixedZeta.flipped()
        and re-fitting chord-length knots, for a single to_RZFourier() call).

        The knot vectors get special-cased: 'uniform' knots depend only on
        the control point counts and p_u/p_v (fixed at construction), never
        on where the dofs actually put the points, so they're computed once,
        ever, and never invalidated by new_x. 'chord' knots do depend on
        point positions and are invalidated by new_x same as the net itself.

        Returns
        -------
        trimmed_ctrl_pts_jim : ndarray, shape (n_v+p_v+1, n_u+p_u+1, 3)
            Control net.
        trimmed_weights_ji : ndarray, shape (n_v+p_v+1, n_u+p_u+1)
            NURBS weights.
        knots_u, knots_v : ndarray
            Knot vectors for the u (poloidal) and v (toroidal) directions.
        """
        p_u = self.p_u
        p_v = self.p_v

        recompute_net = self.new_x or self._control_points_jim is None
        if self.knot_parametrization == "chord":
            recompute_knots = recompute_net or self._knots_u is None
        else:
            recompute_knots = self._knots_u is None

        if recompute_net:
            point_list, w_list = self._get_control_points_xyz(return_w=True)

            control_points_jim = np.array(point_list)
            w_list_jim = np.array(w_list)

            n_u = control_points_jim.shape[1] - 1
            n_v = control_points_jim.shape[0] - 1

            if recompute_knots:
                if self.knot_parametrization == "chord":
                    # u knots: chord-length per row, averaged across rows. Each
                    # row (cross section) can have differently-spaced control
                    # points, but the whole surface has one shared u-knot-vector,
                    # so average the per-row chord-length knots -- standard
                    # technique for tensor-product/lofted NURBS surfaces
                    # (averaging preserves monotonicity and the
                    # knots[p]==0/knots[n+p+1]==2pi endpoints, since every row's
                    # knots satisfy those).
                    knots_u = np.mean(
                        [
                            chord_length_knots(control_points_jim[j], p_u)
                            for j in range(n_v + 1)
                        ],
                        axis=0,
                    )
                    # v knots: chord-length between row centroids (one
                    # representative point per cross section)
                    row_centroids = control_points_jim.mean(axis=1)
                    knots_v = chord_length_knots(row_centroids, p_v)
                else:
                    knots_u = uniform_knots(n_u, p_u)
                    knots_v = uniform_knots(n_v, p_v)
                self._knots_u = knots_u
                self._knots_v = knots_v

            # Periodic wraparound: tile p_u/p_v points from each end onto the
            # opposite side, in both directions. Reflection symmetry doesn't
            # depend on p_u/p_v's parity -- the mirror is already baked into
            # _get_control_points_xyz's reflect-and-tile (cs.flipped(),
            # cs_angle negation, cs_zeta reflection), not into how u/v map to
            # theta/zeta, so it survives regardless of where the knots fall.
            # See "Chord-length knots break stellarator symmetry.md" in the
            # Obsidian vault.
            control_points_jim = np.concatenate(
                [
                    control_points_jim[:, -p_u:, :],
                    control_points_jim,
                    control_points_jim[:, :p_u, :],
                ],
                axis=1,
            )
            w_list_jim = np.concatenate(
                [w_list_jim[:, -p_u:], w_list_jim, w_list_jim[:, :p_u]], axis=1
            )
            control_points_jim = np.concatenate(
                [
                    control_points_jim[-p_v:, :, :],
                    control_points_jim,
                    control_points_jim[:p_v, :, :],
                ],
                axis=0,
            )
            w_list_jim = np.concatenate(
                [w_list_jim[-p_v:, :], w_list_jim, w_list_jim[:p_v, :]], axis=0
            )
            self._control_points_jim = control_points_jim
            self._w_list_jim = w_list_jim

        if self.new_x:
            self.new_x = False

        return (
            self._control_points_jim,
            self._w_list_jim,
            self._knots_u,
            self._knots_v,
        )

    def surf_callable(
        self,
        u,
        v,
    ):
        """
        Evaluate the spline surface at (a set of) u, v pairs within a field
        period.

        Parameters
        ----------
        u, v : array-like
            Parametric coordinates, in [0, 2*pi).

        Returns
        -------
        x_surf, y_surf, z_surf : ndarray
            Surface (X, Y, Z) at the given (u, v).
        """
        trimmed_ctrl_pts_jim, trimmed_weights_ji, knots_u, knots_v = (
            self._control_net_and_knots()
        )
        p_u = self.p_u
        p_v = self.p_v

        u_basis = b_p(knots_u, p_u, u)
        v_basis = b_p(knots_v, p_v, v)

        tp_basis = np.einsum("xj,xi->xji", v_basis, u_basis)
        w_tp_basis = np.einsum("xji,ji->xji", tp_basis, trimmed_weights_ji)
        summed_w_tp_basis = np.einsum("xji->x", w_tp_basis)
        nurbs_tp_basis = np.einsum(
            "xji,x->xji", w_tp_basis, 1 / summed_w_tp_basis
        )

        surf = np.einsum("xji,jim->xm", nurbs_tp_basis, trimmed_ctrl_pts_jim)
        x_surf = surf[:, 0]
        y_surf = surf[:, 1]
        z_surf = surf[:, 2]

        return x_surf, y_surf, z_surf

    def gamma_lin(self, data, quadpoints_phi, quadpoints_theta):
        r"""
        Evaluate (X, Y, Z) at paired (phi, theta) points -- data[i] =
        Gamma(phi[i], theta[i]) -- via a Newton root-find for the NURBS
        v-parameter (see gamma_eval_benchmark.ipynb, examples/2_Intermediate,
        for the derivation and a benchmark against oversample+CloughTocher
        interpolation).

        Parameters
        ----------
        data : ndarray, shape (N, 3)
            Filled in place with (X, Y, Z).
        quadpoints_phi, quadpoints_theta : ndarray, shape (N,)
            Fractions in [0, 1), per the Surface convention.
        """
        trimmed_ctrl_pts_jim, trimmed_weights_ji, knots_u, knots_v = (
            self._control_net_and_knots()
        )
        p_u = self.p_u
        p_v = self.p_v

        theta = np.asarray(quadpoints_theta) * 2 * np.pi
        phi = np.asarray(quadpoints_phi) * 2 * np.pi

        u_basis_row = b_p(knots_u, p_u, theta)  # (N, n_u_basis)

        # collapse the u-direction: Q_* are the "u-fixed" 1D-curve-in-v coefficients,
        # one row per query point -- no grid broadcasting, points are already paired
        wp = (
            trimmed_weights_ji[:, :, None] * trimmed_ctrl_pts_jim
        )  # (n_v_ext, n_u_ext, 3)
        ww = trimmed_weights_ji  # (n_v_ext, n_u_ext)
        Q_pos = np.einsum("ki,jim->kjm", u_basis_row, wp)  # (N, n_v_ext, 3)
        Q_w = np.einsum("ki,ji->kj", u_basis_row, ww)  # (N, n_v_ext)

        def _xyz_and_derivs(v):
            # wrap into the valid periodic domain (rather than clip) so scipy's
            # own, otherwise-unconstrained iterate always gets a well-defined
            # basis evaluation
            v_wrapped = v % (2 * np.pi)
            basis_v, dbasis_v = b_p_deriv(knots_v, p_v, v_wrapped)
            Nx = np.einsum("mj,mj->m", basis_v, Q_pos[:, :, 0])
            Ny = np.einsum("mj,mj->m", basis_v, Q_pos[:, :, 1])
            D = np.einsum("mj,mj->m", basis_v, Q_w)
            dNx = np.einsum("mj,mj->m", dbasis_v, Q_pos[:, :, 0])
            dNy = np.einsum("mj,mj->m", dbasis_v, Q_pos[:, :, 1])
            dD = np.einsum("mj,mj->m", dbasis_v, Q_w)
            return Nx, Ny, D, dNx, dNy, dD

        def func(v):
            Nx, Ny, D, *_ = _xyz_and_derivs(v)
            X, Y = Nx / D, Ny / D
            zeta_cur = np.arctan2(Y, X) % (2 * np.pi)
            return ((zeta_cur - phi + np.pi) % (2 * np.pi)) - np.pi

        def fprime(v):
            Nx, Ny, D, dNx, dNy, dD = _xyz_and_derivs(v)
            X, Y = Nx / D, Ny / D
            dX = (dNx * D - Nx * dD) / D**2
            dY = (dNy * D - Ny * dD) / D**2
            return (X * dY - Y * dX) / (X**2 + Y**2)

        v_sol = newton(
            func, x0=phi.copy(), fprime=fprime, tol=1e-12, maxiter=50
        )

        v_wrapped = v_sol % (2 * np.pi)
        basis_v = b_p(knots_v, p_v, v_wrapped)
        Nx = np.einsum("mj,mj->m", basis_v, Q_pos[:, :, 0])
        Ny = np.einsum("mj,mj->m", basis_v, Q_pos[:, :, 1])
        Nz = np.einsum("mj,mj->m", basis_v, Q_pos[:, :, 2])
        D = np.einsum("mj,mj->m", basis_v, Q_w)

        data[:, 0] = Nx / D
        data[:, 1] = Ny / D
        data[:, 2] = Nz / D

    def gamma_impl(self, data, quadpoints_phi, quadpoints_theta):
        r"""
        Evaluate (X, Y, Z) on a tensor product grid of phi x theta points. A
        thin wrapper around `gamma_lin` (same pattern as
        `SurfaceHenneberg.gamma_impl`).

        Parameters
        ----------
        data : ndarray, shape (n_phi, n_theta, 3)
            Filled in place, matching the Surface convention (phi axis
            first).
        quadpoints_phi, quadpoints_theta : ndarray, 1D
            Fractions in [0, 1).
        """
        nphi = len(quadpoints_phi)
        ntheta = len(quadpoints_theta)
        phi2d, theta2d = np.meshgrid(quadpoints_phi, quadpoints_theta)
        data1d = np.zeros((nphi * ntheta, 3))
        self.gamma_lin(
            data1d,
            np.reshape(phi2d, (nphi * ntheta,)),
            np.reshape(theta2d, (nphi * ntheta,)),
        )
        for xyz in range(3):
            data[:, :, xyz] = np.reshape(data1d[:, xyz], (ntheta, nphi)).T

    def centroid_axis_callable(
        self,
        a,
    ):
        xyz_list = self.get_xyz_centroids()
        p_a = self.p_v

        centroids_im = np.array(xyz_list)

        # Periodic wraparound, matching _control_net_and_knots: tile p_a
        # points from each end onto the opposite side, with knots widened
        # to match.
        if self.knot_parametrization == "chord":
            knots_a = chord_length_knots(centroids_im, p_a)
        else:
            knots_a = uniform_knots(centroids_im.shape[0] - 1, p_a)
        centroids_im = np.concatenate(
            [centroids_im[-p_a:], centroids_im, centroids_im[:p_a]], axis=0
        )

        a_basis = b_p(knots_a, p_a, a)

        # einsum, not @ -- matmul spuriously raises divide-by-zero/overflow
        # RuntimeWarnings on some BLAS backends when a_basis contains
        # subnormal values, even though the result is correct.
        x_centroid = np.einsum("ij,j->i", a_basis, centroids_im[:, 0])
        y_centroid = np.einsum("ij,j->i", a_basis, centroids_im[:, 1])
        z_centroid = np.einsum("ij,j->i", a_basis, centroids_im[:, 2])

        return x_centroid, y_centroid, z_centroid

    def centroid_axis_derivative_callable(self, a):
        r"""
        d(x,y,z)/da for `centroid_axis_callable`'s curve, needed by
        `fsolve_centroid_axis_from_zetas`'s Newton solve for a given target zeta.
        Mirrors `centroid_axis_callable` exactly, swapping `b_p` for
        `b_p_deriv`.
        """
        xyz_list = self.get_xyz_centroids()
        p_a = self.p_v

        centroids_im = np.array(xyz_list)

        if self.knot_parametrization == "chord":
            knots_a = chord_length_knots(centroids_im, p_a)
        else:
            knots_a = uniform_knots(centroids_im.shape[0] - 1, p_a)
        centroids_im = np.concatenate(
            [centroids_im[-p_a:], centroids_im, centroids_im[:p_a]], axis=0
        )

        _, da_basis = b_p_deriv(knots_a, p_a, a)

        dx_da = np.einsum("ij,j->i", da_basis, centroids_im[:, 0])
        dy_da = np.einsum("ij,j->i", da_basis, centroids_im[:, 1])
        dz_da = np.einsum("ij,j->i", da_basis, centroids_im[:, 2])

        return dx_da, dy_da, dz_da

    def fsolve_centroid_axis_from_zetas(self, zeta_surf, offset):
        def f0(a, target):
            a = a % (2 * np.pi)
            x, y, z = self.centroid_axis_callable(a)
            zeta = np.arctan2(y, x) % (2 * np.pi)
            return zeta - target

        def f1(a, target):
            a = a % (2 * np.pi)
            x, y, z = self.centroid_axis_callable(a)
            dx_da, dy_da, dz_da = self.centroid_axis_derivative_callable(a)
            return 1 / (1 + (y / x) ** 2) * ((dy_da / x) - (y / x**2) * dx_da)

        a_star = []
        for target in zeta_surf.flatten():
            a_star.append(
                fsolve(
                    func=f0,
                    x0=(target - offset) % (2 * np.pi),
                    fprime=f1,
                    args=target,
                )
            )
        a_star = np.array(a_star).flatten() % (2 * np.pi)

        x_star, y_star, z_star = self.centroid_axis_callable(a_star)

        return x_star, y_star, z_star

    def uniform_tz_interp(
        self,
        nu=32,
        nv=32,
        nu_interp=64,
        nv_interp=64,
        plot=False,
        _fsolve=False,
    ):
        """
        Return R, Z on a (theta_a, zeta) grid, where theta_a is the
        conventional polar angle about the centroid and zeta is the
        toroidal angle. Evaluates R, Z on a uniform (u, v) grid, interpolates
        to a (theta_p, zeta) grid (theta_p: polar angle about the centroid
        in a plane of constant zeta, mirrored for stellarator symmetry),
        then interpolates again onto the final (theta_a, zeta) grid.

        Parameters
        ----------
        nu, nv : int
            Shape of the output arrays.
        nu_interp, nv_interp : int
            Shape of the intermediate (u, v) grid used for interpolation.
        plot : bool
            Whether to plot the points making up the equal-arclength grid.
        _fsolve : bool
            Accepted for signature compatibility with related methods; not
            used in this function body.
        """

        # Exact evaluation at the target (u, zeta) grid via gamma_lin's
        # Newton solve, instead of oversampling surf_callable on a (u, v)
        # mesh and reading off an approximate zeta = atan2(y, x). u is used
        # directly as theta (see gamma_lin's docstring: simsopt only
        # requires zeta/phi to be a real physical angle, theta's convention
        # is otherwise unconstrained), so this hits the target zeta with
        # zero error. The interpolation onto a *uniform physical* theta
        # below is still genuine interpolation -- gamma_lin can't shortcut
        # that part, since it only solves for exact zeta hits, not exact
        # theta hits (see gamma_eval_benchmark.ipynb / the conversation this
        # came from for why).

        u = np.linspace(0, 2 * np.pi, nu_interp, endpoint=True)
        v = np.linspace(0, 2 * np.pi, nv_interp, endpoint=True)
        v_grid = np.tile(v, (nu_interp, 1))

        # gamma_impl is the grid-shaped wrapper around gamma_lin (same
        # reasoning as exact_tz_interp); its (phi, theta) axis order is
        # (nv_interp, nu_interp) here, transposed to match this function's
        # (nu_interp, nv_interp) = (u, v) convention used below.
        data = np.zeros((nv_interp, nu_interp, 3))
        self.gamma_impl(data, v / (2 * np.pi), u / (2 * np.pi))
        x_surf = data[:, :, 0].T
        y_surf = data[:, :, 1].T
        z_surf = data[:, :, 2].T

        zeta_surf = (
            v_grid  # exact by construction, no need to recover via atan2
        )
        R_surf = np.sqrt(x_surf**2 + y_surf**2)

        # Axis reference must be evaluated at the SAME physical zeta as
        # x_surf/y_surf/z_surf's columns (now exact, via gamma_lin's Newton
        # solve) -- centroid_axis_callable's own raw parameter is *not* the
        # physical zeta (same "raw parameter != physical angle" issue
        # gamma_lin fixes for the surface itself), so it must go through
        # its own Newton solve (fsolve_centroid_axis_from_zetas) rather than being
        # evaluated directly at `v`. Skipping this and evaluating at the
        # raw parameter directly leaves the axis and surface systematically
        # misaligned in zeta, corrupting theta_surf below.
        x_axis, y_axis, z_axis = self.fsolve_centroid_axis_from_zetas(
            v, offset=0.0
        )

        R_axis = np.sqrt(x_axis**2 + y_axis**2)

        theta_surf = np.arctan2(z_surf - z_axis, R_surf - R_axis) % (2 * np.pi)

        if nv % (2 * self.nfp) != 0:
            raise ValueError("nv_intermediate must be divisible by 2*nfp. ")

        nu_tz = nu
        nv_tz = nv // (2 * self.nfp)

        tz_points = np.vstack(
            (
                np.concatenate(
                    [
                        theta_surf.flatten() - 2 * np.pi,
                        theta_surf.flatten(),
                        theta_surf.flatten() + 2 * np.pi,
                    ]
                ),
                np.tile(zeta_surf.flatten(), 3),
            )
        )
        zeta_eval, theta_eval = np.meshgrid(
            np.linspace(
                np.pi / self.nfp, 2 * np.pi / self.nfp, nv_tz + 1, endpoint=True
            ),
            np.linspace(0, 2 * np.pi, nu_tz, endpoint=False),
        )
        eval_grid = np.vstack((theta_eval.flatten(), zeta_eval.flatten()))

        R_tz_callable = CloughTocher2DInterpolator(
            points=tz_points.T,
            values=np.tile(R_surf.flatten(), 3),
        )
        z_tz_callable = CloughTocher2DInterpolator(
            points=tz_points.T,
            values=np.tile(z_surf.flatten(), 3),
        )

        R_on_tz_grid = R_tz_callable(eval_grid.T).reshape(nu_tz, nv_tz + 1)
        z_on_tz_grid = z_tz_callable(eval_grid.T).reshape(nu_tz, nv_tz + 1)

        if plot:
            x_on_tz_grid = R_on_tz_grid * np.cos(zeta_eval)
            y_on_tz_grid = R_on_tz_grid * np.sin(zeta_eval)
            z_on_tz_grid = z_on_tz_grid
            fig3d, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.scatter(x_on_tz_grid, y_on_tz_grid, z_on_tz_grid)
            ax.set_box_aspect((1, 1, 1))
            ax.set_ylim(-1, 1)
            ax.set_xlim(-1, 1)
            ax.set_zlim(-1, 1)

        if plot:
            numCols = 5
            numRows = 2
            plotNum = 1
            nzeta_cs = 9
            zeta_cs = np.linspace(
                0, 2 * np.pi / self.nfp, num=nzeta_cs, endpoint=True
            )
            theta_cs_eval = np.linspace(0, 2 * np.pi, 64)

            fig = plt.figure("Poincare Plots", figsize=(14, 7))
            fig.patch.set_facecolor("white")
            plt.subplot(numRows, numCols, plotNum)

            plotNum += 1
            for ind in range(nzeta_cs):
                if zeta_cs[ind] >= np.pi / self.nfp:
                    plt.subplot(numRows, numCols, ind + 1)
                    plt.title(r"$\phi =$" + str(zeta_cs[ind]))
                    plt.gca().set_aspect("equal", adjustable="box")
                    plt.plot()
                    point = np.vstack(
                        (
                            theta_cs_eval,
                            zeta_cs[ind] * np.ones_like(theta_cs_eval),
                        )
                    ).T
                    R_cs = R_tz_callable(point)
                    z_cs = z_tz_callable(point)
                    plt.plot(R_cs, z_cs, "r--")
                    plt.gca().set_aspect("equal", adjustable="box")
                    plt.xlabel("R")
                    plt.ylabel("Z")
                else:
                    plt.subplot(numRows, numCols, ind + 1)
                    plt.title(r"$\phi =$" + str(zeta_cs[ind]))
                    plt.gca().set_aspect("equal", adjustable="box")
                    plt.plot()
                    point = np.vstack(
                        (
                            theta_cs_eval,
                            (np.pi - zeta_cs[ind])
                            * np.ones_like(theta_cs_eval),
                        )
                    ).T
                    R_cs = R_tz_callable(point)[::-1]
                    z_cs = -z_tz_callable(point)[::-1]
                    plt.plot(R_cs, z_cs, "r--")
                    plt.gca().set_aspect("equal", adjustable="box")
                    plt.xlabel("R")
                    plt.ylabel("Z")

        if plot:
            x_on_tz_grid = R_on_tz_grid * np.cos(zeta_eval)
            y_on_tz_grid = R_on_tz_grid * np.sin(zeta_eval)
            z_on_tz_grid = z_on_tz_grid
            # for theta, i in enumerate(ulist):

            ax.scatter(x_on_tz_grid, y_on_tz_grid, z_on_tz_grid)
            ax.set_box_aspect((1, 1, 1))
            ax.set_ylim(-1, 1)
            ax.set_xlim(-1, 1)
            ax.set_zlim(-1, 1)

        R_flipped = np.roll(R_on_tz_grid[::-1, -1:0:-1], 1, axis=0)
        z_flipped = np.roll(-z_on_tz_grid[::-1, -1:0:-1], 1, axis=0)
        R_1fp = np.concatenate([R_flipped, R_on_tz_grid[:, :-1]], axis=1)
        z_1fp = np.concatenate([z_flipped, z_on_tz_grid[:, :-1]], axis=1)
        R_full = np.concatenate([R_1fp] * self.nfp, axis=1)
        z_full = np.concatenate([z_1fp] * self.nfp, axis=1)
        zeta_full, theta_full = np.meshgrid(
            np.linspace(0, 2 * np.pi, nv, endpoint=False),
            np.linspace(0, 2 * np.pi, nu, endpoint=False),
        )

        # print(theta_full)

        return R_full.T, z_full.T, zeta_full, theta_full

    def exact_tz_interp(self, nu=None, nv=None, plot=False, ax=None):
        r"""
        Return R, Z on a (theta, zeta) grid with NO interpolation at all --
        theta is the spline's own u parameter, used directly with no
        reparametrization (simsopt only requires zeta/phi to be a real
        physical toroidal angle; theta's specific convention is otherwise
        unconstrained -- see gamma_lin's docstring and
        gamma_eval_benchmark.ipynb, examples/2_Intermediate). zeta is hit
        exactly via gamma_lin's Newton root-find for v.

        This is the only *_tz_interp method with zero approximation error,
        at the cost that "theta" here doesn't mean a uniform physical angle
        or an arclength-normalized one the way uniform_tz_interp's/
        arclength_tz_interp's theta do. If you need either of those
        conventions specifically, interpolation is unavoidable -- see those
        two methods, whose own first evaluation stage now also uses this
        same exact gamma_lin evaluation, just followed by a genuine
        theta-reparametrization step this method skips entirely.

        Parameters
        ----------
        nu, nv : int, optional
            Shape of the output arrays. Defaults to `2*nfp*16`.
        plot : bool
            Whether to scatter-plot the resulting points.
        ax : matplotlib 3D axis, optional
            Axis to plot on, if `plot`.
        """
        nu = 2 * self.nfp * 16 if nu is None else nu
        nv = 2 * self.nfp * 16 if nv is None else nv

        # gamma_impl is the grid-shaped wrapper around gamma_lin -- this
        # method needs a (zeta, theta) grid for the Fourier transform, so
        # call it directly instead of hand-rolling the same
        # meshgrid+flatten+reshape gamma_impl already does. Its (phi,
        # theta) axis convention (phi first) already matches this method's
        # (zeta, theta) return convention, so no transpose is needed here.
        quadpoints_theta = np.linspace(0, 1, nu, endpoint=False)
        quadpoints_phi = np.linspace(0, 1, nv, endpoint=False)

        data = np.zeros((nv, nu, 3))
        self.gamma_impl(data, quadpoints_phi, quadpoints_theta)
        R_zt = np.sqrt(data[:, :, 0] ** 2 + data[:, :, 1] ** 2)
        z_zt = data[:, :, 2]

        zeta_full, theta_full = np.meshgrid(
            np.linspace(0, 2 * np.pi, nv, endpoint=False),
            np.linspace(0, 2 * np.pi, nu, endpoint=False),
        )

        if plot:
            x_full = R_zt.T * np.cos(zeta_full)
            y_full = R_zt.T * np.sin(zeta_full)
            z_full = z_zt.T
            if ax is None:
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.scatter(x_full, y_full, z_full, s=1)
            ax.set_box_aspect((1, 1, 1))
            ax.set_ylim(-1, 1)
            ax.set_xlim(-1, 1)
            ax.set_zlim(-1, 1)

        return R_zt, z_zt, zeta_full, theta_full

    def arclength_tz_interp(
        self,
        nu=None,
        nv=None,
        nu_interp=None,
        nv_interp=None,
        plot=False,
        ax=None,
        _fsolve=False,
    ):
        r"""
        Return R, Z on a (theta_a, zeta) grid, where theta_a is the poloidal
        angle demarking unit arc length on the curve and zeta is the
        toroidal angle. Evaluates R, Z on a uniform (u, v) grid, interpolates
        to a (theta_p, zeta) grid (theta_p: polar angle about the centroid
        in a plane of constant zeta, mirrored for stellarator symmetry),
        then interpolates again onto the final (theta_a, zeta) grid.

        Parameters
        ----------
        nu, nv : int, optional
            Shape of the output arrays. Defaults to `2*nfp*16`.
        nu_interp, nv_interp : int, optional
            Shape of the intermediate (u, v) grid used for interpolation.
            Defaults to `2*nfp*16`.
        plot : bool
            Whether to plot the points making up the equal-arclength grid.
        ax : matplotlib 3D axis, optional
            Axis to plot on, if `plot`.
        _fsolve : bool
            Accepted for signature compatibility with related methods; not
            used in this function body.
        """
        # Creating grid to interpolate u and v on

        nu = 2 * self.nfp * 16 if nu is None else nu
        nv = 2 * self.nfp * 16 if nv is None else nv
        nu_interp = 2 * self.nfp * 16 if nu_interp is None else nu_interp
        nv_interp = 2 * self.nfp * 16 if nv_interp is None else nv_interp

        if nv_interp % (2 * self.nfp) != 0:
            raise ValueError("nv_intermediate must be divisible by 2*nfp. ")

        nu_uz = nu_interp
        nv_uz = nv_interp // (2 * self.nfp)

        zeta_eval, theta_eval = np.meshgrid(
            np.linspace(
                np.pi / self.nfp, 2 * np.pi / self.nfp, nv_uz, endpoint=True
            ),
            np.linspace(0, 2 * np.pi, nu_uz, endpoint=True),
        )

        # Exact evaluation at the target (u, zeta) grid via gamma_impl (the
        # grid-shaped wrapper around gamma_lin's Newton solve for v),
        # instead of oversampling surf_callable on a (u, v) mesh and
        # CloughTocher-regridding onto this grid. u is used directly as
        # theta (gamma_lin's convention -- see its docstring).
        # R_uz_callable/z_uz_callable are still built, now from this exact
        # data rather than an approximate oversample, because the theta=0
        # root-find below genuinely needs values at arbitrary u, not just
        # this grid -- that's real interpolation gamma_lin can't shortcut,
        # since it only solves for exact zeta hits, not exact theta hits.
        zeta_1d = zeta_eval[0, :]
        theta_1d = theta_eval[:, 0]
        data = np.zeros((nv_uz, nu_uz, 3))
        self.gamma_impl(data, zeta_1d / (2 * np.pi), theta_1d / (2 * np.pi))
        R_on_uz_grid = np.sqrt(data[:, :, 0] ** 2 + data[:, :, 1] ** 2).T
        z_on_uz_grid = data[:, :, 2].T

        zeta_1d_halfgrid = zeta_eval[0, :]
        ulist = []

        # finding theta=0 point (the outboard, Z=Z_axis crossing of the
        # cross section -- the standard VMEC/stellarator convention. Not
        # the same as u=0: in this class's own (R,theta) convention,
        # R = r_paxis - r_cs*cos(theta), so u=0 is actually the *inboard*
        # crossing, not outboard -- using u=0 directly here would silently
        # break the rbs=zbc=0 stellarator-symmetry assumption ft() makes,
        # unless u=0 happened to coincide with the true symmetry point,
        # which isn't guaranteed for a general chord-length-parametrized
        # cross section.)

        # Exact (u, zeta) evaluation via gamma_lin's own Newton solve --
        # used for both the root-find objective and the outboard check
        # below
        def _exact_Rz(u, zeta):
            u_arr = np.atleast_1d(u).astype(float)
            zeta_arr = np.broadcast_to(np.atleast_1d(zeta), u_arr.shape).astype(
                float
            )
            data = np.zeros((u_arr.size, 3))
            self.gamma_lin(data, zeta_arr / (2 * np.pi), u_arr / (2 * np.pi))
            R = np.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2)
            Z = data[:, 2]
            return R, Z

        # Exact axis (R, Z) at the target zeta values via
        # fsolve_centroid_axis_from_zetas's Newton solve
        x_axis0, y_axis0, z_axis_1d = self.fsolve_centroid_axis_from_zetas(
            zeta_1d_halfgrid, offset=0.0
        )
        R_axis_1d = np.sqrt(x_axis0**2 + y_axis0**2)
        R_axis_on_uz_grid = np.tile(R_axis_1d, (nu_uz, 1))
        z_axis_on_uz_grid = np.tile(z_axis_1d, (nu_uz, 1))

        def _is_outboard(x, zeta_val, r_axis_val):
            R, _ = _exact_Rz(x, zeta_val)
            return R[0] > r_axis_val

        for i, zeta in enumerate(zeta_1d_halfgrid):
            zs = (z_on_uz_grid - z_axis_on_uz_grid)[:, i]
            u_eval = theta_eval[:, i]
            switch_indices = np.logical_xor(zs > 0, np.roll(zs > 0, 1))
            a = u_eval[np.roll(switch_indices, -1)]
            b = u_eval[np.roll(switch_indices, 0)]

            if switch_indices[0]:
                a = u_eval[np.roll(switch_indices, -1)]
                b = np.roll(u_eval[np.roll(switch_indices, 0)], -1)
            else:
                a = u_eval[np.roll(switch_indices, -1)]
                b = u_eval[np.roll(switch_indices, 0)]

            z_axis_i = z_axis_1d[i]
            R_axis_i = R_axis_1d[i]

            def f(x, zeta, z_axis_i=z_axis_i):
                _, Z = _exact_Rz(x, zeta)
                return Z[0] - z_axis_i

            nfails = 0
            nsucc = 0
            nattempts = 0

            for k, _ in enumerate(a):
                u_theta0, r = bisect(
                    f, a=a[k], b=b[k], args=zeta, full_output=True
                )

                if r.converged and _is_outboard(u_theta0, zeta, R_axis_i):
                    ulist.append(u_theta0)
                    nsucc += 1
                    nattempts += 1
                    break
                else:
                    nfails += 1
                    nattempts += 1

            if nfails == len(a):
                warnings.warn(
                    "arclength_tz_interp: exact theta=0 root-find found no "
                    f"converged outboard crossing at zeta={zeta:.6g} "
                    f"(column {i}/{len(zeta_1d_halfgrid)}); falling back to "
                    "the coarse-grid point closest to Z=Z_axis on the "
                    "outboard side. Results near this zeta may be less "
                    "accurate than elsewhere -- consider increasing "
                    "nu_interp/nv_interp if this appears often.",
                    stacklevel=2,
                )
                u_feasible = u_eval[
                    (R_on_uz_grid - R_axis_on_uz_grid)[:, i] > 0
                ]
                zs_feasible = zs[(R_on_uz_grid - R_axis_on_uz_grid)[:, i] > 0]
                ulist.append(u_feasible[np.argmin(zs_feasible)])

            assert nsucc + nfails == nattempts
        ulist = np.array(ulist).flatten()
        # print(ulist)

        R_0, z_0 = _exact_Rz(ulist, zeta_1d_halfgrid)
        R_0 = R_0.reshape(1, nv_uz)
        z_0 = z_0.reshape(1, nv_uz)

        # reparametrizing on arclength

        # sorting by u, starting from theta=0 point
        theta0_eval = (theta_eval - np.outer(np.ones(nu_uz), ulist)) % (
            2 * np.pi
        )
        sorted_theta0_indices = np.argsort(theta0_eval, axis=0)

        R_on_uz_grid = np.take_along_axis(
            R_on_uz_grid, sorted_theta0_indices, axis=0
        )  # R_on_arclength_grid[sorted_theta_indices]
        z_on_uz_grid = np.take_along_axis(
            z_on_uz_grid, sorted_theta0_indices, axis=0
        )  # z_on_arclength_grid[sorted_theta_indices]

        # adding zero arclength point to start of R array
        R_on_uz_grid = np.insert(R_on_uz_grid, 0, R_0, axis=0)
        z_on_uz_grid = np.insert(z_on_uz_grid, 0, z_0, axis=0)

        # copying zero arclength point to end of array
        R_on_uz_grid = np.concatenate([R_on_uz_grid, R_0], axis=0)
        z_on_uz_grid = np.concatenate([z_on_uz_grid, z_0], axis=0)

        dist_to_next_u = np.sqrt(
            (R_on_uz_grid[1:, :] - R_on_uz_grid[0:-1, :]) ** 2
            + (z_on_uz_grid[1:, :] - z_on_uz_grid[0:-1, :]) ** 2
        )
        dist_to_next_u = np.insert(
            dist_to_next_u, 0, np.zeros_like(R_on_uz_grid[0, :]), axis=0
        )
        u_arclength = np.cumsum(dist_to_next_u, axis=0)
        col_max = np.outer(np.ones(u_arclength.shape[0]), u_arclength[-1, :])

        u_arclength_normalized = (u_arclength / col_max) * 2 * np.pi

        zeta_extended = np.concatenate([zeta_eval, zeta_eval[:2, :]], axis=0)

        if nv % (2 * self.nfp) != 0:
            raise ValueError("nv must be divisible by 2*nfp. ")

        nv_final = nv // (2 * self.nfp)
        nu_final = nu

        arclength_points = np.vstack(
            (u_arclength_normalized.flatten(), zeta_extended.flatten())
        )
        zeta_ffeval, theta_ffeval = np.meshgrid(
            np.linspace(
                np.pi / self.nfp,
                2 * np.pi / self.nfp,
                nv_final + 1,
                endpoint=True,
            ),
            np.linspace(0, 2 * np.pi, nu_final, endpoint=False),
        )
        eval_grid = np.vstack((theta_ffeval.flatten(), zeta_ffeval.flatten()))

        R_az_callable = CloughTocher2DInterpolator(
            points=arclength_points.T,
            values=R_on_uz_grid.flatten(),
        )
        z_az_callable = CloughTocher2DInterpolator(
            points=arclength_points.T,
            values=z_on_uz_grid.flatten(),
        )

        R_on_az_grid = R_az_callable(eval_grid.T).reshape(
            nu_final, nv_final + 1
        )
        z_on_az_grid = z_az_callable(eval_grid.T).reshape(
            nu_final, nv_final + 1
        )

        R_flipped = np.roll(R_on_az_grid[::-1, -1:0:-1], 1, axis=0)
        z_flipped = np.roll(-z_on_az_grid[::-1, -1:0:-1], 1, axis=0)
        R_1fp = np.concatenate([R_flipped, R_on_az_grid[:, :-1]], axis=1)
        z_1fp = np.concatenate([z_flipped, z_on_az_grid[:, :-1]], axis=1)
        R_full = np.concatenate([R_1fp] * self.nfp, axis=1)
        z_full = np.concatenate([z_1fp] * self.nfp, axis=1)
        zeta_full, theta_full = np.meshgrid(
            np.linspace(0, 2 * np.pi, nv, endpoint=False),
            np.linspace(0, 2 * np.pi, nu_final, endpoint=False),
        )

        if plot:
            x_on_az_grid = R_full * np.cos(zeta_full)
            y_on_az_grid = R_full * np.sin(zeta_full)
            z_on_az_grid = z_full

            ax.scatter(x_on_az_grid, y_on_az_grid, z_on_az_grid, s=1)
            ax.plot_wireframe(
                x_on_az_grid, y_on_az_grid, z_on_az_grid, alpha=0.05
            )

            ax.set_box_aspect((1, 1, 1))
            ax.set_ylim(-1, 1)
            ax.set_xlim(-1, 1)
            ax.set_zlim(-1, 1)

        return R_full.T, z_full.T, zeta_full, theta_full

    def ft(
        self,
        R=None,
        z=None,
        zeta=None,
        theta=None,
        nu=None,
        nv=None,
        nv_interp=None,
        nu_interp=None,
        plot_ft=False,
        ft_ax=None,
        plot_intermediate=False,
        intermediate_ax=None,
        _fsolve=False,
        collocation="exact",
        spec_cond=True,
        spec_cond_options={
            "plot": False,
            "ftol": 1e-4,
            "Mtol": 1.1,
            "shapetol": None,
            "niters": 5000,
            "verbose": False,
            "cutoff": 1e-6,
        },
    ):
        r"""
        Fourier transform from spline surface to Fourier coefficients for
        a VMEC surface.

        Returns
        -------
        rbc, zbs : ndarray
            VMEC Fourier coefficients.
        """
        nu = 2 * self.nfp * 16 if nu is None else nu
        nv = 2 * self.nfp * 16 if nv is None else nv
        nu_interp = 2 * self.nfp * 16 if nu_interp is None else nu_interp
        nv_interp = 2 * self.nfp * 16 if nv_interp is None else nv_interp
        assert nv % 2 == 0, "nv must be even"

        if (
            np.any(R is None)
            or np.any(z is None)
            or np.any(zeta is None)
            or np.any(theta is None)
        ):
            # tic = time.perf_counter()
            if collocation == "uniform":
                R_on_tz_grid, z_on_tz_grid, zeta_eval, theta_eval = (
                    self.uniform_tz_interp(
                        nu=nu,
                        nv=nv,
                        nv_interp=nv_interp,
                        nu_interp=nu_interp,
                        # plot=plot_intermediate,
                        # ax=intermediate_ax,
                        _fsolve=_fsolve,
                    )
                )
            elif collocation == "arclength":
                R_on_tz_grid, z_on_tz_grid, zeta_eval, theta_eval = (
                    self.arclength_tz_interp(
                        nu=nu,
                        nv=nv,
                        nv_interp=nv_interp,
                        nu_interp=nu_interp,
                        plot=plot_intermediate,
                        ax=intermediate_ax,
                        _fsolve=_fsolve,
                    )
                )
            elif collocation == "exact":
                # theta = the spline's own u parameter, no reparametrization
                # -- zero interpolation error, but not a uniform physical
                # angle or an arclength-normalized one (see
                # exact_tz_interp's docstring)
                R_on_tz_grid, z_on_tz_grid, zeta_eval, theta_eval = (
                    self.exact_tz_interp(
                        nu=nu,
                        nv=nv,
                        plot=plot_intermediate,
                        ax=intermediate_ax,
                    )
                )
            # toc = time.perf_counter()
            # print(f"Created equispaced grid in {toc - tic:0.4f} seconds")
        else:
            R_on_tz_grid, z_on_tz_grid, zeta_eval, theta_eval = (
                R,
                z,
                zeta,
                theta,
            )

        M, N = self.M, self.N

        # Fourier transform
        cosnmtz = np.array(
            [
                [
                    np.cos(m * theta_eval - n * ((zeta_eval) * self.nfp))
                    for m in range(0, M + 1)
                ]
                for n in range(-N, N + 1)
            ],
        )
        sinnmtz = np.array(
            [
                [
                    np.sin(m * theta_eval - n * ((zeta_eval) * self.nfp))
                    for m in range(0, M + 1)
                ]
                for n in range(-N, N + 1)
            ],
        )

        rbc_in = np.einsum("nmtz,tz->nm", cosnmtz, R_on_tz_grid.T) / (nu * nv)
        rbc_in[:N, 0] = 0
        rbc_in[:] *= 2
        rbc_in[N, 0] /= 2

        rbs_in = np.einsum("nmtz,tz->nm", sinnmtz, R_on_tz_grid.T) / (nu * nv)
        rbs_in[:N, 0] = 0
        rbs_in[:] *= 2
        rbs_in[N, 0] /= 2

        zbc_in = np.einsum("nmtz,tz->nm", cosnmtz, z_on_tz_grid.T) / (nu * nv)
        zbc_in[:N, 0] = 0
        zbc_in[:] *= 2
        zbc_in[N, 0] /= 2

        # print(f'zbc: {zbc}')
        zbs_in = np.einsum("nmtz,tz->nm", sinnmtz, z_on_tz_grid.T) / (nu * nv)
        zbs_in[:N, 0] = 0
        zbs_in[:] *= 2
        zbs_in[N, 0] /= 2

        rbc, zbs = rbc_in, zbs_in

        if plot_ft:

            def boundary_poincare_plot(
                rbc,
                zbs,
                phi,
                N,
                M,
                nfp,
                ntheta=200,
            ):
                xn = np.arange(-N, N + 1, 1)
                xm = np.arange(0, M + 1, 1)

                ntheta = 200
                theta = np.linspace(0, 2 * np.pi, num=ntheta)

                R = np.zeros((ntheta, 1))
                Z = np.zeros((ntheta, 1))

                for i in range(rbc.shape[0]):
                    for j in range(rbc.shape[1]):
                        if rbc[i, j] != 0 or zbs[i, j] != 0:
                            angle = xm[j] * theta - xn[i] * phi * nfp
                            R = R + rbc[i, j] * np.cos(
                                angle
                            )  # /(np.abs(i) + np.abs(j))
                            Z = Z + zbs[i, j] * np.sin(
                                angle
                            )  # /(np.abs(i) + np.abs(j))
                return R.flatten(), Z.flatten()

            n_rows = 2
            n_cols = 4
            figsize = (14.5, 8.1)
            fig_poincare, axes = plt.subplots(
                n_rows, n_cols, figsize=figsize, subplot_kw={"aspect": "equal"}
            )
            axes = axes.flatten()
            phi_array = np.linspace(0, np.pi / 2, 5)
            for k, phi in enumerate(phi_array):
                cs_xyz = self.cross_section(phi / (2 * np.pi), thetas=200)
                R_spline_plot = np.sqrt(cs_xyz[:, 0] ** 2 + cs_xyz[:, 1] ** 2)
                Z_spline_plot = cs_xyz[:, 2]
                axes[k].plot(
                    R_spline_plot,
                    Z_spline_plot,
                    "k--",
                    lw=1,
                    label="Spline (ground truth)",
                )
                R_ft, Z_ft = boundary_poincare_plot(
                    rbc, zbs, phi, self.N, self.M, self.nfp
                )
                axes[k].plot(R_ft, Z_ft, lw=1, label="FT")
                axes[k].legend()

        return rbc, zbs

    def centroid_axis_fourier_coeffs(self, N=6, nv=300, plot=False):
        """
        Return Fourier coefficients for the centroid axis, to be used as an
        initial guess in the VMEC input.

        Parameters
        ----------
        N : int
            Number of Fourier modes.
        nv : int
            Number of points used to sample the axis before fitting.
        plot : bool
            Whether to plot the fit.

        Returns
        -------
        r_n, z_n : ndarray, shape (2*N+1,)
            Fourier coefficients of the centroid axis (R, Z).
        """
        vlist = np.linspace(0, 2 * np.pi / self.nfp, nv, endpoint=False)
        x_ax, y_ax, z_ax = self.centroid_axis_callable(vlist)
        zeta = np.arctan2(y_ax, x_ax) % (2 * np.pi / self.nfp)
        zeta = np.concatenate((zeta - 2 * np.pi, zeta, zeta + 2 * np.pi))
        zeta_eval = np.linspace(0, 2 * np.pi / self.nfp, nv)
        x_zeta = griddata(
            points=zeta,
            values=np.tile(x_ax, 3),
            xi=zeta_eval,
            method="cubic",
        )
        y_zeta = griddata(
            points=zeta,
            values=np.tile(y_ax, 3),
            xi=zeta_eval,
            method="cubic",
        )
        z_zeta = griddata(
            points=zeta,
            values=np.tile(z_ax, 3),
            xi=zeta_eval,
            method="cubic",
        )
        if plot:
            fig = plt.figure("Axis plot")
            fig.patch.set_facecolor("white")
            ax = fig.add_subplot(projection="3d", azim=0, elev=90)
            ax.plot(x_zeta, y_zeta, z_zeta)

        r_zeta = np.sqrt(x_zeta**2 + y_zeta**2)
        cosnz = np.array(
            [np.cos(-n * zeta_eval * self.nfp) for n in range(-N, N + 1)]
        )
        sinnz = np.array(
            [np.sin(-n * zeta_eval * self.nfp) for n in range(-N, N + 1)]
        )
        r_n = np.einsum("nz,z->n", cosnz, r_zeta) / nv
        z_n = np.einsum("nz,z->n", sinnz, z_zeta) / nv

        if plot:
            zeta_plot = np.linspace(0, 2 * np.pi / self.nfp, 200)
            # plotting fourier transformed axis
            r = np.zeros_like(zeta_plot)
            z = np.zeros_like(zeta_plot)
            for i, n in enumerate(range(-N, N + 1)):
                r += r_n[i] * np.cos(-n * zeta_plot * self.nfp)
                z += z_n[i] * np.sin(-n * zeta_plot * self.nfp)
            x = r * np.cos(zeta_plot)
            y = r * np.sin(zeta_plot)
            ax.plot(x, y, z, "r--")
            plt.show()

        return r_n, z_n

    def to_RZFourier(
        self,
        R=None,
        z=None,
        zeta=None,
        theta=None,
        interp=False,
        M=None,
        N=None,
        nu=None,
        nv=None,
        nu_interp=None,
        nv_interp=None,
        plot=False,
        collocation="exact",
        spec_cond="variational",
        spec_cond_options=None,
    ):
        # print('to_RZFourier called')
        if M is None and N is None:
            M, N = self.M, self.N

        rbc, zbs = self.ft(
            R,
            z,
            zeta,
            theta,
            nu,
            nv,
            nv_interp,
            nu_interp,
            plot,
            _fsolve=interp,
            collocation=collocation,
            spec_cond=spec_cond,
            spec_cond_options=spec_cond_options,
        )

        surf = SurfaceRZFourier(nfp=self.nfp, ntor=N, mpol=M)

        for m in range(0, M + 1):
            for n in range(-N, N + 1):
                if m == 0 and n < 0:
                    continue
                else:
                    surf.set_rc(m, n, rbc[n + N, m])
                    surf.set_zs(m, n, zbs[n + N, m])

        if spec_cond == "variational":
            default_options = {
                "plot": False,
                "ftol": 1e-4,
                "Mtol": 1.1,
                "shapetol": 1e-3,
                "niters": 400,
                "verbose": False,
                "cutoff": 1e-8,
            }
            options = (
                spec_cond_options
                if spec_cond_options is not None
                else default_options
            )
            surf = surf.variational_spec_cond(**options)
        elif spec_cond == "direct":
            default_options = {
                "verbose": False,
                "method": "trf",
                "Fourier_continuation": False,
            }
            options = (
                spec_cond_options
                if spec_cond_options is not None
                else default_options
            )
            surf.condense_spectrum(**options)

        return surf

    def to_RZFourier_inner(
        self,
        interp=False,
    ):
        M, N = self.M, self.N

        if interp:
            rbc, zbs = self.to_vmec_interp(fsolve=True)
        else:
            rbc, zbs = self.to_vmec_interp()

        surf = SurfaceRZFourier(nfp=self.nfp, ntor=N, mpol=M)

        for m in range(0, M + 1):
            for n in range(-N, N + 1):
                if m == 0 and n < 0:
                    continue
                else:
                    surf.set_rc(m, n, rbc[n + N, m])
                    surf.set_zs(m, n, zbs[n + N, m])

        # vmec = vmec_from_surf(surf)
        # vmec.run()

        return surf

    def write_inequality_constraints(
        self,
        maxval=np.inf,
        constrain_radii=False,
        axis_r_max=10.0,
        cs_r_max=1.0,
    ):
        """
        Build linear inequality constraints lb <= A @ dofs <= ub, intended
        as a less restrictive alternative to box-bounding every dof
        directly -- in particular the angle dofs, where independent box
        bounds on each theta_k can't express "these must stay in order",
        so the optimizer is free to walk them past each other and fold
        the cross section's control polygon over on itself. A linear
        ordering constraint between consecutive thetas can express that
        directly.

        All constraints are built over free dofs only. Where one endpoint
        of a would-be two-dof constraint happens to be fixed, its fixed
        value is folded into the bound as a constant instead of being
        given a matrix row (there's no free dof for that entry, but the
        remaining free endpoint still needs the bound).

        Parameters
        ----------
        maxval : float
            Upper bound used for otherwise-unbounded constraints.
        constrain_radii : bool
            Every cross-section radius is always constrained to be >= 0
            and <= cs_r_max (both basic physical/scale sanity checks --
            unconditional, not gated by this flag; observed during
            optimization: without an upper bound, the cross-section and
            axis radii can grow without limit together). If
            constrain_radii is additionally True, each radius is also
            upper-bounded (for cross sections after the first) by the
            pseudo-axis's own r_ctrl at the *matching* index specifically
            -- a tighter, per-cross-section refinement on top of the
            unconditional cs_r_max cap. Off by default: that extra,
            index-matched behavior isn't actually necessary to prevent
            self-intersection (a properly-ordered cross section can still
            self-intersect via its radii alone, and unbounded radii don't
            by themselves cause self-intersection -- a real
            non-self-intersection constraint is a separate, nonlinear
            thing, not yet implemented here). This option also inherits a
            pre-existing limitation: it matches cross-section index i
            directly against PseudoAxis1:r_axis_{i}, which is only
            meaningful when axis_points == n_cs.
        axis_r_max : float
            Upper bound on each pseudo-axis r_ctrl value -- unconditional,
            same motivation as cs_r_max below (observed unbounded growth
            during optimization).
        cs_r_max : float
            Upper bound on each cross-section radius -- unconditional,
            same motivation as axis_r_max above. A fixed constant rather
            than a bound relative to another dof (e.g. the axis's own
            r_ctrl) specifically because SLSQP's line search can probe
            trial points that don't respect every constraint
            simultaneously -- a dof-relative bound can still be blown
            through if the reference dof is *also* moving in the same
            wild trial step.

        Returns
        -------
        A : ndarray
            Constraint matrix.
        lb, ub : ndarray
            Lower and upper bounds.
        constraint_titles : list of str
            Human-readable label for each constraint row.
        """
        dofs = self.dof_names
        indices_dict = dict(zip(dofs, range(len(dofs))))

        if self.cs_basis != "polar":
            return np.zeros((0, len(dofs))), np.array([]), np.array([]), []

        constraints_list = []
        lb = []
        ub = []
        constraint_titles = []

        # Name prefixes ("PseudoAxis1", "CrossSectionFixedZeta1", ...) are
        # NOT reliably "1" -- simsopt's auto-naming counter is global and
        # per-class, incrementing across every instance ever constructed
        # in the process, not per-SurfaceBSpline. Hardcoding the prefix
        # (as this method used to) works only for the first SurfaceBSpline
        # built in a given process and KeyErrors for every one after --
        # use each object's own .name instead.
        axis_name_prefix = self.axis.name

        for i in range(1, self.axis_points - 1):
            temp = np.zeros(len(dofs))
            temp[indices_dict[f"{axis_name_prefix}:z_axis_{i}"]] = 1
            constraints_list.append(temp)
            constraint_titles.append(f"-1 < {axis_name_prefix}:z_axis_{i} < 1")
            lb.append(-1)
            ub.append(1)

        # range(self.axis_points), not range(1, ...): r_axis_0 is free by
        # default (only z_axis_0/zeta_axis_0 are fixed in PseudoAxis), so
        # it needs this bound too -- guard on indices_dict for scripts
        # that do fix it explicitly (e.g. axis.fix("r_axis_0")).
        for i in range(self.axis_points):
            name = f"{axis_name_prefix}:r_axis_{i}"
            if name not in indices_dict:
                continue
            temp = np.zeros(len(dofs))
            temp[indices_dict[name]] = 1
            constraints_list.append(temp)
            constraint_titles.append(f"0 < {name} < {axis_r_max}")
            lb.append(0)
            ub.append(axis_r_max)

        # r >= 0 is a basic physical necessity (a radius can't be
        # negative), not a self-intersection-prevention measure, so it's
        # unconditional -- unlike constrain_radii's upper-bound behavior
        # below, this isn't optional.
        for i, cs in enumerate(self.cs_list):
            for j in range(cs.n_pts):
                name = f"{cs.name}:r_{j}"
                if name not in indices_dict:
                    continue
                temp = np.zeros(len(dofs))
                temp[indices_dict[name]] = 1
                constraints_list.append(temp)
                constraint_titles.append(f"0 <= {name}")
                lb.append(0)
                ub.append(maxval)

        # r_cs <= cs_r_max for every cross-section radius -- unconditional
        # (same reasoning as r >= 0 above): without some upper bound, the
        # cross-section and axis radii have been observed to grow without
        # limit together during optimization. A fixed numeric cap (rather
        # than tying it to another dof's current value, e.g. the axis's
        # own r_ctrl) matters here specifically because SLSQP's line
        # search can probe trial points that don't respect every
        # constraint simultaneously -- a dof-relative bound can still be
        # blown through if the dof it's relative to is *also* moving in
        # the same wild trial step, where a fixed constant can't.
        for cs in self.cs_list:
            for j in range(cs.n_pts):
                name = f"{cs.name}:r_{j}"
                if name not in indices_dict:
                    continue
                temp = np.zeros(len(dofs))
                temp[indices_dict[name]] = 1
                constraints_list.append(temp)
                constraint_titles.append(f"{name} <= {cs_r_max}")
                lb.append(-maxval)
                ub.append(cs_r_max)

        if constrain_radii:
            for i, cs in enumerate(self.cs_list):
                for j in range(cs.n_pts):
                    name = f"{cs.name}:r_{j}"
                    if name not in indices_dict:
                        continue
                    axis_r_name = f"{axis_name_prefix}:r_axis_{i}"
                    if i > 0 and axis_r_name in indices_dict:
                        temp = np.zeros(len(dofs))
                        temp[indices_dict[name]] = -1
                        temp[indices_dict[axis_r_name]] = 1
                        constraints_list.append(temp)
                        constraint_titles.append(
                            f"0 < {axis_r_name} - {name} < {maxval}"
                        )
                        lb.append(0)
                        ub.append(maxval)

        # theta ordering: theta_0 >= 0, theta_k <= theta_{k+1} for each
        # consecutive pair, theta_{n-1} <= 2*pi -- applies regardless of
        # cs_equispaced/z_sym, since it's just as important to keep an
        # equispaced cross section's *free* dofs (theta_0/global angle
        # aside) from reordering as a non-equispaced one's.
        for cs in self.cs_list:
            n_pts = cs.n_pts
            theta_names = [f"{cs.name}:theta_{k}" for k in range(n_pts)]

            def theta_value(k, cs=cs):
                return cs.get(f"theta_{k}")

            if theta_names[0] in indices_dict:
                temp = np.zeros(len(dofs))
                temp[indices_dict[theta_names[0]]] = 1
                constraints_list.append(temp)
                constraint_titles.append(f"0 <= {theta_names[0]}")
                lb.append(0)
                ub.append(maxval)

            for k in range(n_pts - 1):
                name_k, name_k1 = theta_names[k], theta_names[k + 1]
                free_k = name_k in indices_dict
                free_k1 = name_k1 in indices_dict
                if not free_k and not free_k1:
                    continue
                temp = np.zeros(len(dofs))
                if free_k:
                    temp[indices_dict[name_k]] = -1
                if free_k1:
                    temp[indices_dict[name_k1]] = 1
                constraints_list.append(temp)
                constraint_titles.append(f"{name_k} <= {name_k1}")
                if free_k and free_k1:
                    lb.append(0)
                    ub.append(maxval)
                elif free_k1:
                    # theta_k fixed at a constant c -> theta_k1 >= c
                    lb.append(theta_value(k))
                    ub.append(maxval)
                else:
                    # theta_k1 fixed at a constant c -> theta_k <= c, and
                    # the row is -theta_k, so the bound is -c
                    lb.append(-theta_value(k + 1))
                    ub.append(maxval)

            if theta_names[-1] in indices_dict:
                temp = np.zeros(len(dofs))
                temp[indices_dict[theta_names[-1]]] = 1
                constraints_list.append(temp)
                constraint_titles.append(f"{theta_names[-1]} <= 2*pi")
                lb.append(-maxval)
                ub.append(2 * np.pi)

        A = (
            np.array(constraints_list)
            if constraints_list
            else np.zeros((0, len(dofs)))
        )
        lb = np.array(lb)
        ub = np.array(ub)

        return A, lb, ub, constraint_titles

    def write_ub_constraints(self):
        """
        DEPRECATED: has the same not-fully-verified-correct constraint
        logic as write_inequality_constraints did before it was reworked
        (see that method's docstring), but hasn't itself been reworked
        yet. Use write_inequality_constraints instead -- it's the one
        that's been checked over. Kept only for reference/until a
        single-sided-bound variant is actually needed again.

        Build the linear inequality constraints A @ dofs <= b_ub.

        Returns
        -------
        A : ndarray
            Constraint matrix.
        b_ub : ndarray
            Upper bounds.
        constraint_titles : list of str
            Human-readable label for each constraint row.
        """
        warnings.warn(
            "write_ub_constraints is deprecated and has known, unresolved "
            "correctness issues -- use write_inequality_constraints "
            "instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        dofs = self.dof_names
        indices_dict = dict(zip(dofs, range(len(self.dof_names))))

        # print(indices_dict)

        constraints_list = []
        A_ub = []
        # lb = []
        # ub = []
        constraint_titles = []

        if self.cs_basis == "polar":
            for i in range(1, self.axis_points - 1):
                temp = np.zeros(len(dofs))
                temp[indices_dict[f"PseudoAxis1:z_axis_{i}"]] = 1
                constraints_list.append(np.copy(temp))
                constraint_titles.append(f"PseudoAxis1:z_axis_{i} < 1")
                A_ub.append(1)
                temp = np.zeros(len(dofs))
                temp[indices_dict[f"PseudoAxis1:z_axis_{i}"]] = -1
                constraints_list.append(np.copy(temp))
                constraint_titles.append(f"-PseudoAxis1:z_axis_{i} < 1")
                A_ub.append(1)

            for i in range(1, self.axis_points):
                temp = np.zeros(len(dofs))
                temp[indices_dict[f"PseudoAxis1:r_axis_{i}"]] = -1
                constraints_list.append(np.copy(temp))
                constraint_titles.append(f"PseudoAxis1:r_axis_{i} > 0")
                A_ub.append(0)

            # radii in cross section
            if self.axis_angles_fixed:
                """
                Make sure that the radii for a given cross section does not exceed half of the pseudo axis
                radius at the same zeta
                """
                for i, cs in enumerate(
                    self.cs_list
                ):  # TODO change if indexing ever gets fixed
                    for j in range(cs.n_pts):
                        temp = np.zeros(len(dofs))
                        temp[
                            indices_dict[f"CrossSectionFixedZeta{i + 1}:r_{j}"]
                        ] = -1
                        constraints_list.append(np.copy(temp))
                        constraint_titles.append(
                            f"- CrossSectionFixedZeta{i + 1}:r_{j} < 0"
                        )
                        A_ub.append(0)

            # thetas in cross section
            if self.cs_equispaced == False:
                for i, cs in enumerate(self.cs_list):
                    if cs.z_sym:
                        max_angle = np.pi
                        # for j in range(1, cs.n_pts-1):
                        temp = np.zeros(len(dofs))
                        temp[
                            indices_dict[
                                f"CrossSectionFixedZeta{i + 1}:theta_{1}"
                            ]
                        ] = -1
                        constraints_list.append(np.copy(temp))
                        constraint_titles.append(
                            f"-CrossSectionFixedZeta{i + 1}:theta_{1} < {0}"
                        )
                        A_ub.append(0)
                        for j in range(1, cs.n_pts - 2):
                            temp = np.zeros(len(dofs))
                            temp[
                                indices_dict[
                                    f"CrossSectionFixedZeta{i + 1}:theta_{j}"
                                ]
                            ] = 1
                            temp[
                                indices_dict[
                                    f"CrossSectionFixedZeta{i + 1}:theta_{j + 1}"
                                ]
                            ] = -1
                            constraint_titles.append(
                                f"CrossSectionFixedZeta{i + 1}:theta_{j} - CrossSectionFixedZeta{i + 1}:theta_{j + 1} < 0"
                            )
                            constraints_list.append(np.copy(temp))
                            A_ub.append(0)
                        temp = np.zeros(len(dofs))
                        temp[
                            indices_dict[
                                f"CrossSectionFixedZeta{i + 1}:theta_{cs.n_pts - 2}"
                            ]
                        ] = -1
                        constraints_list.append(np.copy(temp))
                        constraint_titles.append(
                            f"-CrossSectionFixedZeta{i + 1}:theta_{cs.n_pts - 2} < {0}"
                        )
                        A_ub.append(0)
                    else:
                        min_angle = 0
                        max_angle = np.pi

                        temp = np.zeros(len(dofs))
                        temp[
                            indices_dict[
                                f"CrossSectionFixedZeta{i + 1}:theta_{0}"
                            ]
                        ] = -1
                        constraint_titles.append(
                            f"-CrossSectionFixedZeta{i + 1}:theta_{0} < {min_angle}"
                        )
                        constraints_list.append(np.copy(temp))
                        A_ub.append(min_angle)
                        for j in range(0, (cs.n_pts // 2) - 1):
                            temp = np.zeros(len(dofs))
                            temp[
                                indices_dict[
                                    f"CrossSectionFixedZeta{i + 1}:theta_{j}"
                                ]
                            ] = 1
                            temp[
                                indices_dict[
                                    f"CrossSectionFixedZeta{i + 1}:theta_{j + 1}"
                                ]
                            ] = -1
                            constraint_titles.append(
                                f"CrossSectionFixedZeta{i + 1}:theta_{j} - CrossSectionFixedZeta{i + 1}:theta_{j + 1}< 0"
                            )
                            constraints_list.append(np.copy(temp))
                            A_ub.append(0)
                        temp = np.zeros(len(dofs))
                        temp[
                            indices_dict[
                                f"CrossSectionFixedZeta{i + 1}:theta_{(cs.n_pts // 2) - 1}"
                            ]
                        ] = 1
                        constraint_titles.append(
                            f"CrossSectionFixedZeta{i + 1}:theta_{(cs.n_pts // 2) - 1} < {max_angle}"
                        )
                        constraints_list.append(np.copy(temp))
                        A_ub.append(max_angle)

                        min_angle = np.pi
                        max_angle = 2 * np.pi
                        temp = np.zeros(len(dofs))
                        temp[
                            indices_dict[
                                f"CrossSectionFixedZeta{i + 1}:theta_{(cs.n_pts // 2)}"
                            ]
                        ] = -1
                        # print(f'CrossSectionFixedZeta{i+1}:theta_{(cs.n_pts // 2)}')
                        constraint_titles.append(
                            f"-CrossSectionFixedZeta{i + 1}:theta_{(cs.n_pts // 2)} < {-max_angle}"
                        )
                        constraints_list.append(np.copy(temp))
                        A_ub.append(-max_angle)
                        for j in range(cs.n_pts // 2, cs.n_pts - 1):
                            temp = np.zeros(len(dofs))
                            temp[
                                indices_dict[
                                    f"CrossSectionFixedZeta{i + 1}:theta_{j}"
                                ]
                            ] = 1
                            temp[
                                indices_dict[
                                    f"CrossSectionFixedZeta{i + 1}:theta_{j + 1}"
                                ]
                            ] = -1
                            constraint_titles.append(
                                f"CrossSectionFixedZeta{i + 1}:theta_{j} - CrossSectionFixedZeta{i + 1}:theta_{j + 1} < {0}"
                            )
                            constraints_list.append(np.copy(temp))
                            A_ub.append(0)
                        temp = np.zeros(len(dofs))
                        temp[
                            indices_dict[
                                f"CrossSectionFixedZeta{i + 1}:theta_{cs.n_pts - 1}"
                            ]
                        ] = 1
                        constraints_list.append(np.copy(temp))
                        constraint_titles.append(
                            f"CrossSectionFixedZeta{i + 1}:theta_{cs.n_pts - 1} < {max_angle}"
                        )
                        A_ub.append(max_angle)

        if self.cs_basis == "cartesian":
            raise NotImplementedError

        A = np.array(constraints_list)

        return A, A_ub, constraint_titles

    def set_dofs_from_vec(self, dofs):
        # assume that dofs were written for an object whose indices start at 1
        start_idx = 0
        end_idx = 0
        for i, cs in enumerate(self.cs_list):
            end_idx += len(cs.x)
            cs.x = dofs[start_idx:end_idx]
            start_idx = end_idx
        axis_end = end_idx + len(self.axis.x)
        self.axis.x = dofs[end_idx:axis_end]
        self.local_x = dofs[axis_end:]
        return None

    def plot(
        self,
        _surf=True,
        _surf_points=False,
        _ctrl_points=True,
        _ctrl_points_full=True,
        _pseudo_axis=True,
        _pseudo_axis_ctrl_pts=True,
        _centroid_axis=True,
        _rtz_vectors=True,
        _RZ_vectors=False,
        ax=None,
        _surf_kwargs={"alpha": 0.3, "rcount": 64, "ccount": 64},
        _surf_points_kwargs={"color": "k", "marker": "."},
        _ctrl_points_kwargs={"color": "g", "marker": ".", "ls": "--"},
        _pseudo_axis_kwargs={"color": "g", "ls": "-"},
        _pseudo_axis_ctrl_pts_kwargs={"color": "c", "marker": "*"},
        _centroid_axis_kwargs={"color": "r", "ls": "--"},
        _rtz_vectors_kwargs={},
        _RZ_vectors_kwargs={},
    ):
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        xyz_list = self._get_control_points_xyz()

        # Generating surface
        if _surf or _surf_points:
            ax.set_aspect("equal")
            ax.set_ylim(-1, 1)
            ax.set_xlim(-1, 1)
            ax.set_zlim(-1, 1)

            nu = _surf_kwargs["rcount"]
            nv = _surf_kwargs["ccount"]

            u = np.linspace(0, 2 * np.pi, nu, endpoint=True)
            v = np.linspace(0, 2 * np.pi, nv, endpoint=True)
            v_grid, u_grid = np.meshgrid(v, u)

            x_surf, y_surf, z_surf = self.surf_callable(
                u_grid.flatten(), v_grid.flatten()
            )
            x_surf = x_surf.reshape(nu, nv)
            y_surf = y_surf.reshape(nu, nv)
            z_surf = z_surf.reshape(nu, nv)

        #####################################################################
        # plotting surface

        if _surf:
            ax.plot_surface(x_surf, y_surf, z_surf, **_surf_kwargs)
        if _surf_points:
            ax.plot(x_surf, y_surf, z_surf, **_surf_points_kwargs)
        #####################################################################
        # plotting control points
        if _ctrl_points:
            points = np.array(np.vstack(xyz_list))
            if _ctrl_points_full:
                for i in range(0, self.nfp * (2 * (self.n_cs - 1))):
                    ax.plot(
                        np.append(
                            points[
                                i * (self.points_per_cs) : (i + 1)
                                * (self.points_per_cs),
                                0,
                            ],
                            points[i * (self.points_per_cs), 0],
                        ),
                        np.append(
                            points[
                                i * (self.points_per_cs) : (i + 1)
                                * (self.points_per_cs),
                                1,
                            ],
                            points[i * (self.points_per_cs), 1],
                        ),
                        np.append(
                            points[
                                i * (self.points_per_cs) : (i + 1)
                                * (self.points_per_cs),
                                2,
                            ],
                            points[i * (self.points_per_cs), 2],
                        ),
                        **_ctrl_points_kwargs,
                    )
            else:
                for i in range(0, self.n_cs):
                    ax.plot(
                        np.append(
                            points[
                                i * (self.points_per_cs) : (i + 1)
                                * (self.points_per_cs),
                                0,
                            ],
                            points[i * (self.points_per_cs), 0],
                        ),
                        np.append(
                            points[
                                i * (self.points_per_cs) : (i + 1)
                                * (self.points_per_cs),
                                1,
                            ],
                            points[i * (self.points_per_cs), 1],
                        ),
                        np.append(
                            points[
                                i * (self.points_per_cs) : (i + 1)
                                * (self.points_per_cs),
                                2,
                            ],
                            points[i * (self.points_per_cs), 2],
                        ),
                        **_ctrl_points_kwargs,
                    )

        #####################################################################
        # plotting vectors from axis to control points
        if _rtz_vectors:
            cs_zeta, cs_angles = self.get_cs_zeta_angle()
            rtz = np.array(self.get_rtz_full_device()).reshape(-1, 3)
            if _ctrl_points_full:
                pass
            else:
                rtz = rtz[: (self.points_per_cs * self.n_cs), :]
            r_ctrl, theta_ctrl, zeta_ctrl = rtz[:, 0], rtz[:, 1], rtz[:, 2]
            # matches _get_control_points_xyz's offset formula exactly
            # (including which local basis -- fixed R-Z plane or the axis's
            # own Bishop frame, per use_bishop_frame) so these quivers
            # actually point at the real control points instead of where
            # the old, hardcoded (-R_hat, Z_hat) basis would have put them.
            axis_pos, e1, e2 = self._axis_local_basis(zeta_ctrl)
            offset = (
                r_ctrl[:, None] * np.cos(theta_ctrl)[:, None] * e1
                + r_ctrl[:, None] * np.sin(theta_ctrl)[:, None] * e2
            )

            loc_x, loc_y, loc_z = axis_pos[:, 0], axis_pos[:, 1], axis_pos[:, 2]
            dir_x, dir_y, dir_z = offset[:, 0], offset[:, 1], offset[:, 2]

            ax.quiver(
                loc_x, loc_y, loc_z, dir_x, dir_y, dir_z, **_rtz_vectors_kwargs
            )

        if _RZ_vectors:
            xyz = np.array(self._get_control_points_xyz()).reshape(-1, 3)
            x_ctrl, y_ctrl, z_ctrl = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            ax.quiver(
                np.zeros_like(x_ctrl),
                np.zeros_like(x_ctrl),
                np.zeros_like(x_ctrl),
                x_ctrl,
                y_ctrl,
                z_ctrl,
                alpha=0.25,
            )

        #####################################################################
        # plotting axis
        if _pseudo_axis:
            for i in range(1, self.nfp + 1):
                phi = np.linspace(
                    (i - 1) * 2 * np.pi / self.nfp,
                    i * 2 * np.pi / self.nfp,
                    200,
                )
                r_paxis, z_paxis = self._axis_rz(phi)
                ax.plot(
                    r_paxis * np.cos(phi),
                    r_paxis * np.sin(phi),
                    z_paxis,
                    **_pseudo_axis_kwargs,
                )
            rax_ctrl = np.append(self.axis.r_ctrl, self.axis.r_ctrl[-2:0:-1])
            rax_ctrl = np.tile(rax_ctrl, self.nfp)
            zax_ctrl = np.append(self.axis.z_ctrl, -self.axis.z_ctrl[-2:0:-1])
            zax_ctrl = np.tile(zax_ctrl, self.nfp)
            zetaax_ctrl_1fp = np.append(
                self.axis.zeta_ctrl,
                (2 * np.pi / self.nfp) - self.axis.zeta_ctrl[-2:0:-1],
            )
            zetaax_ctrl = np.copy(zetaax_ctrl_1fp)
            for i in range(1, self.nfp):
                zetaax_ctrl = np.append(
                    zetaax_ctrl, zetaax_ctrl_1fp + i * (2 * np.pi / self.nfp)
                )

            xax_ctrl = rax_ctrl * np.cos(zetaax_ctrl)
            yax_ctrl = rax_ctrl * np.sin(zetaax_ctrl)
            zax_ctrl = zax_ctrl

            xax_ctrl = np.append(xax_ctrl, xax_ctrl[0])
            yax_ctrl = np.append(yax_ctrl, yax_ctrl[0])
            zax_ctrl = np.append(zax_ctrl, zax_ctrl[0])

        if _pseudo_axis_ctrl_pts:
            ax.plot(
                xax_ctrl, yax_ctrl, zax_ctrl, **_pseudo_axis_ctrl_pts_kwargs
            )

        #####################################################################
        # Centroid axis
        if _centroid_axis:
            # x_centroid = np.zeros_like(a_basis[:, i])
            # x_centroid = np.zeros_like(a_basis[:, i])
            na = nv
            a = np.linspace(0, 2 * np.pi, na)
            x_centroid, y_centroid, z_centroid = self.centroid_axis_callable(a)

            ax.plot(x_centroid, y_centroid, z_centroid, **_centroid_axis_kwargs)
        ax._axis3don = False
