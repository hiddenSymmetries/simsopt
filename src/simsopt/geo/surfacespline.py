import matplotlib.pyplot as plt
import numpy as np
import simsoptpp as sopp
from scipy.interpolate import griddata
from scipy.optimize import newton

from .._core import Optimizable
from .._core.optimizable import DOFs
from ..geo import SurfaceRZFourier
from ..util.spline_helpers import (
    b_p,
    b_p_deriv,
    b_p_deriv2,
    double_reflection_rmf,
    uniform_knots,
)
from .curve import Curve
from .plotting import fix_matplotlib_3d
from .surface import Surface

__all__ = ["CrossSectionFixedZeta", "PseudoAxis", "SurfaceBSpline"]

PINK = "#ff8fb8"
SEAFOAM = "#5fcfa8"
SEAFOAM_DARK = "#3a9e7c"
FADED = 0.3  # alpha of the control net outside one half field period


class CrossSectionFixedZeta(Optimizable):
    r"""
    Cross section of control points in local polar coordinates, initialized to a circle.

    Parameters
    ----------
    zeta_index : int
        Index of the cross section in the parent `SurfaceBSpline`.
    n_ctrl_pts : int
        Number of control points (including endpoints if z_sym).
    z_sym : bool
        Whether the cross section is up-down symmetric.
    equispaced : bool
        Whether to fix the polar-angle dofs.
    default_r : float
        Initial radius.
    nurbs : bool
        Whether the weights are free dofs.
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
                    lower_for_last = (
                        new_bounds[-2] if len(new_bounds) >= 2 else 0.0
                    )
                    dofs.update_bounds(
                        f"theta_{n_pts - 1}", (lower_for_last, max_angle)
                    )

    def set_dofs_impl(self, v):
        """
        Set r_ctrl, theta_ctrl and w_ctrl from v, keeping them in sync when full_x is reassigned.

        Parameters
        ----------
        v : ndarray
            Full dof vector [r_ctrl, theta_ctrl, w_ctrl].
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
        """
        Return dof names for the radii, angles and weights.

        Parameters
        ----------
        n_pts : int
            Number of stored control points.

        Returns
        -------
        list of str
            Names r_i, theta_i and w_i.
        """
        namelist = []
        for i in range(n_pts):
            namelist.append(f"r_{i}")
        for i in range(n_pts):
            namelist.append(f"theta_{i}")
        for i in range(n_pts):
            namelist.append(f"w_{i}")
        return namelist

    def get_r_ctrl_full(self):
        """
        Return control-point radii for the full cross section, mirrored if z_sym.

        Returns
        -------
        ndarray
            Radii of all control points.
        """
        if self.z_sym:
            if self.n_ctrl_pts % 2 == 1:
                full = np.concatenate((self.r_ctrl, self.r_ctrl[:0:-1]))
            else:
                full = np.concatenate((self.r_ctrl, self.r_ctrl[-2:0:-1]))
        else:
            full = self.r_ctrl
        return full

    def get_theta_ctrl_full(self):
        """
        Return control-point polar angles for the full cross section, mirrored if z_sym.

        Returns
        -------
        ndarray
            Polar angles of all control points.
        """
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
        """
        Return control-point weights for the full cross section, mirrored if z_sym.

        Returns
        -------
        ndarray
            Weights of all control points.
        """
        if self.z_sym:
            if self.n_ctrl_pts % 2 == 1:
                full = np.concatenate((self.w_ctrl, self.w_ctrl[:0:-1]))
            else:
                full = np.concatenate((self.w_ctrl, self.w_ctrl[-2:0:-1]))
        else:
            full = self.w_ctrl
        return full

    def flipped(self):
        """
        Return the up-down mirror image of this cross section.

        Returns
        -------
        CrossSectionFixedZeta
            self if z_sym, otherwise a new mirrored cross section with the same fixed dofs.
        """
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
            # fix() by name: passing a bool would be treated as a boolean mask.
            for is_fixed, name in zip(fixed_list, self.local_full_dof_names):
                if is_fixed:
                    flipped_cs.fix(name)
                    self.fix(name)
            return flipped_cs


class PseudoAxis(sopp.Curve, Curve):
    r"""
    B-spline pseudo-axis with control points in cylindrical coordinates (dofs include both endpoints).

    Parameters
    ----------
    n_ctrl_pts : int
        Number of control points.
    p : int
        B-spline degree.
    nfp : int
        Number of field periods.
    stellsym : bool
        Whether the axis is stellarator symmetric.
    axis_angles_fixed : bool
        Whether to fix the toroidal-angle dofs of the control points.
    quadpoints : int or array-like
        Number of uniform quadpoints, or explicit quadpoints in [0, 1).
    """

    def __init__(
        self,
        n_ctrl_pts=2,
        p=3,
        nfp=2,
        stellsym=True,
        axis_angles_fixed=False,
        quadpoints=61,
    ):
        self.n_ctrl_pts = n_ctrl_pts
        self.stellsym = stellsym
        self.nfp = nfp
        self.p = p

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

        # _control_net_and_knots cache
        self._centroids_im = None
        self._knots_a = None

        sopp.Curve.__init__(self, quadpoints)
        Curve.__init__(
            self, dofs=dofs, external_dof_setter=PseudoAxis.set_dofs_impl
        )

    def num_dofs(self):
        """
        Return the total number of dofs.

        Returns
        -------
        int
            Length of full_x.
        """
        return len(self.full_x)

    def get_dofs(self):
        """
        Return the full dof vector.

        Returns
        -------
        ndarray
            full_x.
        """
        return self.full_x

    def _get_control_points_xyz(self):
        """
        Return the Cartesian control points over the full device.

        Returns
        -------
        ndarray, shape (n, 3)
            Control points after stellarator-symmetric reflection and nfp tiling.
        """

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
        Return the cached periodic control net (rebuilt when dofs change) and uniform knot vector (built once).

        Returns
        -------
        centroids_im : ndarray, shape (n + 2*p, 3)
            Control points padded with p points on each end.
        knots_a : ndarray
            Uniform knot vector.
        """
        p = self.p
        recompute_net = self.new_x or self._centroids_im is None
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
                self._knots_a = uniform_knots(centroids_im.shape[0] - 1, p)

            # periodic wraparound: pad p points from each end onto the other
            centroids_im = np.concatenate(
                [centroids_im[-p:], centroids_im, centroids_im[:p]], axis=0
            )
            self._centroids_im = centroids_im

        if self.new_x:
            self.new_x = False

        return self._centroids_im, self._knots_a

    def _solve_v(self, phi, trimmed_ctrl_pts_im, knots_a):
        """
        Newton-solve for the spline parameter v at which atan2(Y(v), X(v)) == phi.

        Parameters
        ----------
        phi : ndarray
            Target toroidal angles in radians.
        trimmed_ctrl_pts_im : ndarray, shape (n, 3)
            Padded control net from `_control_net_and_knots`.
        knots_a : ndarray
            Knot vector from `_control_net_and_knots`.

        Returns
        -------
        ndarray
            Spline parameter v in [0, 2*pi) for each phi.
        """
        p = self.p

        def _xy_and_derivs(v):
            # wrap (not clip) so unconstrained Newton iterates stay in the domain
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
        Evaluate (X, Y, Z) and up to max_deriv t-derivatives, via implicit differentiation of v(t).

        Parameters
        ----------
        quadpoints : array-like
            Points t in [0, 1), with toroidal angle phi = 2*pi*t.
        max_deriv : int
            Highest derivative order to return (0, 1 or 2).

        Returns
        -------
        tuple of ndarray
            X, Y, Z, then dX/dt, dY/dt, dZ/dt if max_deriv >= 1, then second derivatives if max_deriv == 2.
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

        # G = d(phi)/dv; dv/dt = 2*pi / G
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

        # G = N/D; d2v/dt2 = -4*pi^2 * G'(v) / G^3
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
        Check that the toroidal angle increases monotonically in v, as `_solve_v` requires.

        Parameters
        ----------
        n_check : int
            Number of uniformly spaced v values to check.

        Returns
        -------
        bool
            True if d(phi)/dv > 0 at every checked point.
        """
        trimmed_ctrl_pts_im, knots_a = self._control_net_and_knots()
        p = self.p
        v = np.linspace(0, 2 * np.pi, n_check, endpoint=False)
        basis_a, dbasis_a = b_p_deriv(knots_a, p, v)
        X = np.einsum("i,ti->t", trimmed_ctrl_pts_im[:, 0], basis_a)
        Y = np.einsum("i,ti->t", trimmed_ctrl_pts_im[:, 1], basis_a)
        dX = np.einsum("i,ti->t", trimmed_ctrl_pts_im[:, 0], dbasis_a)
        dY = np.einsum("i,ti->t", trimmed_ctrl_pts_im[:, 1], dbasis_a)
        # sign(dphi/dv) == sign(X*dY - Y*dX)
        return bool(np.all(X * dY - Y * dX > 0))

    def gamma_impl(self, data, quadpoints):
        """
        Evaluate the axis position.

        Parameters
        ----------
        data : ndarray, shape (len(quadpoints), 3)
            Output array, filled in place.
        quadpoints : array-like
            Points t in [0, 1).
        """
        X, Y, Z = self._gamma_and_derivs(quadpoints, max_deriv=0)
        data[:, 0] = X
        data[:, 1] = Y
        data[:, 2] = Z

    def gammadash_impl(self, data):
        """
        Evaluate the first derivative of the axis position with respect to t.

        Parameters
        ----------
        data : ndarray, shape (len(quadpoints), 3)
            Output array, filled in place.
        """
        _, _, _, dXdt, dYdt, dZdt = self._gamma_and_derivs(
            self.quadpoints, max_deriv=1
        )
        data[:, 0] = dXdt
        data[:, 1] = dYdt
        data[:, 2] = dZdt

    def gammadashdash_impl(self, data):
        """
        Evaluate the second derivative of the axis position with respect to t.

        Parameters
        ----------
        data : ndarray, shape (len(quadpoints), 3)
            Output array, filled in place.
        """
        (_, _, _, _, _, _, d2Xdt2, d2Ydt2, d2Zdt2) = self._gamma_and_derivs(
            self.quadpoints, max_deriv=2
        )
        data[:, 0] = d2Xdt2
        data[:, 1] = d2Ydt2
        data[:, 2] = d2Zdt2

    def bishop_frame(self, quadpoints=None, n_prop=2000):
        r"""
        Bishop (rotation-minimizing) frame, made nfp-periodic by a linear twist correction.

        Uses the double-reflection method (Wang, Juttler, Zheng, Liu, ACM TOG 27(1), 2008).
        The frame is not stellarator symmetric within a field period.

        Parameters
        ----------
        quadpoints : array-like or None
            Points t in [0, 1); defaults to self.quadpoints.
        n_prop : int
            Number of propagation grid points per field period.

        Returns
        -------
        T, N, B : ndarray, each (n, 3)
            Unit tangent, normal, and binormal.
        """
        if quadpoints is None:
            quadpoints = self.quadpoints
        quadpoints = np.asarray(quadpoints)
        nfp = self.nfp

        eps = np.finfo(float).eps
        t_grid = np.linspace(eps, 1.0 / nfp - eps, n_prop)

        # splice requested points into the grid, skipping near-duplicates (the RMF step divides by their spacing)
        snap_tol = 1e-9
        k = np.floor(quadpoints * nfp).astype(int)
        t_local = quadpoints - k / nfp

        t_req = np.unique(t_local)
        j = np.clip(np.searchsorted(t_grid, t_req), 1, n_prop - 1)
        d_grid = np.minimum(
            np.abs(t_req - t_grid[j - 1]), np.abs(t_grid[j] - t_req)
        )
        t_extra = t_req[d_grid > snap_tol]
        if len(t_extra) > 1:
            t_extra = t_extra[
                np.concatenate([[True], np.diff(t_extra) > snap_tol])
            ]
        t_dense = np.sort(np.concatenate([t_grid, t_extra]))
        X, Y, Z, dX, dY, dZ = self._gamma_and_derivs(t_dense, max_deriv=1)
        gamma_dense = np.vstack([X, Y, Z]).T
        gammadash_dense = np.vstack([dX, dY, dZ]).T

        # seed N with R-hat at the first point, projected orthogonal to the tangent
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

        # angle between propagated N at the period's end and the rotated starting N
        N_pred_end = Rzp @ N_dense[0]
        B_pred_end = Rzp @ B_dense[0]
        cos_defect = N_dense[-1] @ N_pred_end
        sin_defect = N_dense[-1] @ B_pred_end
        rot_defect = np.arctan2(sin_defect, cos_defect)

        # beta(t) - beta(t + 1/nfp) = rot_defect, so the frame tiles without a seam
        beta = -rot_defect * nfp * t_dense
        cB, sB = np.cos(beta), np.sin(beta)
        N_corr = cB[:, None] * N_dense + sB[:, None] * B_dense

        j = np.clip(np.searchsorted(t_dense, t_local), 1, len(t_dense) - 1)
        j = np.where(
            np.abs(t_local - t_dense[j - 1]) < np.abs(t_dense[j] - t_local),
            j - 1,
            j,
        )
        N_local = N_corr[j]

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
        Set r_ctrl, z_ctrl and zeta_ctrl from v.

        Parameters
        ----------
        v : ndarray
            Full dof vector [r_ctrl, z_ctrl, zeta_ctrl].
        """
        # not num_dofs(): full_x isn't available yet when called from Optimizable.__init__
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
        """
        Return dof names for the axis control points.

        Returns
        -------
        list of str
            Names r_axis_i, z_axis_i and zeta_axis_i.
        """
        name_list = [
            f"{j}_axis_{i}"
            for j in ("r", "z", "zeta")
            for i in range(self.n_ctrl_pts)
        ]
        return name_list


class SurfaceBSpline(sopp.Surface, Surface):
    r"""
    B-spline surface built from a `PseudoAxis` and polar control-point cross sections.

    Notes
    -----
    Cross sections (`CrossSectionFixedZeta`) define control points in local coordinates at fixed toroidal angle.
    """

    def __init__(
        self,
        axis_points=4,
        n_cs=2,
        points_per_cs=7,
        toroidal_equispaced=True,
        poloidal_equispaced=False,
        axis_angles_fixed=False,
        cs_global_angle_free=False,
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
        use_bishop_frame=False,
    ):
        """
        Parameters
        ----------
        axis_points : int
            Number of axis control points.
        n_cs : int
            Cross sections per half period.
        points_per_cs : int
            Number of control points per cross section.
        toroidal_equispaced : bool
            Whether to fix the toroidal positions (cs_zeta) of the cross sections.
        poloidal_equispaced : bool
            Whether to fix the polar angles of the control points in each cross section.
        axis_angles_fixed : bool
            Whether to fix the toroidal angles of the axis control points.
        cs_global_angle_free : bool
            Whether each cross section's rigid rotation angle (cs_angle) is a free dof.
        nfp : int
            Number of field periods.
        M, N : int
            Default poloidal and toroidal resolution for `to_RZFourier`.
        p_u, p_v : int
            B-spline degree in the poloidal (u) and toroidal (v) directions.
        default_r : float
            Initial cross-section radius.
        stellsym : bool
            Stellarator symmetry; only True is implemented.
        cs_basis : str
            Cross-section coordinates; only "polar" is implemented.
        nurbs : bool
            Whether the control-point weights are free dofs.
        dofs : DOFs, optional
            Precomputed cs_zeta and cs_angle dofs; built from the defaults if None.
        quadpoints_phi, quadpoints_theta : array-like, optional
            Surface quadpoints; simsopt defaults if None.
        use_bishop_frame : bool
            If True, place cross sections in the axis's Bishop frame instead of the fixed-zeta R-Z plane.
        """
        if stellsym:
            max_angle = np.pi / nfp
        else:
            raise NotImplementedError
            max_angle = 2 * np.pi

        self.axis_points = axis_points
        self.points_per_cs = points_per_cs
        self.toroidal_equispaced = toroidal_equispaced
        self.poloidal_equispaced = poloidal_equispaced
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
        self.use_bishop_frame = use_bishop_frame

        if dofs is None:
            # Spacing h makes the gap across the zeta = pi/nfp mirror plane equal the neighbor gap.
            h_zeta = max_angle / (n_cs - 0.5)
            cs_zeta = h_zeta * np.arange(n_cs)
            cs_angles = np.zeros(n_cs)

            self.cs_zeta = cs_zeta
            self.cs_angles = cs_angles

            dofs = np.append(self.cs_zeta, self.cs_angles)

            names = [f"cs_zeta{i}" for i in range(n_cs)] + [
                f"cs_angle{i}" for i in range(n_cs)
            ]
            # cs_zeta bounds: halfway to each neighbor, strictly inside the mirror plane
            zeta_lb = [(n - 0.5) * h_zeta for n in range(n_cs)]
            zeta_ub = [
                min((n + 0.5) * h_zeta, max_angle - 0.25 * h_zeta)
                for n in range(n_cs)
            ]
            dofs = DOFs(
                dofs,
                names,
                [not toroidal_equispaced] * n_cs
                + [cs_global_angle_free] * n_cs,
                zeta_lb + [-2 * np.pi / points_per_cs] * n_cs,
                zeta_ub + [2 * np.pi / points_per_cs] * n_cs,
            )

        # the zeta = 0 cross section is z_sym and pinned
        dofs.fix("cs_angle0")
        dofs.fix("cs_zeta0")

        self.axis = PseudoAxis(
            n_ctrl_pts=axis_points,
            nfp=nfp,
            stellsym=stellsym,
            axis_angles_fixed=axis_angles_fixed,
        )

        self.cs_list = []

        for i in range(n_cs):
            if cs_basis == "polar":
                cross_section = CrossSectionFixedZeta(
                    zeta_index=i,
                    n_ctrl_pts=points_per_cs,
                    equispaced=poloidal_equispaced,
                    default_r=default_r,
                    z_sym=(i == 0),
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

        # _control_net_and_knots cache
        self._control_points_jim = None
        self._w_list_jim = None
        self._knots_u = None
        self._knots_v = None

        # dofs live on self.axis/self.cs_list; depends_on makes them Optimizable ancestors
        Surface.__init__(
            self,
            dofs=dofs,
            external_dof_setter=SurfaceBSpline.set_dofs_impl,
            depends_on=[self.axis] + self.cs_list,
        )

    def num_dofs(self):
        """
        Return the total number of dofs.

        Returns
        -------
        int
            Length of full_x.
        """
        return len(self.full_x)

    def get_dofs(self):
        """
        Return the full dof vector.

        Returns
        -------
        ndarray
            full_x.
        """
        return self.full_x

    def set_dofs_impl(self, v):
        """
        Clear the cached geometry after a dof change.

        Parameters
        ----------
        v : ndarray
            Local dof vector; unused, since the dofs live on self.axis and self.cs_list.
        """
        # dofs already live on self.axis/self.cs_list; only the C++ cache needs clearing
        self.invalidate_cache()

    def recompute_bell(self, parent=None):
        """
        Clear the cached geometry when a parent's dofs change.

        Parameters
        ----------
        parent : Optimizable, optional
            The parent whose dofs changed; unused.
        """
        # dof changes on depends_on parents arrive here, not via set_dofs_impl
        self.invalidate_cache()

    def get_cs_zeta_angle(self):
        """
        Return the toroidal positions and rotation angles of the stored cross sections.

        Returns
        -------
        zeta_list, cs_angle_list : ndarray, each (n_cs,)
            cs_zeta and cs_angle dof values.
        """
        zeta_list = np.array(
            [self.get(f"cs_zeta{i}") for i in range(self.n_cs)]
        )
        cs_angle_list = np.array(
            [self.get(f"cs_angle{i}") for i in range(self.n_cs)]
        )
        return zeta_list, cs_angle_list

    def _axis_rz(self, zeta):
        """
        Return the axis (R, Z) at the given toroidal angles.

        Parameters
        ----------
        zeta : array-like
            Toroidal angles in radians.

        Returns
        -------
        r_axis, z_axis : ndarray
            Axis R and Z at each zeta.
        """
        data = np.zeros((len(zeta), 3))
        self.axis.gamma_impl(data, np.asarray(zeta) / (2 * np.pi))
        r_axis = np.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2)
        z_axis = data[:, 2]
        return r_axis, z_axis

    def _axis_local_basis(self, zeta):
        """
        Return the axis position and the basis (e1, e2) that cross-section offsets are placed in.

        Parameters
        ----------
        zeta : array-like
            Toroidal angles in radians.

        Returns
        -------
        axis_pos, e1, e2 : ndarray, each (len(zeta), 3)
            Axis position and basis: (-R_hat, Z_hat), or (-N, -B) with the Bishop frame.
        """
        zeta = np.asarray(zeta)
        if self.use_bishop_frame:
            data = np.zeros((len(zeta), 3))
            self.axis.gamma_impl(data, zeta / (2 * np.pi))
            _, N, B = self.axis.bishop_frame(zeta / (2 * np.pi))
            # signs flipped so a planar axis matches the non-Bishop basis
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
        Return the Cartesian control points of every cross section over the full device.

        Parameters
        ----------
        return_w : bool
            Whether to also return the weights.

        Returns
        -------
        point_list : list of ndarray
            One (points_per_cs, 3) array per cross section.
        w_list : list of ndarray
            Matching weights; only returned if return_w.
        """
        point_list = []
        w_list = []

        cs_zeta, cs_angles = self.get_cs_zeta_angle()

        cs_zeta_1fp = np.append(
            cs_zeta, (2 * np.pi / self.nfp) - cs_zeta[:0:-1]
        )
        cs_zeta_full = np.concatenate(
            [cs_zeta_1fp + n * (2 * np.pi / self.nfp) for n in range(self.nfp)]
        )

        axis_pos, e1, e2 = self._axis_local_basis(cs_zeta_full)

        cs_list_1fp = [
            cs if (i // self.n_cs) == 0 else cs.flipped()
            for i, cs in enumerate(self.cs_list + self.cs_list[:0:-1])
        ]
        cs_list_full = np.tile(cs_list_1fp, self.nfp)

        cs_angle_1fp = [
            angle if (i // self.n_cs) == 0 else -angle
            for i, angle in enumerate(np.append(cs_angles, cs_angles[:0:-1]))
        ]
        cs_angle_full = np.tile(cs_angle_1fp, self.nfp)

        for i, cs in enumerate(cs_list_full):
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
        Return the centroid of each cross section's control points over the full device.

        Returns
        -------
        list of ndarray
            One (3,) Cartesian centroid per cross section.
        """
        cs_zeta, cs_angles = self.get_cs_zeta_angle()
        point_list = []
        cs_zeta_1fp = np.append(
            cs_zeta, (2 * np.pi / self.nfp) - cs_zeta[:0:-1]
        )
        cs_zeta_full = np.concatenate(
            [cs_zeta_1fp + n * (2 * np.pi / self.nfp) for n in range(self.nfp)]
        )

        axis_pos, e1, e2 = self._axis_local_basis(cs_zeta_full)

        cs_list_1fp = [
            cs if (i // self.n_cs) == 0 else cs.flipped()
            for i, cs in enumerate(self.cs_list + self.cs_list[:0:-1])
        ]
        cs_list_full = np.tile(cs_list_1fp, self.nfp)

        cs_angle_1fp = [
            angle if (i // self.n_cs) == 0 else -angle
            for i, angle in enumerate(np.append(cs_angles, cs_angles[:0:-1]))
        ]
        cs_angle_full = np.tile(cs_angle_1fp, self.nfp)

        for i, cs in enumerate(cs_list_full):
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
        Return the local (r, theta, zeta) of every control point over the full device.

        Returns
        -------
        list of ndarray
            One (points_per_cs, 3) array per cross section.
        """
        point_list = []

        cs_zeta, cs_angles = self.get_cs_zeta_angle()

        cs_zeta_1fp = np.append(
            cs_zeta, (2 * np.pi / self.nfp) - cs_zeta[:0:-1]
        )
        cs_zeta_full = np.concatenate(
            [cs_zeta_1fp + n * (2 * np.pi / self.nfp) for n in range(self.nfp)]
        )

        cs_list_1fp = [
            cs if (i // self.n_cs) == 0 else cs.flipped()
            for i, cs in enumerate(self.cs_list + self.cs_list[:0:-1])
        ]
        cs_list_full = np.tile(cs_list_1fp, self.nfp)

        cs_angle_1fp = [
            angle if (i // self.n_cs) == 0 else -angle
            for i, angle in enumerate(np.append(cs_angles, cs_angles[:0:-1]))
        ]
        cs_angle_full = np.tile(cs_angle_1fp, self.nfp)

        for i, cs in enumerate(cs_list_full):
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
        Return the cached periodic control net, weights and knot vectors; the net is rebuilt when dofs change.

        Returns
        -------
        control_points_jim : ndarray, shape (n_v + 1 + 2*p_v, n_u + 1 + 2*p_u, 3)
            Control net padded periodically in both directions.
        w_list_jim : ndarray, shape (n_v + 1 + 2*p_v, n_u + 1 + 2*p_u)
            Matching weights.
        knots_u, knots_v : ndarray
            Uniform knot vectors in u (poloidal) and v (toroidal).
        """
        p_u = self.p_u
        p_v = self.p_v

        recompute_net = self.new_x or self._control_points_jim is None
        recompute_knots = self._knots_u is None

        if recompute_net:
            point_list, w_list = self._get_control_points_xyz(return_w=True)

            control_points_jim = np.array(point_list)
            w_list_jim = np.array(w_list)

            n_u = control_points_jim.shape[1] - 1
            n_v = control_points_jim.shape[0] - 1

            if recompute_knots:
                knots_u = uniform_knots(n_u, p_u)
                knots_v = uniform_knots(n_v, p_v)
                self._knots_u = knots_u
                self._knots_v = knots_v

            # Periodic wraparound: tile p_u/p_v points from each end onto the opposite side.
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
        Evaluate surface (X, Y, Z) at paired spline parameters u, v.

        Parameters
        ----------
        u : ndarray, shape (N,)
            Poloidal spline parameter in [0, 2*pi).
        v : ndarray, shape (N,)
            Toroidal spline parameter in [0, 2*pi).

        Returns
        -------
        x_surf, y_surf, z_surf : ndarray, shape (N,)
            Cartesian coordinates of the surface points.
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
        Fill data[i] with (X, Y, Z) at paired (phi[i], theta[i]), Newton-solving for v so the toroidal angle is phi.

        Parameters
        ----------
        data : ndarray, shape (N, 3)
            Output array, filled in place.
        quadpoints_phi : ndarray, shape (N,)
            Toroidal angles as fractions in [0, 1).
        quadpoints_theta : ndarray, shape (N,)
            Poloidal spline parameters as fractions in [0, 1).
        """
        trimmed_ctrl_pts_jim, trimmed_weights_ji, knots_u, knots_v = (
            self._control_net_and_knots()
        )
        p_u = self.p_u
        p_v = self.p_v

        # u = -2*pi*theta so the normal points outward, as for other simsopt surfaces
        theta = (-np.asarray(quadpoints_theta) * 2 * np.pi) % (2 * np.pi)
        phi = np.asarray(quadpoints_phi) * 2 * np.pi

        u_basis_row = b_p(knots_u, p_u, theta)  # (N, n_u_basis)

        # collapse the u-direction: one 1D curve in v per query point
        wp = (
            trimmed_weights_ji[:, :, None] * trimmed_ctrl_pts_jim
        )  # (n_v_ext, n_u_ext, 3)
        ww = trimmed_weights_ji  # (n_v_ext, n_u_ext)
        Q_pos = np.einsum("ki,jim->kjm", u_basis_row, wp)  # (N, n_v_ext, 3)
        Q_w = np.einsum("ki,ji->kj", u_basis_row, ww)  # (N, n_v_ext)

        def _xyz_and_derivs(v):
            # wrap (not clip) so Newton iterates stay in the periodic domain
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
        Fill data with (X, Y, Z) on the phi x theta grid, via `gamma_lin`.

        Parameters
        ----------
        data : ndarray, shape (nphi, ntheta, 3)
            Output array, filled in place.
        quadpoints_phi : ndarray, shape (nphi,)
            Toroidal angles as fractions in [0, 1).
        quadpoints_theta : ndarray, shape (ntheta,)
            Poloidal spline parameters as fractions in [0, 1).
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

    def _gamma_and_derivs(self, quadpoints_phi, quadpoints_theta, max_deriv=2):
        r"""
        Gamma and its phi/theta derivatives up to order max_deriv at paired (phi, theta) points.

        Parameters
        ----------
        quadpoints_phi : ndarray, shape (N,)
            Toroidal angles as fractions in [0, 1).
        quadpoints_theta : ndarray, shape (N,)
            Poloidal spline parameters as fractions in [0, 1).
        max_deriv : int
            Highest derivative order to compute: 0, 1, or 2.

        Returns
        -------
        ndarray or tuple of ndarray, each shape (N, 3)
            gamma if max_deriv == 0; else (gamma, gammadash1, gammadash2), plus (gammadash1dash1, gammadash1dash2, gammadash2dash2) if max_deriv >= 2.
        """
        trimmed_ctrl_pts_jim, trimmed_weights_ji, knots_u, knots_v = (
            self._control_net_and_knots()
        )
        p_u = self.p_u
        p_v = self.p_v

        # u = -2*pi*theta so the normal points outward, as for other simsopt surfaces
        theta = (-np.asarray(quadpoints_theta) * 2 * np.pi) % (2 * np.pi)
        phi = np.asarray(quadpoints_phi) * 2 * np.pi

        wp = trimmed_weights_ji[:, :, None] * trimmed_ctrl_pts_jim
        ww = trimmed_weights_ji

        if max_deriv >= 2:
            basis_u, dbasis_u, d2basis_u = b_p_deriv2(knots_u, p_u, theta)
        elif max_deriv == 1:
            basis_u, dbasis_u = b_p_deriv(knots_u, p_u, theta)
        else:
            basis_u = b_p(knots_u, p_u, theta)
        if max_deriv >= 1:
            # d/d(2*pi*theta) = -d/du; second derivatives are unchanged
            dbasis_u = -dbasis_u

        # Contract the u index at each query theta, leaving a rational curve in v.
        Q = np.einsum("ki,jim->kjm", basis_u, wp)
        Q_w = np.einsum("ki,ji->kj", basis_u, ww)
        if max_deriv >= 1:
            Q_du = np.einsum("ki,jim->kjm", dbasis_u, wp)
            Q_w_du = np.einsum("ki,ji->kj", dbasis_u, ww)
        if max_deriv >= 2:
            Q_duu = np.einsum("ki,jim->kjm", d2basis_u, wp)
            Q_w_duu = np.einsum("ki,ji->kj", d2basis_u, ww)

        def _xy_and_dv(v):
            v_wrapped = v % (2 * np.pi)
            basis_v, dbasis_v = b_p_deriv(knots_v, p_v, v_wrapped)
            Nx = np.einsum("mj,mj->m", basis_v, Q[:, :, 0])
            Ny = np.einsum("mj,mj->m", basis_v, Q[:, :, 1])
            D = np.einsum("mj,mj->m", basis_v, Q_w)
            dNx = np.einsum("mj,mj->m", dbasis_v, Q[:, :, 0])
            dNy = np.einsum("mj,mj->m", dbasis_v, Q[:, :, 1])
            dD = np.einsum("mj,mj->m", dbasis_v, Q_w)
            return Nx, Ny, D, dNx, dNy, dD

        def func(v):
            Nx, Ny, D, *_ = _xy_and_dv(v)
            X, Y = Nx / D, Ny / D
            zeta_cur = np.arctan2(Y, X) % (2 * np.pi)
            return ((zeta_cur - phi + np.pi) % (2 * np.pi)) - np.pi

        def fprime(v):
            Nx, Ny, D, dNx, dNy, dD = _xy_and_dv(v)
            X, Y = Nx / D, Ny / D
            dX = (dNx * D - Nx * dD) / D**2
            dY = (dNy * D - Ny * dD) / D**2
            return (X * dY - Y * dX) / (X**2 + Y**2)

        v_sol = newton(
            func, x0=phi.copy(), fprime=fprime, tol=1e-12, maxiter=50
        )
        v_wrapped = v_sol % (2 * np.pi)

        if max_deriv >= 2:
            basis_v, dbasis_v, d2basis_v = b_p_deriv2(knots_v, p_v, v_wrapped)
        elif max_deriv == 1:
            basis_v, dbasis_v = b_p_deriv(knots_v, p_v, v_wrapped)
        else:
            basis_v = b_p(knots_v, p_v, v_wrapped)

        Nx = np.einsum("mj,mj->m", basis_v, Q[:, :, 0])
        Ny = np.einsum("mj,mj->m", basis_v, Q[:, :, 1])
        Nz = np.einsum("mj,mj->m", basis_v, Q[:, :, 2])
        D = np.einsum("mj,mj->m", basis_v, Q_w)

        X, Y, Z = Nx / D, Ny / D, Nz / D
        out = np.stack([X, Y, Z], axis=1)
        if max_deriv == 0:
            return out

        # First u, v partials of the rational surface (quotient rule).
        Nx_v = np.einsum("mj,mj->m", dbasis_v, Q[:, :, 0])
        Ny_v = np.einsum("mj,mj->m", dbasis_v, Q[:, :, 1])
        Nz_v = np.einsum("mj,mj->m", dbasis_v, Q[:, :, 2])
        D_v = np.einsum("mj,mj->m", dbasis_v, Q_w)
        X_v = (Nx_v - X * D_v) / D
        Y_v = (Ny_v - Y * D_v) / D
        Z_v = (Nz_v - Z * D_v) / D

        Nx_u = np.einsum("mj,mj->m", basis_v, Q_du[:, :, 0])
        Ny_u = np.einsum("mj,mj->m", basis_v, Q_du[:, :, 1])
        Nz_u = np.einsum("mj,mj->m", basis_v, Q_du[:, :, 2])
        D_u = np.einsum("mj,mj->m", basis_v, Q_w_du)
        X_u = (Nx_u - X * D_u) / D
        Y_u = (Ny_u - Y * D_u) / D
        Z_u = (Nz_u - Z * D_u) / D

        # Partials of zeta = atan2(Y, X).
        rho2 = X**2 + Y**2
        zeta_v = (X * Y_v - Y * X_v) / rho2
        zeta_u = (X * Y_u - Y * X_u) / rho2

        Gamma_v = np.stack([X_v, Y_v, Z_v], axis=1)
        Gamma_u = np.stack([X_u, Y_u, Z_u], axis=1)

        gammadash1 = Gamma_v * (2 * np.pi / zeta_v)[:, None]
        gammadash2 = (
            2 * np.pi * (Gamma_u - (zeta_u / zeta_v)[:, None] * Gamma_v)
        )

        out = (out, gammadash1, gammadash2)
        if max_deriv == 1:
            return out

        # Second u, v partials of the rational surface (Piegl & Tiller, The NURBS Book).
        Nx_vv = np.einsum("mj,mj->m", d2basis_v, Q[:, :, 0])
        Ny_vv = np.einsum("mj,mj->m", d2basis_v, Q[:, :, 1])
        Nz_vv = np.einsum("mj,mj->m", d2basis_v, Q[:, :, 2])
        D_vv = np.einsum("mj,mj->m", d2basis_v, Q_w)
        X_vv = (Nx_vv - 2 * X_v * D_v - X * D_vv) / D
        Y_vv = (Ny_vv - 2 * Y_v * D_v - Y * D_vv) / D
        Z_vv = (Nz_vv - 2 * Z_v * D_v - Z * D_vv) / D

        Nx_uu = np.einsum("mj,mj->m", basis_v, Q_duu[:, :, 0])
        Ny_uu = np.einsum("mj,mj->m", basis_v, Q_duu[:, :, 1])
        Nz_uu = np.einsum("mj,mj->m", basis_v, Q_duu[:, :, 2])
        D_uu = np.einsum("mj,mj->m", basis_v, Q_w_duu)
        X_uu = (Nx_uu - 2 * X_u * D_u - X * D_uu) / D
        Y_uu = (Ny_uu - 2 * Y_u * D_u - Y * D_uu) / D
        Z_uu = (Nz_uu - 2 * Z_u * D_u - Z * D_uu) / D

        Nx_uv = np.einsum("mj,mj->m", dbasis_v, Q_du[:, :, 0])
        Ny_uv = np.einsum("mj,mj->m", dbasis_v, Q_du[:, :, 1])
        Nz_uv = np.einsum("mj,mj->m", dbasis_v, Q_du[:, :, 2])
        D_uv = np.einsum("mj,mj->m", dbasis_v, Q_w_du)
        X_uv = (Nx_uv - X_u * D_v - X_v * D_u - X * D_uv) / D
        Y_uv = (Ny_uv - Y_u * D_v - Y_v * D_u - Y * D_uv) / D
        Z_uv = (Nz_uv - Z_u * D_v - Z_v * D_u - Z * D_uv) / D

        # Second partials of zeta, via zeta = Im(log(X + iY)).
        rho2_u = 2 * (X * X_u + Y * Y_u)
        rho2_v = 2 * (X * X_v + Y * Y_v)
        zeta_uu = (X * Y_uu - Y * X_uu) / rho2 - (rho2_u * zeta_u) / rho2
        zeta_vv = (X * Y_vv - Y * X_vv) / rho2 - (rho2_v * zeta_v) / rho2
        zeta_uv = (X * Y_uv - Y * X_uv) / rho2 - (
            rho2_u * zeta_v + rho2_v * zeta_u
        ) / (2 * rho2)

        # Implicit derivatives of v(theta, phi) from zeta(2*pi*theta, v) = 2*pi*phi.
        v_phi = 2 * np.pi / zeta_v
        v_theta = -2 * np.pi * zeta_u / zeta_v
        v_phiphi = -4 * np.pi**2 * zeta_vv / zeta_v**3
        v_thetatheta = (
            -4
            * np.pi**2
            * (
                zeta_uu * zeta_v**2
                - 2 * zeta_uv * zeta_u * zeta_v
                + zeta_vv * zeta_u**2
            )
            / zeta_v**3
        )
        v_phitheta = (
            -4 * np.pi**2 * (zeta_v * zeta_uv - zeta_u * zeta_vv) / zeta_v**3
        )

        Gamma_uu = np.stack([X_uu, Y_uu, Z_uu], axis=1)
        Gamma_uv = np.stack([X_uv, Y_uv, Z_uv], axis=1)
        Gamma_vv = np.stack([X_vv, Y_vv, Z_vv], axis=1)

        gammadash1dash1 = (
            Gamma_vv * (v_phi**2)[:, None] + Gamma_v * v_phiphi[:, None]
        )
        gammadash2dash2 = (
            4 * np.pi**2 * Gamma_uu
            + (4 * np.pi * v_theta)[:, None] * Gamma_uv
            + Gamma_vv * (v_theta**2)[:, None]
            + Gamma_v * v_thetatheta[:, None]
        )
        gammadash1dash2 = (
            2 * np.pi * Gamma_uv + Gamma_vv * v_theta[:, None]
        ) * v_phi[:, None] + Gamma_v * v_phitheta[:, None]

        out = out + (gammadash1dash1, gammadash1dash2, gammadash2dash2)
        return out

    def _dash_on_grid(self, max_deriv):
        """
        Evaluate `_gamma_and_derivs` on the quadpoint grid.

        Parameters
        ----------
        max_deriv : int
            Highest derivative order to compute: 0, 1, or 2.

        Returns
        -------
        result : ndarray or tuple of ndarray
            Output of `_gamma_and_derivs` on the flattened (nphi * ntheta) grid.
        reshape3 : callable
            Maps a flattened (nphi * ntheta, 3) array to shape (nphi, ntheta, 3).
        """
        nphi = len(self.quadpoints_phi)
        ntheta = len(self.quadpoints_theta)
        phi2d, theta2d = np.meshgrid(self.quadpoints_phi, self.quadpoints_theta)
        result = self._gamma_and_derivs(
            np.reshape(phi2d, (nphi * ntheta,)),
            np.reshape(theta2d, (nphi * ntheta,)),
            max_deriv=max_deriv,
        )

        def reshape3(arr1d):
            out = np.zeros((nphi, ntheta, 3))
            for xyz in range(3):
                out[:, :, xyz] = np.reshape(arr1d[:, xyz], (ntheta, nphi)).T
            return out

        return result, reshape3

    def gammadash1_impl(self, data):
        """
        Fill data with d(gamma)/d(phi) on the quadpoint grid.

        Parameters
        ----------
        data : ndarray, shape (nphi, ntheta, 3)
            Output array, filled in place.
        """
        (_, g1, _g2), reshape3 = self._dash_on_grid(max_deriv=1)
        data[:, :, :] = reshape3(g1)

    def gammadash2_impl(self, data):
        """
        Fill data with d(gamma)/d(theta) on the quadpoint grid.

        Parameters
        ----------
        data : ndarray, shape (nphi, ntheta, 3)
            Output array, filled in place.
        """
        (_, _g1, g2), reshape3 = self._dash_on_grid(max_deriv=1)
        data[:, :, :] = reshape3(g2)

    def gammadash1dash1_impl(self, data):
        """
        Fill data with d2(gamma)/d(phi)2 on the quadpoint grid.

        Parameters
        ----------
        data : ndarray, shape (nphi, ntheta, 3)
            Output array, filled in place.
        """
        (_, _g1, _g2, g11, _g12, _g22), reshape3 = self._dash_on_grid(
            max_deriv=2
        )
        data[:, :, :] = reshape3(g11)

    def gammadash1dash2_impl(self, data):
        """
        Fill data with d2(gamma)/d(phi)d(theta) on the quadpoint grid.

        Parameters
        ----------
        data : ndarray, shape (nphi, ntheta, 3)
            Output array, filled in place.
        """
        (_, _g1, _g2, _g11, g12, _g22), reshape3 = self._dash_on_grid(
            max_deriv=2
        )
        data[:, :, :] = reshape3(g12)

    def gammadash2dash2_impl(self, data):
        """
        Fill data with d2(gamma)/d(theta)2 on the quadpoint grid.

        Parameters
        ----------
        data : ndarray, shape (nphi, ntheta, 3)
            Output array, filled in place.
        """
        (_, _g1, _g2, _g11, _g12, g22), reshape3 = self._dash_on_grid(
            max_deriv=2
        )
        data[:, :, :] = reshape3(g22)

    def centroid_axis_callable(
        self,
        a,
    ):
        """
        Evaluate the periodic B-spline through the cross-section centroids.

        Parameters
        ----------
        a : ndarray, shape (N,)
            Centroid-axis spline parameter in [0, 2*pi).

        Returns
        -------
        x_centroid, y_centroid, z_centroid : ndarray, shape (N,)
            Cartesian coordinates of the centroid axis.
        """
        xyz_list = self.get_xyz_centroids()
        p_a = self.p_v

        centroids_im = np.array(xyz_list)

        # Periodic wraparound: tile p_a points from each end onto the other.
        knots_a = uniform_knots(centroids_im.shape[0] - 1, p_a)
        centroids_im = np.concatenate(
            [centroids_im[-p_a:], centroids_im, centroids_im[:p_a]], axis=0
        )

        a_basis = b_p(knots_a, p_a, a)

        # einsum, not @: matmul raises spurious RuntimeWarnings on subnormal basis values.
        x_centroid = np.einsum("ij,j->i", a_basis, centroids_im[:, 0])
        y_centroid = np.einsum("ij,j->i", a_basis, centroids_im[:, 1])
        z_centroid = np.einsum("ij,j->i", a_basis, centroids_im[:, 2])

        return x_centroid, y_centroid, z_centroid

    def exact_tz_interp(self, nu=None, nv=None, plot=False, ax=None):
        r"""
        Return R, Z on a (theta, zeta) grid with no interpolation, using the spline u parameter as theta.

        Parameters
        ----------
        nu, nv : int, optional
            Shape of the output arrays. Defaults to `2*nfp*16`.
        plot : bool
            Whether to scatter-plot the resulting points.
        ax : matplotlib 3D axis, optional
            Axis to plot on, if `plot`.

        Returns
        -------
        R, z : ndarray, shape (nv, nu)
            Cylindrical R and Z at each (zeta, theta) grid point.
        zeta, theta : ndarray, shape (nu, nv)
            Meshgrid of toroidal and poloidal angles in [0, 2*pi).
        """
        nu = 2 * self.nfp * 16 if nu is None else nu
        nv = 2 * self.nfp * 16 if nv is None else nv

        # gamma_impl's (phi, theta) order already matches the (zeta, theta) return order.
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

    def ft(
        self,
        R=None,
        z=None,
        zeta=None,
        theta=None,
        nu=None,
        nv=None,
        plot_ft=False,
        ft_ax=None,
        plot_intermediate=False,
        intermediate_ax=None,
    ):
        r"""
        Fourier-transform the spline surface into VMEC coefficients.

        Parameters
        ----------
        R, z, zeta, theta : ndarray, optional
            Precomputed collocation grid; computed with `exact_tz_interp` if any is None.
        nu, nv : int, optional
            Collocation grid shape, also used to normalize the transform. Defaults to `2*nfp*16`.
        plot_ft : bool
            Plot the Fourier reconstruction against the spline.
        ft_ax : sequence of Axes, optional
            Axes for `plot_ft`, passed to `plot_ft_vs_ground_truth`.
        plot_intermediate : bool
            Plot the collocation points.
        intermediate_ax : matplotlib 3D axis, optional
            Axis for `plot_intermediate`.

        Returns
        -------
        rbc, zbs : ndarray, shape (2*N+1, M+1)
            VMEC Fourier coefficients.
        """
        nu = 2 * self.nfp * 16 if nu is None else nu
        nv = 2 * self.nfp * 16 if nv is None else nv
        assert nv % 2 == 0, "nv must be even"

        if (
            np.any(R is None)
            or np.any(z is None)
            or np.any(zeta is None)
            or np.any(theta is None)
        ):
            # theta = spline u: no interpolation error, but not a physical angle.
            R_on_tz_grid, z_on_tz_grid, zeta_eval, theta_eval = (
                self.exact_tz_interp(
                    nu=nu,
                    nv=nv,
                    plot=plot_intermediate,
                    ax=intermediate_ax,
                )
            )
        else:
            R_on_tz_grid, z_on_tz_grid, zeta_eval, theta_eval = (
                R,
                z,
                zeta,
                theta,
            )

        M, N = self.M, self.N

        # Discrete transform; m=0, n<0 modes are zeroed and folded into n>0.
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

        zbs_in = np.einsum("nmtz,tz->nm", sinnmtz, z_on_tz_grid.T) / (nu * nv)
        zbs_in[:N, 0] = 0
        zbs_in[:] *= 2
        zbs_in[N, 0] /= 2

        rbc, zbs = rbc_in, zbs_in

        if plot_ft:
            self.plot_ft_vs_ground_truth(rbc, zbs, ax=ft_ax)

        return rbc, zbs

    def plot_ft_vs_ground_truth(self, rbc, zbs, phi_array=None, ax=None):
        """
        Plot spline cross sections against the (rbc, zbs) Fourier reconstruction.

        Parameters
        ----------
        rbc, zbs : ndarray
            VMEC Fourier coefficients, as returned by `ft`.
        phi_array : array-like, optional
            Toroidal angles to plot, one subplot each. Defaults to 5 angles in [0, pi/2].
        ax : sequence of Axes, optional
            One Axes per angle. Defaults to a new 2x4 grid.

        Returns
        -------
        axes : sequence of Axes
            The Axes plotted on.
        """

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

            theta = np.linspace(0, 2 * np.pi, num=ntheta)

            R = np.zeros((ntheta, 1))
            Z = np.zeros((ntheta, 1))

            for i in range(rbc.shape[0]):
                for j in range(rbc.shape[1]):
                    if rbc[i, j] != 0 or zbs[i, j] != 0:
                        angle = xm[j] * theta - xn[i] * phi * nfp
                        R = R + rbc[i, j] * np.cos(angle)
                        Z = Z + zbs[i, j] * np.sin(angle)
            return R.flatten(), Z.flatten()

        if phi_array is None:
            phi_array = np.linspace(0, np.pi / 2, 5)
        if ax is None:
            _, axes = plt.subplots(
                2, 4, figsize=(14.5, 8.1), subplot_kw={"aspect": "equal"}
            )
            axes = axes.flatten()
        else:
            axes = ax

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

        return axes

    def centroid_axis_fourier_coeffs(self, N=6, nv=300, plot=False):
        """
        Fourier coefficients of the centroid axis, e.g. as a VMEC initial guess.

        Parameters
        ----------
        N : int
            Number of Fourier modes.
        nv : int
            Number of axis samples used for the fit.
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
        nu=None,
        nv=None,
        plot=False,
        spec_cond="variational",
        spec_cond_options=None,
    ):
        """
        Convert this spline to a spectrally condensed `SurfaceRZFourier`.

        Parameters
        ----------
        R, z, zeta, theta : ndarray, optional
            Precomputed collocation grid passed to `ft`.
        nu, nv : int, optional
            Grid sizes passed to `ft`.
        plot : bool
            Plot the Fourier fit against the spline.
        spec_cond : str
            "variational", "direct", or anything else to skip condensation.
        spec_cond_options : dict, optional
            Keyword arguments for the chosen condensation method.

        Returns
        -------
        SurfaceRZFourier
            If condensation raises RuntimeError, the raw fit is plotted before re-raising.
        """
        M, N = self.M, self.N

        rbc, zbs = self.ft(
            R,
            z,
            zeta,
            theta,
            nu,
            nv,
            plot_ft=plot,
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
                "shapetol": 5e-3,
                "niters": 400,
                "verbose": False,
                "cutoff": 1e-8,
            }
            options = (
                spec_cond_options
                if spec_cond_options is not None
                else default_options
            )
            try:
                surf = surf.variational_spec_cond(**options)
            except RuntimeError:
                # Usually a shapetol tuning issue; show the raw fit to help diagnose.
                self.plot_ft_vs_ground_truth(rbc, zbs)
                plt.show()
                raise
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
        else:
            surf = surf

        return surf

    def write_inequality_constraints(
        self,
        maxval=np.inf,
        axis_r_max=10.0,
        axis_r_fixed_max=1.0,
        fix_axis_angles=False,
        z_axis_max=0.5,
        cs_axis_ratio=None,
        cs_angle_step_max=None,
    ):
        """
        Linear constraints lb <= A @ x <= ub on the radii and axis z, over free dofs only.

        Theta ordering is enforced by CrossSectionFixedZeta's box bounds, not here.

        Parameters
        ----------
        maxval : float
            Upper bound used for otherwise-unbounded constraints.
        axis_r_max : float
            Upper bound on r_axis_0 when it is free.
        axis_r_fixed_max : float
            Upper bound on every r_axis_i (i >= 1) when r_axis_0 is fixed.
        fix_axis_angles : bool
            If True, fix all axis zeta dofs first (mutates the surface's dof vector).
        z_axis_max : float
            Bound on |z_axis_i| for interior axis points.
        cs_angle_step_max : float or None
            Bound on the angle step between neighboring cross sections, including across zeta = pi/nfp.
        cs_axis_ratio : float or None
            If set to a, use a * r_cs_j <= mean(r_axis) instead of r_cs_j <= min(r_axis).

        Returns
        -------
        A : ndarray, shape (n_constraints, n_free_dofs)
            Constraint matrix.
        lb, ub : ndarray
            Lower and upper bounds.
        constraint_titles : list of str
            Human-readable label for each constraint row.
        """
        if fix_axis_angles:
            for i in range(self.axis_points):
                if f"{self.axis.name}:zeta_axis_{i}" in self.dof_names:
                    self.axis.fix(f"zeta_axis_{i}")

        dofs = self.dof_names
        indices_dict = dict(zip(dofs, range(len(dofs))))

        if self.cs_basis != "polar":
            return np.zeros((0, len(dofs))), np.array([]), np.array([]), []

        constraints_list = []
        lb = []
        ub = []
        constraint_titles = []

        # Use .name, not a hardcoded "PseudoAxis1": simsopt's name counter is process-global.
        axis_name_prefix = self.axis.name

        for i in range(1, self.axis_points - 1):
            temp = np.zeros(len(dofs))
            temp[indices_dict[f"{axis_name_prefix}:z_axis_{i}"]] = 1
            constraints_list.append(temp)
            constraint_titles.append(
                f"-{z_axis_max} < {axis_name_prefix}:z_axis_{i} < {z_axis_max}"
            )
            lb.append(-z_axis_max)
            ub.append(z_axis_max)

        def axis_r_name(i):
            return f"{axis_name_prefix}:r_axis_{i}"

        def add_row(coeffs, lo, hi, title):
            temp = np.zeros(len(dofs))
            for name, c in coeffs.items():
                temp[indices_dict[name]] = c
            constraints_list.append(temp)
            constraint_titles.append(title)
            lb.append(lo)
            ub.append(hi)

        # Axis radii: r_axis_i <= r_axis_0 if free, else <= axis_r_fixed_max.
        r0_free = axis_r_name(0) in indices_dict
        if r0_free:
            add_row(
                {axis_r_name(0): 1},
                0,
                axis_r_max,
                f"0 < {axis_r_name(0)} < {axis_r_max}",
            )
        for i in range(1, self.axis_points):
            if axis_r_name(i) not in indices_dict:
                continue
            if r0_free:
                add_row(
                    {axis_r_name(i): 1}, 0, maxval, f"0 <= {axis_r_name(i)}"
                )
                add_row(
                    {axis_r_name(0): 1, axis_r_name(i): -1},
                    0,
                    maxval,
                    f"{axis_r_name(i)} <= {axis_r_name(0)}",
                )
            else:
                add_row(
                    {axis_r_name(i): 1},
                    0,
                    axis_r_fixed_max,
                    f"0 < {axis_r_name(i)} < {axis_r_fixed_max}",
                )

        # Cross-section radii: 0 <= r_j <= r_axis_m (r_axis_0 rows are redundant).
        axis_ms = range(1 if self.axis_points > 1 else 0, self.axis_points)
        all_axis = range(self.axis_points)
        for cs in self.cs_list:
            for j in range(cs.n_pts):
                name = f"{cs.name}:r_{j}"
                cs_free = name in indices_dict
                if cs_free:
                    add_row({name: 1}, 0, maxval, f"0 <= {name}")
                cs_val = None if cs_free else cs.get(f"r_{j}")
                if cs_axis_ratio is not None:
                    # a * r_cs - mean(r_axis) <= 0, fixed values as constants
                    n_ax = self.axis_points
                    coeffs = {name: cs_axis_ratio} if cs_free else {}
                    const = 0.0 if cs_free else -cs_axis_ratio * cs_val
                    for m in all_axis:
                        ax_name = axis_r_name(m)
                        if ax_name in indices_dict:
                            coeffs[ax_name] = (
                                coeffs.get(ax_name, 0) - 1.0 / n_ax
                            )
                        else:
                            const += self.axis.get(f"r_axis_{m}") / n_ax
                    if coeffs:
                        add_row(
                            coeffs,
                            -maxval,
                            const,
                            f"{cs_axis_ratio}*{name} <= mean(r_axis)",
                        )
                    continue
                for m in axis_ms:
                    ax_name = axis_r_name(m)
                    ax_free = ax_name in indices_dict
                    title = f"{name} <= {ax_name}"
                    if cs_free and ax_free:
                        add_row({ax_name: 1, name: -1}, 0, maxval, title)
                    elif cs_free:
                        # axis radius fixed at a constant c -> r_j <= c
                        add_row(
                            {name: 1},
                            -maxval,
                            self.axis.get(f"r_axis_{m}"),
                            title,
                        )
                    elif ax_free:
                        # cross-section radius fixed at c -> r_axis_m >= c
                        add_row({ax_name: 1}, cs_val, maxval, title)

        if cs_angle_step_max is not None:
            angle_name = lambda k: f"{self.name}:cs_angle{k}"  # noqa: E731
            for k in range(self.n_cs - 1):
                n0, n1 = angle_name(k), angle_name(k + 1)
                f0, f1 = n0 in indices_dict, n1 in indices_dict
                if not (f0 or f1):
                    continue
                title = (
                    f"|cs_angle{k + 1} - cs_angle{k}| <= {cs_angle_step_max}"
                )
                if f0 and f1:
                    add_row(
                        {n1: 1, n0: -1},
                        -cs_angle_step_max,
                        cs_angle_step_max,
                        title,
                    )
                elif f1:
                    c0 = self.get(f"cs_angle{k}")
                    add_row(
                        {n1: 1},
                        c0 - cs_angle_step_max,
                        c0 + cs_angle_step_max,
                        title,
                    )
                else:
                    c1 = self.get(f"cs_angle{k + 1}")
                    add_row(
                        {n0: 1},
                        c1 - cs_angle_step_max,
                        c1 + cs_angle_step_max,
                        title,
                    )
            # Across zeta = pi/nfp the last cross section (angle a) meets its mirror (-a).
            n_last = angle_name(self.n_cs - 1)
            if n_last in indices_dict:
                add_row(
                    {n_last: 2},
                    -cs_angle_step_max,
                    cs_angle_step_max,
                    f"|2*cs_angle{self.n_cs - 1}| <= {cs_angle_step_max} "
                    "(step across the zeta = pi/nfp mirror plane)",
                )

        A = (
            np.array(constraints_list)
            if constraints_list
            else np.zeros((0, len(dofs)))
        )
        lb = np.array(lb)
        ub = np.array(ub)

        return A, lb, ub, constraint_titles

    def plot(
        self,
        engine="matplotlib",
        ax=None,
        show=True,
        close=True,
        axis_equal=True,
        plot_axis=True,
        plot_control_net=True,
        **kwargs,
    ):
        """
        Plot the surface, with the pseudo-axis and control net for matplotlib and plotly.

        Parameters
        ----------
        engine : str
            "matplotlib", "plotly" or "mayavi"; mayavi draws only the surface.
        ax : mpl_toolkits.mplot3d.Axes3D or plotly.graph_objects.Figure, optional
            Axes (matplotlib) or figure (plotly) to draw on; created if None.
        show : bool
            Show the figure at the end.
        close : bool
            Close the surface seams; see `Surface.plot`.
        axis_equal : bool
            Give the x, y and z axes equal scale (matplotlib only; plotly always does).
        plot_axis : bool
            Plot the pseudo-axis.
        plot_control_net : bool
            Plot the control points and net, faded outside one half field period.
        **kwargs
            Passed to `Surface.plot` for the surface (defaults: pink, alpha/opacity 0.3).

        Returns
        -------
        ax
            The matplotlib axes or plotly figure plotted on.
        """
        if engine not in ("matplotlib", "plotly"):
            return super().plot(
                engine=engine,
                ax=ax,
                show=show,
                close=close,
                axis_equal=axis_equal,
                **kwargs,
            )

        kwargs.setdefault("color", PINK)
        if engine == "matplotlib":
            kwargs.setdefault("alpha", 0.3)
        else:
            kwargs.setdefault("opacity", 0.3)
            kwargs.setdefault("showscale", False)
        ax = super().plot(
            engine=engine,
            ax=ax,
            show=False,
            close=close,
            axis_equal=False,
            **kwargs,
        )

        if plot_axis:
            phi = np.linspace(0, 2 * np.pi, 400)
            r_axis, z_axis = self._axis_rz(phi)
            axis_xyz = np.stack(
                [r_axis * np.cos(phi), r_axis * np.sin(phi), z_axis], axis=1
            )

        if plot_control_net:
            # (n_cross_sections, points_per_cs, 3); rows are ordered in zeta, and
            # the first n_cs rows span one half field period (drawn opaque).
            net = np.array(self._get_control_points_xyz())
            half = self.n_cs
            points = {
                1.0: net[:half].reshape(-1, 3),
                FADED: net[half:].reshape(-1, 3),
            }
            net = np.concatenate([net, net[:, :1]], axis=1)
            net = np.concatenate([net, net[:1]], axis=0)
            columns = net.transpose(1, 0, 2)
            lines = {
                1.0: list(net[:half]) + [col[:half] for col in columns],
                FADED: list(net[half:-1])
                + [col[half - 1 :] for col in columns],
            }

        if engine == "matplotlib":
            if plot_axis:
                ax.plot(*axis_xyz.T, color="k", lw=1)
            if plot_control_net:
                for alpha in (1.0, FADED):
                    ax.scatter(
                        *points[alpha].T,
                        color=SEAFOAM_DARK,
                        s=3,
                        depthshade=False,
                        alpha=alpha,
                    )
                    for line in lines[alpha]:
                        ax.plot(*line.T, color=SEAFOAM, lw=0.8, alpha=alpha)
            if axis_equal:
                fix_matplotlib_3d(ax)
            if show:
                plt.show()
        else:
            import plotly.graph_objects as go

            if plot_axis:
                ax.add_trace(
                    go.Scatter3d(
                        x=axis_xyz[:, 0],
                        y=axis_xyz[:, 1],
                        z=axis_xyz[:, 2],
                        mode="lines",
                        line=dict(color="black", width=3),
                        name="axis",
                        showlegend=False,
                    )
                )
            if plot_control_net:
                for alpha in (1.0, FADED):
                    # one trace per style; NaN rows separate the individual lines
                    segs = np.concatenate(
                        [
                            np.vstack([line, np.full((1, 3), np.nan)])
                            for line in lines[alpha]
                        ]
                    )
                    ax.add_trace(
                        go.Scatter3d(
                            x=segs[:, 0],
                            y=segs[:, 1],
                            z=segs[:, 2],
                            mode="lines",
                            opacity=alpha,
                            line=dict(color=SEAFOAM, width=2),
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
                    ax.add_trace(
                        go.Scatter3d(
                            x=points[alpha][:, 0],
                            y=points[alpha][:, 1],
                            z=points[alpha][:, 2],
                            mode="markers",
                            opacity=alpha,
                            marker=dict(color=SEAFOAM_DARK, size=2),
                            name="control points",
                            showlegend=False,
                        )
                    )
            if show:
                ax.show()
        return ax
