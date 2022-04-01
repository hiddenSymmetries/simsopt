import logging

from matplotlib import pyplot as plt
import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from scipy.sparse import csr_matrix, lil_matrix
import time

logger = logging.getLogger(__name__)


class PermanentMagnetOptimizer:
    r"""
        ``PermanentMagnetOptimizer`` is a class for solving the permanent
        magnet optimization problem for stellarators. The class
        takes as input two toroidal surfaces specified as SurfaceRZFourier
        objects, and initializes a set of points (in cylindrical coordinates)
        between these surfaces. It finishes initialization by pre-computing
        a number of quantities required for the optimization such as the
        geometric factor in the dipole part of the magnetic field and the 
        target Bfield that is a sum of the coil and plasma magnetic fields.
        It then provides functions for solving several variations of the
        optimization problem.

        Args:
            plasma_boundary:  SurfaceRZFourier object representing 
                              the plasma boundary surface.  
            rz_inner_surface: SurfaceRZFourier object representing 
                              the inner toroidal surface of the volume.
                              Defaults to the plasma boundary, extended
                              by the boundary normal vector projected on
                              to the (R, Z) plane to keep the phi = constant
                              cross-section the same as the plasma surface. 
            rz_outer_surface: SurfaceRZFourier object representing 
                              the outer toroidal surface of the volume.
                              Defaults to the inner surface, extended
                              by the boundary normal vector projected on
                              to the (R, Z) plane to keep the phi = constant
                              cross-section the same as the inner surface. 
            plasma_offset:    Offset to use for generating the inner toroidal surface.
            coil_offset:      Offset to use for generating the outer toroidal surface.
            B_plasma_surface: Magnetic field (coils and plasma) at the plasma
                              boundary. Typically this will be the optimized plasma
                              magnetic field from a stage-1 optimization, and the
                              optimized coils from a basic stage-2 optimization. 
                              This variable must be specified to run the permanent 
                              magnet optimization.
    """

    def __init__(
        self, plasma_boundary, rz_inner_surface=None, 
        rz_outer_surface=None, plasma_offset=0.1, 
        coil_offset=0.1, B_plasma_surface=None,
    ):
        self.plasma_offset = plasma_offset
        self.coil_offset = coil_offset
        self.B_plasma_surface = B_plasma_surface
        if not isinstance(plasma_boundary, SurfaceRZFourier):
            raise ValueError(
                'Plasma boundary must be specified as SurfaceRZFourier class.'
            )
        else:
            self.plasma_boundary = plasma_boundary
            self.phi = self.plasma_boundary.quadpoints_phi
            self.nphi = len(self.phi)
            self.theta = self.plasma_boundary.quadpoints_theta
            self.ntheta = len(self.theta)

        # If the inner surface is not specified, make default surface.
        if rz_inner_surface is None:
            print(
                "Inner toroidal surface not specified, defaulting to "
                "extending the plasma boundary shape using the normal "
                " vectors at the quadrature locations. The normal vectors "
                " are projected onto the RZ plane so that the new surface "
                " lies entirely on the original phi = constant cross-section. "
            )
            self._set_inner_rz_surface()
        else:
            self.rz_inner_surface = rz_inner_surface

        # If the outer surface is not specified, make default surface. 
        if rz_outer_surface is None:
            print(
                "Outer toroidal surface not specified, defaulting to "
                "extending the inner toroidal surface shape using the normal "
                " vectors at the quadrature locations. The normal vectors "
                " are projected onto the RZ plane so that the new surface "
                " lies entirely on the original phi = constant cross-section. "
            )
            self._set_outer_rz_surface()
        else:
            self.rz_outer_surface = rz_outer_surface
        if not isinstance(self.rz_inner_surface, SurfaceRZFourier):
            raise ValueError("Inner surface is not SurfaceRZFourier object.")
        if not isinstance(self.rz_inner_surface, SurfaceRZFourier):
            raise ValueError("Outer surface is not SurfaceRZFourier object.")

        # check the inner and outer surface are same size
        # and defined at the same (theta, phi) coordinate locations
        if len(self.rz_inner_surface.quadpoints_theta) != len(self.rz_outer_surface.quadpoints_theta):
            raise ValueError(
                "Inner and outer toroidal surfaces have different number of "
                "poloidal quadrature points."
            )
        if len(self.rz_inner_surface.quadpoints_phi) != len(self.rz_outer_surface.quadpoints_phi):
            raise ValueError(
                "Inner and outer toroidal surfaces have different number of "
                "toroidal quadrature points."
            )
        if np.any(self.rz_inner_surface.quadpoints_theta != self.rz_outer_surface.quadpoints_theta):
            raise ValueError(
                "Inner and outer toroidal surfaces must be defined at the "
                "same poloidal quadrature points."
            )
        if np.any(self.rz_inner_surface.quadpoints_phi != self.rz_outer_surface.quadpoints_phi):
            raise ValueError(
                "Inner and outer toroidal surfaces must be defined at the "
                "same toroidal quadrature points."
            )

        # Get (R, Z) coordinates of the three boundaries
        xyz_plasma = self.plasma_boundary.gamma()
        self.r_plasma = np.sqrt(xyz_plasma[:, :, 0] ** 2 + xyz_plasma[:, :, 1] ** 2)
        self.z_plasma = xyz_plasma[:, :, 2]
        xyz_inner = self.rz_inner_surface.gamma()
        self.r_inner = np.sqrt(xyz_inner[:, :, 0] ** 2 + xyz_inner[:, :, 1] ** 2)
        self.z_inner = xyz_inner[:, :, 2]
        xyz_outer = self.rz_outer_surface.gamma()
        self.r_outer = np.sqrt(xyz_outer[:, :, 0] ** 2 + xyz_outer[:, :, 1] ** 2)
        self.z_outer = xyz_outer[:, :, 2]

        r_max = np.max(self.r_outer)
        r_min = np.min(self.r_outer)
        z_max = np.max(self.z_outer)
        z_min = np.min(self.z_outer)

        # Initialize uniform grid of curved, square bricks
        Delta_r = 0.05
        Nr = int((r_max - r_min) / Delta_r)
        self.Nr = Nr
        Delta_z = Delta_r
        Nz = int((z_max - z_min) / Delta_z)
        self.Nz = Nz
        phi = self.rz_outer_surface.quadpoints_phi
        Nphi = len(phi)
        print('Largest possible dipole dimension is = {0:.2f}'.format(
            (
                r_max - self.plasma_boundary.get_rc(0, 0)
            ) * (phi[1] - phi[0]) * 2 * np.pi
        )
        )
        print('dR = {0:.2f}'.format(Delta_r))
        print('dZ = {0:.2f}'.format(Delta_z))
        norm = self.plasma_boundary.unitnormal()
        phi = self.plasma_boundary.quadpoints_phi
        norms = []
        for i in range(Nphi):
            rot_matrix = [[np.cos(phi[i]), np.sin(phi[i]), 0],
                          [-np.sin(phi[i]), np.cos(phi[i]), 0],
                          [0, 0, 1]]

            norms.append((rot_matrix @ norm[i, :, :].T).T)
        norms = np.array(norms)
        print(
            'Approximate closest distance between plasma and inner surface = {0:.2f}'.format(
                np.min(np.sqrt(norms[:, :, 0] ** 2 + norms[:, :, 2] ** 2) * self.plasma_offset)
            )    
        )
        self.plasma_unitnormal_cylindrical = norms
        R = np.linspace(r_min, r_max, Nr)
        Z = np.linspace(z_min, z_max, Nz)

        # Make 3D mesh
        R, Phi, Z = np.meshgrid(R, phi, Z, indexing='ij')
        self.RPhiZ = np.transpose(np.array([R, Phi, Z]), [1, 2, 3, 0])

        # Have the uniform grid, now need to loop through and eliminate cells. 
        self.final_RZ_grid = self._make_final_surface()

        # Compute the maximum allowable magnetic moment m_max
        phi = self.rz_inner_surface.quadpoints_phi
        B_max = 1.4
        mu0 = 4 * np.pi * 1e-7
        dipole_grid_r = np.zeros(self.ndipoles)
        running_tally = 0
        for i in range(self.nphi):
            radii = np.ravel(np.array(self.final_RZ_grid[i])[:, 0])
            len_radii = len(radii)
            dipole_grid_r[running_tally:running_tally + len_radii] = radii
            running_tally += len_radii
        cell_vol = dipole_grid_r * Delta_r * Delta_z * (phi[1] - phi[0]) * 2 * np.pi
        # FAMUS paper says m_max = B_r / (mu0 * cell_vol) but it 
        # should be m_max = B_r * cell_vol / mu0  (just from units)
        self.m_maxima = B_max * cell_vol / mu0

        # Compute the geometric factor for the A matrix in the optimization
        self._compute_geometric_factor()

        # optionally plot the plasma boundary + inner/outer surfaces
        self._plot_surfaces()

    def _set_inner_rz_surface(self):
        """
            If the inner toroidal surface was not specified, this function
            is called and simply takes each (r, z) quadrature point on the 
            plasma boundary and shifts it by self.plasma_offset at constant
            theta value. 
        """
        # make copy of plasma boundary
        mpol = self.plasma_boundary.mpol
        ntor = self.plasma_boundary.ntor
        nfp = self.plasma_boundary.nfp
        stellsym = self.plasma_boundary.stellsym
        range_surf = self.plasma_boundary.range 
        rz_inner_surface = SurfaceRZFourier(
            mpol=mpol, ntor=ntor, nfp=nfp,
            stellsym=stellsym, range=range_surf,
            quadpoints_theta=self.theta, 
            quadpoints_phi=self.phi
        )
        for i in range(mpol + 1):
            for j in range(-ntor, ntor + 1):
                rz_inner_surface.set_rc(i, j, self.plasma_boundary.get_rc(i, j))
                rz_inner_surface.set_rs(i, j, self.plasma_boundary.get_rs(i, j))
                rz_inner_surface.set_zc(i, j, self.plasma_boundary.get_zc(i, j))
                rz_inner_surface.set_zs(i, j, self.plasma_boundary.get_zs(i, j))

        # extend via the normal vector
        rz_inner_surface.extend_via_projected_normal(self.phi, self.plasma_offset)
        self.rz_inner_surface = rz_inner_surface

    def _set_outer_rz_surface(self):
        """
            If the outer toroidal surface was not specified, this function
            is called and simply takes each (r, z) quadrature point on the 
            inner toroidal surface and shifts it by self.coil_offset at constant
            theta value. 
        """
        theta = self.rz_inner_surface.quadpoints_theta
        phi = self.rz_inner_surface.quadpoints_phi

        # make copy of plasma boundary
        mpol = self.rz_inner_surface.mpol
        ntor = self.rz_inner_surface.ntor
        nfp = self.rz_inner_surface.nfp
        stellsym = self.rz_inner_surface.stellsym
        range_surf = self.rz_inner_surface.range 
        quadpoints_theta = self.rz_inner_surface.quadpoints_theta 
        quadpoints_phi = self.rz_inner_surface.quadpoints_phi 
        rz_outer_surface = SurfaceRZFourier(
            mpol=mpol, ntor=ntor, nfp=nfp,
            stellsym=stellsym, range=range_surf,
            quadpoints_theta=quadpoints_theta, 
            quadpoints_phi=quadpoints_phi
        ) 
        for i in range(mpol + 1):
            for j in range(-ntor, ntor + 1):
                rz_outer_surface.set_rc(i, j, self.rz_inner_surface.get_rc(i, j))
                rz_outer_surface.set_rs(i, j, self.rz_inner_surface.get_rs(i, j))
                rz_outer_surface.set_zc(i, j, self.rz_inner_surface.get_zc(i, j))
                rz_outer_surface.set_zs(i, j, self.rz_inner_surface.get_zs(i, j))

        # extend via the normal vector
        rz_outer_surface.extend_via_projected_normal(self.phi, self.coil_offset)
        self.rz_outer_surface = rz_outer_surface

    def _plot_surfaces(self):
        """
            Simple plotting function for debugging the permanent
            magnet gridding procedure.
        """
        plt.figure()
        for i, ind in enumerate([0, 1, 29, 30, 31]):
            plt.subplot(1, 5, i + 1)
            plt.title(r'$\phi = ${0:.2f}'.format(2 * np.pi * self.phi[ind]))
            plt.scatter(self.r_plasma[ind, :], self.z_plasma[ind, :], label='Plasma surface')
            plt.scatter(self.r_inner[ind, :], self.z_inner[ind, :], label='Inner surface')
            plt.scatter(self.r_outer[ind, :], self.z_outer[ind, :], label='Outer surface')
            plt.scatter(
                np.array(self.final_RZ_grid[ind])[:, 0], 
                np.array(self.final_RZ_grid[ind])[:, 1], 
                label='Final grid'
            )
            plt.legend()
            plt.grid(True)
        plt.show()

    def _make_final_surface(self):
        """
            Takes the uniform RZ grid initialized earlier, and loops through
            and creates a final set of points which lie between the
            inner and outer toroidal surfaces corresponding to the permanent
            magnet surface. 

            For each toroidal cross-section:
            For each dipole location:
            1. Find nearest point from dipole to the inner surface
            2. Find nearest point from dipole to the outer surface
            3. Select nearest point that is closest to the dipole
            4. Get normal vector of this inner/outer surface point
            5. Draw ray from dipole location in the direction of this normal vector
            6. If closest point between inner surface and the ray is the 
               start of the ray, conclude point is outside the inner surface. 
            7. If closest point between outer surface and the ray is the
               start of the ray, conclude point is outside the outer surface. 
            8. If Step 4 was True but Step 5 was False, add the point to the final grid.
        """
        phi_inner = self.rz_inner_surface.quadpoints_phi
        normal_inner = self.rz_inner_surface.unitnormal()
        normal_outer = self.rz_outer_surface.unitnormal()
        Nray = 2000
        total_points = 0

        new_grids = []
        for i in range(len(phi_inner)):
            # Get (R, Z) locations of the points with respect to the magnetic axis
            Rpoint = np.ravel(self.RPhiZ[:, i, :, 0])
            Zpoint = np.ravel(self.RPhiZ[:, i, :, 2])

            new_grids_i = []
            for j in range(len(Rpoint)):
                # find nearest point on inner/outer toroidal surface
                dist_inner = (self.r_inner[i, :] - Rpoint[j]) ** 2 + (self.z_inner[i, :] - Zpoint[j]) ** 2
                inner_loc = dist_inner.argmin()
                dist_outer = (self.r_outer[i, :] - Rpoint[j]) ** 2 + (self.z_outer[i, :] - Zpoint[j]) ** 2
                outer_loc = dist_outer.argmin()
                if dist_inner[inner_loc] < dist_outer[outer_loc]:
                    nearest_loc = inner_loc
                    ray_direction = normal_inner[i, nearest_loc, :]
                else:
                    nearest_loc = outer_loc
                    ray_direction = normal_outer[i, nearest_loc, :]

                # rotate ray in (r, phi, z) coordinates and set phi component to zero
                # so that we keep everything in the same phi = constant cross-section
                rot_matrix = [[np.cos(phi_inner[i]), np.sin(phi_inner[i]), 0],
                              [-np.sin(phi_inner[i]), np.cos(phi_inner[i]), 0],
                              [0, 0, 1]]
                ray_direction = rot_matrix @ ray_direction 
                ray_direction = ray_direction[[0, 2]] / np.sqrt(ray_direction[0] ** 2 + ray_direction[2] ** 2)

                ray_equation = np.outer(
                    [Rpoint[j], 
                     Zpoint[j]], 
                    np.ones(Nray)
                ) + np.outer(ray_direction, np.linspace(0, 3, Nray))
                nearest_loc_inner = (
                    (self.r_inner[i, inner_loc] - ray_equation[0, :]
                     ) ** 2 + (
                        self.z_inner[i, inner_loc] - ray_equation[1, :]) ** 2
                ).argmin()

                # nearest distance from the inner surface to the ray should be just the original point
                if nearest_loc_inner != 0:
                    continue
                nearest_loc_outer = (
                    (self.r_outer[i, outer_loc] - ray_equation[0, :]
                     ) ** 2 + (
                        self.z_outer[i, outer_loc] - ray_equation[1, :]) ** 2
                ).argmin()

                # nearest distance from the outer surface to the ray should be NOT be the original point
                if nearest_loc_outer != 0:
                    total_points += 1
                    new_grids_i.append(np.array([Rpoint[j], Zpoint[j]]))
            new_grids.append(new_grids_i)
        self.ndipoles = total_points
        return new_grids

    def _compute_geometric_factor(self):
        """ 
            Computes the geometric factor in the expression for the
            total magnetic field as a sum over all the dipoles. Only
            needs to computed once, before the optimization. The 
            geometric factor at each plasma surface quadrature point
            is a sum of the contributions from each of the i dipoles.
            The dipole i contribution is,
            math::
                g_i(\phi, \theta) = \mu_0(\frac{3r_i\cdot N}{|r_i|^5}r_i - \frac{N}{|r_i|^3}) / 4 * pi
            and these factors are stacked together and reshaped in a matrix.
            The end result is that the matrix A in the ||Am - b||^2 part
            of the optimization is computed here.
        """
        phi = self.phi
        geo_factor = np.zeros((self.nphi, self.ntheta, self.ndipoles, 3))

        for i in range(self.nphi):
            phi_plasma = phi[i] 
            for j in range(self.ntheta):
                normal_plasma = self.plasma_unitnormal_cylindrical[i, j, :]
                R_plasma = np.array([self.r_plasma[i, j], phi_plasma, self.z_plasma[i, j]])
                running_tally = 0
                for k in range(self.nphi):
                    dipole_grid_r = np.ravel(np.array(self.final_RZ_grid[k])[:, 0])
                    dipole_grid_z = np.ravel(np.array(self.final_RZ_grid[k])[:, 0])
                    phi_dipole = phi[k] * np.ones(len(dipole_grid_r))
                    R_dipole = np.array([dipole_grid_r, phi_dipole, dipole_grid_z]).T
                    R_dist = self._cyl_dist(R_plasma, R_dipole) 
                    R_diff = R_dipole - np.outer(np.ones(R_dipole.shape[0]), R_plasma)
                    geo_factor[i, j, 
                               running_tally: running_tally + len(dipole_grid_r), :
                               ] = (3 * (np.array([R_diff @ normal_plasma,
                                                   R_diff @ normal_plasma,
                                                   R_diff @ normal_plasma
                                                   ]).T * R_diff).T / R_dist ** 5 - np.outer(
                                   normal_plasma, 1.0 / R_dist ** 3)
                    ).T
                    running_tally += len(dipole_grid_r)
        mu0 = 4 * np.pi * 1e-7
        dphi = (phi[1] - phi[0]) * 2 * np.pi
        dtheta = (self.theta[1] - self.theta[0]) * 2 * np.pi
        geo_factor = np.reshape(geo_factor, (self.nphi * self.ntheta, self.ndipoles * 3)) * mu0 / (4 * np.pi)
        self.A_obj = geo_factor * np.sqrt(dphi * dtheta)
        # Make histogram of the element values in A_obj
        A_hist = np.ravel(self.A_obj)
        plt.figure()
        plt.hist(A_hist, bins=100)
        plt.show()

    def _cyl_dist(self, plasma_vec, dipole_vec):
        """
            Computes cylindrical distances between a single point on the
            plasma boundary and a list of points corresponding to the dipoles
            that are on the same phi = constant cross-section.
        """
        radial_term = plasma_vec[0] ** 2 + dipole_vec[:, 0] ** 2 
        angular_term = - 2 * plasma_vec[0] * dipole_vec[:, 0] * np.cos(plasma_vec[1] - dipole_vec[:, 1])
        axial_term = (plasma_vec[2] - dipole_vec[:, 2]) ** 2
        return np.sqrt(radial_term + angular_term + axial_term)

    def _compute_cylindrical_norm(self, boundary_vec):
        """
            Computes cylindrical norm of any vector (in cylindrical coordinates)
            that is on the plasma boundary, with shape (nphi, ntheta, 3).
        """
        radial_term = boundary_vec[:, :, 0] ** 2
        axial_term = boundary_vec[:, :, 2] ** 2
        return np.sqrt(radial_term + axial_term)

    def _prox_l0(self, threshold):
        """Proximal operator for L0 regularization."""
        self.m_proxy = self.m * (np.abs(self.m) > threshold * self.nu)

    def _prox_l1(self, threshold):
        """Proximal operator for L1 regularization."""
        self.m_proxy = np.sign(self.m) * np.maximum(np.abs(self.m) - threshold * self.nu, 0)

    def _projection_L2_balls(self, x):
        """
        Project the vector x onto a series of L2 balls in R3.
        """
        N = len(x) // 3
        x_shaped = x.reshape(N, 3)
        denom = np.maximum(
            1,
            np.sqrt(x_shaped[:, 0] ** 2 + x_shaped[:, 1] ** 2 + x_shaped[:, 2] ** 2) / self.m_maxima
        )
        return np.divide(x_shaped, np.array([denom, denom, denom]).T).reshape(3 * N)

    def _phi_MwPGP(self, x, g):
        """
        phi(x_i, g_i) = g_i(x_i) is not on the L2 ball,
        otherwise set it to zero
        """
        N = len(x) // 3
        x_shaped = x.reshape(N, 3)
        check_active = np.isclose(
            x_shaped[:, 0] ** 2 + x_shaped[:, 1] ** 2 + x_shaped[:, 2] ** 2,
            self.m_maxima ** 2
        )
        # if triplet is in the active set (on the L2 unit ball)
        # then zero out those three indices
        g_shaped = np.copy(g).reshape((N, 3))
        if np.any(check_active):
            g_shaped[check_active, :] = 0.0
        return g_shaped.reshape(3 * N)

    def _beta_tilde(self, x, g, alpha):
        """
        beta_tilde(x_i, g_i, alpha) = 0_i if the ith triplet
        is not on the L2 ball, otherwise is equal to different
        values depending on the orientation of g.
        """
        N = len(x) // 3
        x_shaped = x.reshape((N, 3))
        check_active = np.isclose(
            x_shaped[:, 0] ** 2 + x_shaped[:, 1] ** 2 + x_shaped[:, 2] ** 2,
            self.m_maxima ** 2
        )
        # if triplet is NOT in the active set (on the L2 unit ball)
        # then zero out those three indices
        beta = np.zeros((N, 3))
        if np.any(~check_active):
            gradient_sphere = 2 * x_shaped
            denom_n = np.linalg.norm(gradient_sphere, axis=1, ord=2)
            n = np.divide(
                gradient_sphere,
                np.array([denom_n, denom_n, denom_n]).T
            )
            g_inds = np.ravel(np.where(~check_active))
            g_shaped = np.copy(g).reshape(N, 3)
            nTg = np.diag(n @ g_shaped.T)
            check_nTg_positive = (nTg > 0)
            beta_inds = np.logical_and(check_active, check_nTg_positive)
            beta_not_inds = np.logical_and(check_active, ~check_nTg_positive)
            N_not_inds = np.sum(np.array(beta_not_inds, dtype=int))
            if np.any(beta_inds):
                beta[beta_inds, :] = g_shaped[beta_inds, :]
            if np.any(beta_not_inds):
                beta[beta_not_inds, :] = self.g_reduced_gradient(
                    x_shaped[beta_not_inds, :].reshape(3 * N_not_inds),
                    alpha,
                    g_shaped[beta_not_inds, :].reshape(3 * N_not_inds)
                ).reshape(N_not_inds, 3)
        return beta.reshape(3 * N)

    def _g_reduced_gradient(self, x, alpha, g):
        """
        The reduced gradient of G is simply the
        gradient step in the L2-projected direction.
        """
        return (x - self._projection_L2_balls(x - alpha * g)) / alpha

    def _g_reduced_projected_gradient(self, x, alpha, g):
        """
        The reduced projected gradient of G is the
        gradient step in the L2-projected direction,
        subject to some modifications.
    """
        return self._phi_MwPGP(x, g) + self._beta_tilde(x, g, alpha)

    def _find_max_alphaf(self, x, p):
        """
        Solve a quadratic equation to determine the largest
        step size alphaf such that the entirety of x - alpha * p
        lives in the convex space defined by the intersection
        of the N L2 balls defined in R3, so that
        (x[0] - alpha * p[0]) ** 2 + (x[1] - alpha * p[1]) ** 2
        + (x[2] - alpha * p[2]) ** 2 <= 1.
        """
        N = len(x) // 3
        x_shaped = x.reshape((N, 3))
        p_shaped = p.reshape((N, 3))
        a = p_shaped[:, 0] ** 2 + p_shaped[:, 1] ** 2 + p_shaped[:, 2] ** 2
        b = - 2 * (
            x_shaped[:, 0] * p_shaped[:, 0] + x_shaped[:, 1] * p_shaped[:, 1] + x_shaped[:, 2] * p_shaped[:, 2]
        )
        c = (x_shaped[:, 0] ** 2 + x_shaped[:, 1] ** 2 + x_shaped[:, 2] ** 2) - self.m_maxima ** 2

        # Need to think harder about the line below
        p_nonzero_inds = np.ravel(np.where(a > 0))
        if len(p_nonzero_inds) != 0:
            a = a[p_nonzero_inds]
            b = b[p_nonzero_inds]
            c = c[p_nonzero_inds]
            if np.all((b ** 2 - 4 * a * c) >= 0.0):
                alphaf_plus = np.min((-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a))
            else:
                alphaf_plus = 1e100
            # alphaf_minus = np.min((-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a))
        else:
            alphaf_plus = 0.0
        return alphaf_plus

    def _MwPGP(self, alpha=None, delta=0.5, epsilon=1e-4, 
               relax_and_split=False, max_iter=500):
        """
        Run the MwPGP algorithm defined in: Bouchala, Jiří, et al.
        On the solution of convex QPQC problems with elliptic and
        other separable constraints with strong curvature.
        Applied Mathematics and Computation 247 (2014): 848-864.

        Here we solve the more specific optimization problem,
        min_x ||Ax - b||^2 + R(x)
        with a convex regularizer R(x) and subject
        to many spherical constraints.

        Note there is a typo in one of their definitions of
        alpha_cg, it should be alpha_cg = (A.T @ b.T) @ p / (p.T @ (A.T @ A) @ p)
        I think.
        Also, everywhere we need to replace A -> A.T @ A because they solve
        the optimization problem with x^T @ A @ x + ...
        """
        A = self.A_obj
        b = self.b_obj
        x0 = self.m

        # Need to put ATA everywhere since original MwPGP algorithm 
        # assumes that the problem looks like x^T A x
        ATA = A.T @ A
        print('ATA avg = ', np.mean(ATA))
        ATb = A.T @ b
        #if relax_and_split:
        #    ATA += np.eye(ATA.shape[0]) / self.nu
        #    ATb += 2.0 * self.m_proxy / self.nu
        alpha_max = 2.0 / np.linalg.norm(ATA, ord=2)
        if alpha is None:
            alpha = alpha_max  # - epsilon
        elif alpha > alpha_max or alpha < 0:
            print('Warning, invalid alpha value passed to MwPGP, '
                  'overwriting this value with the default.')
            alpha = alpha_max  # - epsilon
        print('alpha_MwPGP = ', alpha)
        g = ATA @ x0 - ATb
        p = self._phi_MwPGP(x0, g)
        g_alpha_P = self._g_reduced_projected_gradient(x0, alpha, g)
        norm_g_alpha_P = np.linalg.norm(g_alpha_P, ord=2)
        k = 0
        x_k = x0
        delta_fac = np.sqrt(2 * delta)
        # objective_history = []
        start = time.time()
        while k < max_iter:
            if k % max(int(max_iter / 100), 1) == 0:
                cost = np.linalg.norm(A @ x_k - b, ord=2) ** 2
                print(k, cost)

            if (delta_fac * norm_g_alpha_P <= np.linalg.norm(
                    self._phi_MwPGP(x_k, g))):
                ATAp = ATA @ p
                alpha_cg = g.T @ p / (p.T @ ATAp)
                alpha_f = self._find_max_alphaf(x_k, p)  # not quite working yet?
                if alpha_cg < alpha_f:
                    # Take a conjugate gradient step
                    x_k1 = x_k - alpha_cg * p
                    g = g - alpha_cg * ATAp
                    gamma = self._phi_MwPGP(x_k1, g).T @ ATAp / (p.T @ ATAp)
                    p = self._phi_MwPGP(x_k1, g) - gamma * p
                else:
                    # Take a mixed projected gradient step
                    x_k1 = self._projection_L2_balls((x_k - alpha_f * p) - alpha * (g - alpha_f * ATAp))
                    g = ATA @ x_k1 - ATb
                    p = self._phi_MwPGP(x_k1, g)
            else:
                # Take a projected gradient step
                x_k1 = self._projection_L2_balls(x_k - alpha * g)
                g = ATA @ x_k1 - ATb
                p = self._phi_MwPGP(x_k1, g)
            k = k + 1
            if np.max(abs(x_k - x_k1)) < epsilon:
                print('MwPGP finished early, at iteration ', k)
                break
            # objective_history.append(cost)
            x_k = x_k1
        end = time.time()
        cost = np.linalg.norm(A @ x_k - b, ord=2) ** 2  # / len(A) ** 2
        print('Error after MwPGP iterations = ', cost)
        print('Total time for MwPGP = ', end - start)
        self.m = x_k 

    def _optimize(self, m0=None, epsilon=1e-4, lambda_reg=None, nu=1e20,
                  regularizer=None, lambda_nonconvex=0,
                  max_iter_MwPGP=50, max_iter_RS=4, 
                  nonconvex_or_nonsmooth_terms=False):
        # Initialize initial guess for the dipole strengths
        if m0 is not None:
            if len(m0) == self.ndipoles * 3:
                self.m = m0
            else:
                raise ValueError(
                    'Initial dipole guess is incorrect shape --'
                    ' guess must be 1D with shape (ndipoles * 3).'
                )
        else:
            self.m = np.random.rand(self.ndipoles * 3) * np.max(self.m_maxima)

        # Initialize 'b' vector in ||Am - b||^2 part of the optimization,
        # corresponding to the normal component of the target fields.
        if self.B_plasma_surface.shape != (self.nphi, self.ntheta, 3):
            raise ValueError('Magnetic field surface data is incorrect shape.')
        Bs = self.B_plasma_surface
        self.b_obj = np.sum(Bs * self.plasma_boundary.unitnormal(), axis=2).reshape(self.nphi * self.ntheta)
        ave_Bn = np.mean(np.abs(self.b_obj))
        Bmag = np.linalg.norm(Bs, axis=-1, ord=2).reshape(self.nphi * self.ntheta)
        ave_BnB = np.mean(np.abs(self.b_obj) / Bmag)
        total_Bn = np.sum(np.abs(self.b_obj) ** 2)

        # Print out the bulk optimization paramaters 
        dipole_error = np.linalg.norm(self.A_obj @ self.m, ord=2) ** 2
        total_error = np.linalg.norm(self.A_obj @ self.m - self.b_obj, ord=2) ** 2
        print('Number of phi quadrature points on plasma surface = ', self.nphi)
        print('Number of theta quadrature points on plasma surface = ', self.ntheta)
        print('<B * n> without the permanent magnets = {0:.5f}'.format(ave_Bn)) 
        print('<B * n / |B| > without the permanent magnets = {0:.5f}'.format(ave_BnB)) 
        print(r'$|b|_2^2 = |B * n|_2^2$ without the permanent magnets = {0:.5f}'.format(total_Bn))
        print(r'Initial $|Am_0|_2^2 = |B_M * n|_2^2$ without the coils/plasma = {0:.7f}'.format(dipole_error))
        print('Number of dipoles = ', self.ndipoles)
        print('Inner toroidal surface offset from plasma surface = ', self.plasma_offset)
        print('Outer toroidal surface offset from inner toroidal surface = ', self.coil_offset)
        print('Maximum dipole moment = ', np.max(self.m_maxima))
        print('Shape of A matrix = ', self.A_obj.shape)
        print('Shape of b vector = ', self.b_obj.shape)
        print('Initial error on plasma surface = {0:.5f}'.format(total_error))
        # Begin optimization
        if nonconvex_or_nonsmooth_terms:
            # Relax-and-split algorithm
            self.nu = nu
            self.m_proxy = self.m
            for i in range(max_iter_RS):
                # update self.m
                self._MwPGP(epsilon=epsilon, max_iter=max_iter_MwPGP, relax_and_split=True)
                # update self.m_proxy
                self._prox_l0(lambda_nonconvex)
        else:
            self._MwPGP(delta=1e100, epsilon=epsilon, max_iter=max_iter_MwPGP)
        ave_Bn = np.mean(np.abs(self.A_obj @ self.m - self.b_obj))
        ave_BnB = np.mean(np.abs((self.A_obj @ self.m - self.b_obj)) / Bmag)  # using original Bmag without PMs
        print(self.m, self.m_maxima, np.max(self.m))
        print('<B * n> with the optimized permanent magnets = {0:.5f}'.format(ave_Bn)) 
        print('<B * n / |B| > without the permanent magnets = {0:.5f}'.format(ave_BnB)) 

