import logging

from matplotlib import pyplot as plt
import warnings
import numpy as np
from .surfacerzfourier import SurfaceRZFourier

logger = logging.getLogger(__name__)


class PermanentMagnetOptimizer():
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
            nphi: Number of grid points :math:`\phi_j` in the toroidal angle :math:`\phi`.
            nr: Number of grid points :math:`r_j` in the radial coordinate :math:`r`.
            nz: Number of grid points :math:`z_j` in the axial angle :math:`z`.
            rz_inner_surface: SurfaceRZFourier object representing 
                              the inner toroidal surface of the volume.
            rz_outer_surface: SurfaceRZFourier object representing 
                              the outer toroidal surface of the volume.
    """

    def __init__(
        self, plasma_boundary, rz_inner_surface=None, 
        rz_outer_surface=None, plasma_offset=None, coil_offset=None,
    ):
        self.plasma_offset = plasma_offset
        self.coil_offset = coil_offset
        if not isinstance(plasma_boundary, SurfaceRZFourier):
            raise ValueError(
                'Plasma boundary must be specified as SurfaceRZFourier class.'
            )
        else:
            self.plasma_boundary = plasma_boundary

        # If the inner surface is not specified, make default surface.
        if rz_inner_surface is None:
            print(
                "Inner toroidal surface not specified, defaulting to "
                "the plasma boundary shape."
            )
            if plasma_offset is None:
                print(
                    "Volume initialized without a given offset to use "
                    "for defining the inner surface. Defaulting to 10 cm."
                )
                self.plasma_offset = 0.1

            self._set_inner_rz_surface()
        else:
            self.rz_inner_surface = rz_inner_surface

        # If the outer surface is not specified, make default surface. 
        if rz_outer_surface is None:
            print(
                "Outer toroidal surface not specified, defaulting to "
                "using the locus of points whose closest distance to "
                "the inner surface in their respective poloidal plane "
                "is equal to a given radial extent."
            )
            if coil_offset is None:
                print(
                    "Volume initialized without a given offset to use "
                    "for defining the outer surface. Defaulting to 10 cm."
                )
                self.coil_offset = 0.1
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

        xyz_outer = self.rz_outer_surface.gamma()
        r_outer = np.sqrt(xyz_outer[:, :, 0] ** 2 + xyz_outer[:, :, 1] ** 2)
        z_outer = xyz_outer[:, :, 2]
        r_max = np.max(r_outer)
        r_min = np.min(r_outer)
        z_max = np.max(z_outer)
        z_min = np.min(z_outer)

        # Initialize uniform grid of curved, square bricks
        Delta_r = 0.02
        Nr = int((r_max - r_min) / Delta_r)
        self.Nr = Nr
        Delta_z = Delta_r
        Nz = int((z_max - z_min) / Delta_z)
        self.Nz = Nz
        phi = self.rz_outer_surface.quadpoints_phi
        Nphi = len(phi)
        print('Largest dipole size is = {0:.2f}'.format(
            (
                r_max - self.plasma_boundary.get_rc(0, 0)
            ) * (phi[1] - phi[0]) * 2 * np.pi
        )
        )
        R = np.linspace(r_min, r_max, Nr)
        Phi = self.rz_outer_surface.quadpoints_phi
        Z = np.linspace(z_min, z_max, Nz)

        # Make 3D mesh
        R, Phi, Z = np.meshgrid(R, Phi, Z, indexing='ij')
        self.RPhiZ = np.transpose(np.array([R, Phi, Z]), [1, 2, 3, 0])

        # Have the uniform grid, now need to loop through and eliminate cells. 
        self.final_RZ_grid = self._make_final_surface()

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
        theta = self.plasma_boundary.quadpoints_theta
        phi = self.plasma_boundary.quadpoints_phi
        npol = len(theta)

        # make copy of plasma boundary
        mpol = self.plasma_boundary.mpol
        ntor = self.plasma_boundary.ntor
        nfp = self.plasma_boundary.nfp
        stellsym = self.plasma_boundary.stellsym
        range_surf = self.plasma_boundary.range 
        theta_min = np.min(theta)
        theta_max = np.max(theta)
        dtheta = theta[1] - theta[0]
        phi_min = np.min(phi)
        phi_max = np.max(phi)
        quadpoints_theta = theta
        quadpoints_phi = phi
        rz_inner_surface = SurfaceRZFourier(
            mpol=mpol, ntor=ntor, nfp=nfp,
            stellsym=stellsym, range=range_surf,
            quadpoints_theta=quadpoints_theta, 
            quadpoints_phi=quadpoints_phi
        )
        for i in range(mpol + 1):
            for j in range(-ntor, ntor + 1):
                rz_inner_surface.set_rc(i, j, self.plasma_boundary.get_rc(i, j))
                rz_inner_surface.set_rs(i, j, self.plasma_boundary.get_rs(i, j))
                rz_inner_surface.set_zc(i, j, self.plasma_boundary.get_zc(i, j))
                rz_inner_surface.set_zs(i, j, self.plasma_boundary.get_zs(i, j))

        # extend via the normal vector
        rz_inner_surface.extend_via_normal(self.plasma_offset)
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
        npol = len(theta)
        ntor = len(phi)

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
        rz_outer_surface.extend_via_normal(self.coil_offset)
        self.rz_outer_surface = rz_outer_surface

    def _plot_surfaces(self):
        """
            Simple plotting function for debugging the permanent
            magnet gridding procedure.
        """
        plt.figure()
        for i, ind in enumerate([0, 10, 30]):
            plt.subplot(1, 3, i + 1)
            plt.title(r'$\phi = ${0:.2f}'.format(2 * np.pi * self.plasma_boundary.quadpoints_phi[ind]))
            xyz_plasma = self.plasma_boundary.gamma()
            plt.scatter(np.sqrt(xyz_plasma[ind, :, 0] ** 2 + xyz_plasma[ind, :, 1] ** 2), xyz_plasma[ind, :, 2], label='Plasma surface')
            xyz_plasma = self.rz_inner_surface.gamma()
            plt.scatter(np.sqrt(xyz_plasma[ind, :, 0] ** 2 + xyz_plasma[ind, :, 1] ** 2), xyz_plasma[ind, :, 2], label='Inner surface')
            xyz_plasma = self.rz_outer_surface.gamma()
            plt.scatter(np.sqrt(xyz_plasma[ind, :, 0] ** 2 + xyz_plasma[ind, :, 1] ** 2), xyz_plasma[ind, :, 2], label='Outer surface')
            plt.scatter(np.array(self.final_RZ_grid[ind])[:, 0], np.array(self.final_RZ_grid[ind])[:, 1], label='Final grid')
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
            2. Get normal vector of this inner surface point
            3. Draw ray from dipole location in the direction of this normal vector
            4. If closest point between inner surface and the ray is the 
               start of the ray, conclude point is outside the inner surface. 
            5. If closest point between outer surface and the ray is the
               start of the ray, conclude point is outside the outer surface. 
            6. If Step 4 was True but Step 5 was False, add the point to the final grid.
        """
        phi_inner = self.rz_inner_surface.quadpoints_phi
        xyz_inner = self.rz_inner_surface.gamma()
        r_inner = np.sqrt(xyz_inner[:, :, 0] ** 2 + xyz_inner[:, :, 1] ** 2)
        z_inner = xyz_inner[:, :, 2]
        xyz_outer = self.rz_outer_surface.gamma()
        r_outer = np.sqrt(xyz_outer[:, :, 0] ** 2 + xyz_outer[:, :, 1] ** 2)
        z_outer = xyz_outer[:, :, 2]
        normal_inner = self.rz_inner_surface.unitnormal()
        Nray = 2000
        total_points = 0

        new_grids = []
        for i in range(len(phi_inner)):
            # Get (R, Z) locations of the points with respect to the magnetic axis
            Rpoint = np.ravel(self.RPhiZ[:, i, :, 0])
            Zpoint = np.ravel(self.RPhiZ[:, i, :, 2])

            # find nearest point on inner toroidal surface
            new_grids_i = []
            for j in range(len(Rpoint)):
                nearest_loc = ((r_inner[i, :] - Rpoint[j]) ** 2 + (z_inner[i, :] - Zpoint[j]) ** 2).argmin()
                ray_direction = normal_inner[i, nearest_loc, :]

                # rotate ray in (r, phi, z) coordinates and set phi component to zero
                # so that we keep everything in the same phi = constant cross-section
                rot_matrix = [[np.cos(phi_inner[i]), np.sin(phi_inner[i]), 0],
                              [-np.sin(phi_inner[i]), np.cos(phi_inner[i]), 0],
                              [0, 0, 1]]
                ray_direction = rot_matrix @ ray_direction 
                ray_direction = ray_direction[[0, 2]]

                ray_equation = np.outer(
                    [Rpoint[j], 
                     Zpoint[j]], 
                    np.ones(Nray)
                ) + np.outer(ray_direction, np.linspace(0, 3, Nray))
                nearest_loc_inner = ((r_inner[i, nearest_loc] - ray_equation[0, :]) ** 2 + (z_inner[i, nearest_loc] - ray_equation[1, :]) ** 2).argmin()

                # nearest distance from the inner surface to the ray should be just the original point
                if nearest_loc_inner != 0:
                    continue
                nearest_loc_outer = ((r_outer[i, nearest_loc] - ray_equation[0, :]) ** 2 + (z_outer[i, nearest_loc] - ray_equation[1, :]) ** 2).argmin()

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
            needs to computed once, before the optimization.
            math::
                g(\phi, \theta) = \mu_0(\frac{3r_i\cdot N}{|r_i|^5}r_i - \frac{N}{|r_i|^3}) / 4 * pi
        """
        normal_plasma_boundary = self.plasma_boundary.unitnormal()

        phi = self.plasma_boundary.quadpoints_phi
        nphi = len(phi)
        ntheta = len(self.plasma_boundary.quadpoints_theta)
        geo_factor = np.zeros((nphi, ntheta, self.ndipoles, 3))
        xyz_plasma = self.plasma_boundary.gamma()

        for i in range(nphi):
            # rotate normal vector to (r, phi, z) coordinates
            rot_matrix = [[np.cos(phi[i]), np.sin(phi[i]), 0],
                          [-np.sin(phi[i]), np.cos(phi[i]), 0],
                          [0, 0, 1]]
            phi_plasma = phi[i] 
            for j in range(ntheta):
                normal_plasma = rot_matrix @ normal_plasma_boundary[i, j, :]
                r_plasma = np.sqrt(xyz_plasma[i, j, 0] ** 2 + xyz_plasma[i, j, 1] ** 2)
                z_plasma = xyz_plasma[i, j, 2]
                R_plasma = np.array([r_plasma, phi_plasma, z_plasma])
                for k in range(nphi):
                    dipole_grid_r = np.ravel(np.array(self.final_RZ_grid[k])[:, 0])
                    dipole_grid_z = np.ravel(np.array(self.final_RZ_grid[k])[:, 0])
                    phi_dipole = phi[k] * np.ones(len(dipole_grid_r))
                    R_dipole = np.array([dipole_grid_r, phi_dipole, dipole_grid_z]).T
                    R_dist = self._cyl_dist(R_plasma, R_dipole) 
                    R_diff = R_dipole - np.outer(np.ones(R_dipole.shape[0]), R_plasma)
                    geo_factor[i, j, 
                               k * len(dipole_grid_r): (k + 1) * len(dipole_grid_r), :
                               ] = (3 * (np.array([R_diff @ normal_plasma,
                                                   R_diff @ normal_plasma,
                                                   R_diff @ normal_plasma
                                                   ]).T * R_diff).T / R_dist ** 5 - np.outer(
                                   normal_plasma, 1.0 / R_dist ** 3)
                    ).T
        self.geo_factor = geo_factor

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
