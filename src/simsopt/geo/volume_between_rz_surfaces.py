import logging

from matplotlib import pyplot as plt
from copy import deepcopy
import warnings
import numpy as np
import f90nml
from .surfacerzfourier import SurfaceRZFourier

logger = logging.getLogger(__name__)


class volume_between_two_RZsurfaces():
    r"""
    ``volume_between_two_RZsurfaces`` is a volume (a grid) between two toroidal
    surfaces that is represented in cylindrical coordinates.

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

        # Define the set of radial coordinates r_inner(theta, phi), r_outer(theta, phi)
        # and similarly for the z-component. 
        #self.quadpoints_r = np.hstack(
        #        (self.rz_inner_surface.quadpoints_r, 
        #         self.rz_outer_surface.quadpoints_r)
        #    )
        #self.quadpoints_z = np.hstack(
        #        (self.rz_inner_surface.quadpoints_z, 
        #         self.rz_outer_surface.quadpoints_z)
        #    )

        # Define (r, phi, z) mesh between the two surfaces by initializing
        # cylindrical grid, looping through the cells, and eliminating
        # any cells not between the two surfaces.
        xyz_outer = self.rz_outer_surface.gamma()
        r_outer = np.sqrt(xyz_outer[:, :, 0] ** 2 + xyz_outer[:, :, 1] ** 2)
        z_outer = xyz_outer[:, :, 2]
        r_max = np.max(r_outer)
        r_min = np.min(r_outer)
        z_max = np.max(z_outer)
        z_min = np.min(z_outer)
        phi_max = np.max(self.rz_outer_surface.quadpoints_phi)
        phi_min = np.min(self.rz_outer_surface.quadpoints_phi)
        Delta_r = 0.02
        Nr = int((r_max - r_min) / Delta_r)
        self.Nr = Nr
        Delta_z = Delta_r
        Nz = int((z_max - z_min) / Delta_z)
        self.Nz = Nz
        Nphi = 40
        print(Nr, Nphi, Nz)
        R = np.linspace(r_min, r_max, Nr)
        Phi = self.rz_outer_surface.quadpoints_phi  # np.linspace(phi_min, phi_max, Nphi)
        Z = np.linspace(z_min, z_max, Nz)
        R, Phi, Z = np.meshgrid(R, Phi, Z, indexing='ij')
        self.R = R
        self.Z = Z
        self.Phi = Phi
        self.RPhiZ = np.transpose(np.array([R, Phi, Z]), [1, 2, 3, 0])  # np.vstack((np.vstack((np.ravel(R), np.ravel(Phi))), np.ravel(Z)))
        print(self.RPhiZ.shape, Phi.shape)
        # Have the uniform grid, now need to loop through and eliminate cells. 
        self.fixed_grid = self._check_points_between_surfaces()
        #self.fixed_grid = np.reshape(self.RPhiZ, (32 * Nz * Nr, 3))
        print(np.shape(self.fixed_grid[0]))
        # optionally plot the plasma boundary + inner/outer surfaces
        # in a number of toroidal slices
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
        quadpoints_theta = theta  # np.linspace(theta_min, theta_max + dtheta, npol * 2)
        quadpoints_phi = phi  # np.linspace(phi_min, phi_max, len(phi) * 2)
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
            plt.scatter(np.array(self.fixed_grid[ind])[:, 0], np.array(self.fixed_grid[ind])[:, 1], label='Final grid')
            plt.legend()
            plt.grid(True)
            #plt.scatter(self.RPhiZ[:, ind, :, 0], self.RPhiZ[:, ind, :, 2])
            #ntor = self.rz_inner_surface.ntor
            #ntor_vals = np.arange(-ntor, ntor + 1)
            #R_axis = 0
            #Z_axis = 0
            #for k in range(-ntor, ntor + 1):
            #    R_axis += self.plasma_boundary.get_rc(0, k) * np.cos(k * self.plasma_boundary.quadpoints_phi[ind]) 
            #    Z_axis += self.plasma_boundary.get_zs(0, k) * np.sin( - k * self.plasma_boundary.quadpoints_phi[ind]) 
            #plt.scatter(R_axis, Z_axis, color='k')
        plt.show()

    def _check_points_between_surfaces(self):
        theta_inner = self.rz_inner_surface.quadpoints_theta
        phi_inner = self.rz_inner_surface.quadpoints_phi
        xyz_inner = self.rz_inner_surface.gamma()
        r_inner = np.sqrt(xyz_inner[:, :, 0] ** 2 + xyz_inner[:, :, 1] ** 2)
        z_inner = xyz_inner[:, :, 2]
        theta_outer = self.rz_outer_surface.quadpoints_theta
        phi_outer = self.rz_outer_surface.quadpoints_phi
        xyz_outer = self.rz_outer_surface.gamma()
        r_outer = np.sqrt(xyz_outer[:, :, 0] ** 2 + xyz_outer[:, :, 1] ** 2)
        z_outer = xyz_outer[:, :, 2]
        normal_inner = self.rz_inner_surface.unitnormal()
        print(np.shape(normal_inner))
        Nray = 2000
        # For each toroidal cross-section, find theta of the location
        # find nearest theta value on inner and outer surfaces, and compare distances
        good_inds = np.zeros((len(phi_inner), self.Nr * self.Nz), dtype=bool)
        new_grids = []
        for i in range(len(phi_inner)):
            # Get (R, Z) locations of the points with respect to the magnetic axis
            Rpoint = np.ravel(self.RPhiZ[:, i, :, 0])  # - self.rz_inner_surface.get_rc(0, 0))
            Zpoint = np.ravel(self.RPhiZ[:, i, :, 2])  # - self.rz_inner_surface.get_zs(1, 0))
            # find nearest point on inner toroidal surface
            new_grids_i = []
            for j in range(len(Rpoint)):
                nearest_loc = ((r_inner[i, :] - Rpoint[j]) ** 2 + (z_inner[i, :] - Zpoint[j]) ** 2).argmin()
                ray_direction = normal_inner[i, nearest_loc, :]
                # rotate ray in (r, phi, z) coordinates and set phi component to zero
                rot_matrix = [[np.cos(phi_inner[i]), np.sin(phi_inner[i]), 0],
                              [-np.sin(phi_inner[i]), np.cos(phi_inner[i]), 0],
                              [0, 0, 1]]
                ray_direction = rot_matrix @ ray_direction 
                ray_direction = ray_direction[[0, 2]]
                # should be dimension (2, 1000)
                ray_equation = np.outer([Rpoint[j], Zpoint[j]], np.ones(Nray)) + np.outer(ray_direction, np.linspace(0, 3, Nray))
                nearest_loc_inner = ((r_inner[i, nearest_loc] - ray_equation[0, :]) ** 2 + (z_inner[i, nearest_loc] - ray_equation[1, :]) ** 2).argmin()
                # nearest distance from the inner surface to the ray should be just the original point
                if nearest_loc_inner != 0:
                    continue
                nearest_loc_outer = ((r_outer[i, nearest_loc] - ray_equation[0, :]) ** 2 + (z_outer[i, nearest_loc] - ray_equation[1, :]) ** 2).argmin()
                # nearest distance from the outer surface to the ray should be NOT be the original point
                if nearest_loc_outer != 0:
                    new_grids_i.append(np.array([Rpoint[j], Zpoint[j]]))
            new_grids.append(new_grids_i)
            #theta_point = np.arctan2(Zpoint, Rpoint - self.rz_inner_surface.get_rc(0, 0))
            # Normalize thetas to [0, 1]
            #print(np.max(theta_point), np.max(theta_inner))
            #print(np.max(theta_point - np.min(theta_point)), np.max(theta_inner))
            #theta_point = (theta_point - np.min(theta_point)) / (2 * np.pi)
            #print(np.max(theta_point), np.max(theta_inner))
            #for j, theta in enumerate(theta_point):
            #    theta_ind = np.abs(theta_point[j] - theta_inner).argmin()
            # print(theta_point[j], theta_inner[theta_ind])
           #     dist_inner = np.sqrt(
           #         (r_inner[i, theta_ind] - Rpoint[j]) ** 2
           #         + (z_inner[i, theta_ind] - Zpoint[j]) ** 2
           #     )
           #     dist_outer = np.sqrt(
           #         (r_outer[i, theta_ind] - Rpoint[j]) ** 2
           #         + (z_outer[i, theta_ind] - Zpoint[j]) ** 2
           #     )
           #     dist_surfaces = np.sqrt(
           ##         (r_inner[i, theta_ind] - r_outer[i, theta_ind]) ** 2
           #         + (z_inner[i, theta_ind] - z_outer[i, theta_ind]) ** 2
           #     )
           #     if dist_inner <= dist_surfaces and dist_outer <= dist_surfaces:
           #         print(i, j, dist_inner, dist_outer, dist_surfaces)
           #         print(r_inner[i, theta_ind], r_outer[i, theta_ind], Rpoint[j])
           #         good_inds[i, j] = True
           # good_inds_i = np.reshape(good_inds[i, :], (self.Nr, self.Nz))
           # new_grid_i = self.RPhiZ[:, i, :, :]
           # new_grids.append(new_grid_i[good_inds_i, :])
        return new_grids
