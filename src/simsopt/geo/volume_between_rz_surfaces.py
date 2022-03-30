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
                self.plasma_offset = 0.4

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
        # check no intersections between the surfaces
        self._check_no_intersections()

        # Define the set of radial coordinates r_inner(theta, phi), r_outer(theta, phi)
        # and similarly for the z-component. 
        self.quadpoints_r = np.hstack(
                (self.rz_inner_surface.quadpoints_r, 
                 self.rz_outer_surface.quadpoints_r)
            )
        self.quadpoints_z = np.hstack(
                (self.rz_inner_surface.quadpoints_z, 
                 self.rz_outer_surface.quadpoints_z)
            )

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
        ntor = len(phi)
        # r_inner = self.plasma_boundary.quadpoints_r 
        # z_inner = self.plasma_boundary.quadpoints_z

        # make copy of plasma boundary
        mpol = self.plasma_boundary.mpol
        ntor = self.plasma_boundary.ntor
        nfp = self.plasma_boundary.nfp
        stellsym = self.plasma_boundary.stellsym
        range_surf = self.plasma_boundary.range 
        quadpoints_theta = self.plasma_boundary.quadpoints_theta 
        quadpoints_phi = self.plasma_boundary.quadpoints_phi 
        rz_inner_surface = SurfaceRZFourier(
            mpol=mpol, ntor=ntor, nfp=nfp,
            stellsym=stellsym, range=range_surf,
            quadpoints_theta=quadpoints_theta, 
            quadpoints_phi=quadpoints_phi
        ) 

        for j in range(ntor):
            rz_inner_surface.set_rc(1, j, rz_inner_surface.get_rc(1, j) + self.plasma_offset)
            rz_inner_surface.set_zs(1, j, rz_inner_surface.get_zs(1, j) + self.plasma_offset)
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
        # r_outer = self.rz_inner_surface.quadpoints_r 
        # z_outer = self.rz_inner_surface.quadpoints_z

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

        for j in range(ntor):
            rz_outer_surface.set_rc(1, j, 
                rz_outer_surface.get_rc(1, j) + self.coil_offset + self.plasma_offset
            )
            rz_outer_surface.set_zs(1, j, 
                rz_outer_surface.get_zs(1, j) + self.coil_offset + self.plasma_offset
            )
        self.rz_outer_surface = rz_outer_surface

    def _check_no_intersections(self):
        """
            At each r(theta, phi), z(theta, phi) quadrature point, check
            that the distance from the center of the inner surface
            to the outer surface is 
            larger than the same distance to the inner surface.
        """
        npol = len(self.rz_inner_surface.quadpoints_theta)
        ntor = len(self.rz_inner_surface.quadpoints_phi)
        center_inner_surface_r = self.rz_inner_surface.get_rc(0, 0)
        if not self.rz_inner_surface.stellsym:
            center_inner_surface_z = self.rz_inner_surface.get_zc(0, 0)
        else:
            center_inner_surface_z = 0.0
        for i in range(npol):
            for j in range(ntor):
                #print(
                #    i, j,
                #    self.plasma_boundary.quadpoints_r[i, j],
                #    self.rz_inner_surface.quadpoints_r[i, j],
                #    self.rz_outer_surface.quadpoints_r[i, j],
                #    self.plasma_boundary.quadpoints_z[i, j],
                #    self.rz_inner_surface.quadpoints_z[i, j],
                #    self.rz_outer_surface.quadpoints_z[i, j],
                #)
                rz_dist_inner = (
                        (self.rz_inner_surface.quadpoints_r[i, j] - center_inner_surface_r) ** 2 + 
                        (self.rz_inner_surface.quadpoints_z[i, j] - center_inner_surface_z) ** 2
                    )
                rz_dist_outer = (
                        (self.rz_outer_surface.quadpoints_r[i, j] - center_inner_surface_r) ** 2 + 
                        (self.rz_outer_surface.quadpoints_z[i, j] - center_inner_surface_z) ** 2
                    )
                #if rz_dist_inner >= rz_dist_outer:
                #    raise ValueError(
                #        "The distance from the origin to the inner surface is "
                #        "larger than that to the outer surface at one or more "
                #        "of the quadrature points!"
                #    )

    def _plot_surfaces(self):
        print(2 * np.pi * self.plasma_boundary.quadpoints_theta)
        print(2 * np.pi * self.plasma_boundary.quadpoints_phi)
        plt.figure()
        for i, ind in enumerate([0, 10, 30]):
            plt.subplot(1, 3, i + 1)
            plt.title(r'$\phi = $' + str(2 * np.pi * self.plasma_boundary.quadpoints_phi[ind]))
            plt.scatter(
                self.plasma_boundary.quadpoints_r[:, ind], 
                self.plasma_boundary.quadpoints_z[:, ind],
                label='Plasma surface'
            )
            plt.scatter(
                self.rz_inner_surface.quadpoints_r[:, ind], 
                self.rz_inner_surface.quadpoints_z[:, ind],
                label='Inner PM surface'
            )
            plt.scatter(
                self.rz_outer_surface.quadpoints_r[:, ind], 
                self.rz_outer_surface.quadpoints_z[:, ind],
                label='Outer PM surface'
            )
            plt.legend()
            plt.grid(True)
        plt.show()
