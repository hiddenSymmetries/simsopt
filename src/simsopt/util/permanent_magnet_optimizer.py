import logging
from matplotlib import pyplot as plt
import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from scipy.sparse import csr_matrix
import simsoptpp as sopp

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
            dr:               Radial and axial grid spacing in the permanent magnet manifold.
    """

    def __init__(
        self, plasma_boundary, rz_inner_surface=None, 
        rz_outer_surface=None, plasma_offset=0.1, 
        coil_offset=0.1, B_plasma_surface=None, dr=0.1,
    ):
        if plasma_offset <= 0 or coil_offset <= 0:
            raise ValueError('permanent magnets must be offset from the plasma')

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

        # If dr <= 0 raise error
        if dr <= 0:
            raise ValueError('dr grid spacing must be > 0')

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
        if not isinstance(self.rz_inner_surface, SurfaceRZFourier):
            raise ValueError("Inner surface is not SurfaceRZFourier object.")

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
        if not isinstance(self.rz_outer_surface, SurfaceRZFourier):
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
        self.phi_plasma = np.arctan2(xyz_plasma[:, :, 1], xyz_plasma[:, :, 0])
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
        Delta_r = dr 
        Nr = int((r_max - r_min) / Delta_r)
        self.Nr = Nr
        Delta_z = Delta_r
        Nz = int((z_max - z_min) / Delta_z)
        self.Nz = Nz
        phi = 2 * np.pi * np.copy(self.rz_outer_surface.quadpoints_phi)
        print('Largest possible dipole dimension is = {0:.2f}'.format(
            (
                r_max - self.plasma_boundary.get_rc(0, 0)
            ) * (phi[1] - phi[0])
        )
        )
        print('dR = {0:.2f}'.format(Delta_r))
        print('dZ = {0:.2f}'.format(Delta_z))
        norm = self.plasma_boundary.unitnormal()
        norms = np.zeros(norm.shape)
        for i in range(self.nphi):
            rot_matrix = [[np.cos(phi[i]), np.sin(phi[i]), 0],
                          [-np.sin(phi[i]), np.cos(phi[i]), 0],
                          [0, 0, 1]]
            for j in range(self.ntheta):
                norms[i, j, :] = rot_matrix @ norm[i, j, :]
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
        B_max = 1.4
        mu0 = 4 * np.pi * 1e-7
        dipole_grid_r = np.zeros(self.ndipoles)
        dipole_grid_phi = np.zeros(self.ndipoles)
        dipole_grid_z = np.zeros(self.ndipoles)
        running_tally = 0
        for i in range(self.nphi):
            radii = np.ravel(np.array(self.final_RZ_grid[i])[:, 0])
            z_coords = np.ravel(np.array(self.final_RZ_grid[i])[:, 1])
            len_radii = len(radii)
            dipole_grid_r[running_tally:running_tally + len_radii] = radii
            dipole_grid_phi[running_tally:running_tally + len_radii] = phi[i]
            dipole_grid_z[running_tally:running_tally + len_radii] = z_coords
            running_tally += len_radii
        self.dipole_grid = np.array([dipole_grid_r, dipole_grid_phi, dipole_grid_z]).T
        cell_vol = dipole_grid_r * Delta_r * Delta_z * (phi[1] - phi[0])

        # FAMUS paper says m_max = B_r / (mu0 * cell_vol) but it 
        # should be m_max = B_r * cell_vol / mu0  (just from units)
        self.m_maxima = B_max * cell_vol / mu0

        # Compute the geometric factor for the A matrix in the optimization
        self._compute_geometric_factor()

        # optionally plot the plasma boundary + inner/outer surfaces
        self._plot_surfaces()

    def _plot_final_dipoles(self):
        plt.figure()
        ax = plt.axes(projection="3d")
        colors = []
        dipoles = dipoles.reshape(pm_opt.ndipoles, 3)
        for i in range(pm_opt.ndipoles):
            colors.append(np.sqrt(dipoles[i, 0] ** 2 + dipoles[i, 1] ** 2 + dipoles[i, 2] ** 2))
        sax = ax.scatter(dipole_grid[:, 0], dipole_grid[:, 1], dipole_grid[:, 2], c=colors)
        plt.colorbar(sax)
        plt.axis('off')
        plt.grid(None)

        plt.figure(figsize=(14, 14))
        for i, ind in enumerate([0, 5, 20, 31]):
            plt.subplot(2, 2, i + 1)
            plt.title(r'$\phi = ${0:.2f}$^o$'.format(360 * pm_opt.phi[ind]))
            r_plasma = np.hstack((pm_opt.r_plasma[ind, :], pm_opt.r_plasma[ind, 0]))
            z_plasma = np.hstack((pm_opt.z_plasma[ind, :], pm_opt.z_plasma[ind, 0]))
            r_inner = np.hstack((pm_opt.r_inner[ind, :], pm_opt.r_inner[ind, 0]))
            z_inner = np.hstack((pm_opt.z_inner[ind, :], pm_opt.z_inner[ind, 0]))
            r_outer = np.hstack((pm_opt.r_outer[ind, :], pm_opt.r_outer[ind, 0]))
            z_outer = np.hstack((pm_opt.z_outer[ind, :], pm_opt.z_outer[ind, 0]))

            plt.plot(r_plasma, z_plasma, label='Plasma surface', linewidth=2)
            plt.plot(r_inner, z_inner, label='Inner surface', linewidth=2)
            plt.plot(r_outer, z_outer, label='Outer surface', linewidth=2)

            running_tally = 0
            for k in range(ind):
                running_tally += len(np.array(pm_opt.final_RZ_grid[k])[:, 0])
            colors = []
            dipoles_i = dipoles[running_tally:running_tally + len(np.array(pm_opt.final_RZ_grid[ind])[:, 0]), :]
            for j in range(len(dipoles_i)):
                colors.append(np.sqrt(dipoles_i[j, 0] ** 2 + dipoles_i[j, 1] ** 2 + dipoles_i[j, 2] ** 2))

            sax = plt.scatter(
                np.array(pm_opt.final_RZ_grid[ind])[:, 0],
                np.array(pm_opt.final_RZ_grid[ind])[:, 1],
                c=colors,
                label='PMs'
            )
            plt.colorbar(sax)
            print(i, ind, len(np.array(pm_opt.final_RZ_grid[ind])[:, 0]), dipoles_i.shape)
            plt.quiver(
                np.array(pm_opt.final_RZ_grid[ind])[:, 0],
                np.array(pm_opt.final_RZ_grid[ind])[:, 1],
                dipoles_i[:, 0],
                dipoles_i[:, 2],
            )
            plt.xlabel('R (m)')
            plt.ylabel('Z (m)')
            if i == 0:
                plt.legend()
            plt.grid(True)
        plt.savefig('grids_permanent_magnets.png')

    def _toVTK(
        self, vtkname, dim=(1) 
    ):
        """write dipole data into a VTK file. Function taken and editted from 
           ciaoxiang's CoilPy library. 
        Args:
            vtkname (str): VTK filename, will be appended with .vts or .vtu.
            dim (tuple, optional): Dimension information if saved as structured grids. Defaults to (1).
        """
        from pyevtk.hl import gridToVTK, pointsToVTK

        dim = np.atleast_1d(dim)
        if len(dim) == 1:  # save as points
            print("write VTK as points")
            data = {"m": (self.mx, self.my, self.mz)}
            if not self.old:
                data.update({"rho": self.pho ** self.momentq})
            data.update(kwargs)
            pointsToVTK(
                vtkname, self.ox, self.oy, self.oz, data=data
            )  # .update(kwargs))
        else:  # save as surfaces
            assert len(dim) == 3
            print("write VTK as closed surface")
            if close:
                # manually close the gap
                phi = 2 * np.pi / self.nfp

                def map_toroidal(vec):
                    rotate = np.array(
                        [
                            [np.cos(phi), np.sin(phi), 0],
                            [-np.sin(phi), np.cos(phi), 0],
                            [0, 0, 1],
                        ]
                    )
                    return np.matmul(vec, rotate)

                data_array = {
                    "ox": self.ox,
                    "oy": self.oy,
                    "oz": self.oz,
                    "mx": self.mx,
                    "my": self.my,
                    "mz": self.mz,
                    "Ic": self.Ic,
                    "rho": self.pho ** self.momentq,
                }
                data_array.update(kwargs)
                nr, nz, nt = dim
                for key in list(data_array.keys()):
                    new_vec = np.zeros((nr, nz + 1, nt + 1))
                    for ir in range(nr):
                        new_vec[ir, :, :] = map_matrix(
                            np.reshape(data_array[key], dim)[ir, :, :]
                        )
                    if toroidal:
                        data_array[key] = new_vec
                    else:
                        if ntnz:
                            data_array[key] = np.ascontiguousarray(new_vec[:, :, :-1])
                        else:
                            data_array[key] = np.ascontiguousarray(new_vec[:, :-1, :])
                ox = np.copy(data_array["ox"])
                oy = np.copy(data_array["oy"])
                oz = np.copy(data_array["oz"])
                del data_array["ox"]
                del data_array["oy"]
                del data_array["oz"]
                data_array["m"] = (data_array["mx"], data_array["my"], data_array["mz"])
                if toroidal and self.nfp >= 1:  # not quite sure if should include nfp=1
                    for ir in range(nr):
                        if ntnz:
                            xyz = map_toroidal(
                                np.transpose([ox[ir, :, 0], oy[ir, :, 0], oz[ir, :, 0]])
                            )
                            ox[ir, :, nz] = xyz[:, 0]
                            oy[ir, :, nz] = xyz[:, 1]
                            oz[ir, :, nz] = xyz[:, 2]
                            moment = map_toroidal(
                                np.transpose(
                                    [
                                        data_array["mx"][ir, :, 0],
                                        data_array["my"][ir, :, 0],
                                        data_array["mz"][ir, :, 0],
                                    ]
                                )
                            )
                            data_array["m"][0][ir, :, nz] = moment[:, 0]
                            data_array["m"][1][ir, :, nz] = moment[:, 1]
                            data_array["m"][2][ir, :, nz] = moment[:, 2]
                        else:
                            xyz = map_toroidal(
                                np.transpose([ox[ir, 0, :], oy[ir, 0, :], oz[ir, 0, :]])
                            )
                            ox[ir, nz, :] = xyz[:, 0]
                            oy[ir, nz, :] = xyz[:, 1]
                            oz[ir, nz, :] = xyz[:, 2]
                            moment = map_toroidal(
                                np.transpose(
                                    [
                                        data_array["mx"][ir, 0, :],
                                        data_array["my"][ir, 0, :],
                                        data_array["mz"][ir, 0, :],
                                    ]
                                )
                            )
                            data_array["m"][0][ir, nz, :] = moment[:, 0]
                            data_array["m"][1][ir, nz, :] = moment[:, 1]
                            data_array["m"][2][ir, nz, :] = moment[:, 2]
                del data_array["mx"]
                del data_array["my"]
                del data_array["mz"]
                gridToVTK(vtkname, ox, oy, oz, pointData=data_array)
                return
            else:
                ox = np.reshape(self.ox[: self.num], dim)
                oy = np.reshape(self.oy[: self.num], dim)
                oz = np.reshape(self.oz[: self.num], dim)
                mx = np.reshape(self.mx[: self.num], dim)
                my = np.reshape(self.my[: self.num], dim)
                mz = np.reshape(self.mz[: self.num], dim)
                rho = np.reshape(self.pho[: self.num] ** self.momentq, dim)
                Ic = np.reshape(self.Ic[: self.num], dim)
            data = {"m": (mx, my, mz), "rho": rho, "Ic": Ic}
            data.update(kwargs)
            gridToVTK(vtkname, ox, oy, oz, pointData=data)
        return 

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
        plt.figure(figsize=(14, 14))
        for i, ind in enumerate([0, 5, 20, 31]):
            plt.subplot(2, 2, i + 1)
            plt.title(r'$\phi = ${0:.2f}$^o$'.format(360 * self.phi[ind]))
            r_plasma = np.hstack((self.r_plasma[ind, :], self.r_plasma[ind, 0]))
            z_plasma = np.hstack((self.z_plasma[ind, :], self.z_plasma[ind, 0]))
            r_inner = np.hstack((self.r_inner[ind, :], self.r_inner[ind, 0]))
            z_inner = np.hstack((self.z_inner[ind, :], self.z_inner[ind, 0]))
            r_outer = np.hstack((self.r_outer[ind, :], self.r_outer[ind, 0]))
            z_outer = np.hstack((self.z_outer[ind, :], self.z_outer[ind, 0]))

            plt.plot(r_plasma, z_plasma, label='Plasma surface', linewidth=2)
            plt.plot(r_inner, z_inner, label='Inner surface', linewidth=2)
            plt.plot(r_outer, z_outer, label='Outer surface', linewidth=2)

            #plt.plot(self.r_plasma[ind, :], self.z_plasma[ind, :], label='Plasma surface', linewidth=3)
            #plt.plot(self.r_inner[ind, :], self.z_inner[ind, :], label='Inner surface', linewidth=3)
            #plt.plot(self.r_outer[ind, :], self.z_outer[ind, :], label='Outer surface', linewidth=3)
            plt.scatter(
                np.array(self.final_RZ_grid[ind])[:, 0], 
                np.array(self.final_RZ_grid[ind])[:, 1], 
                label='Final grid',
                c='k'
            )
            # plt.scatter(np.ravel(self.RPhiZ[:, i, :, 0]), np.ravel(self.RPhiZ[:, i, :, 2]), c='k')
            if i == 0:
                plt.legend()
            plt.grid(True)
        plt.savefig('grids_permanent_magnets.png')

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
        phi_inner = 2 * np.pi * np.copy(self.rz_inner_surface.quadpoints_phi)
        normal_inner = self.rz_inner_surface.unitnormal()
        normal_outer = self.rz_outer_surface.unitnormal()
        Nray = 2000
        total_points = 0

        new_grids = []
        for i in range(len(phi_inner)):
            # Get (R, Z) locations of the points with respect to the magnetic axis
            Rpoint = np.ravel(self.RPhiZ[:, i, :, 0])
            Zpoint = np.ravel(self.RPhiZ[:, i, :, 2])

            # rotate normal vectors in (r, phi, z) coordinates and set phi component to zero
            # so that we keep everything in the same phi = constant cross-section
            rot_matrix = [[np.cos(phi_inner[i]), np.sin(phi_inner[i]), 0],
                          [-np.sin(phi_inner[i]), np.cos(phi_inner[i]), 0],
                          [0, 0, 1]]
            for j in range(normal_inner.shape[1]):
                normal_inner[i, j, :] = rot_matrix @ normal_inner[i, j, :]
                normal_outer[i, j, :] = rot_matrix @ normal_outer[i, j, :]
            normal_inner[i, :, 1] = 0.0
            normal_inner[i, :, 0] = normal_inner[i, :, 0] / np.sqrt(normal_inner[i, :, 0] ** 2 + normal_inner[i, :, 2] ** 2)
            normal_inner[i, :, 2] = normal_inner[i, :, 2] / np.sqrt(normal_inner[i, :, 0] ** 2 + normal_inner[i, :, 2] ** 2)
            normal_outer[i, :, 1] = 0.0
            normal_outer[i, :, 0] = normal_outer[i, :, 0] / np.sqrt(normal_outer[i, :, 0] ** 2 + normal_outer[i, :, 2] ** 2)
            normal_outer[i, :, 2] = normal_outer[i, :, 2] / np.sqrt(normal_outer[i, :, 0] ** 2 + normal_outer[i, :, 2] ** 2)

            # Find nearest (R, Z) points on the surface
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

                ray_equation = np.outer(
                    [Rpoint[j], 
                     Zpoint[j]], 
                    np.ones(Nray)
                ) + np.outer(ray_direction[[0, 2]], np.linspace(0, 2, Nray))
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
        phi = 2 * np.pi * self.plasma_boundary.quadpoints_phi
        nfp = self.plasma_boundary.nfp
        stellsym = self.plasma_boundary.stellsym 
        if stellsym:
            nsym = nfp * 2
        else:
            nsym = nfp
        geo_factor = np.zeros((self.nphi, self.ntheta, self.ndipoles, 3, nsym))

        # Loops over all the field period contributions from every quad point
        for i in range(self.nphi):
            for j in range(self.ntheta):
                normal_plasma_xyz = self.plasma_boundary.unitnormal()[i, j, :]
                R_plasma_xyz = self.plasma_boundary.gamma()[i, j, :]
                running_tally = 0
                for k in range(self.nphi):
                    dipole_grid_r = np.ravel(np.array(self.final_RZ_grid[k])[:, 0])
                    dipole_grid_z = np.ravel(np.array(self.final_RZ_grid[k])[:, 1])
                    for fp in range(nfp):
                        phi_sym = phi[k] + (2 * np.pi / nfp) * fp
                        dipole_grid_x = dipole_grid_r * np.cos(phi_sym)
                        dipole_grid_y = dipole_grid_r * np.sin(phi_sym)
                        R_dipole_xyz = np.array([dipole_grid_x, dipole_grid_y, dipole_grid_z]).T
                        R_dist_xyz = np.sqrt((R_plasma_xyz[0] - R_dipole_xyz[:, 0]) ** 2 + (R_plasma_xyz[1] - R_dipole_xyz[:, 1]) ** 2 + (R_plasma_xyz[2] - R_dipole_xyz[:, 2]) ** 2)
                        R_diff_xyz = np.zeros((len(dipole_grid_r), 3))
                        for jj in range(len(dipole_grid_r)):
                            R_diff_xyz[jj, :] = R_plasma_xyz - R_dipole_xyz[jj, :]
                        RdotN_xyz = R_diff_xyz @ normal_plasma_xyz
                        for kk in range(3):
                            geo_factor[i, j, 
                                       running_tally:running_tally + len(dipole_grid_r), kk, fp
                                       ] = 3.0 * RdotN_xyz / R_dist_xyz ** 5 * R_diff_xyz[:, kk] - normal_plasma_xyz[kk] / R_dist_xyz ** 3
                        if stellsym:
                            # Need to put magnets at (R, -phi, -Z), equivalently (X, -Y, -Z) 
                            # and then multiple geo_factor
                            # because stellarator symmetric field
                            # transforms like (mr, mphi, mz) -> (-mr, mphi, mz)
                            # mx = mr cos(phi) - mphi sin(phi)
                            # my = mphi cos(phi) + mr sin(phi)
                            # so geo factor must be transformed into cylindrical, altered
                            # and then transformed back
                            R_dipole_stellsym = np.copy(R_dipole_xyz)
                            R_dipole_stellsym[:, 1] = - R_dipole_xyz[:, 1]
                            R_dipole_stellsym[:, 2] = - R_dipole_xyz[:, 2]
                            R_dist_stellsym = np.sqrt((R_plasma_xyz[0] - R_dipole_stellsym[:, 0]) ** 2 + (R_plasma_xyz[1] - R_dipole_stellsym[:, 1]) ** 2 + (R_plasma_xyz[2] - R_dipole_stellsym[:, 2]) ** 2)
                            R_diff_stellsym = np.zeros((len(dipole_grid_r), 3))
                            for jj in range(len(dipole_grid_r)):
                                R_diff_stellsym[jj, :] = R_plasma_xyz - R_dipole_stellsym[jj, :]
                            RdotN_stellsym = R_diff_stellsym @ normal_plasma_xyz
                            # get geo_factor in cartesian
                            for kk in range(3):
                                geo_factor[i, j, 
                                           running_tally:running_tally + len(dipole_grid_r), kk, fp + nfp
                                           ] = (3.0 * RdotN_stellsym / R_dist_stellsym ** 5 * R_diff_stellsym[:, kk] - normal_plasma_xyz[kk] / R_dist_stellsym ** 3)
                            # rotate into cylindrical
                            geo_factor_r = geo_factor[i, j, 
                                                      running_tally:running_tally + len(dipole_grid_r), 0, fp + nfp
                                                      ] * np.cos(phi_sym) + geo_factor[i, j, 
                                                                                       running_tally:running_tally + len(dipole_grid_r), 1, fp + nfp
                                                                                       ] * np.sin(phi_sym)
                            geo_factor_phi = -geo_factor[i, j, 
                                                         running_tally:running_tally + len(dipole_grid_r), 0, fp + nfp
                                                         ] * np.sin(phi_sym) + geo_factor[i, j, 
                                                                                          running_tally:running_tally + len(dipole_grid_r), 1, fp + nfp
                                                                                          ] * np.cos(phi_sym)
                            # get back into cartesian, but with sign of geo_factor_r flipped
                            geo_factor_x = -geo_factor_r * np.cos(phi_sym) - geo_factor_phi * np.sin(phi_sym)
                            geo_factor_y = -geo_factor_r * np.sin(phi_sym) + geo_factor_phi * np.cos(phi_sym)
                            # set final matrix
                            geo_factor[i, j, 
                                       running_tally:running_tally + len(dipole_grid_r), 0, fp + nfp
                                       ] = geo_factor_x
                            geo_factor[i, j, 
                                       running_tally:running_tally + len(dipole_grid_r), 1, fp + nfp
                                       ] = geo_factor_y
                    running_tally += len(dipole_grid_r)

        # Sum over the matrix contributions from each part of the torus
        mu_fac = 1e-7
        geo_factor = np.sum(geo_factor, axis=-1)
        geo_factor_flat = np.reshape(geo_factor, (self.nphi * self.ntheta, self.ndipoles * 3)) * mu_fac
        geo_factor = np.reshape(geo_factor, (self.nphi * self.ntheta, self.ndipoles, 3)) * mu_fac
        dphi = (self.phi[1] - self.phi[0]) * 2 * np.pi
        dtheta = (self.theta[1] - self.theta[0]) * 2 * np.pi
        self.A_obj = geo_factor_flat * np.sqrt(dphi * dtheta)
        self.A_obj_expanded = geo_factor * np.sqrt(dphi * dtheta)

        # Initialize 'b' vector in 0.5 * ||Am - b||^2 part of the optimization,
        # corresponding to the normal component of the target fields. Note
        # the factor of two in the least-squares term: 0.5 * m.T @ (A.T @ A) @ m - b.T @ m
        if self.B_plasma_surface.shape != (self.nphi, self.ntheta, 3):
            raise ValueError('Magnetic field surface data is incorrect shape.')
        Bs = self.B_plasma_surface

        # minus sign below because ||Ax - b||^2 term but original
        # term is integral(B_P + B_C + B_M)?
        self.b_obj = - np.sum(
            Bs * self.plasma_boundary.unitnormal(), axis=2
        ).reshape(self.nphi * self.ntheta) * np.sqrt(dphi * dtheta)
        self.ATb = (self.A_obj.transpose()).dot(self.b_obj)
        self.ATA = (self.A_obj).T @ self.A_obj 
        self.ATA_scale = np.linalg.norm(self.ATA, ord=2)

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

    def _prox_l0(self, m, reg_l0, nu):
        """Proximal operator for L0 regularization."""
        return m * (np.abs(m) > np.sqrt(2 * reg_l0 * nu))

    def _prox_l1(self, m, reg_l1, nu):
        """Proximal operator for L1 regularization."""
        return np.sign(m) * np.maximum(np.abs(m) - reg_l1 * nu, 0)

    def _projection_L2_balls(self, x, m_maxima):
        """
        Project the vector x onto a series of L2 balls in R3.
        """
        N = len(x) // 3
        x_shaped = x.reshape(N, 3)
        denom = np.maximum(
            1,
            np.sqrt(x_shaped[:, 0] ** 2 + x_shaped[:, 1] ** 2 + x_shaped[:, 2] ** 2) / m_maxima 
        )
        return np.divide(x_shaped, np.array([denom, denom, denom]).T).reshape(3 * N)

    def _optimize(self, m0=None, epsilon=1e-4, nu=1e3,
                  reg_l0=0, reg_l1=0, reg_l2=0, reg_l2_shifted=0, 
                  max_iter_MwPGP=50, max_iter_RS=4, verbose=True,
                  geometric_threshold=1e-50, 
                  ): 
        """ 
            Perform the permanent magnet optimization problem, 
            phrased as a relax-and-split formulation that 
            solves the convex and nonconvex parts separately. 
            This allows for speedy algorithms for both parts, 
            the imposition of convex equality and inequality
            constraints (including the required constraint on
            the strengths of the dipole moments). 

            Args:
                m0: Initial guess for the permanent magnet
                    dipole moments. Defaults to a random 
                    starting guess between [0, m_max].
                epsilon: Error tolerance for the convex 
                    part of the algorithm (MwPGP).
                nu: Hyperparameter used for the relax-and-split
                    least-squares. Set nu >> 1 to reduce the
                    importance of nonconvexity in the problem.
                reg_l0: Regularization value for the L0
                    nonconvex term in the optimization. This value 
                    is automatically scaled based on the max dipole
                    moment values, so that reg_l0 = 1 corresponds 
                    to reg_l0 = np.max(m_maxima). It follows that
                    users should choose reg_l0 in [0, 1]. 
                reg_l1: Regularization value for the L1
                    nonsmooth term in the optimization,
                reg_l2: Regularization value for any convex
                    regularizers in the optimization problem,
                    such as the often-used L2 norm.
                reg_l2_shifted: Regularization value for the L2
                    smooth and convex term in the optimization, 
                    shifted by the vector of maximum dipole magnitudes. 
                max_iter_MwPGP: Maximum iterations to perform during
                    a run of the convex part of the relax-and-split
                    algorithm (MwPGP). 
                max_iter_RS: Maximum iterations to perform of the 
                    overall relax-and-split algorithm. Therefore, 
                    also the number of times that MwPGP is called,
                    and the number of times a prox is computed.
                verbose: Prints out all the loss term errors separately.
                geometric_threshold: Threshold value for A.T * A matrix
                    appearing in the optimization. Any elements in |A.T * A|
                    below this value are truncated off. 
        """
        # Initialize initial guess for the dipole strengths
        if m0 is not None:
            if len(m0) != self.ndipoles * 3:
                raise ValueError(
                    'Initial dipole guess is incorrect shape --'
                    ' guess must be 1D with shape (ndipoles * 3).'
                )
        else:
            m0 = self._projection_L2_balls(
                np.linalg.pinv(self.A_obj) @ self.b_obj, 
                self.m_maxima
            )
            # temporary check
            m0 = np.zeros(m0.shape)

        print('L2 regularization being used with coefficient = {0:.2e}'.format(reg_l2))
        print('Shifted L2 regularization being used with coefficient = {0:.2e}'.format(reg_l2_shifted))

        # scale regularizers to the largest scale of ATA (~1e-6)
        # to avoid regularization >> ||Am - b|| ** 2 term in the optimization
        # prox uses reg_l0 * nu for the threshold
        # so normalization below allows reg_l0 and reg_l1 
        # values to be exactly the thresholds used in 
        # calculation of the prox
        if reg_l0 < 0 or reg_l0 > 1:
            raise ValueError(
                'L0 regularization must be between 0 and 1. This '
                'value is automatically scaled to the largest of the '
                'dipole maximum values, so reg_l0 = 1 should basically '
                'truncate all the dipoles to zero. '
            )
        reg_l0 = reg_l0 * self.ATA_scale * np.max(self.m_maxima) ** 2 / (2 * nu)
        reg_l1 = reg_l1 * self.ATA_scale / nu
        nu = nu / self.ATA_scale

        reg_l2 = reg_l2  # * self.ATA_scale
        reg_l2_shifted = reg_l2_shifted  # * self.ATA_scale
        ATA = self.ATA + 2 * (reg_l2 + reg_l2_shifted) * np.eye(self.ATA.shape[0])

        # if using relax and split, add that contribution to ATA
        if reg_l0 > 0.0 or reg_l1 > 0.0: 
            ATA += np.eye(ATA.shape[0]) / nu

        # Add shifted L2 contribution to ATb
        ATb = self.ATb + reg_l2_shifted * np.ravel(np.outer(self.m_maxima, np.ones(3)))

        # get optimal alpha value for the MwPGP algorithm
        alpha_max = 2.0 / np.linalg.norm(ATA, ord=2)

        # Truncate all terms with magnitude < geometric_threshold
        ATA = (np.abs(ATA) > geometric_threshold) * ATA

        # Optionally make ATA a sparse matrix for memory savings
        ATA_sparse = csr_matrix(ATA)

        print('Total number of elements in ATA = ', len(np.ravel(ATA_sparse.toarray())))
        print('Number of nonzero elements in ATA = ', ATA_sparse.count_nonzero())
        print('Percent of elements in ATA that are nonzero = ', 
              ATA_sparse.count_nonzero() / len(np.ravel(ATA_sparse.toarray()))
              )
        # Print out initial errors and the bulk optimization paramaters 
        ave_Bn = np.mean(np.abs(self.b_obj))
        Bmag = np.linalg.norm(self.B_plasma_surface, axis=-1, ord=2).reshape(self.nphi * self.ntheta)
        ave_BnB = np.mean(np.abs(self.b_obj) / Bmag)
        total_Bn = np.sum(np.abs(self.b_obj) ** 2)
        dipole_error = np.linalg.norm(self.A_obj.dot(m0), ord=2) ** 2
        total_error = np.linalg.norm(self.A_obj.dot(m0) - self.b_obj, ord=2) ** 2
        print('Number of phi quadrature points on plasma surface = ', self.nphi)
        print('Number of theta quadrature points on plasma surface = ', self.ntheta)
        print('<B * n> without the permanent magnets = {0:.4e}'.format(ave_Bn)) 
        print('<B * n / |B| > without the permanent magnets = {0:.4e}'.format(ave_BnB)) 
        print(r'$|b|_2^2 = |B * n|_2^2$ without the permanent magnets = {0:.4e}'.format(total_Bn))
        print(r'Initial $|Am_0|_2^2 = |B_M * n|_2^2$ without the coils/plasma = {0:.4e}'.format(dipole_error))
        print('Number of dipoles = ', self.ndipoles)
        print('Inner toroidal surface offset from plasma surface = ', self.plasma_offset)
        print('Outer toroidal surface offset from inner toroidal surface = ', self.coil_offset)
        print('Maximum dipole moment = ', np.max(self.m_maxima))
        print('Shape of A matrix = ', self.A_obj.shape)
        print('Shape of b vector = ', self.b_obj.shape)
        print('Initial error on plasma surface = {0:.4e}'.format(total_error))

        # Begin optimization
        m_proxy = m0
        err_RS = []
        if reg_l0 > 0.0 or reg_l1 > 0.0: 
            # Relax-and-split algorithm
            if reg_l0 > 0.0:
                prox = self._prox_l0
            elif reg_l1 > 0.0:
                prox = self._prox_l1
            m = m0
            for i in range(max_iter_RS):
                # update m
                MwPGP_hist, _, m_hist, m = sopp.MwPGP_algorithm(
                    A_obj=self.A_obj_expanded,
                    b_obj=self.b_obj,
                    ATA=np.reshape(ATA, (self.ndipoles, 3, self.ndipoles, 3)),
                    ATb=np.reshape(ATb, (self.ndipoles, 3)),
                    m_proxy=m_proxy.reshape(self.ndipoles, 3),
                    m0=m.reshape(self.ndipoles, 3),
                    m_maxima=self.m_maxima,
                    alpha=alpha_max,
                    nu=nu,
                    epsilon=epsilon,
                    max_iter=max_iter_MwPGP,
                    verbose=True,
                    reg_l0=reg_l0,
                    reg_l1=reg_l1,
                    reg_l2=reg_l2,
                    reg_l2_shifted=reg_l2_shifted,
                )
                m = np.ravel(m)
                err_RS.append(MwPGP_hist[-1])

                # update m_proxy
                m_proxy = prox(m, reg_l0, nu)
                if np.linalg.norm(m - m_proxy) < epsilon:
                    print('Relax-and-split finished early, at iteration ', i)
            # Default here is to use the sparse version of m from relax-and-split
            m = m_proxy
        else:
            MwPGP_hist, _, m_hist, m = sopp.MwPGP_algorithm(
                A_obj=self.A_obj_expanded,
                b_obj=self.b_obj,
                ATA=np.reshape(ATA, (self.ndipoles, 3, self.ndipoles, 3)),
                ATb=np.reshape(ATb, (self.ndipoles, 3)),
                m_proxy=m0.reshape(self.ndipoles, 3),
                m0=m0.reshape(self.ndipoles, 3),
                m_maxima=self.m_maxima,
                alpha=alpha_max,
                epsilon=epsilon,
                max_iter=max_iter_MwPGP,
                verbose=True,
                reg_l0=reg_l0,
                reg_l1=reg_l1,
                reg_l2=reg_l2,
                reg_l2_shifted=reg_l2_shifted,
            )    
            m = np.ravel(m)
            m_proxy = m

        # Compute metrics with permanent magnet results
        ave_Bn_proxy = np.mean(np.abs(self.A_obj.dot(m_proxy) - self.b_obj))
        ave_Bn = np.mean(np.abs(self.A_obj.dot(m) - self.b_obj))
        ave_BnB = np.mean(np.abs((self.A_obj.dot(m_proxy) - self.b_obj)) / Bmag)  # using original Bmag without PMs
        print(np.max(self.m_maxima), np.max(m_proxy))
        print('<B * n> with the optimized permanent magnets = {0:.8e}'.format(ave_Bn)) 
        print('<B * n> with the sparsified permanent magnets = {0:.8e}'.format(ave_Bn_proxy)) 
        print('<B * n / |B| > with the permanent magnets = {0:.8e}'.format(ave_BnB)) 
        #print('A * m = ', np.tensordot(self.A_obj_expanded, m.reshape(self.ndipoles, 3), axes=([-2, -1], [-2, -1])).reshape(self.nphi, self.ntheta))
        #print('b_obj = ', self.b_obj.reshape(self.nphi, self.ntheta))
        self.m = m
        m_vec = m.reshape(self.ndipoles, 3)
        # print(self.dipole_grid[:, 0] ** 2 + self.dipole_grid[:, 2] ** 2)
        #print(m_vec[:, 0] ** 2 + m_vec[:, 1] ** 2, m_vec[:, 2] ** 2) 
        #print(m)
        self.m_proxy = m_proxy
        return MwPGP_hist, err_RS, m_hist, m_proxy