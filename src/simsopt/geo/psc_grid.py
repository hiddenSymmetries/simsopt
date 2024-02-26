from pathlib import Path
import warnings

import numpy as np
from pyevtk.hl import pointsToVTK

from . import Surface
import simsoptpp as sopp
from simsopt.field import BiotSavart
from scipy.integrate import simpson, dblquad, IntegrationWarning

__all__ = ['PSCgrid']


class PSCgrid:
    r"""
    ``PSCgrid`` is a class for setting up the grid,
    plasma surface, and other objects needed to perform PSC
    optimization for stellarators. The class
    takes as input two toroidal surfaces specified as SurfaceRZFourier
    objects, and initializes a set of points (in Cartesian coordinates)
    between these surfaces.

    Args:
        plasma_boundary: Surface class object 
            Representing the plasma boundary surface. Gets converted
            into SurfaceRZFourier object for ease of use.
        Bn: 2D numpy array, shape (ntheta_quadpoints, nphi_quadpoints)
            Magnetic field (coils and plasma) at the plasma
            boundary. Typically this will be the optimized plasma
            magnetic field from a stage-1 optimization, and the
            optimized coils from a basic stage-2 optimization.
            This variable must be specified to run PSC optimization.
    """

    def __init__(self):
        self.mu0 = 4 * np.pi * 1e-7

    def _setup_uniform_grid(self):
        """
        Initializes a uniform grid in cartesian coordinates and sets
        some important grid variables for later.
        """
        # Get (X, Y, Z) coordinates of the two boundaries
        self.xyz_inner = self.inner_toroidal_surface.gamma().reshape(-1, 3)
        self.xyz_outer = self.outer_toroidal_surface.gamma().reshape(-1, 3)
        x_outer = self.xyz_outer[:, 0]
        y_outer = self.xyz_outer[:, 1]
        z_outer = self.xyz_outer[:, 2]

        x_max = np.max(x_outer)
        x_min = np.min(x_outer)
        y_max = np.max(y_outer)
        y_min = np.min(y_outer)
        z_max = np.max(z_outer)
        z_min = np.min(z_outer)
        z_max = max(z_max, abs(z_min))
        print(x_min, x_max, y_min, y_max, z_min, z_max)

        # Initialize uniform grid
        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz
        self.dx = (x_max - x_min) / (Nx - 1)
        self.dy = (y_max - y_min) / (Ny - 1)
        self.dz = 2 * z_max / (Nz - 1)
        print(Nx, Ny, Nz, self.dx, self.dy, self.dz)
        Nmin = min(self.dx, min(self.dy, self.dz))
        self.R = Nmin / 4.0
        self.a = self.R / 10.0
        print('Major radius of the coils is R = ', self.R)
        print('Coils are spaced so that every coil of radius R '
              ' is at least 2R away from the next coil'
        )

        # Extra work below so that the stitching with the symmetries is done in
        # such a way that the reflected cells are still dx and dy away from
        # the old cells.
        #### Note that Cartesian cells can only do nfp = 2, 4, 6, ... 
        #### and correctly be rotated to have the right symmetries
        # if (self.plasma_boundary.nfp % 2) == 0:
        #     X = np.linspace(self.dx / 2.0, (x_max - x_min) + self.dx / 2.0, Nx, endpoint=True)
        #     Y = np.linspace(self.dy / 2.0, (y_max - y_min) + self.dy / 2.0, Ny, endpoint=True)
        # else:
        X = np.linspace(x_min, x_max, Nx, endpoint=True)
        Y = np.linspace(y_min, y_max, Ny, endpoint=True)
        Z = np.linspace(-z_max, z_max, Nz, endpoint=True)

        # Make 3D mesh
        X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')
        self.xyz_uniform = np.transpose(np.array([X, Y, Z]), [1, 2, 3, 0]).reshape(Nx * Ny * Nz, 3)

        # # Extra work for nfp = 4 to chop off half of the originally nfp = 2 uniform grid
        # if self.plasma_boundary.nfp == 4:
        #     inds = []
        #     for i in range(Nx):
        #         for j in range(Ny):
        #             for k in range(Nz):
        #                 if X[i, j, k] < Y[i, j, k]:
        #                     inds.append(int(i * Ny * Nz + j * Nz + k))
        #     good_inds = np.setdiff1d(np.arange(Nx * Ny * Nz), inds)
        #     self.xyz_uniform = self.xyz_uniform[good_inds, :]
        # else:
        #     # Get (R, Z) coordinates of the outer boundary
        #     rphiz_outer = np.array(
        #         [np.sqrt(self.xyz_outer[:, 0] ** 2 + self.xyz_outer[:, 1] ** 2), 
        #          np.arctan2(self.xyz_outer[:, 1], self.xyz_outer[:, 0]),
        #          self.xyz_outer[:, 2]]
        #     ).T

        #     r_max = np.max(rphiz_outer[:, 0])
        #     r_min = np.min(rphiz_outer[:, 0])
        #     z_max = np.max(rphiz_outer[:, 2])
        #     z_min = np.min(rphiz_outer[:, 2])

        #     # Initialize uniform grid of curved, square bricks
        #     Nr = int((r_max - r_min) / self.dr)
        #     self.Nr = Nr
        #     self.dz = self.dr
        #     Nz = int((z_max - z_min) / self.dz)
        #     self.Nz = Nz
        #     phi = 2 * np.pi * np.copy(self.plasma_boundary.quadpoints_phi)
        #     R = np.linspace(r_min, r_max, Nr)
        #     Z = np.linspace(z_min, z_max, Nz)

        #     # Make 3D mesh
        #     R, Phi, Z = np.meshgrid(R, phi, Z, indexing='ij')
        #     X = R * np.cos(Phi)
        #     Y = R * np.sin(Phi)
        #     self.xyz_uniform = np.transpose(np.array([X, Y, Z]), [1, 2, 3, 0]).reshape(-1, 3)

        # Save uniform grid before we start chopping off parts.
        contig = np.ascontiguousarray
        pointsToVTK('uniform_grid', contig(self.xyz_uniform[:, 0]),
                    contig(self.xyz_uniform[:, 1]), contig(self.xyz_uniform[:, 2]))

    @classmethod
    def geo_setup_between_toroidal_surfaces(
        cls, 
        plasma_boundary : Surface,
        Bn,
        B_TF,
        inner_toroidal_surface: Surface, 
        outer_toroidal_surface: Surface,
        **kwargs,
    ):
        """
        Function to initialize a SIMSOPT PermanentMagnetGrid from a 
        volume defined by two toroidal surfaces. These must be specified
        directly. Often a good choice is made by extending the plasma 
        boundary by its normal vectors.

        Args
        ----------
        inner_toroidal_surface: Surface class object 
            Representing the inner toroidal surface of the volume.
            Gets converted into SurfaceRZFourier object for 
            ease of use.
        outer_toroidal_surface: Surface object representing
            the outer toroidal surface of the volume. Typically 
            want this to have same quadrature points as the inner
            surface for a functional grid setup. 
            Gets converted into SurfaceRZFourier object for 
            ease of use.
        kwargs: The following are valid keyword arguments.
            Nx: int
                Number of points in x to use in a cartesian grid, taken between the 
                inner and outer toroidal surfaces. Used only if the
                coordinate_flag = cartesian, then Nx is the x-size of the
                rectangular cubes in the grid.
            Ny: int
                Number of points in y to use in a cartesian grid, taken between the 
                inner and outer toroidal surfaces. Used only if the
                coordinate_flag = cartesian, then Ny is the y-size of the
                rectangular cubes in the grid.
            Nz: int
                Number of points in z to use in a cartesian grid, taken between the 
                inner and outer toroidal surfaces. Used only if the
                coordinate_flag = cartesian, then Nz is the z-size of the
                rectangular cubes in the grid.
            Nt: int
                Number of turns of the coil. 
        Returns
        -------
        psc_grid: An initialized PSCgrid class object.

        """
        
        psc_grid = cls() 
        Bn = np.array(Bn)
        if len(Bn.shape) != 2: 
            raise ValueError('Normal magnetic field surface data is incorrect shape.')
        psc_grid.Bn = Bn
        
        if not isinstance(B_TF, BiotSavart):
            raise ValueError('The magnetic field from the energized coils '
                ' must be passed as a BiotSavart object.'
        )
        psc_grid.B_TF = B_TF
        psc_grid.plasma_boundary = plasma_boundary.to_RZFourier()
        psc_grid.nphi = len(psc_grid.plasma_boundary.quadpoints_phi)
        psc_grid.ntheta = len(psc_grid.plasma_boundary.quadpoints_theta)
        Nx = kwargs.pop("Nx", 10)
        Ny = Nx  # kwargs.pop("Ny", 10)
        Nz = Nx  # kwargs.pop("Nz", 10)
        if Nx <= 0 or Ny <= 0 or Nz <= 0:
            raise ValueError('Nx, Ny, and Nz should be positive integers')
        psc_grid.Nx = Nx
        psc_grid.Ny = Ny
        psc_grid.Nz = Nz
        psc_grid.inner_toroidal_surface = inner_toroidal_surface.to_RZFourier()
        psc_grid.outer_toroidal_surface = outer_toroidal_surface.to_RZFourier()    
        warnings.warn(
            'Plasma boundary and inner and outer toroidal surfaces should '
            'all have the same "range" parameter in order for a permanent'
            ' magnet grid to be correctly initialized.'
        )        

        Nt = kwargs.pop("Nt", 1)
        psc_grid.Nt = Nt
        # Have the uniform grid, now need to loop through and eliminate cells.
        contig = np.ascontiguousarray
        normal_inner = inner_toroidal_surface.unitnormal().reshape(-1, 3)   
        normal_outer = outer_toroidal_surface.unitnormal().reshape(-1, 3)   
        psc_grid._setup_uniform_grid()
        psc_grid.grid_xyz = sopp.define_a_uniform_cartesian_grid_between_two_toroidal_surfaces(
            contig(normal_inner), 
            contig(normal_outer), 
            contig(psc_grid.xyz_uniform), 
            contig(psc_grid.xyz_inner), 
            contig(psc_grid.xyz_outer))
        inds = np.ravel(np.logical_not(np.all(psc_grid.grid_xyz == 0.0, axis=-1)))
        psc_grid.grid_xyz = psc_grid.grid_xyz[inds, :]
        psc_grid.num_psc = psc_grid.grid_xyz.shape[0]
        # psc_grid.pm_phi = np.arctan2(psc_grid.grid_xyz[:, 1], psc_grid.grid_xyz[:, 0])
        pointsToVTK('psc_grid',
                    contig(psc_grid.grid_xyz[:, 0]),
                    contig(psc_grid.grid_xyz[:, 1]),
                    contig(psc_grid.grid_xyz[:, 2]))
        print('Number of PSC locations = ', len(psc_grid.grid_xyz))

        # PSC coil geometry determined by its center point in grid_xyz
        # and its alpha and delta angles, which we initialize randomly here.
        psc_grid.alphas = np.random.rand(psc_grid.num_psc) * 2 * np.pi
        psc_grid.deltas = np.random.rand(psc_grid.num_psc) * 2 * np.pi

        psc_grid.coil_normals = np.array(
            [np.cos(psc_grid.alphas) * np.sin(psc_grid.deltas),
             -np.sin(psc_grid.alphas),
             np.cos(psc_grid.alphas) * np.cos(psc_grid.deltas)]
        ).T
        
        # Initialize curve objects corresponding to each PSC coil for 
        # plotting in 3D
        psc_grid.fluxes(psc_grid.alphas, psc_grid.deltas)
        psc_grid.inductances(psc_grid.alphas, psc_grid.deltas)
        psc_grid.plot_curves()

        # psc_grid._optimization_setup()
        return psc_grid
    
    @classmethod
    def geo_setup_manual(
        cls, 
        points,
        R,
        a,
        alphas,
        deltas,
        **kwargs,
    ):
        
        psc_grid = cls()
        psc_grid.grid_xyz = points
        psc_grid.R = R
        psc_grid.a = a
        psc_grid.alphas = alphas
        psc_grid.deltas = deltas
        Nt = kwargs.pop("Nt", 1)
        psc_grid.Nt = Nt
        psc_grid.num_psc = psc_grid.grid_xyz.shape[0]
        
        Bn = kwargs.pop("Bn", np.zeros((1, 3)))
        Bn = np.array(Bn)
        if len(Bn.shape) != 2: 
            raise ValueError('Normal magnetic field surface data is incorrect shape.')
        psc_grid.Bn = Bn
        
        B_TF = kwargs.pop("B_TF", np.zeros((1, 3)))
        psc_grid.B_TF = B_TF
        
        contig = np.ascontiguousarray
        pointsToVTK('psc_grid',
                    contig(psc_grid.grid_xyz[:, 0]),
                    contig(psc_grid.grid_xyz[:, 1]),
                    contig(psc_grid.grid_xyz[:, 2]))
        print('Number of PSC locations = ', len(psc_grid.grid_xyz))

        # PSC coil geometry determined by its center point in grid_xyz
        # and its alpha and delta angles, which we initialize randomly here.
        # psc_grid.alphas = np.random.rand(psc_grid.num_psc) * 2 * np.pi
        # psc_grid.deltas = np.random.rand(psc_grid.num_psc) * 2 * np.pi
        psc_grid.coil_normals = np.array(
            [np.cos(alphas) * np.sin(deltas),
             -np.sin(alphas),
             np.cos(alphas) * np.cos(deltas)]
        ).T
        
        # Initialize curve objects corresponding to each PSC coil for 
        # plotting in 3D
        if isinstance(B_TF, BiotSavart):
            psc_grid.fluxes(psc_grid.alphas, psc_grid.deltas)
        psc_grid.inductances(psc_grid.alphas, psc_grid.deltas)
        psc_grid.plot_curves()
        
        return psc_grid

    def plot_curves(self):
        
        from . import CurvePlanarFourier, curves_to_vtk
        from pyevtk.hl import pointsToVTK

        # r(\phi) = \sum_{m=0}^{\text{order}} r_{c,m}\cos(m \phi) + \sum_{m=1}^{\text{order}} r_{s,m}\sin(m \phi).
        # [r_{c,0}, \cdots, r_{c,\text{order}}, r_{s,1}, \cdots, r_{s,\text{order}}, q_0, q_i, q_j, q_k, x_{\text{center}}, y_{\text{center}}, z_{\text{center}}]
    
        order = 1
        ncoils = self.num_psc
        # self.I = np.zeros(ncoils)
    
        # Set the degrees of freedom in the coil objects
        # base_currents = [Current(coilcurrents[i]) for i in range(ncoils)]
        ppp = 40
        curves = [CurvePlanarFourier(order*ppp, order, nfp=1, stellsym=False) for i in range(ncoils)]
        for ic in range(ncoils):
            xyz = self.grid_xyz[ic, :]
            dofs = np.zeros(10)
            dofs[0] = self.R
            dofs[1] = 0.0  # self.R
            dofs[2] = 0.0  #self.R
            # Conversion from Euler angles in 3-2-1 body sequence to 
            # quaternions: 
            # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
            dofs[3] = np.cos(self.alphas[ic] / 2.0) * np.cos(self.deltas[ic] / 2.0)
            dofs[4] = np.sin(self.alphas[ic] / 2.0) * np.cos(self.deltas[ic] / 2.0)
            dofs[5] = np.cos(self.alphas[ic] / 2.0) * np.sin(self.deltas[ic] / 2.0)
            dofs[6] = -np.sin(self.alphas[ic] / 2.0) * np.sin(self.deltas[ic] / 2.0)
            # Now specify the center 
            dofs[7] = xyz[0]
            dofs[8] = xyz[1]
            dofs[9] = xyz[2] 
            curves[ic].set_dofs(dofs)
        curves_to_vtk(curves, "psc_curves")
        self.curves = curves
        
        contig = np.ascontiguousarray
        if isinstance(self.B_TF, BiotSavart):
            self.B_TF.set_points(self.grid_xyz)
            B = self.B_TF.B()
            pointsToVTK('curve_centers', 
                        contig(self.grid_xyz[:, 0]),
                        contig(self.grid_xyz[:, 1]), 
                        contig(self.grid_xyz[:, 2]),
                        data={"n": (contig(self.coil_normals[:, 0]), 
                                    contig(self.coil_normals[:, 1]),
                                    contig(self.coil_normals[:, 2])),
                              "psi": contig(self.psi),
                              "B_TF": (contig(B[:, 0]), 
                                       contig(B[:, 1]),
                                       contig(B[:, 2])),
                              },
            )
        else:
            pointsToVTK('curve_centers', 
                        contig(self.grid_xyz[:, 0]),
                        contig(self.grid_xyz[:, 1]), 
                        contig(self.grid_xyz[:, 2]),
                        data={"n": (contig(self.coil_normals[:, 0]), 
                                    contig(self.coil_normals[:, 1]),
                                    contig(self.coil_normals[:, 2])),
                              },
            )
        # return coils
    
    def inductances(self, alphas, deltas):
        """ Calculate the inductance matrix needed for the PSC forward problem """
        points = self.grid_xyz
        nphi = 20
        phi = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
        dphi = phi[1] - phi[0]
        R = self.R
        r = self.a
        ncoils = self.num_psc
        # print(r, R, alphas, deltas, phi)
        L = np.zeros((ncoils, ncoils))
        for i in range(ncoils):
    	    # Loop through all the PSCs, using all the symmetries
            for j in range(i + 1, ncoils):
                xj = points[j, 0] / R
                yj = points[j, 1] / R
                zj = points[j, 2] / R
                cd = np.cos(deltas[i] - deltas[j])
                sd = np.sin(deltas[i] - deltas[j])
                ca = np.cos(alphas[i] - alphas[j])
                sa = np.sin(alphas[i] - alphas[j])
                integrand = 0.0
                for k in range(nphi):
                    ck = np.cos(phi[k])
                    sk = np.sin(phi[k])
                    for kk in range(nphi):
                        ckk = np.cos(phi[kk])
                        skk = np.sin(phi[kk])
                        integrand += (ck * ckk * ca + sk * skk * cd - ck * skk * sa * sd) / np.sqrt(
                            (xj + ckk - ck * cd - sk * sa * sd) ** 2 + (
                                yj + skk - sk * ca) ** 2 + (
                                    zj + ck * sd - sk * sa * cd) ** 2)
                L[i, j] = integrand * dphi ** 2 / (4.0 * np.pi)
                # def integrand(yy, xx):
                #     ck = np.cos(xx)
                #     sk = np.sin(xx)
                #     ckk = np.cos(yy)
                #     skk = np.sin(yy)
                #     return (ck * ckk * ca + sk * skk * cd - ck * skk * sa * sd) / np.sqrt(
                #                  (xj + ckk - ck * cd - sk * sa * sd) ** 2 + (
                #                      yj + skk - sk * ca) ** 2 + (
                #                          zj + ck * sd - sk * sa * cd) ** 2)
                # integral, err = dblquad(integrand, 0, 2 * np.pi, 0, 2 * np.pi)
                # L[i, j] = integral / (4.0 * np.pi)
        # symmetrize and scale
        L = L + L.T
        np.fill_diagonal(L, np.log(8.0 * R / r) - 2.0)
        self.L = L * self.mu0 * R * self.Nt ** 2
        
    def fluxes(self, alphas, deltas):
        import warnings
        warnings.filterwarnings("error")
        
        # rho = np.linspace(0, self.R, 1000)
        # drho = rho[1] - rho[0]
        # theta = np.linspace(0, 2 * np.pi, 1000)
        # dtheta = theta[1] - theta[0]
        # rho, theta = np.meshgrid(rho, theta, indexing='ij')
        fluxes = np.zeros(len(alphas))
        centers = self.grid_xyz
        bs = self.B_TF
        plot_points = []
        Bzs = []
        for j in range(len(alphas)):
            
            # xyz_coil = np.array([x * np.ones(rho.shape) + rho * np.cos(theta),
            #                      y * np.ones(rho.shape) + rho * np.sin(theta),
            #                      z * np.ones(rho.shape)]).T.reshape(-1, 3)
            # bs.set_points(xyz_coil)
            # Bz = bs.B() @ self.coil_normals[j, :]
            
            def Bz_func(yy, xx):
                x0 = xx * np.cos(yy)
                y0 = xx * np.sin(yy)
                z0 = 0.0
                x = x0 * np.cos(
                    deltas[j]
                    ) + y0 * np.sin(
                        alphas[j]
                        ) * np.sin(
                            deltas[j]
                            ) + z0 * np.cos(
                                alphas[j]
                                ) * np.sin(deltas[j]) + centers[j, 0]
                y = y0 * np.cos(
                        alphas[j]
                        ) - z0 * np.sin(
                                alphas[j]
                                ) + centers[j, 1]
                z = -x0 * np.sin(
                    deltas[j]
                    ) + y0 * np.sin(
                        alphas[j]
                        ) * np.cos(
                            deltas[j]
                            ) + z0 * np.cos(
                                alphas[j]
                                ) * np.cos(deltas[j]) + centers[j, 2]
                bs.set_points(np.array([x, y, z]).reshape(-1, 3))
                if j == 0:
                    plot_points.append(np.array([x, y, z]).reshape(-1, 3))
                    Bzs.append(bs.B().reshape(-1, 3))
                Bz = bs.B().reshape(-1, 3) @ self.coil_normals[j, :]
                return Bz * xx  # .reshape(-1)
            
            print(1.8020846512773543e-08 / 2.5998449603217048e-08 * np.pi)
            # fluxes[j] += np.sum(np.sum(Bz_func(theta, rho))) * drho * dtheta
            # fluxes[j] += simpson(simpson(Bz.reshape(rho.shape) * rho, rho0, axis=0), theta0)
            # fluxes[j] += gauss_weights @ (gauss_weights @ (Bz.reshape(rho.shape) * rho))
            try:
                integral, err = dblquad(Bz_func, 0, self.R, 0, 2 * np.pi)
            except IntegrationWarning:
                print('Integral is divergent')
            fluxes[j] = integral
        
        contig = np.ascontiguousarray
        plot_points = np.array(plot_points).reshape(-1, 3)
        Bzs = np.array(Bzs).reshape(-1, 3)
        print(plot_points.shape)
        pointsToVTK('flux_int_points', contig(plot_points[:, 0]),
                    contig(plot_points[:, 1]), contig(plot_points[:, 2]),
                    data={"Bz": (contig(Bzs[:, 0]), 
                                 contig(Bzs[:, 1]),
                                 contig(Bzs[:, 2]),),},)
                
        self.psi = fluxes  # * self.R / 2.0 * np.pi
        