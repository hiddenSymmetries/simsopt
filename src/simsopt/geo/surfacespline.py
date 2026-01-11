from .._core import Optimizable
from .._core.optimizable import DOFs
from ..geo import SurfaceRZFourier
from ..mhd import Vmec
from ..util.mpi import MpiPartition

import numpy as np
from scipy.interpolate import CubicSpline, Akima1DInterpolator, CloughTocher2DInterpolator
import matplotlib. pyplot as plt
from scipy.interpolate import griddata
from scipy.optimize import fsolve, bisect
from simsopt.util import MpiPartition

import matplotlib
matplotlib.use('QtAgg')

mpi = MpiPartition()

__all__ = ['CrossSectionFixedZeta', 'PseudoAxis', 'SurfaceBSpline']

class CrossSectionFixedZeta(Optimizable):
    r"""
    Class for toroidal control point cross section, using local polar
    coordinates to describe control points. 

    Initializes to a polygon of control points, approximating a 
    circle. 
    
    Args:
        zeta_index: index of the cross-section as tracked by the 
        parent `PseudoAxisSurface`
        n_ctrl_pts: number of control points per cross-section
        (includes endpoints for z-symmetric cross sections)
        z_sym: whether the cross section is up-down symmetric
        equispaced: whether to fix the polar angle degrees of
        freedom of the control vectors
        default_r: default radius
        nurbs: whether or not to use nurbs
    """
    def __init__(
        self,
        zeta_index,
        n_ctrl_pts=7,
        z_sym=False,
        equispaced=False,
        default_r=0.3,
        nurbs=False
    ):
        self.zeta_index = zeta_index
        self.n_ctrl_pts = n_ctrl_pts 
        self.z_sym = z_sym
        self.nurbs = nurbs
        if z_sym:
            n_pts = (n_ctrl_pts // 2) + 1
            max_angle = np.pi
            # default behaviour: circular 
            r_ctrl = default_r*np.ones(n_pts)
            theta_ctrl = np.arange(0, n_pts, 1) * 2*np.pi/n_ctrl_pts
            w_ctrl = 0.5*np.ones(n_pts)
            cs_dofs = np.concatenate([r_ctrl, theta_ctrl, w_ctrl])

            assert len(cs_dofs) == 3*((n_ctrl_pts // 2) + 1)
            if not n_ctrl_pts % 2:
                assert theta_ctrl[-1] == np.pi, f'theta_ctrl: {theta_ctrl}'
        else:
            n_pts = n_ctrl_pts
            max_angle = 2*np.pi
            # default behaviour: circular 
            r_ctrl = default_r*np.ones(n_pts)
            theta_ctrl=np.linspace(0, max_angle, n_pts+1)[:-1]
            w_ctrl = 0.5*np.ones(n_pts)
            cs_dofs = np.concatenate([r_ctrl, theta_ctrl, w_ctrl])

            assert len(cs_dofs) == 3*n_ctrl_pts

        self.r_ctrl = cs_dofs[:n_pts]
        self.theta_ctrl = cs_dofs[n_pts:2*n_pts:]
        self.w_ctrl = cs_dofs[2*n_pts:]
        self.n_pts = n_pts

        # naming dofs
        names = self._name_dofs(n_pts)

        if equispaced:
            dofs = DOFs(
                cs_dofs,
                names,
                [True]*(n_pts) + [False]*(n_pts) + [nurbs]*(n_pts),
                [0] * (n_pts) + (np.linspace(0, max_angle, n_pts+1)[:-1] - 0.5*np.linspace(0, max_angle, n_pts+1)[1]).tolist() + [0] * (n_pts),
                [1] * (n_pts) + (np.linspace(0, max_angle, n_pts+1)[1:] - 0.5*np.linspace(0, max_angle, n_pts+1)[1]).tolist() + [1] * (n_pts)
            )
            super().__init__(dofs=dofs)
        else:
            dofs = DOFs(
                cs_dofs,
                names,
                [True]*(n_pts) + [True]*(n_pts) + [False] + [nurbs]*(n_pts-1), # fixing one of the weights in the cross section
                [0] * (n_pts) + (np.linspace(0, max_angle, n_pts+1)[:-1] - 0.5*np.linspace(0, max_angle, n_pts+1)[1]).tolist() + [0] * (n_pts),
                [1] * (n_pts) + (np.linspace(0, max_angle, n_pts+1)[1:] - 0.5*np.linspace(0, max_angle, n_pts+1)[1]).tolist() + [1] * (n_pts)
            )
            super().__init__(dofs=dofs)     
            if z_sym:
                assert self.get('theta_0') == 0, f"theta_0 = {self.get('theta_0')}"
                dofs.fix(f'theta_0')
                if not n_ctrl_pts % 2:
                    assert self.get(f'theta_{n_pts-1}') == np.pi, f"theta_0 = {self.get(f'theta_{n_pts-1}')}"
                    dofs.fix(f'theta_{n_pts-1}')
                    new_bounds = np.linspace(0, max_angle, n_pts-1)
                    for k in range(1, n_pts-1):
                        dofs.update_bounds(f'theta_{k}', (new_bounds[k-1], new_bounds[k]))

    def _name_dofs(self, n_pts):
        namelist = []
        for i in range(n_pts):
            namelist.append(f'r_{i}')
        for i in range(n_pts):
            namelist.append(f'theta_{i}')
        for i in range(n_pts):
            namelist.append(f'w_{i}')
        return namelist

    def get_r_ctrl_full(self):
        #print(f'n_pts: {len(self.r_ctrl)}')
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
                full = np.concatenate((self.theta_ctrl, 2*np.pi-self.theta_ctrl[:0:-1]))
            else:
                full = np.concatenate((self.theta_ctrl, 2*np.pi-self.theta_ctrl[-2:0:-1]))
        else:
            full = self.theta_ctrl
        return full

    def get_w_ctrl_full(self):
        #print(f'n_pts: {len(self.r_ctrl)}')
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
            ws_flipped = self.w_ctrl[:0:-1]
            theta_flipped = 2*np.pi - np.insert(self.theta_ctrl[:0:-1], 0, self.theta_ctrl[0])
            if self.nurbs:
                dofs_flipped = np.concatenate((r_flipped, theta_flipped, ws_flipped))
            else:
                dofs_flipped = np.concatenate((r_flipped, theta_flipped))
            # print(f'flipped dofs: {dofs_flipped}')
            flipped_cs = CrossSectionFixedZeta(
                zeta_index=self.zeta_index,
                n_ctrl_pts=self.n_ctrl_pts,
                z_sym=False,
                nurbs=self.nurbs
            )
            flipped_cs.x = dofs_flipped
            return flipped_cs

class PseudoAxis(Optimizable):
    r"""
    Class for a pseudo-axis around which to build a spline surface.
    The number of dofs includes BOTH endpoints. Control points are 
    vectors described in cylindrical coordinates. 

    Args:
        n_ctrl_pts: number of control points
        nfp: number of field periods
        stellsym: whether axis is stellarator symmetric
        axis_angles_fixed: whether to freeze dofs for polar
        angle of control points. 
    """
    def __init__(
        self, 
        n_ctrl_pts=2,
        nfp=2,
        stellsym=True,
        axis_angles_fixed=False
    ):
        self.n_ctrl_pts=n_ctrl_pts
        self.stellsym=stellsym
        self.nfp=nfp

        if not stellsym:
            max_angle=2*np.pi/nfp
        else:
            max_angle=np.pi/nfp
        
        # default behaviour: circular axis
        r_ctrl = np.ones(n_ctrl_pts)
        z_ctrl = np.zeros(n_ctrl_pts)
        zeta_ctrl = np.linspace(0, max_angle, n_ctrl_pts)       
        axisdofs = np.concatenate([r_ctrl, z_ctrl, zeta_ctrl])

        self.r_ctrl = axisdofs[:n_ctrl_pts]
        self.z_ctrl = axisdofs[n_ctrl_pts:2*n_ctrl_pts]
        self.zeta_ctrl = axisdofs[2*n_ctrl_pts:]

        # naming dofs
        names = self._name_dofs()
        dofs = DOFs(
            axisdofs,
            names,
            [True]*len(r_ctrl) + [True]*len(z_ctrl) + [not axis_angles_fixed]*len(zeta_ctrl),
            [0.3] * len(r_ctrl) + [-1] * len(z_ctrl) + (np.linspace(0, max_angle, len(zeta_ctrl)) - (0.5*max_angle)/(len(zeta_ctrl)-1)).tolist(),
            [1.2] * len(r_ctrl) + [1] * len(z_ctrl) + (np.linspace(0, max_angle, len(zeta_ctrl)) + (0.5*max_angle)/(len(zeta_ctrl)-1)).tolist(), 
        )

        dofs.set('r_axis_0', 1)
        dofs.set('z_axis_0', 0)
        dofs.set('zeta_axis_0', 0)
        dofs.fix('r_axis_0')
        dofs.fix('z_axis_0')
        dofs.fix('zeta_axis_0')
        if stellsym:
            dofs.set(f'z_axis_{n_ctrl_pts - 1}', 0)
            dofs.set(f'zeta_axis_{n_ctrl_pts - 1}', np.pi/self.nfp)
            dofs.fix(f'z_axis_{n_ctrl_pts - 1}')
            dofs.fix(f'zeta_axis_{n_ctrl_pts - 1}')

        super().__init__(dofs=dofs)

    def __call__(
            self,
            zeta,
            method='bspline'
    ):
        if method=='bspline':
            return self.axis_spline_callable(zeta)
        
        else:
            # defining interpolant
            if self.stellsym:
                r_ctrl=np.append(self.r_ctrl, self.r_ctrl[-2::-1])
                z_ctrl=np.append(self.z_ctrl, -self.z_ctrl[-2::-1])
                zeta_ctrl=np.append(self.zeta_ctrl, (2*np.pi/self.nfp)-self.zeta_ctrl[-2::-1])

            print(f'zeta_ctrl: {zeta_ctrl}')

            R = Akima1DInterpolator(
                x=zeta_ctrl,
                y=r_ctrl
            )
            z = Akima1DInterpolator(
                x=zeta_ctrl,
                y=z_ctrl
            )

            zeta = zeta % (2*np.pi/self.nfp)
            return R(zeta), z(zeta)

    def axis_spline_callable(
            self,
            x,
            p=3,
            n_interp=200,
            plot=False
        ):
        x = x%(2*np.pi/self.nfp)
        if self.stellsym:
            r_ctrl_1fp=np.append(self.r_ctrl, self.r_ctrl[-2::-1])[:-1]
            z_ctrl_1fp=np.append(self.z_ctrl, -self.z_ctrl[-2::-1])[:-1]
            zeta_ctrl_1fp=np.append(self.zeta_ctrl, (2*np.pi/self.nfp)-self.zeta_ctrl[-2::-1])[:-1]
        
        r_ctrl = np.tile(r_ctrl_1fp, self.nfp)
        z_ctrl = np.tile(z_ctrl_1fp, self.nfp)
        zeta_ctrl = np.concatenate(
            [zeta_ctrl_1fp + n*2*np.pi/self.nfp for n in range(self.nfp)]
        )
        x_ctrl = r_ctrl*np.cos(zeta_ctrl)
        y_ctrl = r_ctrl*np.sin(zeta_ctrl)

        xyz_list = np.vstack((x_ctrl, y_ctrl, z_ctrl)).T#[:-1]
        centroids_im = np.array(xyz_list)

        # a basis
        n = centroids_im.shape[0] - 1
        centroids_im = np.concatenate([centroids_im, centroids_im[:p, :]], axis = 0)
        n_knots_a = n + 2*p + 2

        interval_a = (2*np.pi) / (n + 1)
        knots_a = -p * interval_a + np.arange(0, n + 2*p + 2) * interval_a

        #assert len(knots_a) == n + p + 2
        assert np.isclose(knots_a[p], 0)
        assert np.isclose(knots_a[n + p + 1], 2*np.pi), f'knots_a[n + p + 1]: {knots_a[n + p + 1]}'

        domain = np.linspace(0, 2*np.pi, n_interp, endpoint=False)

        a_basis = b_p(knots_a, p, domain)

        axis = np.einsum('ix,ti->tx', centroids_im, a_basis)
        x_axis = axis[:, 0]
        y_axis = axis[:, 1]
        z_axis = axis[:, 2]

        zeta_axis = np.arctan2(y_axis, x_axis) % (2*np.pi)

        inds = zeta_axis.argsort()
        sorted_zeta_axis = zeta_axis[inds]
        sorted_x_axis = x_axis[inds]
        sorted_y_axis = y_axis[inds]
        sorted_z_axis = z_axis[inds]

        R_axis = np.sqrt(sorted_x_axis**2 + sorted_y_axis**2)
        ext_R_axis = np.tile(R_axis, 3)
        ext_z_axis = np.tile(sorted_z_axis, 3)
        ext_zeta = np.concatenate([sorted_zeta_axis-2*np.pi, sorted_zeta_axis, sorted_zeta_axis+2*np.pi])

        R_interpolant = CubicSpline(
            x = ext_zeta,
            y = ext_R_axis
        )(x)

        z_interpolant = CubicSpline(
            x = ext_zeta, 
            y = ext_z_axis
        )(x)

        if plot:
            fig = plt.figure("axis_bspline",figsize=(14,7))
            ax = fig.add_subplot(projection='3d',azim=0, elev=90)
            ax.scatter(x_ctrl, y_ctrl, z_ctrl)
            ax.scatter(x_axis, y_axis, z_axis)
            ax.scatter(R_interpolant*np.cos(x), R_interpolant*np.sin(x), z_interpolant)
            ax.set_box_aspect((1, 1, 1))
            ax.set_ylim(-1, 1)
            ax.set_xlim(-1, 1)
            ax.set_zlim(-1, 1)
            plt.show()  

        return R_interpolant, z_interpolant

    def _name_dofs(self):
        name_list = [f'{j}_axis_{i}' for j in ('r', 'z', 'zeta') for i in range(self.n_ctrl_pts)]
        return name_list
       
class SurfaceBSpline(Optimizable):#(sopp.Surface, Surface):#
    r"""
    Class for a B-spline surface, as described in Ali et. al (manuscript
    in progress). The main benefits of this representation are
    threefold:
    - Easy to box-bound to a space of diverse but feasible
    stellarator shapes
    - Local control 
    - Can be constrained to be unique

    The `PseudoAxisSurface` class is a composite of two other
    classes: the `PseudoAxis`, which is a B-spline curve intended to
    be constrained to lie in the interior of the control points, and
    the control point cross sections (`CrossSectionFixedZeta` or
    `CrossSectionFixedZetaCartesian`), which define the control
    points in local cartesian or polar coordinates in planes of
    constant toroidal angle, respectively. 
    """
    def __init__(
        self, 
        axis_points=4, 
        points_per_cs=7,
        cs_equispaced=True,
        axis_angles_fixed=False,
        cs_global_angle_free=False,
        n_cs=2,
        nfp=2,
        M=12,
        N=12,
        p_u = 2,
        p_v = 2,
        default_r=0.5,
        stellsym=True,
        cs_basis='polar',
        nurbs=False
    ):
        '''
        axis_points: number of points for the axis spline
        points_per_cs: number of points per cross section
        n_cs: number of toroidal cross sections per half field period
        '''
        if stellsym:
            max_angle = np.pi/nfp
        else:
            raise NotImplementedError
            max_angle = 2 * np.pi
        self.M = M
        self.N = N
        self.nfp = nfp
        self.n_cs = n_cs
        self.points_per_cs = points_per_cs
        self.p_u = p_u
        self.p_v = p_v
        self.axis_angles_fixed = axis_angles_fixed
        self.cs_basis = cs_basis
        self.cs_equispaced = cs_equispaced
        self.cs_global_angle_free = cs_global_angle_free
        self.axis_points = axis_points

        # create equidistant points in zeta
        cs_zeta = np.linspace(0, max_angle, n_cs)
        # all angles zero
        cs_angles = np.zeros(n_cs)

        self.cs_zeta = cs_zeta
        self.cs_angles = cs_angles

        cs_dofs = np.array([None] * n_cs)

        if stellsym & np.all(cs_dofs != None):
            assert len(cs_dofs[0]) == 2*((points_per_cs//2) + 1)
            assert len(cs_dofs[-1]) == 2*((points_per_cs//2) + 1)

        dofs = np.append(self.cs_zeta, self.cs_angles)

        names = [f'cs_zeta{i}' for i in range(n_cs)] + [f'cs_angle{i}' for i in range(n_cs)]
        dofs = DOFs(
            dofs,
            names,
            [False]*n_cs + [cs_global_angle_free]*n_cs,
            [(n-1)*max_angle/(n_cs-2) for n in range(n_cs)] + [-2*np.pi/points_per_cs] * n_cs,
            [(n)*max_angle/(n_cs-2) for n in range(n_cs)] + [2*np.pi/points_per_cs] * n_cs
        )

        dofs.fix('cs_angle0')
        dofs.fix(f'cs_angle{n_cs-1}')
        dofs.fix('cs_zeta0')
        dofs.fix(f'cs_zeta{n_cs-1}')

        self.axis = PseudoAxis(
                n_ctrl_pts=axis_points,
                nfp=nfp,
                stellsym=stellsym,
                axis_angles_fixed=axis_angles_fixed
            )

        self.cs_list = []

        for i in range(n_cs):
            if cs_basis == 'polar':
                cross_section = CrossSectionFixedZeta(
                    zeta_index=i,
                    n_ctrl_pts=points_per_cs,
                    equispaced=cs_equispaced,
                    default_r=default_r,
                    z_sym=((i == 0) or (i == n_cs-1)),
                    nurbs=nurbs
                )
            elif cs_basis == 'cartesian':
                raise NotImplementedError('see dev branch')
            self.cs_list.append(cross_section)

        Optimizable.__init__(self, dofs=dofs, depends_on=[self.axis] + self.cs_list)

    def get_cs_zeta_angle(self):
        zeta_list = np.array([self.get(f'cs_zeta{i}') for i in range(self.n_cs)])
        cs_angle_list = np.array([self.get(f'cs_angle{i}') for i in range(self.n_cs)])
        return zeta_list, cs_angle_list

    def get_xyz_full_device(
            self,
            return_w=False
        ):
        '''
        Return a list of (X,Y,Z) tuples for control in each cross section points over the entire device domain
        '''
        point_list = []
        w_list = []

        cs_zeta, cs_angles = self.get_cs_zeta_angle()

        cs_zeta_1fp = np.append(cs_zeta, (2*np.pi/self.nfp)-cs_zeta[-2:0:-1])
        cs_zeta_full = np.concatenate([cs_zeta_1fp + n*(2*np.pi/self.nfp) for n in range(self.nfp)])

        r_paxis, z_paxis = self.axis(cs_zeta_full)

        cs_list_1fp = [cs if (i//self.n_cs) == 0 else cs.flipped() for i, cs in enumerate(self.cs_list + self.cs_list[-2:0:-1])] #!!!!
        cs_list_full = np.tile(cs_list_1fp, self.nfp)

        cs_angle_1fp = [angle if (i//self.n_cs) == 0 else -angle for i, angle in enumerate(np.append(cs_angles, cs_angles[-2:0:-1]))]
        cs_angle_full = np.tile(cs_angle_1fp, self.nfp)

        for i, cs in enumerate(cs_list_full):
            # point by point in each cross section
            cs_r_ctrl_full = cs.get_r_ctrl_full()
            cs_theta_ctrl_full = cs.get_theta_ctrl_full()
            cs_w_ctrl_full = cs.get_w_ctrl_full()
            zeta = cs_zeta_full[i]
            cs_pointlist = []
            for j, r_cs in enumerate(cs_r_ctrl_full):
                theta = cs_theta_ctrl_full[j]
                point_x = (r_paxis[i] - r_cs * np.cos(theta+cs_angle_full[i]))*np.cos(zeta)
                point_y = (r_paxis[i] - r_cs * np.cos(theta+cs_angle_full[i]))*np.sin(zeta)
                point_z = z_paxis[i] + r_cs * np.sin(theta+cs_angle_full[i])
                new_point = np.array([point_x, point_y, point_z]).T
                cs_pointlist.append(new_point)
            point_list.append(np.array(cs_pointlist))
            w_list.append(cs_w_ctrl_full)
        
        if return_w:
            return point_list, w_list
        else:
            return point_list

    def get_xyz_centroids(self):
        '''
        Return a list of (X,Y,Z) tuples for control in each cross section points over the entire device domain
        '''
        cs_zeta, cs_angles = self.get_cs_zeta_angle()
        point_list = []
        # cross section by cross section
        cs_zeta_1fp = np.append(cs_zeta, (2*np.pi/self.nfp)-cs_zeta[-2:0:-1])
        cs_zeta_full = np.concatenate([cs_zeta_1fp + n*(2*np.pi/self.nfp) for n in range(self.nfp)])

        r_paxis, z_paxis = self.axis(cs_zeta_full)
        #print(self.axis(cs_zeta_full))

        cs_list_1fp = [cs if (i//self.n_cs) == 0 else cs.flipped() for i, cs in enumerate(self.cs_list + self.cs_list[-2:0:-1])] #!!!!
        cs_list_full = np.tile(cs_list_1fp, self.nfp)

        cs_angle_1fp = [angle if (i//self.n_cs) == 0 else -angle for i, angle in enumerate(np.append(cs_angles, cs_angles[-2:0:-1]))]
        cs_angle_full = np.tile(cs_angle_1fp, self.nfp)

        for i, cs in enumerate(cs_list_full):
            # point by point in each cross section
            zeta = cs_zeta_full[i]
            cs_pointlist = []
            point_x = 0
            point_y = 0
            point_z = 0
            cs_r_ctrl_full = cs.get_r_ctrl_full()
            cs_theta_ctrl_full = cs.get_theta_ctrl_full()
            for j, r_cs in enumerate(cs_r_ctrl_full):
                theta = cs_theta_ctrl_full[j]
                point_x += (r_paxis[i] - r_cs * np.cos(theta+cs_angle_full[i]))*np.cos(zeta)
                point_y += (r_paxis[i] - r_cs * np.cos(theta+cs_angle_full[i]))*np.sin(zeta)
                point_z += z_paxis[i] + r_cs * np.sin(theta+cs_angle_full[i])
            new_point = np.array([point_x, point_y, point_z]).T
            new_point = new_point/len(cs_r_ctrl_full)
            point_list.append(np.array(new_point))
        return point_list

    def get_rtz_full_device(self):
        '''
        Return toroidal coordinate (e.g., (r,t,z) tuples) for each control point over the entire
        device domain. 
        '''
        point_list = []

        cs_zeta, cs_angles = self.get_cs_zeta_angle()

        cs_zeta_1fp = np.append(cs_zeta, (2*np.pi/self.nfp)-cs_zeta[-2:0:-1])
        cs_zeta_full = np.concatenate([cs_zeta_1fp + n*(2*np.pi/self.nfp) for n in range(self.nfp)])

        cs_list_1fp = [cs if (i//self.n_cs) == 0 else cs.flipped() for i, cs in enumerate(self.cs_list + self.cs_list[-2:0:-1])] #!!!!
        cs_list_full = np.tile(cs_list_1fp, self.nfp)

        cs_angle_1fp = [angle if (i//self.n_cs) == 0 else -angle for i, angle in enumerate(np.append(cs_angles, cs_angles[-2:0:-1]))]
        cs_angle_full = np.tile(cs_angle_1fp, self.nfp)

        for i, cs in enumerate(cs_list_full):
            # point by point in each cross section
            zeta = cs_zeta_full[i]
            cs_pointlist = []
            cs_r_ctrl_full = cs.get_r_ctrl_full()
            cs_theta_ctrl_full = cs.get_theta_ctrl_full()
            for j, r_cs in enumerate(cs_r_ctrl_full):
                point_r = r_cs
                point_theta = cs_theta_ctrl_full[j]+cs_angle_full[i]
                point_zeta = zeta
                new_point = np.array([point_r, point_theta, point_zeta]).T
                cs_pointlist.append(new_point)
            point_list.append(np.array(cs_pointlist))
        return point_list
   
    def plot(
            self,
            _surf = True,
            _surf_points = False,
            _ctrl_points = True,
            _ctrl_points_full=True,
            _pseudo_axis = True,
            _pseudo_axis_ctrl_pts=True,
            _centroid_axis = True,
            _rtz_vectors = True,
            _RZ_vectors = False,
            ax = None,
            _surf_kwargs = {'alpha':0.3, 'rcount':64, 'ccount':64},
            _surf_points_kwargs = {'color':'k', 'marker':'.'},
            _ctrl_points_kwargs = {'color':'g', 'marker':'.', 'ls':'--'},
            _pseudo_axis_kwargs = {'color':'g', 'ls':'-'},
            _pseudo_axis_ctrl_pts_kwargs={'color':'c', 'marker':'*'},
            _centroid_axis_kwargs = {'color':'r', 'ls':'--'},
            _rtz_vectors_kwargs = {},
            _RZ_vectors_kwargs = {},
        ): 
        if ax == None:  
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        xyz_list = self.get_xyz_full_device()

        # Generating surface
        if _surf or _surf_points:
            ax.set_aspect('equal')
            ax.set_ylim(-1, 1)
            ax.set_xlim(-1, 1)
            ax.set_zlim(-1, 1)

            nu = _surf_kwargs['rcount']
            nv = _surf_kwargs['ccount']

            u = np.linspace(0, 2*np.pi, nu, endpoint = True)
            v = np.linspace(0, 2*np.pi, nv, endpoint = True)
            v_grid, u_grid = np.meshgrid(v, u)

            x_surf, y_surf, z_surf = self.surf_callable(u_grid.flatten(), v_grid.flatten())
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
                for i in range(0, self.nfp * (2*(self.n_cs-1))):
                    ax.plot(
                        np.append(points[i*(self.points_per_cs):(i+1)*(self.points_per_cs), 0], points[i*(self.points_per_cs), 0]),
                        np.append(points[i*(self.points_per_cs):(i+1)*(self.points_per_cs), 1], points[i*(self.points_per_cs), 1]),
                        np.append(points[i*(self.points_per_cs):(i+1)*(self.points_per_cs), 2], points[i*(self.points_per_cs), 2]),
                        **_ctrl_points_kwargs
                    )
            else:              
                for i in range(0, self.n_cs):
                    ax.plot(
                        np.append(points[i*(self.points_per_cs):(i+1)*(self.points_per_cs), 0], points[i*(self.points_per_cs), 0]),
                        np.append(points[i*(self.points_per_cs):(i+1)*(self.points_per_cs), 1], points[i*(self.points_per_cs), 1]),
                        np.append(points[i*(self.points_per_cs):(i+1)*(self.points_per_cs), 2], points[i*(self.points_per_cs), 2]),
                        **_ctrl_points_kwargs
                    )


        #####################################################################
        # plotting vectors from axis to control points
        if _rtz_vectors:
            cs_zeta, cs_angles = self.get_cs_zeta_angle()
            rtz = np.array(self.get_rtz_full_device()).reshape(-1, 3)
            if _ctrl_points_full:
                pass
            else:
                rtz = rtz[:(self.points_per_cs * self.n_cs), :]
            r_ctrl, theta_ctrl, zeta_ctrl = rtz[:,0], rtz[:,1], rtz[:,2]
            r_paxis, z_paxis = self.axis(zeta_ctrl)

            dir_x = np.cos(zeta_ctrl)*(-r_ctrl*np.cos(theta_ctrl))
            dir_y = np.sin(zeta_ctrl)*(-r_ctrl*np.cos(theta_ctrl))
            dir_z = r_ctrl*np.sin(theta_ctrl)

            loc_x = r_paxis*np.cos(zeta_ctrl)
            loc_y = r_paxis*np.sin(zeta_ctrl)
            loc_z = z_paxis

            ax.quiver(loc_x, loc_y, loc_z, dir_x, dir_y, dir_z, **_rtz_vectors_kwargs)

        if _RZ_vectors:
            xyz = np.array(self.get_xyz_full_device()).reshape(-1,3)
            x_ctrl, y_ctrl, z_ctrl = xyz[:,0], xyz[:,1], xyz[:,2]
            ax.quiver(np.zeros_like(x_ctrl), np.zeros_like(x_ctrl), np.zeros_like(x_ctrl), x_ctrl, y_ctrl, z_ctrl, alpha = 0.25)

        #####################################################################
        # plotting axis
        if _pseudo_axis:
            
            x = lambda phi: self.axis(phi)[0] * np.cos(phi)
            y = lambda phi: self.axis(phi)[0] * np.sin(phi)
            for i in range(1, self.nfp+1):
                phi = np.linspace((i-1)*2*np.pi/self.nfp, i*2*np.pi/self.nfp, 200)
                ax.plot(x(phi), y(phi), self.axis(phi)[1], **_pseudo_axis_kwargs)
            rax_ctrl=np.append(self.axis.r_ctrl, self.axis.r_ctrl[-2:0:-1])
            rax_ctrl=np.tile(rax_ctrl, self.nfp)
            zax_ctrl=np.append(self.axis.z_ctrl, -self.axis.z_ctrl[-2:0:-1])
            zax_ctrl=np.tile(zax_ctrl, self.nfp)
            zetaax_ctrl_1fp=np.append(self.axis.zeta_ctrl, (2*np.pi/self.nfp)-self.axis.zeta_ctrl[-2:0:-1])
            zetaax_ctrl = np.copy(zetaax_ctrl_1fp)
            for i in range(1, self.nfp):
                zetaax_ctrl = np.append(zetaax_ctrl, zetaax_ctrl_1fp+i*(2*np.pi/self.nfp))

            xax_ctrl = rax_ctrl * np.cos(zetaax_ctrl)
            yax_ctrl = rax_ctrl * np.sin(zetaax_ctrl)
            zax_ctrl = zax_ctrl

            xax_ctrl = np.append(xax_ctrl, xax_ctrl[0])
            yax_ctrl = np.append(yax_ctrl, yax_ctrl[0])
            zax_ctrl = np.append(zax_ctrl, zax_ctrl[0])

        if _pseudo_axis_ctrl_pts:
            ax.plot(xax_ctrl, yax_ctrl, zax_ctrl, **_pseudo_axis_ctrl_pts_kwargs)
        
        #####################################################################
        # Centroid axis
        if _centroid_axis:
            # x_centroid = np.zeros_like(a_basis[:, i])
            # x_centroid = np.zeros_like(a_basis[:, i])
            na = nv
            a = np.linspace(0, 2*np.pi, na)
            x_centroid, y_centroid, z_centroid = self.centroid_axis_callable(a)

            ax.plot(x_centroid, y_centroid, z_centroid, **_centroid_axis_kwargs)
        ax._axis3don = False

    def surf_callable(
            self, 
            u, 
            v,
        ):
        '''
        Evaluate the spline surface at (a set of) u, v pairs within a field period. 
        '''
        point_list, w_list = self.get_xyz_full_device(return_w=True)

        p_u = self.p_u
        p_v = self.p_v

        control_points_jim = np.array(point_list)            
        w_list_jim = np.array(w_list)
        
        n_u = control_points_jim.shape[1] - 1
        n_v = control_points_jim.shape[0] - 1

        # u basis
        control_points_jim = np.concatenate([control_points_jim, control_points_jim[:, :p_u, :]], axis = 1)
        w_list_jim = np.concatenate([w_list_jim, w_list_jim[:, :p_u]], axis = 1)
        n_knots_u = n_u + 2*p_u + 2

        interval_u = (2*np.pi) / (n_u + 1)
        knots_u = -p_u*interval_u + np.arange(0, n_u+2*p_u+2)*interval_u

        #assert(len(knots_u)==n_knots_u), f"len(knots_u) = {len(knots_u)}"
        assert np.isclose(knots_u[p_u], 0), f"knots_u[p_u] = {knots_u[p_u]}"
        assert np.isclose(knots_u[n_u + p_u + 1], 2 * np.pi), f"knots_u[n_u + p_u + 1] = {knots_u[n_u + p_u + 1]}"
        u_basis = b_p(knots_u, p_u, u)

        # v basis - closed
        
        control_points_jim = np.concatenate([control_points_jim, control_points_jim[:p_v, :, :]], axis = 0)
        w_list_jim = np.concatenate([w_list_jim, w_list_jim[:p_v, :]], axis = 0)
        n_knots_v = n_v + 2*p_v + 2

        interval_v = (2*np.pi) / (n_v + 1)
        knots_v = -p_v*interval_v + np.arange(0, n_v+2*p_v+2)*interval_v

        #assert(len(knots_v)==n_knots_v), f"len(knots_v) = {len(knots_v)}"
        assert np.isclose(knots_v[p_v],0), f"knots_v[p_v] = {knots_v[p_v]}"
        assert np.isclose(knots_v[n_v + p_v + 1], 2*np.pi), f"knots_v[n_v - p_v + 1] = {knots_v[n_v + p_v + 1]}"
        v_basis = b_p(knots_v, p_v, v)

        trimmed_ctrl_pts_jim = control_points_jim[:n_v+p_v+1, :n_u+p_u+1, :]
        trimmed_weights_ji = w_list_jim[:n_v+p_v+1, :n_u+p_u+1]
        tp_basis = np.einsum('xj,xi->xji', v_basis, u_basis)
        w_tp_basis = np.einsum('xji,ji->xji',tp_basis, trimmed_weights_ji)
        summed_w_tp_basis = np.einsum('xji->x',w_tp_basis)
        nurbs_tp_basis = np.einsum('xji,x->xji', w_tp_basis, 1/summed_w_tp_basis)

        surf = np.einsum('xji,jim->xm', nurbs_tp_basis, trimmed_ctrl_pts_jim)
        x_surf = surf[:, 0]
        y_surf = surf[:, 1]
        z_surf = surf[:, 2]

        return x_surf, y_surf, z_surf

    def centroid_axis_callable(
            self,
            a,
        ):
        xyz_list = self.get_xyz_centroids()
        p_a = self.p_v

        centroids_im = np.array(xyz_list)

        # a basis
        n_a = centroids_im.shape[0] - 1
        centroids_im = np.concatenate([centroids_im, centroids_im[:p_a, :]], axis = 0)
        n_knots_a = n_a + 2*p_a + 2

        interval_a = (2*np.pi) / (n_a+1)
        knots_a = -p_a*interval_a + np.arange(0, n_a+2*p_a+2)*interval_a

        assert np.isclose(knots_a[p_a], 0), f"knots_u[p_u] = {knots_a[p_a]}"
        assert np.isclose(knots_a[n_a + p_a + 1], 2 * np.pi), f"knots_u[n_u + p_u + 1] = {knots_a[n_a + p_a + 1]}"

        a_basis = b_p(knots_a, p_a, a)

        x_centroid = np.zeros_like(a_basis[:, 0])
        y_centroid = np.zeros_like(a_basis[:, 0])
        z_centroid = np.zeros_like(a_basis[:, 0])

        for i in range(n_a+p_a+1):
            x_centroid += centroids_im[i, 0] * a_basis[:, i]
            y_centroid += centroids_im[i, 1] * a_basis[:, i]
            z_centroid += centroids_im[i, 2] * a_basis[:, i]

        return x_centroid, y_centroid, z_centroid

    def fsolve_axis_from_zetas(
            self, 
            zeta_surf,
            offset
        ):
        def f0(a, target):
            a = a % (2 * np.pi)
            x, y, z = self.centroid_axis_callable(a)
            zeta = np.arctan2(y, x) % (2*np.pi)
            return zeta - target
        def f1(a, target):
            a = a % (2 * np.pi)
            x, y, z = self.centroid_axis_callable(a)
            dx_da, dy_da, dz_da = self.centroid_axis_derivative_callable(a)
            return (1/(1+(y/x)**2) * ((dy_da/x) - (y/x**2)*dx_da))
        a_star = []
        for target in zeta_surf.flatten():
            a_star.append(fsolve(
                func = f0, 
                x0 = (target - offset) % (2*np.pi),
                fprime = f1,
                args = target
            ))
        a_star = np.array(a_star).flatten() % (2*np.pi)
        
        x_star, y_star, z_star = self.centroid_axis_callable(a_star)

        return x_star, y_star, z_star
    
    def uniform_tz_interp(
            self,
            nu = 32,
            nv = 32,
            nu_interp = 64,
            nv_interp = 64,
            plot=False,
            _fsolve=False,
        ):
        '''
        Return R, Z values computed on a (theta_a, zeta) grid, where theta_a is the conventional polar angle
        about the centroid, and zeta is the toroidal polar angle. First evaluates the R, Z values on a uniform (u,v) grid,
        then interpolates to obtain R, Z on a (theta_p, zeta) grid, where theta_p is the polar angle about the centroid
        in a plane of constant zeta. The points are mirrored to make sure that the collocation points themselves are
        stellarator symmetric. Finally, another interpolation to finally evaluate R, Z on a (theta_a, zeta)
        grid is performed. 

        Args:
            nu,nv: The shape of the outputted array is (nu, nv)
            nu_interp,nv_interp: The (u, v) grid is generated with the shape (nu_interp, nv_interp)
            nu_intermediate, nv_intermediate: The intermediate grid with polar theta_p has the shape (nu_eq, 2*nv_eq + 1). 
            note that nv_intermediate is the number of points in a half-field period
            plot: Boolean to turn on plotting of the points making up the equal arclength grid
        '''

        # Creating grid to interpolate u and v on 

        u = np.linspace(0, 2*np.pi, nu_interp, endpoint = True)
        v = np.linspace(0, 2*np.pi, nv_interp, endpoint = True)
        v_grid, u_grid = np.meshgrid(v, u)

        x_surf, y_surf, z_surf = self.surf_callable(u_grid.flatten(), v_grid.flatten())
        print(f'type(nv_interp): {type(nv_interp)}')
        print(f'type(nu_interp): {type(nu_interp)}')

        x_surf = x_surf.reshape(nu_interp, nv_interp)
        y_surf = y_surf.reshape(nu_interp, nv_interp)
        z_surf = z_surf.reshape(nu_interp, nv_interp)

        zeta_surf= np.arctan2(y_surf, x_surf) % (2*np.pi)
        R_surf = np.sqrt(x_surf**2 + y_surf**2)

        x_axis, y_axis, z_axis = self.centroid_axis_callable(np.linspace(0, 2*np.pi, nv_interp, endpoint=False))

        R_axis = np.sqrt(x_axis**2 + y_axis**2)

        theta_surf = np.arctan2(z_surf-z_axis, R_surf-R_axis) % (2*np.pi)

        if nv%(2*self.nfp) != 0:
            raise ValueError('nv_intermediate must be divisible by 2*nfp. ')

        nu_tz = nu
        nv_tz = nv//(2*self.nfp)

        tz_points = np.vstack((
            np.concatenate([theta_surf.flatten()-2*np.pi, theta_surf.flatten(), theta_surf.flatten()+2*np.pi]),
            np.tile(zeta_surf.flatten(), 3)
        ))
        zeta_eval, theta_eval = np.meshgrid(np.linspace(np.pi/self.nfp, 2*np.pi/self.nfp, nv_tz+1, endpoint=True), np.linspace(0, 2*np.pi, nu_tz, endpoint=False))
        eval_grid = np.vstack((theta_eval.flatten(), zeta_eval.flatten()))

        R_tz_callable = CloughTocher2DInterpolator(
            points=tz_points.T,
            values=np.tile(R_surf.flatten(), 3),
        )
        z_tz_callable = CloughTocher2DInterpolator(
            points=tz_points.T,
            values=np.tile(z_surf.flatten(), 3),
        )

        R_on_tz_grid = R_tz_callable(eval_grid.T).reshape(nu_tz, nv_tz+1)
        z_on_tz_grid = z_tz_callable(eval_grid.T).reshape(nu_tz, nv_tz+1)

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
            zeta_cs = np.linspace(0, 2*np.pi/self.nfp,num=nzeta_cs,endpoint=True)
            theta_cs_eval = np.linspace(0, 2*np.pi, 64)

            fig = plt.figure("Poincare Plots",figsize=(14,7))
            fig.patch.set_facecolor('white')
            plt.subplot(numRows,numCols,plotNum)

            plotNum += 1
            for ind in range(nzeta_cs):
                if zeta_cs[ind] >= np.pi/self.nfp:
                    plt.subplot(numRows,numCols,ind+1)
                    plt.title(r'$\phi =$' + str(zeta_cs[ind]))
                    plt.gca().set_aspect('equal',adjustable='box')
                    plt.plot()
                    point = np.vstack((theta_cs_eval, zeta_cs[ind]*np.ones_like(theta_cs_eval))).T
                    R_cs = R_tz_callable(point)
                    z_cs = z_tz_callable(point)
                    plt.plot(R_cs, z_cs, 'r--')
                    plt.gca().set_aspect('equal',adjustable='box')
                    plt.xlabel('R')
                    plt.ylabel('Z')
                else:
                    plt.subplot(numRows,numCols,ind+1)
                    plt.title(r'$\phi =$' + str(zeta_cs[ind]))
                    plt.gca().set_aspect('equal',adjustable='box')
                    plt.plot()
                    point = np.vstack((theta_cs_eval, (np.pi-zeta_cs[ind])*np.ones_like(theta_cs_eval))).T
                    R_cs = R_tz_callable(point)[::-1]
                    z_cs = -z_tz_callable(point)[::-1]
                    plt.plot(R_cs, z_cs, 'r--')
                    plt.gca().set_aspect('equal',adjustable='box')
                    plt.xlabel('R')
                    plt.ylabel('Z')

        if plot:
            x_on_tz_grid = R_on_tz_grid * np.cos(zeta_eval)
            y_on_tz_grid = R_on_tz_grid * np.sin(zeta_eval)
            z_on_tz_grid = z_on_tz_grid
            #for theta, i in enumerate(ulist):

            ax.scatter(x_on_tz_grid, y_on_tz_grid, z_on_tz_grid)
            ax.set_box_aspect((1, 1, 1))
            ax.set_ylim(-1, 1)
            ax.set_xlim(-1, 1)
            ax.set_zlim(-1, 1)
        
        R_flipped = np.roll(R_on_tz_grid[::-1, -1:0:-1], 1, axis=0)
        z_flipped = np.roll(-z_on_tz_grid[::-1, -1:0:-1], 1, axis = 0)
        R_1fp = np.concatenate([R_flipped, R_on_tz_grid[:, :-1]], axis = 1)
        z_1fp = np.concatenate([z_flipped, z_on_tz_grid[:, :-1]], axis = 1)
        R_full = np.concatenate([R_1fp]*self.nfp, axis=1)
        z_full = np.concatenate([z_1fp]*self.nfp, axis=1)
        zeta_full, theta_full = np.meshgrid(np.linspace(0, 2*np.pi, nv, endpoint=False), np.linspace(0, 2*np.pi, nu, endpoint=False))  

        # print(theta_full)

        return R_full.T, z_full.T, zeta_full, theta_full

    def arclength_tz_interp(
            self,
            nu = 32,
            nv = 32,
            nu_interp = 64,
            nv_interp = 64,
            plot=False,
            ax=None,
            _fsolve=False,
        ):
        r"""
        Return R, Z values computed on a (theta_a, zeta) grid,
        where theta_a is the poloidal angle demarking unit arc
        length on the curve, and zeta is the toroidal polar
        angle. First evaluates the R, Z values on a uniform
        (u,v) grid, then interpolates to obtain R, Z on a
        (theta_p, zeta) grid, where theta_p is the polar angle
        about the centroid in a plane of constant zeta. The
        points are mirrored to make sure that the collocation
        points themselves are stellarator symmetric. Finally,
        another interpolation is performed to evaluate R, Z on
        a (theta_a, zeta) grid.

        Args:
            nu,nv: The shape of the outputted array is
            (nu, nv)
            nu_interp,nv_interp: The (u, v) grid is generated
            with the shape (nu_interp, nv_interp)
            nu_intermediate, nv_intermediate: The intermediate
            grid with polar theta_p has the shape
            (nu_eq, 2*nv_eq + 1).
            Note that nv_intermediate is the number of points
            in a half-field period
            plot: Boolean to turn on plotting of the points
            making up the equal arclength grid
        """
        # Creating grid to interpolate u and v on 

        u = np.linspace(0, 2*np.pi, nu_interp, endpoint = True)
        v = np.linspace(0, 2*np.pi, nv_interp, endpoint = True)
        v_grid, u_grid = np.meshgrid(v, u)

        x_surf, y_surf, z_surf = self.surf_callable(u_grid.flatten(), v_grid.flatten())
        x_surf = x_surf.reshape(nu_interp, nv_interp)
        y_surf = y_surf.reshape(nu_interp, nv_interp)
        z_surf = z_surf.reshape(nu_interp, nv_interp)

        zeta_surf= np.arctan2(y_surf, x_surf) % (2*np.pi)
        R_surf = np.sqrt(x_surf**2 + y_surf**2)

        if nv_interp%(2*self.nfp) != 0:
            raise ValueError('nv_intermediate must be divisible by 2*nfp. ')

        nu_uz = nu_interp
        nv_uz = nv_interp//(2*self.nfp)

        u_zeta_points = np.vstack((u_grid.flatten(), zeta_surf.flatten()))
        zeta_eval, theta_eval = np.meshgrid(np.linspace(np.pi/self.nfp, 2*np.pi/self.nfp, nv_uz, endpoint=True), np.linspace(0, 2*np.pi, nu_uz, endpoint=True))  
        eval_grid = np.vstack((theta_eval.flatten(), zeta_eval.flatten()))

        R_uz_callable = CloughTocher2DInterpolator(
            points=u_zeta_points.T,
            values=R_surf.flatten(),
        )
        z_uz_callable = CloughTocher2DInterpolator(
            points=u_zeta_points.T,
            values=z_surf.flatten(),
        )

        R_on_uz_grid = R_uz_callable(eval_grid.T).reshape(nu_uz, nv_uz)
        z_on_uz_grid = z_uz_callable(eval_grid.T).reshape(nu_uz, nv_uz)

        # obtaining centroid axis as a function of zeta
        x_axis, y_axis, z_axis = self.centroid_axis_callable(np.linspace(0, 2*np.pi, nv_uz, endpoint=False))
        zeta_axis = np.arctan2(y_axis, x_axis) % (2*np.pi)
        indices = np.argsort(zeta_axis)
        x_axis = x_axis[indices]
        y_axis = y_axis[indices]
        R_axis = np.sqrt(x_axis**2+y_axis**2)
        z_axis = z_axis[indices]
        zeta_axis = zeta_axis[indices]
        # print(f'zeta_axis: {zeta_axis}')
        axis_zeta_callable=CubicSpline(zeta_axis, np.vstack([R_axis, z_axis]).T)

        zeta_1d_halfgrid = zeta_eval[0, :]
        ulist = []

        # finding theta=0 point

        axis_on_uz_grid = axis_zeta_callable(zeta_eval)
        R_axis_on_uz_grid = axis_on_uz_grid[:, :, 0].reshape(nu_uz, nv_uz)
        z_axis_on_uz_grid = axis_on_uz_grid[:, :, 1].reshape(nu_uz, nv_uz)

        def check_r_lt_raxis(x, zeta):
            xstar = np.array([x, zeta])
            _R = R_uz_callable(xstar)
            _z = z_uz_callable(xstar)
            R_axis, z_axis = axis_zeta_callable(zeta)
            if _R>R_axis:
                return True
            else:
                return False

        for i, zeta in enumerate(zeta_1d_halfgrid):
            zs = (z_on_uz_grid - z_axis_on_uz_grid)[:, i]
            u_eval = theta_eval[:, i]
            switch_indices = np.logical_xor(zs>0, np.roll(zs>0, 1))
            a = u_eval[np.roll(switch_indices, -1)]
            b = u_eval[np.roll(switch_indices, 0)]

            if switch_indices[0]:
                a = u_eval[np.roll(switch_indices, -1)]
                b = np.roll(u_eval[np.roll(switch_indices, 0)], -1)
            else:
                a = u_eval[np.roll(switch_indices, -1)]
                b = u_eval[np.roll(switch_indices, 0)]

            def f(x, zeta):
                xstar = np.array([x, zeta])
                _z = z_uz_callable(xstar)
                z_axis = z_axis_on_uz_grid[0, i] #TODO: LOOK HERE
                return (_z - z_axis) # (theta)
            nfails = 0
            nsucc = 0
            nattempts = 0

            for k, _ in enumerate(a):
                u_theta0, r = bisect(
                    f,
                    a = a[k],
                    b = b[k],
                    args = zeta,
                    full_output=True
                )

                if (r.converged) and check_r_lt_raxis(u_theta0, zeta):
                    ulist.append(u_theta0)
                    nsucc+=1
                    nattempts+=1
                    break
                else:
                    nfails+=1
                    nattempts+=1

            if nfails == len(a):
                u_feasible = u_eval[(R_on_uz_grid - R_axis_on_uz_grid)[:, i]>0]
                zs_feasible = zs[(R_on_uz_grid - R_axis_on_uz_grid)[:, i]>0]
                ulist.append(u_feasible[np.argmin(zs_feasible)])

            assert nsucc+nfails == nattempts
        ulist = np.array(ulist).flatten()
        # print(ulist)

        xstar = np.vstack([ulist, zeta_1d_halfgrid]).T
        R_0 = R_uz_callable(xstar).reshape(1, nv_uz)
        z_0 = z_uz_callable(xstar).reshape(1, nv_uz)

        # reparametrizing on arclength

        # sorting by u, starting from theta=0 point
        theta0_eval = (theta_eval - np.outer(np.ones(nu_uz), ulist))%(2*np.pi)
        sorted_theta0_indices = np.argsort(theta0_eval, axis=0)

        R_on_uz_grid = np.take_along_axis(R_on_uz_grid, sorted_theta0_indices, axis=0)# R_on_arclength_grid[sorted_theta_indices]
        z_on_uz_grid = np.take_along_axis(z_on_uz_grid, sorted_theta0_indices, axis=0)# z_on_arclength_grid[sorted_theta_indices]

        # adding zero arclength point to start of R array
        R_on_uz_grid = np.insert(R_on_uz_grid, 0, R_0, axis = 0)
        z_on_uz_grid = np.insert(z_on_uz_grid, 0, z_0, axis = 0)

        # copying zero arclength point to end of array 
        R_on_uz_grid = np.concatenate([R_on_uz_grid, R_0], axis = 0)
        z_on_uz_grid = np.concatenate([z_on_uz_grid, z_0], axis = 0)

        dist_to_next_u = np.sqrt((R_on_uz_grid[1:, :] - R_on_uz_grid[0:-1, :])**2 + (z_on_uz_grid[1:, :] - z_on_uz_grid[0:-1, :])**2)
        dist_to_next_u = np.insert(dist_to_next_u, 0, np.zeros_like(R_on_uz_grid[0, :]), axis=0)
        u_arclength = np.cumsum(dist_to_next_u, axis = 0)
        col_max = np.outer(np.ones(u_arclength.shape[0]), u_arclength[-1, :])

        u_arclength_normalized = (u_arclength/col_max) * 2*np.pi

        zeta_extended = np.concatenate([zeta_eval, zeta_eval[:2, :]], axis = 0)

        if nv%(2*self.nfp) != 0:
            raise ValueError('nv must be divisible by 2*nfp. ')

        nv_final = nv//(2*self.nfp)
        nu_final = nu

        arclength_points = np.vstack((u_arclength_normalized.flatten(), zeta_extended.flatten()))
        zeta_ffeval, theta_ffeval = np.meshgrid(np.linspace(np.pi/self.nfp, 2*np.pi/self.nfp, nv_final+1, endpoint=True), np.linspace(0, 2*np.pi, nu_final, endpoint=False))  
        eval_grid = np.vstack((theta_ffeval.flatten(), zeta_ffeval.flatten()))

        R_az_callable = CloughTocher2DInterpolator(
            points=arclength_points.T,
            values=R_on_uz_grid.flatten(),
        )
        z_az_callable = CloughTocher2DInterpolator(
            points=arclength_points.T,
            values=z_on_uz_grid.flatten(),
        )

        R_on_az_grid = R_az_callable(eval_grid.T).reshape(nu_final, nv_final+1)
        z_on_az_grid = z_az_callable(eval_grid.T).reshape(nu_final, nv_final+1)
               
        R_flipped = np.roll(R_on_az_grid[::-1, -1:0:-1], 1, axis=0)
        z_flipped = np.roll(-z_on_az_grid[::-1, -1:0:-1], 1, axis = 0)
        R_1fp = np.concatenate([R_flipped, R_on_az_grid[:, :-1]], axis = 1)
        z_1fp = np.concatenate([z_flipped, z_on_az_grid[:, :-1]], axis = 1)
        R_full = np.concatenate([R_1fp]*self.nfp, axis=1)
        z_full = np.concatenate([z_1fp]*self.nfp, axis=1)
        zeta_full, theta_full = np.meshgrid(np.linspace(0, 2*np.pi, nv, endpoint=False), np.linspace(0, 2*np.pi, nu_final, endpoint=False))  

        if plot:
            x_on_az_grid = R_full * np.cos(zeta_full)
            y_on_az_grid = R_full * np.sin(zeta_full)
            z_on_az_grid = z_full

            ax.scatter(x_on_az_grid, y_on_az_grid, z_on_az_grid, s=1)
            ax.plot_wireframe(x_on_az_grid, y_on_az_grid, z_on_az_grid, alpha = 0.05)


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
            nu=32,
            nv=32,
            nv_interp=32,
            nu_interp=32,
            plot_ft=False,
            ft_ax=None,
            plot_intermediate=False,
            intermediate_ax=None,
            _fsolve=False,
            collocation='arclength',
            spec_cond = True,
            spec_cond_options={
            'plot':False,
            'ftol':1e-4,
            'Mtol':1.1,
            'shapetol':None,
            'niters':5000,
            'verbose':False,
            'cutoff':1e-6
            }
        ):
        r"""
        Performs Fourier transform from spline surface to Fourier
        coefficients for a VMEC surface. 
        """
        assert(nv%2==0), 'nv must be even'

        if np.any(R==None) or np.any(z==None) or np.any(zeta==None) or np.any(theta==None):
            # tic = time.perf_counter()
            if collocation == 'uniform':
                R_on_tz_grid, z_on_tz_grid, zeta_eval, theta_eval = self.uniform_tz_interp(
                    nu = nu,
                    nv = nv,
                    nv_interp = nv_interp,
                    nu_interp = nu_interp,
                    # plot=plot_intermediate,
                    # ax=intermediate_ax,
                    _fsolve=_fsolve,
                )
            elif collocation == 'arclength':
                R_on_tz_grid, z_on_tz_grid, zeta_eval, theta_eval = self.arclength_tz_interp(
                    nu = nu,
                    nv = nv,
                    nv_interp = nv_interp,
                    nu_interp = nu_interp,
                    plot=plot_intermediate,
                    ax=intermediate_ax,
                    _fsolve=_fsolve,
                ) 
            # toc = time.perf_counter()
            # print(f"Created equispaced grid in {toc - tic:0.4f} seconds")
        else:
            R_on_tz_grid, z_on_tz_grid, zeta_eval, theta_eval = R, z, zeta, theta
        
        M, N = self.M, self.N

        # Fourier transform 
        cosnmtz = np.array(
            [[np.cos(m*theta_eval - n*((zeta_eval)*self.nfp)) for m in range(0, M+1)] for n in range(-N, N+1)],
        )
        sinnmtz = np.array(
            [[np.sin(m*theta_eval - n*((zeta_eval)*self.nfp)) for m in range(0, M+1)] for n in range(-N, N+1)],
        )

        rbc_in = np.einsum('nmtz,tz->nm', cosnmtz, R_on_tz_grid.T)/(nu*nv)
        rbc_in[:N, 0] = 0
        rbc_in[:] *= 2
        rbc_in[N, 0] /= 2

        rbs_in = np.einsum('nmtz,tz->nm', sinnmtz, R_on_tz_grid.T)/(nu*nv)
        rbs_in[:N, 0] = 0
        rbs_in[:] *= 2
        rbs_in[N, 0] /= 2

        zbc_in = np.einsum('nmtz,tz->nm', cosnmtz, z_on_tz_grid.T)/(nu*nv)
        zbc_in[:N, 0] = 0
        zbc_in[:] *= 2
        zbc_in[N, 0] /= 2

        # print(f'zbc: {zbc}')
        zbs_in = np.einsum('nmtz,tz->nm', sinnmtz, z_on_tz_grid.T)/(nu*nv)
        zbs_in[:N, 0] = 0
        zbs_in[:] *= 2
        zbs_in[N, 0] /= 2

        rbc, zbs = rbc_in, zbs_in

        rbs = np.zeros_like(rbc)
        zbc = np.zeros_like(zbs)

        if plot_ft:
            alan_plot(rbc, rbs, zbc, zbs, nu, nv, self.M, self.N, self.nfp, ax=ft_ax, poincare=False)
            # self.plot()
        
        return rbc, zbs
  
    def centroid_axis_fourier_coeffs(
            self, 
            N=6,
            nv=300,
            plot=False
        ):
        '''
        Return Fourier coefficients for the centroid axis to be used as an initial guess
        in the vmec input
        '''
        vlist = np.linspace(0, 2*np.pi/self.nfp, nv, endpoint=False)
        x_ax, y_ax, z_ax = self.centroid_axis_callable(vlist)
        zeta = np.arctan2(y_ax, x_ax) % (2*np.pi/self.nfp)
        zeta = np.concatenate((zeta-2*np.pi, zeta, zeta+2*np.pi))
        zeta_eval = np.linspace(0, 2*np.pi/self.nfp, nv)
        x_zeta = griddata(
            points=zeta,
            values=np.tile(x_ax, 3),
            xi=zeta_eval,
            method='cubic',
        )
        y_zeta = griddata(
            points=zeta,
            values=np.tile(y_ax, 3),
            xi=zeta_eval,
            method='cubic',
        )
        z_zeta = griddata(
            points=zeta,
            values=np.tile(z_ax, 3),
            xi=zeta_eval,
            method='cubic',
        )
        if plot:
            fig = plt.figure("Axis plot")
            fig.patch.set_facecolor('white')
            ax = fig.add_subplot(projection='3d',azim=0, elev=90)
            ax.plot(x_zeta, y_zeta, z_zeta)

        r_zeta = np.sqrt(x_zeta**2 + y_zeta**2)
        cosnz = np.array(
            [np.cos(-n*zeta_eval*self.nfp) for n in range(-N, N+1)]
        )
        sinnz = np.array(
            [np.sin(-n*zeta_eval*self.nfp) for n in range(-N, N+1)]
        )
        r_n = np.einsum('nz,z->n', cosnz, r_zeta)/nv
        z_n = np.einsum('nz,z->n', sinnz, z_zeta)/nv

        if plot:
            zeta_plot = np.linspace(0, 2*np.pi/self.nfp, 200)
            #plotting fourier transformed axis
            r = np.zeros_like(zeta_plot)
            z = np.zeros_like(zeta_plot)
            for i, n in enumerate(range(-N, N+1)):
                r += r_n[i]*np.cos(-n*zeta_plot*self.nfp)
                z += z_n[i]*np.sin(-n*zeta_plot*self.nfp)
            x = r*np.cos(zeta_plot)
            y = r*np.sin(zeta_plot)
            ax.plot(x, y, z, 'r--')
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
        nu = 64,
        nv = 64,
        nu_interp = 64,
        nv_interp = 64,
        plot=False,
        collocation='arclength',
        spec_cond=True,
        spec_cond_options={
            'plot':False,
            'ftol':3e-4,
            'Mtol':1.5,
            'shapetol':0.01,
            'niters':400,
            'verbose':True,
            'cutoff':1e-6
        }
    ):
        # print('to_RZFourier called')
        if M==None and N==None:
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
            spec_cond_options=spec_cond_options
        )

        surf = SurfaceRZFourier(nfp=self.nfp, ntor=N, mpol=M)

        for m in range(0, M + 1):
            for n in range(-N, N + 1):
                if m == 0 and n < 0:
                    continue
                else:
                    surf.set_rc(m, n, rbc[n + N, m])
                    surf.set_zs(m, n, zbs[n + N, m])

        if spec_cond:
            surf = surf.variational_spec_cond(
                **spec_cond_options
            )

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

        vmec = vmec_from_surf(surf)
        vmec.run()

        return surf

    def write_inequality_constraints(self, maxval=np.inf):
        '''
        Return a tuple containing lb, ub, A, for inequality constraints
        '''
        dofs = self.dof_names
        indices_dict = dict(zip(dofs, range(len(self.dof_names))))

        #print(indices_dict)

        constraints_list = []
        lb = []
        ub = []
        constraint_titles = []

        if self.cs_basis == 'polar':

            for i in range(1, self.axis_points-1):
                temp = np.zeros(len(dofs))
                temp[indices_dict[f'PseudoAxis1:z_axis_{i}']] = 1
                constraints_list.append(np.copy(temp))
                constraint_titles.append(f'-1 < PseudoAxis1:z_axis_{i} < 1')
                lb.append(-1)
                ub.append(1)

            for i in range(1, self.axis_points):
                temp = np.zeros(len(dofs))
                temp[indices_dict[f'PseudoAxis1:r_axis_{i}']] = 1
                constraints_list.append(np.copy(temp))
                constraint_titles.append(f'0 < PseudoAxis1:r_axis_{i} < 2')
                lb.append(0)
                ub.append(2)

            # radii in cross section
            if self.axis_angles_fixed:
                '''
                Make sure that the radii for a given cross section does not exceed half of the pseudo axis
                radius at the same zeta
                '''
                for i, cs in enumerate(self.cs_list): #TODO change if indexing ever gets fixed
                    for j in range(cs.n_pts):
                        temp = np.zeros(len(dofs))
                        temp[indices_dict[f'CrossSectionFixedZeta{i+1}:r_{j}']] = 1
                        constraints_list.append(np.copy(temp))
                        constraint_titles.append(f'0 < CrossSectionFixedZeta{i+1}:r_{j} < 1')
                        lb.append(0)
                        ub.append(1)
                        if i == 0:
                            temp = np.zeros(len(dofs))
                            temp[indices_dict[f'CrossSectionFixedZeta{i+1}:r_{j}']] = 1
                            constraints_list.append(np.copy(temp))
                            constraint_titles.append(f'0 < CrossSectionFixedZeta{i+1}:r_{j} < 1')
                            lb.append(0)
                            ub.append(maxval)
                        else:
                            temp = np.zeros(len(dofs))
                            temp[indices_dict[f'CrossSectionFixedZeta{i+1}:r_{j}']] = -1
                            temp[indices_dict[f'PseudoAxis1:r_axis_{i}']] = 1
                            constraints_list.append(np.copy(temp))
                            constraint_titles.append(f'0 < PseudoAxis1:r_axis_{i} - CrossSectionFixedZeta{i+1}:r_{j} < 2')
                            lb.append(0)
                            ub.append(maxval)


            # thetas in cross section
            if self.cs_equispaced==False:
                for i, cs in enumerate(self.cs_list):
                    if cs.z_sym:
                        max_angle = np.pi
                        #for j in range(1, cs.n_pts-1):
                        temp = np.zeros(len(dofs))
                        temp[indices_dict[f'CrossSectionFixedZeta{i+1}:theta_{1}']] = 1
                        constraints_list.append(np.copy(temp))
                        constraint_titles.append(f'0 < CrossSectionFixedZeta{i+1}:theta_{1} < {max_angle}')
                        lb.append(0)
                        ub.append(max_angle)
                        for j in range(1, cs.n_pts-2):
                            temp = np.zeros(len(dofs))
                            temp[indices_dict[f'CrossSectionFixedZeta{i+1}:theta_{j}']] = -1
                            temp[indices_dict[f'CrossSectionFixedZeta{i+1}:theta_{j+1}']] = 1
                            constraint_titles.append(f'0 < CrossSectionFixedZeta{i+1}:theta_{j+1} - CrossSectionFixedZeta{i+1}:theta_{j} < 2pi')
                            constraints_list.append(np.copy(temp))
                            lb.append(0)
                            ub.append(2*np.pi)
                        temp = np.zeros(len(dofs))
                        temp[indices_dict[f'CrossSectionFixedZeta{i+1}:theta_{cs.n_pts-2}']] = 1
                        constraints_list.append(np.copy(temp))
                        constraint_titles.append(f'0 < CrossSectionFixedZeta{i+1}:theta_{cs.n_pts-2} < {max_angle}')
                        lb.append(0)
                        ub.append(max_angle)
                    else:
                        min_angle = 0
                        max_angle = np.pi

                        temp = np.zeros(len(dofs))
                        temp[indices_dict[f'CrossSectionFixedZeta{i+1}:theta_{0}']] = 1
                        constraint_titles.append(f'0 < CrossSectionFixedZeta{i+1}:theta_{0} < {max_angle}')
                        constraints_list.append(np.copy(temp))
                        lb.append(min_angle)
                        ub.append(max_angle)                        
                        for j in range(0, (cs.n_pts // 2)-1):
                            temp = np.zeros(len(dofs))
                            temp[indices_dict[f'CrossSectionFixedZeta{i+1}:theta_{j}']] = -1
                            temp[indices_dict[f'CrossSectionFixedZeta{i+1}:theta_{j+1}']] = 1
                            constraint_titles.append(f'0 < CrossSectionFixedZeta{i+1}:theta_{j+1} - CrossSectionFixedZeta{i+1}:theta_{j} < 2')
                            constraints_list.append(np.copy(temp))
                            lb.append(0)
                            ub.append(2*np.pi)
                        temp = np.zeros(len(dofs))
                        temp[indices_dict[f'CrossSectionFixedZeta{i+1}:theta_{(cs.n_pts // 2)-1}']] = 1
                        constraint_titles.append(f'{min_angle} < CrossSectionFixedZeta{i+1}:theta_{(cs.n_pts // 2)-1} < {max_angle}')
                        constraints_list.append(np.copy(temp))
                        lb.append(min_angle)
                        ub.append(max_angle)

                        min_angle = np.pi
                        max_angle = 2*np.pi
                        temp = np.zeros(len(dofs))
                        temp[indices_dict[f'CrossSectionFixedZeta{i+1}:theta_{(cs.n_pts // 2)}']] = 1
                        #print(f'CrossSectionFixedZeta{i+1}:theta_{(cs.n_pts // 2)}')
                        constraint_titles.append(f'{min_angle} < CrossSectionFixedZeta{i+1}:theta_{(cs.n_pts // 2)} < {max_angle}')
                        constraints_list.append(np.copy(temp))
                        lb.append(min_angle)
                        ub.append(max_angle)      
                        for j in range(cs.n_pts // 2, cs.n_pts-1):
                            temp = np.zeros(len(dofs))
                            temp[indices_dict[f'CrossSectionFixedZeta{i+1}:theta_{j}']] = -1
                            temp[indices_dict[f'CrossSectionFixedZeta{i+1}:theta_{j+1}']] = 1
                            constraint_titles.append(f'{0} < CrossSectionFixedZeta{i+1}:theta_{j+1} - CrossSectionFixedZeta{i+1}:theta_{j} < {2}')
                            constraints_list.append(np.copy(temp))
                            lb.append(0)
                            ub.append(2)
                        temp = np.zeros(len(dofs))
                        temp[indices_dict[f'CrossSectionFixedZeta{i+1}:theta_{cs.n_pts-1}']] = 1
                        constraints_list.append(np.copy(temp))
                        constraint_titles.append(f'{min_angle} < CrossSectionFixedZeta{i+1}:theta_{cs.n_pts-1} < {max_angle}')
                        lb.append(min_angle)
                        ub.append(max_angle)

        A = np.array(constraints_list)
        lb = np.array(lb)
        ub = np.array(ub)

        return A, lb, ub, constraint_titles

    def write_ub_constraints(self):
        '''
        Writing constraints in the form Ax <= b_ub
        '''
        dofs = self.dof_names
        indices_dict = dict(zip(dofs, range(len(self.dof_names))))

        #print(indices_dict)

        constraints_list = []
        A_ub = []
        lb = []
        ub = []
        constraint_titles = []

        if self.cs_basis == 'polar':

            for i in range(1, self.axis_points-1):
                temp = np.zeros(len(dofs))
                temp[indices_dict[f'PseudoAxis1:z_axis_{i}']] = 1
                constraints_list.append(np.copy(temp))
                constraint_titles.append(f'PseudoAxis1:z_axis_{i} < 1')
                A_ub.append(1)
                temp = np.zeros(len(dofs))
                temp[indices_dict[f'PseudoAxis1:z_axis_{i}']] = -1
                constraints_list.append(np.copy(temp))
                constraint_titles.append(f'-PseudoAxis1:z_axis_{i} < 1')
                A_ub.append(1)

            for i in range(1, self.axis_points):
                temp = np.zeros(len(dofs))
                temp[indices_dict[f'PseudoAxis1:r_axis_{i}']] = -1
                constraints_list.append(np.copy(temp))
                constraint_titles.append(f'PseudoAxis1:r_axis_{i} > 0')
                A_ub.append(0)

            # radii in cross section
            if self.axis_angles_fixed:
                '''
                Make sure that the radii for a given cross section does not exceed half of the pseudo axis
                radius at the same zeta
                '''
                for i, cs in enumerate(self.cs_list): #TODO change if indexing ever gets fixed
                    for j in range(cs.n_pts):
                        temp = np.zeros(len(dofs))
                        temp[indices_dict[f'CrossSectionFixedZeta{i+1}:r_{j}']] = -1
                        constraints_list.append(np.copy(temp))
                        constraint_titles.append(f'- CrossSectionFixedZeta{i+1}:r_{j} < 0')
                        A_ub.append(0)

            # thetas in cross section
            if self.cs_equispaced==False:
                for i, cs in enumerate(self.cs_list):
                    if cs.z_sym:
                        max_angle = np.pi
                        #for j in range(1, cs.n_pts-1):
                        temp = np.zeros(len(dofs))
                        temp[indices_dict[f'CrossSectionFixedZeta{i+1}:theta_{1}']] = -1
                        constraints_list.append(np.copy(temp))
                        constraint_titles.append(f'-CrossSectionFixedZeta{i+1}:theta_{1} < {0}')
                        A_ub.append(0)
                        for j in range(1, cs.n_pts-2):
                            temp = np.zeros(len(dofs))
                            temp[indices_dict[f'CrossSectionFixedZeta{i+1}:theta_{j}']] = 1
                            temp[indices_dict[f'CrossSectionFixedZeta{i+1}:theta_{j+1}']] = -1
                            constraint_titles.append(f'CrossSectionFixedZeta{i+1}:theta_{j} - CrossSectionFixedZeta{i+1}:theta_{j+1} < 0')
                            constraints_list.append(np.copy(temp))
                            A_ub.append(0)
                        temp = np.zeros(len(dofs))
                        temp[indices_dict[f'CrossSectionFixedZeta{i+1}:theta_{cs.n_pts-2}']] = -1
                        constraints_list.append(np.copy(temp))
                        constraint_titles.append(f'-CrossSectionFixedZeta{i+1}:theta_{cs.n_pts-2} < {0}')
                        A_ub.append(0)
                    else:
                        min_angle = 0
                        max_angle = np.pi

                        temp = np.zeros(len(dofs))
                        temp[indices_dict[f'CrossSectionFixedZeta{i+1}:theta_{0}']] = -1
                        constraint_titles.append(f'-CrossSectionFixedZeta{i+1}:theta_{0} < {min_angle}')
                        constraints_list.append(np.copy(temp))
                        A_ub.append(min_angle)                        
                        for j in range(0, (cs.n_pts // 2)-1):
                            temp = np.zeros(len(dofs))
                            temp[indices_dict[f'CrossSectionFixedZeta{i+1}:theta_{j}']] = 1
                            temp[indices_dict[f'CrossSectionFixedZeta{i+1}:theta_{j+1}']] = -1
                            constraint_titles.append(f'CrossSectionFixedZeta{i+1}:theta_{j} - CrossSectionFixedZeta{i+1}:theta_{j+1}< 0')
                            constraints_list.append(np.copy(temp))
                            A_ub.append(0)
                        temp = np.zeros(len(dofs))
                        temp[indices_dict[f'CrossSectionFixedZeta{i+1}:theta_{(cs.n_pts // 2)-1}']] = 1
                        constraint_titles.append(f'CrossSectionFixedZeta{i+1}:theta_{(cs.n_pts // 2)-1} < {max_angle}')
                        constraints_list.append(np.copy(temp))
                        A_ub.append(max_angle)

                        min_angle = np.pi
                        max_angle = 2*np.pi
                        temp = np.zeros(len(dofs))
                        temp[indices_dict[f'CrossSectionFixedZeta{i+1}:theta_{(cs.n_pts // 2)}']] = -1
                        #print(f'CrossSectionFixedZeta{i+1}:theta_{(cs.n_pts // 2)}')
                        constraint_titles.append(f'-CrossSectionFixedZeta{i+1}:theta_{(cs.n_pts // 2)} < {-max_angle}')
                        constraints_list.append(np.copy(temp))
                        A_ub.append(-max_angle)      
                        for j in range(cs.n_pts // 2, cs.n_pts-1):
                            temp = np.zeros(len(dofs))
                            temp[indices_dict[f'CrossSectionFixedZeta{i+1}:theta_{j}']] = 1
                            temp[indices_dict[f'CrossSectionFixedZeta{i+1}:theta_{j+1}']] = -1
                            constraint_titles.append(f'CrossSectionFixedZeta{i+1}:theta_{j} - CrossSectionFixedZeta{i+1}:theta_{j+1} < {0}')
                            constraints_list.append(np.copy(temp))
                            A_ub.append(0)
                        temp = np.zeros(len(dofs))
                        temp[indices_dict[f'CrossSectionFixedZeta{i+1}:theta_{cs.n_pts-1}']] = 1
                        constraints_list.append(np.copy(temp))
                        constraint_titles.append(f'CrossSectionFixedZeta{i+1}:theta_{cs.n_pts-1} < {max_angle}')
                        A_ub.append(max_angle)

            
        if self.cs_basis == 'cartesian':
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
        axis_end = end_idx+len(self.axis.x)    
        self.axis.x = dofs[end_idx:axis_end]
        self.local_x = dofs[axis_end:]
        return None

def vmec_from_surf(
        nfp,
        surf=None,
        M = 12,
        N = 12,
        ns=13,
        ntheta=32,
        nzeta=32,
        ftol=1e-7,
        phiedge = 1,
        verbose=False,
        niter=3000
    ):
    '''
    Generate VMEC object from any optimizable object with a
    `to_RZFourier` method. 
    '''
    # runtime params
    vmec = Vmec(mpi = mpi, verbose=verbose)
    # N=6
    # r_n, z_n = surf.centroid_axis_fourier_coeffs(N=6)
    # r_n = r_n[N:]
    # z_n = z_n[N:]

    vmec.indata.delt = 9e-1
    # vmec.indata.niter = 2000
    # vmec.indata.nstep = 1e2
    vmec.indata.tcon0 = 2
    vmec.indata.ns_array = np.append(np.array([ns]), np.zeros(99,))
    vmec.indata.niter_array = np.append(np.array([niter]), -1*np.ones(99,))
    vmec.indata.ftol_array = np.append(np.array([ftol]), np.zeros(99,))
    vmec.indata.precon_type = 'none'
    vmec.indata.prec2d_threshold = 1e-19
    # grid params
    vmec.indata.lasym = 0
    vmec.indata.nfp = nfp
    vmec.indata.mpol = M
    vmec.indata.ntor = N
    vmec.indata.ntheta = ntheta
    vmec.indata.nzeta = nzeta
    vmec.indata.phiedge = phiedge
    # free bdry params
    vmec.indata.lfreeb = 0
    vmec.indata.nvacskip = 6
    # pressure params
    vmec.indata.gamma = 0
    vmec.indata.bloat = 1
    vmec.indata.spres_ped = 1
    vmec.indata.pres_scale = 1
    vmec.indata.pmass_type = 'power_series'
    vmec.indata.am = 0
    # current/iota params
    vmec.indata.curtor = 0
    vmec.indata.ncurr = 1
    vmec.indata.piota_type = 'power_series'
    vmec.indata.pcurr_type = 'power_series'

    vmec.boundary = surf
    vmec.set_indata()
    return vmec

def alan_plot(rbc, rbs, zbc, zbs, ntheta, nzeta, M, N, nfp, ax=None, poincare=True):
    if ax is None:
        fig = plt.figure("3D Surface Plot")
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(projection='3d',azim=0, elev=90)
    xn = np.arange(-N,N+1,1)
    xm = np.arange(0,M+1,1)

    ntheta = 200
    nzeta = 9
    theta1D = np.linspace(0,2*np.pi,num=ntheta)
    zeta1D = np.linspace(0,2*np.pi/nfp,num=nzeta) + 2*np.pi/nfp
    zeta2D, theta2D = np.meshgrid(zeta1D,theta1D)

    if poincare:
        fig = plt.figure("Poincare Plots",figsize=(14,7))
        fig.patch.set_facecolor('white')

        R = np.zeros((ntheta,nzeta))
        Z = np.zeros((ntheta,nzeta))

        for i in range(rbc.shape[0]):
            for j in range(rbc.shape[1]):
                if rbc[i,j] !=0 or zbs[i,j] != 0:
                    angle = xm[j]*theta2D - xn[i]*zeta2D*nfp
                    R = R + rbc[i,j]*np.cos(angle)#/(np.abs(i) + np.abs(j))
                    Z = Z + zbs[i,j]*np.sin(angle)#/(np.abs(i) + np.abs(j))
                if rbs[i,j] !=0 or zbc[i,j]:
                    angle = xm[j]*theta2D - xn[i]*zeta2D*nfp
                    R = R + rbs[i,j]*np.sin(angle)
                    Z = Z + zbc[i,j]*np.cos(angle)
        numCols = 5
        numRows = 2
        plotNum = 1
        zeta = np.linspace(0,2 * np.pi/nfp,num=nzeta,endpoint=True)

        plt.subplot(numRows,numCols,plotNum)
        #plt.subplot(1,1,1)
        plotNum += 1
        for ind in range(nzeta):
            plt.subplot(numRows,numCols,ind+1)
            plt.title(r'$\phi =$' + str(zeta[ind]))
            plt.gca().set_aspect('equal',adjustable='box')

            plt.plot(R[:,ind], Z[:,ind], '-')
            plt.plot(R[:,ind], Z[:,ind], '-')
            plt.plot(R[:,ind], Z[:,ind], '-')
            plt.plot(R[:,ind], Z[:,ind], '-')
        plt.gca().set_aspect('equal',adjustable='box')
        plt.xlabel('R')
        plt.ylabel('Z')

    theta1D = np.linspace(0,2*np.pi,num=ntheta)
    zeta1D = np.linspace(0,2*np.pi,num=nzeta) + 2*np.pi/nfp

    zeta2D, theta2D = np.meshgrid(zeta1D,theta1D)

    ntheta = 200
    nzeta = 200
    theta1D = np.linspace(0,2*np.pi,num=ntheta)
    zeta1D = np.linspace(0,2*np.pi,num=nzeta) + 2*np.pi/nfp
    zeta2D, theta2D = np.meshgrid(zeta1D,theta1D)
    R = np.zeros((ntheta,nzeta))
    Z = np.zeros((ntheta,nzeta))

    for i in range(rbc.shape[0]):
        for j in range(rbc.shape[1]):
            if rbc[i,j] !=0 or zbs[i,j] != 0:
                angle = xm[j]*theta2D - xn[i]*zeta2D*nfp
                R = R + rbc[i,j]*np.cos(angle)#/(np.abs(i) + np.abs(j))
                Z = Z + zbs[i,j]*np.sin(angle)#/(np.abs(i) + np.abs(j))
            if rbs[i,j] !=0 or zbc[i,j]:
                angle = xm[j]*theta2D - xn[i]*zeta2D*nfp
                R = R + rbs[i,j]*np.sin(angle)
                Z = Z + zbc[i,j]*np.cos(angle)
    X = R * np.cos(zeta2D)
    Y = R * np.sin(zeta2D)

    ax.plot_surface(X, Y, Z)
    ax.set_box_aspect((1, 1, 1))
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    ax.set_zlim(-1, 1)

def rot_matrix_2d(theta):
    return np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

def b_p(t, p, x, i=None):
    '''
    Compute the B-spline basis function B_pi. 

    ## Inputs:
    t : knot vector \\
    p : degree \\
    x : point at which to evaluate \\
    i : basis of interest (if None, returns all )

    ## Outputs:
    If i is None, an array of shape (x, k) is returned, which consists of the kth p-order B-Spline basis function computed on the array x. 
    If i is an integer, an array of shape (x) is returned, which consists of the ith p-order B-Spline basis function computed on the array x. 
    '''
    b = []

    for deg in range(0, p+1):
        l = len(t)
        if deg == 0:
            # x = x[(x >= t[0]) & (x <= t[-1])]
            x1d = x.copy()
            t1d = t.copy()
            t = np.outer(np.ones(len(x)), t)
            x = np.expand_dims(x, 0)
            b0 = ((x.T >= t[:, :-1]) & (x.T < t[:, 1:]))
            b0[np.isclose(x1d, t1d[-1]), np.isclose(t1d[1:], t1d[-1])] = 1 # accounting for evaluation at rightmost knot
            b.append(b0)
        else: 
            l_term_n = (x.T - t[:, :-deg-1])
            l_term_d = (t[:, deg:-1] - t[:, :-deg-1])
            l_term = b[-1][:, :-1]* np.divide(l_term_n, l_term_d, out=np.zeros_like(l_term_d), where=l_term_d != 0)

            r_term_n = (t[:, deg+1:] - x.T)
            r_term_d = (t[:, deg+1:] - t[:, 1:-deg])
            r_term = b[-1][:, 1:]* np.divide(r_term_n, r_term_d, out=np.zeros_like(r_term_d), where=r_term_d != 0)

            b.append(l_term + r_term)

    if i is None:
        return b[-1]
    else:
        return b[-1][:, i]

def vol_from_boundary(rbc, zbs, M, N, nu, nv, nfp):
    '''
    Compute volume enclosed by a boundary given with VMEC Fourier coefficients.
    Uses a clever trick based on Gauss' identity:
    ∫∫∫ div(A) dV = ∫∫ A . n dA
    Choose some A whose divergence is unity, in this case (in cylindrical coordinates)
    (0, 0, z). Then, 
    V = ∫∫∫dV = ∫∫ zn_z dA. 
    
    :param rbc: RBC coefficients, given in (n,m) format
    :param zbs: ZBS coefficients, given in (n,m) format
    :param M: Max poloidal mode number
    :param N: Max toroidal mode number
    :param nu: Number of points to take in u for quadrature
    :param nv: Number of points to take in v for quadrature
    :param nfp: Number of field periods
    '''
    u_1d = np.linspace(0, 2*np.pi, nu, endpoint=True)
    v_1d = np.linspace(0, 2*np.pi, nv, endpoint=True)        
    v_grid, u_grid = np.meshgrid(v_1d, u_1d)
    cosnmuz = np.array(
        [[np.cos(m*u_grid - n*(nfp*v_grid)) for m in range(0, M+1)] for n in range(-N, N+1)],
    )
    sinnmuz = np.array(
        [[np.sin(m*u_grid - n*(nfp*v_grid)) for m in range(0, M+1)] for n in range(-N, N+1)],
    )
    m = np.array(
        [[m for m in range(0, M+1)] for n in range(-N, N+1)]
    )

    R_uz = np.einsum('nm,nmuz->uz', rbc, cosnmuz)
    Z_uz = np.einsum('nm,nmuz->uz', zbs, sinnmuz)
    duR_uz = np.einsum('nm,nmuz->uz', -m*rbc, sinnmuz)

    integrand = Z_uz * R_uz * duR_uz
    res = (u_1d[1]-u_1d[0])*(v_1d[1]-v_1d[0])*np.sum(integrand)

    return res