import numpy as np
import simsoptpp as sopp
import scipy
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier


def forward_backward(P, L, U, rhs):
    """
    Solve a linear system of the form PLU*adj = rhs
    """
    y = scipy.linalg.solve_triangular(U.T, rhs, lower=True) 
    z = scipy.linalg.solve_triangular(L.T, y, lower=False) 
    adj = P@z 
   
    #  iterative refinement
    yp = scipy.linalg.solve_triangular(U.T, rhs-(P@L@U).T@adj, lower=True) 
    zp = scipy.linalg.solve_triangular(L.T, yp, lower=False) 
    adj += P@zp 
 
    return adj


#class Area(object):
#    """
#    Wrapper class for surface area label.
#    """
#
#    def __init__(self, surface, stellarator):
#        self.stellarator = stellarator
#        self.surface = surface
#    
#    def J(self):
#        """
#        Compute the area of a surface.
#        """
#        return self.surface.area()
#
#    def dJ_by_dsurfacecoefficients(self):
#        """
#        Calculate the derivatives with respect to the surface coefficients.
#        """
#        return self.surface.darea_by_dcoeff()
#
#    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
#        """
#        Calculate the second derivatives with respect to the surface coefficients.
#        """
#        return self.surface.d2area_by_dcoeffdcoeff()
#    
#    def dJ_by_dcoilcoefficients(self):
#        return [np.zeros((c.num_dofs(),)) for c in self.stellarator.coils]
#    
#    def dJ_by_dcoilcurrents(self):
#        return [0. for c in self.stellarator.coils]


#class Volume(object):
#    """
#    Wrapper class for volume label.
#    """
#
#    def __init__(self, surface, stellarator):
#        self.surface = surface
#        self.stellarator = stellarator
#
#    def J(self):
#        """
#        Compute the volume enclosed by the surface.
#        """
#        return self.surface.volume()
#
#    def dJ_by_dsurfacecoefficients(self):
#        """
#        Calculate the derivatives with respect to the surface coefficients.
#        """
#        return self.surface.dvolume_by_dcoeff()
#
#    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
#        """
#        Calculate the second derivatives with respect to the surface coefficients.
#        """
#        return self.surface.d2volume_by_dcoeffdcoeff()
#
#    def dJ_by_dcoilcoefficients(self):
#        return [np.zeros((c.num_dofs(),)) for c in self.stellarator.coils]
#
#    def dJ_by_dcoilcurrents(self):
#        return [0. for c in self.stellarator.coils]

sDIM = 10
class BoozerResidual(object):
    r"""
    """
    def __init__(self, boozer_surface, bs, constraint_weight):
        in_surface=boozer_surface.surface
        self.boozer_surface = boozer_surface
        
        phis   = in_surface.quadpoints_phi
        thetas = in_surface.quadpoints_theta
        s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(in_surface.get_dofs())

        self.constraint_weight=constraint_weight
        self.in_surface = in_surface
        self.surface = s
        self.biotsavart = bs
        self.clear_cached_properties()

    def clear_cached_properties(self):
        self._J = None
        self._dJ_by_dcoefficients = None
        self._dJ_by_dcoilcurrents = None

    def compute(self, compute_derivatives=0):


        self.surface.set_dofs(self.in_surface.get_dofs())
        self.biotsavart.set_points(self.surface.gamma().reshape((-1,3)))
        
        # compute J
        surface = self.surface
        iota = self.boozer_surface.res['iota']
        G = self.boozer_surface.res['G']
        r, = boozer_surface_residual(surface, iota, G, self.biotsavart, derivatives=0)
        r = np.concatenate((r, [np.sqrt(self.constraint_weight)*(self.boozer_surface.label.J()-self.boozer_surface.targetlabel)] ) )
        self._J = 0.5*np.sum(r*r)
        if compute_derivatives > 0:
            r, J = boozer_surface_residual(surface, iota, G, self.biotsavart, derivatives=1)
            
            booz_surf = self.boozer_surface
            iota = booz_surf.res['iota']
            G = booz_surf.res['G']
#            P, L, U = booz_surf.res['PLU']
#            dconstraint_dcoils_vjp = booz_surf.res['dconstraint_dcoils_vjp']
            
            dJ_by_dB = self.dJ_by_dB()
            dJ_by_dcoils = self.biotsavart.B_vjp(dJ_by_dB)

            # dJ_diota, dJ_dG  to the end of dJ_ds are on the end
#            r = np.concatenate((r, [np.sqrt(self.constraint_weight)*(self.boozer_surface.label.J()-self.boozer_surface.targetlabel)] ) )
#            dl = np.zeros((J.shape[1],))
#            dl[:-2] = self.boozer_surface.label.dJ_by_dsurfacecoefficients()
#            J = np.concatenate((J, np.sqrt(self.constraint_weight) * dl[None, :]), axis=0)
#            
#            dJ_ds = J.T@r
#            adj = forward_backward(P, L, U, dJ_ds)
#            
#            adj_times_dg_dcoil, adj_times_dg_dcurr = dconstraint_dcoils_vjp(adj, booz_surf, iota, G, booz_surf.bs)
#            self._dJ_by_dcoefficients = [dj_dc - adj_dg_dc for dj_dc, adj_dg_dc in zip(dJ_by_dcoils, adj_times_dg_dcoil)]
#            
#            dB_by_dcoilcurrents = self.biotsavart.dB_by_dcoilcurrents()
#            dJ_by_dcoilcurrents = [np.sum(dJ_by_dB*dB_dcurr) for dB_dcurr in dB_by_dcoilcurrents]
#            self._dJ_by_dcoilcurrents = [dj_dc - adj_dg_dc for dj_dc, adj_dg_dc in zip(dJ_by_dcoilcurrents, adj_times_dg_dcurr)]

            self._dJ_by_dcoefficients = dJ_by_dcoils
            dB_by_dcoilcurrents = self.biotsavart.dB_by_dcoilcurrents()
            dJ_by_dcoilcurrents = [np.sum(dJ_by_dB*dB_dcurr) for dB_dcurr in dB_by_dcoilcurrents]
            self._dJ_by_dcoilcurrents = dJ_by_dcoilcurrents





    def J(self, compute_derivatives=0):
        assert compute_derivatives >= 0
        if self._J is None:
            self.compute(compute_derivatives=compute_derivatives)
        return self._J

    def dJ_by_dcoefficients(self, compute_derivatives=1):
        assert compute_derivatives >= 1
        if self._dJ_by_dcoefficients is None:
            self.compute(compute_derivatives=compute_derivatives)
        return self._dJ_by_dcoefficients

    def dJ_by_dcoilcurrents(self, compute_derivatives=1):
        assert compute_derivatives >= 1
        if self._dJ_by_dcoilcurrents is None:
            self.compute(compute_derivatives=compute_derivatives)
        return self._dJ_by_dcoilcurrents

    def dJ_by_dB(self):
        """
        Return the partial derivative of the objective with respect to the magnetic field
        """
        
        surface = self.surface
        r, r_dB = boozer_surface_residual_dB(surface, self.boozer_surface.res['iota'], self.boozer_surface.res['G'], self.biotsavart, derivatives=0)
        dJ_by_dB = r[:, None]*r_dB
        dJ_by_dB = np.sum(dJ_by_dB.reshape((-1, 3, 3)), axis=1)
        return dJ_by_dB






class NonQuasiAxisymmetricComponent(object):
    r"""
    This objective decomposes the field magnitude :math:`B(\varphi,\theta)` into quasiaxisymmetric and
    non-quasiaxisymmetric components, then returns a penalty on the latter component.
    
    .. math::
        J &= \frac{1}{2}\int_{\Gamma_{s}} (B-B_{\text{QS}})^2~dS
          &= \frac{1}{2}\int_0^1 \int_0^1 (B - B_{\text{QS}})^2 \|\mathbf n\| ~d\varphi~\d\theta 
    where
    
    .. math::
        B &= \| \mathbf B(\varphi,\theta) \|_2
        B_{\text{QS}} &= \frac{\int_0^1 \int_0^1 B \| n\| ~d\varphi ~d\theta}{\int_0^1 \int_0^1 \|\mathbf n\| ~d\varphi ~d\theta}
    
    """
    def __init__(self, boozer_surface, bs):
        in_surface=boozer_surface.surface
        self.boozer_surface = boozer_surface
        
        phis = np.linspace(0, 1/in_surface.nfp, 2*sDIM, endpoint=False)
        thetas = np.linspace(0, 1., 2*sDIM, endpoint=False)
        s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(in_surface.get_dofs())

        self.in_surface = in_surface
        self.surface = s
        self.biotsavart = bs
        self.clear_cached_properties()

    def clear_cached_properties(self):
        self._J = None
        self._dJ_by_dcoefficients = None
        self._dJ_by_dcoilcurrents = None

    def compute(self, compute_derivatives=0):
        
        self.surface.set_dofs(self.in_surface.get_dofs())
        self.biotsavart.set_points(self.surface.gamma().reshape((-1,3)))
        
        # compute J
        surface = self.surface
        nphi = surface.quadpoints_phi.size
        ntheta = surface.quadpoints_theta.size
        
        B = self.biotsavart.B()
        B = B.reshape((nphi, ntheta, 3))
        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
        
        nor = surface.normal()
        dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)

        B_QS = np.mean(modB * dS, axis=0) / np.mean(dS, axis=0)
        B_nonQS = modB - B_QS[None, :]
        self._J = 0.5 * np.mean(dS * B_nonQS**2) 
        
        #import ipdb;ipdb.set_trace()
        if compute_derivatives > 0:
            booz_surf = self.boozer_surface
            iota = booz_surf.res['iota']
            G = booz_surf.res['G']
            P, L, U = booz_surf.res['PLU']
            dconstraint_dcoils_vjp = booz_surf.res['dconstraint_dcoils_vjp']

            dJ_by_dB = self.dJ_by_dB().reshape((-1, 3))
            dJ_by_dcoils = self.biotsavart.B_vjp(dJ_by_dB)

            # tack on dJ_diota = dJ_dG = 0 to the end of dJ_ds
            dJ_ds = np.concatenate((self.dJ_by_dsurfacecoefficients(), [0., 0.]))
            adj = forward_backward(P, L, U, dJ_ds)
            
            adj_times_dg_dcoil, adj_times_dg_dcurr = dconstraint_dcoils_vjp(adj, booz_surf, iota, G, booz_surf.bs)
            self._dJ_by_dcoefficients = [dj_dc - adj_dg_dc for dj_dc, adj_dg_dc in zip(dJ_by_dcoils, adj_times_dg_dcoil)]
            
            dB_by_dcoilcurrents = self.biotsavart.dB_by_dcoilcurrents()
            dJ_by_dcoilcurrents = [np.sum(dJ_by_dB*dB_dcurr) for dB_dcurr in dB_by_dcoilcurrents]
            self._dJ_by_dcoilcurrents = [dj_dc - adj_dg_dc for dj_dc, adj_dg_dc in zip(dJ_by_dcoilcurrents, adj_times_dg_dcurr)]
    
    def J(self, compute_derivatives=0):
        assert compute_derivatives >= 0
        if self._J is None:
            self.compute(compute_derivatives=compute_derivatives)
        return self._J

    def dJ_by_dcoefficients(self, compute_derivatives=1):
        assert compute_derivatives >= 1
        if self._dJ_by_dcoefficients is None:
            self.compute(compute_derivatives=compute_derivatives)
        return self._dJ_by_dcoefficients

    def dJ_by_dcoilcurrents(self, compute_derivatives=1):
        assert compute_derivatives >= 1
        if self._dJ_by_dcoilcurrents is None:
            self.compute(compute_derivatives=compute_derivatives)
        return self._dJ_by_dcoilcurrents

    def dJ_by_dB(self):
        """
        Return the partial derivative of the objective with respect to the magnetic field
        """
        surface = self.surface
        nphi = surface.quadpoints_phi.size
        ntheta = surface.quadpoints_theta.size

        B = self.biotsavart.B()
        B = B.reshape((nphi, ntheta, 3))
        
        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
        nor = surface.normal()
        dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)

        denom = np.mean(dS, axis=0)
        B_QS = np.mean(modB * dS, axis=0) / denom
        B_nonQS = modB - B_QS[None, :]
        
        dmodB_dB = B / modB[..., None]
        dJ_by_dB = B_nonQS[..., None] * dmodB_dB * dS[:, :, None] / (nphi * ntheta)
        return dJ_by_dB
    
    def dJ_by_dsurfacecoefficients(self):
        """
        Return the partial derivative of the objective with respect to the surface coefficients
        """
        surface = self.surface
        nphi = surface.quadpoints_phi.size
        ntheta = surface.quadpoints_theta.size

        B = self.biotsavart.B()
        B = B.reshape((nphi, ntheta, 3))
        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
        
        nor = surface.normal()
        dnor_dc = surface.dnormal_by_dcoeff()
        dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)
        dS_dc = (nor[:, :, 0, None]*dnor_dc[:, :, 0, :] + nor[:, :, 1, None]*dnor_dc[:, :, 1, :] + nor[:, :, 2, None]*dnor_dc[:, :, 2, :])/dS[:, :, None]

        B_QS = np.mean(modB * dS, axis=0) / np.mean(dS, axis=0)
        B_nonQS = modB - B_QS[None, :]
        
        dB_by_dX = self.biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
        dx_dc = surface.dgamma_by_dcoeff()
        dB_dc = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_dc, optimize=True)
        
        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
        dmodB_dc = (B[:, :, 0, None] * dB_dc[:, :, 0, :] + B[:, :, 1, None] * dB_dc[:, :, 1, :] + B[:, :, 2, None] * dB_dc[:, :, 2, :])/modB[:, :, None]
        
        num = np.mean(modB * dS, axis=0)
        denom = np.mean(dS, axis=0)
        dnum_dc = np.mean(dmodB_dc * dS[..., None] + modB[..., None] * dS_dc, axis=0) 
        ddenom_dc = np.mean(dS_dc, axis=0)
        B_QS_dc = (dnum_dc * denom[:, None] - ddenom_dc * num[:, None])/denom[:, None]**2
        B_nonQS_dc = dmodB_dc - B_QS_dc[None, :]
        
        dJ_by_dc = np.mean(0.5 * dS_dc * B_nonQS[..., None]**2 + dS[..., None] * B_nonQS[..., None] * B_nonQS_dc, axis=(0, 1)) 
        return dJ_by_dc


class Area(object):
    """
    Wrapper class for surface area label.
    """

    def __init__(self, in_surface, stellarator):
        #phis = np.linspace(0, 1/in_surface.nfp, sDIM, endpoint=False)
        phis = np.linspace(0, 1/(2*in_surface.nfp), sDIM, endpoint=False)
        phis += phis[1]/2

        thetas = np.linspace(0, 1., 2*sDIM, endpoint=False)
        s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(in_surface.get_dofs())

        self.in_surface = in_surface


        self.stellarator = stellarator
        self.surface = s
    
    def J(self):
        """
        Compute the area of a surface.
        """
        self.surface.set_dofs(self.in_surface.get_dofs())
        return self.surface.area()

    def dJ_by_dsurfacecoefficients(self):
        """
        Calculate the derivatives with respect to the surface coefficients.
        """
        self.surface.set_dofs(self.in_surface.get_dofs())
        return self.surface.darea_by_dcoeff()

    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
        """
        Calculate the second derivatives with respect to the surface coefficients.
        """
        self.surface.set_dofs(self.in_surface.get_dofs())
        return self.surface.d2area_by_dcoeffdcoeff()
    
    def dJ_by_dcoilcoefficients(self):
        return [np.zeros((c.num_dofs(),)) for c in self.stellarator.coils]
    
    def dJ_by_dcoilcurrents(self):
        return [0. for c in self.stellarator.coils]





class Volume(object):
    """
    Wrapper class for volume label.
    """

    def __init__(self, in_surface, stellarator):
        #phis = np.linspace(0, 1/in_surface.nfp, sDIM, endpoint=False)
        phis = np.linspace(0, 1/(2*in_surface.nfp), sDIM, endpoint=False)
        phis += phis[1]/2

        thetas = np.linspace(0, 1., 2*sDIM, endpoint=False)
        s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(in_surface.get_dofs())

        self.in_surface = in_surface
        self.surface = s
        self.stellarator = stellarator

    def J(self):
        """
        Compute the volume enclosed by the surface.
        """
        self.surface.set_dofs(self.in_surface.get_dofs())
        return self.surface.volume()

    def dJ_by_dsurfacecoefficients(self):
        """
        Calculate the derivatives with respect to the surface coefficients.
        """
        self.surface.set_dofs(self.in_surface.get_dofs())
        return self.surface.dvolume_by_dcoeff()

    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
        """
        Calculate the second derivatives with respect to the surface coefficients.
        """
        self.surface.set_dofs(self.in_surface.get_dofs())
        return self.surface.d2volume_by_dcoeffdcoeff()

    def dJ_by_dcoilcoefficients(self):
        return [np.zeros((c.num_dofs(),)) for c in self.stellarator.coils]

    def dJ_by_dcoilcurrents(self):
        return [0. for c in self.stellarator.coils]

class MajorRadius(object): 
    r"""
    This objective computes the major radius of a toroidal surface with the following formula:
    
    R_major = (1/2*pi) * (V / mean_cross_sectional_area)
    
    where mean_cross_sectional_area = \int^1_0 \int^1_0 z_\theta(xy_\varphi -yx_\varphi) - z_\varphi(xy_\theta-yx_\theta) ~d\theta d\varphi
    
    """
    def __init__(self, boozer_surface, stellarator):
        in_surface = boozer_surface.surface
        #phis = np.linspace(0, 1/in_surface.nfp, sDIM, endpoint=False)
        phis = np.linspace(0, 1/(2*in_surface.nfp), sDIM, endpoint=False)
        phis += phis[1]/2


        thetas = np.linspace(0, 1., 2*sDIM, endpoint=False)
        s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(in_surface.get_dofs())
        
        self.volume = Volume(s, stellarator)
        self.boozer_surface = boozer_surface
        self.in_surface = in_surface
        self.surface = s
        self.biotsavart = boozer_surface.bs
        self.stellarator = stellarator
        self.clear_cached_properties()

    def clear_cached_properties(self):
        self._J = None
        self._dJ_by_dcoefficients = None
        self._dJ_by_dcoilcurrents = None
 
    def compute(self, compute_derivatives=0):

        self.surface.set_dofs(self.in_surface.get_dofs())
        surface = self.surface
    
        g = surface.gamma()
        g1 = surface.gammadash1()
        g2 = surface.gammadash2()
        
        x = g[:, :, 0]
        y = g[:, :, 1]
        
        r = np.sqrt(x**2+y**2)
    
        xvarphi = g1[:, :, 0]
        yvarphi = g1[:, :, 1]
        zvarphi = g1[:, :, 2]
    
        xtheta = g2[:, :, 0]
        ytheta = g2[:, :, 1]
        ztheta = g2[:, :, 2]
    
        mean_area = np.mean((ztheta*(x*yvarphi-y*xvarphi)-zvarphi*(x*ytheta-y*xtheta))/r)/(2*np.pi)
        R_major = self.volume.J() / (2. * np.pi * mean_area)
        self._J = R_major
        
        if compute_derivatives > 0:
            bs = self.biotsavart
            booz_surf = self.boozer_surface
            iota = booz_surf.res['iota']
            G = booz_surf.res['G']
            P, L, U = booz_surf.res['PLU']
            dconstraint_dcoils_vjp = booz_surf.res['dconstraint_dcoils_vjp']
            
            # tack on dJ_diota = dJ_dG = 0 to the end of dJ_ds
            dJ_ds = np.zeros(L.shape[0])
            dj_ds = self.dJ_dsurfacecoefficients()
            dJ_ds[:dj_ds.size] = dj_ds
            adj = forward_backward(P, L, U, dJ_ds)
            
            adj_times_dg_dcoil, adj_times_dg_dcurr = dconstraint_dcoils_vjp(adj, booz_surf, iota, G, bs)
            self._dJ_by_dcoefficients = [- adj_dg_dc for adj_dg_dc in adj_times_dg_dcoil]
            self._dJ_by_dcoilcurrents = [- adj_dg_dc for adj_dg_dc in adj_times_dg_dcurr]

    def dJ_dsurfacecoefficients(self):
        """
        Return the objective value
        """
    
        self.surface.set_dofs(self.in_surface.get_dofs())
        surface = self.surface
    
        g = surface.gamma()
        g1 = surface.gammadash1()
        g2 = surface.gammadash2()
    
        dg_ds = surface.dgamma_by_dcoeff()
        dg1_ds = surface.dgammadash1_by_dcoeff()
        dg2_ds = surface.dgammadash2_by_dcoeff()
    
        x = g[:, :, 0, None]
        y = g[:, :, 1, None]
    
        dx_ds = dg_ds[:, :, 0, :]
        dy_ds = dg_ds[:, :, 1, :]
    
        r = np.sqrt(x**2+y**2)
        dr_ds = (x*dx_ds+y*dy_ds)/r
    
        xvarphi = g1[:, :, 0, None]
        yvarphi = g1[:, :, 1, None]
        zvarphi = g1[:, :, 2, None]
    
        xtheta = g2[:, :, 0, None]
        ytheta = g2[:, :, 1, None]
        ztheta = g2[:, :, 2, None]
    
        dxvarphi_ds = dg1_ds[:, :, 0, :]
        dyvarphi_ds = dg1_ds[:, :, 1, :]
        dzvarphi_ds = dg1_ds[:, :, 2, :]
    
        dxtheta_ds = dg2_ds[:, :, 0, :]
        dytheta_ds = dg2_ds[:, :, 1, :]
        dztheta_ds = dg2_ds[:, :, 2, :]
    
        mean_area = np.mean((1/r) * (ztheta*(x*yvarphi-y*xvarphi)-zvarphi*(x*ytheta-y*xtheta)))
        dmean_area_ds = np.mean((1/(r**2))*((xvarphi * y * ztheta - xtheta * y * zvarphi + x * (-yvarphi * ztheta + ytheta * zvarphi)) * dr_ds + r * (-zvarphi * (ytheta * dx_ds - y * dxtheta_ds - xtheta * dy_ds + x * dytheta_ds) + ztheta * (yvarphi * dx_ds - y * dxvarphi_ds - xvarphi * dy_ds + x * dyvarphi_ds) + (-xvarphi * y + x * yvarphi) * dztheta_ds + (xtheta * y - x * ytheta) * dzvarphi_ds)), axis=(0, 1))
    
        dR_major_ds = (-self.volume.J() * dmean_area_ds + self.volume.dJ_by_dsurfacecoefficients() * mean_area) / mean_area**2
        return dR_major_ds

    def J(self, compute_derivatives=0):
        assert compute_derivatives >= 0
        if self._J is None:
            self.compute(compute_derivatives=compute_derivatives)
        return self._J

    def dJ_by_dcoefficients(self, compute_derivatives=1):
        assert compute_derivatives >= 1
        if self._dJ_by_dcoefficients is None:
            self.compute(compute_derivatives=compute_derivatives)
        return self._dJ_by_dcoefficients

    def dJ_by_dcoilcurrents(self, compute_derivatives=1):
        assert compute_derivatives >= 1
        if self._dJ_by_dcoilcurrents is None:
            self.compute(compute_derivatives=compute_derivatives)
        return self._dJ_by_dcoilcurrents


class ToroidalFlux(object):

    r"""
    Given a surface and Biot Savart kernel, this objective calculates

    .. math::
       J &= \int_{S_{\varphi}} \mathbf{B} \cdot \mathbf{n} ~ds, \\
       &= \int_{S_{\varphi}} \text{curl} \mathbf{A} \cdot \mathbf{n} ~ds, \\
       &= \int_{\partial S_{\varphi}} \mathbf{A} \cdot \mathbf{t}~dl,

    where :math:`S_{\varphi}` is a surface of constant :math:`\varphi`, and :math:`\mathbf A` 
    is the magnetic vector potential.
    """

    def __init__(self, in_surface, biotsavart, stellarator, idx=0, boozer_surface=None):
        phis = np.linspace(0, 1/in_surface.nfp, sDIM, endpoint=False)
        thetas = np.linspace(0, 1., 2*sDIM, endpoint=False)
        self.surface = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        self.surface.set_dofs(in_surface.get_dofs())
        self.in_surface = in_surface




        self.stellarator = stellarator
        self.boozer_surface = boozer_surface
        self.biotsavart = biotsavart
        self.idx = idx
        self.clear_cached_properties()

    def clear_cached_properties(self):
        self._J = None
        self._dJ_by_dcoefficients = None
        self._dJ_by_dcoilcurrents = None

    def J(self, compute_derivatives=0):
        assert compute_derivatives >= 0
        if compute_derivatives > 0:
            assert self.boozer_surface is not None  # you need to initialize with a BoozerSurface to have access to Jacobian
        
        if self._J is None:
            self.compute(compute_derivatives=compute_derivatives)
        return self._J

    def dJ_by_dcoefficients(self, compute_derivatives=1):
        assert compute_derivatives >= 1 and self.boozer_surface is not None
        if self._dJ_by_dcoefficients is None:
            self.compute(compute_derivatives=compute_derivatives)
        return self._dJ_by_dcoefficients

    def dJ_by_dcoilcurrents(self, compute_derivatives=1):
        assert compute_derivatives >= 1 and self.boozer_surface is not None
        if self._dJ_by_dcoilcurrents is None:
            self.compute(compute_derivatives=compute_derivatives)
        return self._dJ_by_dcoilcurrents

    def compute(self, compute_derivatives=0):
        self.surface.set_dofs(self.in_surface.get_dofs())
        x = self.surface.gamma()[self.idx]
        self.biotsavart.set_points(x)
        
        xtheta = self.surface.gammadash2()[self.idx]
        ntheta = self.surface.gamma().shape[1]
        A = self.biotsavart.A()
        self._J = np.sum(A * xtheta)/ntheta
        
        if compute_derivatives > 0:
            booz_surf = self.boozer_surface
            iota = booz_surf.res['iota']
            G = booz_surf.res['G']
            P, L, U = booz_surf.res['PLU']
            dconstraint_dcoils_vjp = booz_surf.res['dconstraint_dcoils_vjp']
            
            dJ_by_dA = self.dJ_by_dA()
            dJ_by_dcoils = self.biotsavart.A_vjp(dJ_by_dA)
            
            # tack on dJ_diota = dJ_dG = 0 to the end of dJ_ds
            dJ_ds = np.zeros(L.shape[0])
            dj_ds = self.dJ_by_dsurfacecoefficients()
            dJ_ds[:dj_ds.size] = dj_ds
            adj = forward_backward(P, L, U, dJ_ds)
        
            adj_times_dg_dcoil, adj_times_dg_dcurr = dconstraint_dcoils_vjp(adj, booz_surf, iota, G, booz_surf.bs)
            self._dJ_by_dcoefficients = [dj_dc - adj_dg_dc for dj_dc, adj_dg_dc in zip(dJ_by_dcoils, adj_times_dg_dcoil)]

            dA_by_dcoilcurrents = self.biotsavart.dA_by_dcoilcurrents()
            dJ_dcurrents = [np.sum(dJ_by_dA*dA_dcurr) for dA_dcurr in dA_by_dcoilcurrents]
           
            self._dJ_by_dcoilcurrents = [dj_dc - adj_dg_dc for dj_dc, adj_dg_dc in zip(dJ_dcurrents, adj_times_dg_dcurr)]

    def dJ_by_dA(self):
        xtheta = self.surface.gammadash2()[self.idx]
        ntheta = self.surface.gamma().shape[1]
        dJ_by_dA = np.zeros((ntheta, 3))
        dJ_by_dA[:, ] = xtheta/ntheta
        return dJ_by_dA

    def dJ_by_dsurfacecoefficients(self):
        """
        Calculate the derivatives with respect to the surface coefficients.
        """
        ntheta = self.surface.gamma().shape[1]
        dA_by_dX = self.biotsavart.dA_by_dX()
        A = self.biotsavart.A()
        dgammadash2 = self.surface.gammadash2()[self.idx, :]
        dgammadash2_by_dc = self.surface.dgammadash2_by_dcoeff()[self.idx, :]

        dx_dc = self.surface.dgamma_by_dcoeff()[self.idx]
        dA_dc = np.sum(dA_by_dX[..., :, None] * dx_dc[..., None, :], axis=1)
        term1 = np.sum(dA_dc * dgammadash2[..., None], axis=(0, 1))
        term2 = np.sum(A[..., None] * dgammadash2_by_dc, axis=(0, 1))

        out = (term1+term2)/ntheta
        return out

    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
        """
        Calculate the second derivatives with respect to the surface coefficients.
        """
        ntheta = self.surface.gamma().shape[1]
        dx_dc = self.surface.dgamma_by_dcoeff()[self.idx]
        d2A_by_dXdX = self.biotsavart.d2A_by_dXdX().reshape((ntheta, 3, 3, 3))
        dA_by_dX = self.biotsavart.dA_by_dX()
        dA_dc = np.sum(dA_by_dX[..., :, None] * dx_dc[..., None, :], axis=1)
        d2A_dcdc = np.einsum('jkpl,jpn,jkm->jlmn', d2A_by_dXdX, dx_dc, dx_dc, optimize=True)

        dgammadash2 = self.surface.gammadash2()[self.idx]
        dgammadash2_by_dc = self.surface.dgammadash2_by_dcoeff()[self.idx]

        term1 = np.sum(d2A_dcdc * dgammadash2[..., None, None], axis=-3)
        term2 = np.sum(dA_dc[..., :, None] * dgammadash2_by_dc[..., None, :], axis=-3)
        term3 = np.sum(dA_dc[..., None, :] * dgammadash2_by_dc[..., :, None], axis=-3)

        out = (1/ntheta) * np.sum(term1+term2+term3, axis=0)
        return out





#class ToroidalFlux(object):
#
#    r"""
#    Given a surface and Biot Savart kernel, this objective calculates
#
#    .. math::
#       J &= \int_{S_{\varphi}} \mathbf{B} \cdot \mathbf{n} ~ds, \\
#       &= \int_{S_{\varphi}} \text{curl} \mathbf{A} \cdot \mathbf{n} ~ds, \\
#       &= \int_{\partial S_{\varphi}} \mathbf{A} \cdot \mathbf{t}~dl,
#
#    where :math:`S_{\varphi}` is a surface of constant :math:`\varphi`, and :math:`\mathbf A` 
#    is the magnetic vector potential.
#    """
#
#    def __init__(self, surface, biotsavart, stellarator, idx=0, boozer_surface=None):
#        self.stellarator = stellarator
#        self.surface = surface
#        self.boozer_surface = boozer_surface
#        self.biotsavart = biotsavart
#        self.idx = idx
#        self.clear_cached_properties()
#
#    def clear_cached_properties(self):
#        self._J = None
#        self._dJ_by_dcoefficients = None
#        self._dJ_by_dcoilcurrents = None
#        x = self.surface.gamma()[self.idx]
#        self.biotsavart.set_points(x)
#
#    def J(self, compute_derivatives=0):
#        assert compute_derivatives >= 0
#        if compute_derivatives > 0:
#            assert self.boozer_surface is not None  # you need to initialize with a BoozerSurface to have access to Jacobian
#        
#        if self._J is None:
#            self.compute(compute_derivatives=compute_derivatives)
#        return self._J
#
#    def dJ_by_dcoefficients(self, compute_derivatives=1):
#        assert compute_derivatives >= 1 and self.boozer_surface is not None
#        if self._dJ_by_dcoefficients is None:
#            self.compute(compute_derivatives=compute_derivatives)
#        return self._dJ_by_dcoefficients
#
#    def dJ_by_dcoilcurrents(self, compute_derivatives=1):
#        assert compute_derivatives >= 1 and self.boozer_surface is not None
#        if self._dJ_by_dcoilcurrents is None:
#            self.compute(compute_derivatives=compute_derivatives)
#        return self._dJ_by_dcoilcurrents
#
#    def compute(self, compute_derivatives=0):
#        xtheta = self.surface.gammadash2()[self.idx]
#        ntheta = self.surface.gamma().shape[1]
#        A = self.biotsavart.A()
#        self._J = np.sum(A * xtheta)/ntheta
#        
#        if compute_derivatives > 0:
#            booz_surf = self.boozer_surface
#            iota = booz_surf.res['iota']
#            G = booz_surf.res['G']
#            P, L, U = booz_surf.res['PLU']
#            dconstraint_dcoils_vjp = booz_surf.res['dconstraint_dcoils_vjp']
#            
#            dJ_by_dA = self.dJ_by_dA()
#            dJ_by_dcoils = self.biotsavart.A_vjp(dJ_by_dA)
#            
#            # tack on dJ_diota = dJ_dG = 0 to the end of dJ_ds
#            dJ_ds = np.zeros(L.shape[0])
#            dj_ds = self.dJ_by_dsurfacecoefficients()
#            dJ_ds[:dj_ds.size] = dj_ds
#            adj = forward_backward(P, L, U, dJ_ds)
#        
#            adj_times_dg_dcoil, adj_times_dg_dcurr = dconstraint_dcoils_vjp(adj, booz_surf, iota, G, booz_surf.bs)
#            self._dJ_by_dcoefficients = [dj_dc - adj_dg_dc for dj_dc, adj_dg_dc in zip(dJ_by_dcoils, adj_times_dg_dcoil)]
#
#            dA_by_dcoilcurrents = self.biotsavart.dA_by_dcoilcurrents()
#            dJ_dcurrents = [np.sum(dJ_by_dA*dA_dcurr) for dA_dcurr in dA_by_dcoilcurrents]
#           
#            self._dJ_by_dcoilcurrents = [dj_dc - adj_dg_dc for dj_dc, adj_dg_dc in zip(dJ_dcurrents, adj_times_dg_dcurr)]
#
#    def dJ_by_dA(self):
#        xtheta = self.surface.gammadash2()[self.idx]
#        ntheta = self.surface.gamma().shape[1]
#        dJ_by_dA = np.zeros((ntheta, 3))
#        dJ_by_dA[:, ] = xtheta/ntheta
#        return dJ_by_dA
#
#    def dJ_by_dsurfacecoefficients(self):
#        """
#        Calculate the derivatives with respect to the surface coefficients.
#        """
#        ntheta = self.surface.gamma().shape[1]
#        dA_by_dX = self.biotsavart.dA_by_dX()
#        A = self.biotsavart.A()
#        dgammadash2 = self.surface.gammadash2()[self.idx, :]
#        dgammadash2_by_dc = self.surface.dgammadash2_by_dcoeff()[self.idx, :]
#
#        dx_dc = self.surface.dgamma_by_dcoeff()[self.idx]
#        dA_dc = np.sum(dA_by_dX[..., :, None] * dx_dc[..., None, :], axis=1)
#        term1 = np.sum(dA_dc * dgammadash2[..., None], axis=(0, 1))
#        term2 = np.sum(A[..., None] * dgammadash2_by_dc, axis=(0, 1))
#
#        out = (term1+term2)/ntheta
#        return out
#
#    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
#        """
#        Calculate the second derivatives with respect to the surface coefficients.
#        """
#        ntheta = self.surface.gamma().shape[1]
#        dx_dc = self.surface.dgamma_by_dcoeff()[self.idx]
#        d2A_by_dXdX = self.biotsavart.d2A_by_dXdX().reshape((ntheta, 3, 3, 3))
#        dA_by_dX = self.biotsavart.dA_by_dX()
#        dA_dc = np.sum(dA_by_dX[..., :, None] * dx_dc[..., None, :], axis=1)
#        d2A_dcdc = np.einsum('jkpl,jpn,jkm->jlmn', d2A_by_dXdX, dx_dc, dx_dc, optimize=True)
#
#        dgammadash2 = self.surface.gammadash2()[self.idx]
#        dgammadash2_by_dc = self.surface.dgammadash2_by_dcoeff()[self.idx]
#
#        term1 = np.sum(d2A_dcdc * dgammadash2[..., None, None], axis=-3)
#        term2 = np.sum(dA_dc[..., :, None] * dgammadash2_by_dc[..., None, :], axis=-3)
#        term3 = np.sum(dA_dc[..., None, :] * dgammadash2_by_dc[..., :, None], axis=-3)
#
#        out = (1/ntheta) * np.sum(term1+term2+term3, axis=0)
#        return out


def boozer_surface_dexactresidual_dcoils_dcurrents_vjp(lm, booz_surf, iota, G, biotsavart):
    """
    For a given surface with points x on it, this function computes the
    vector-Jacobian product of \lm^T * dresidual_dcoils:

    lm^T dresidual_dcoils    = [G*lm - lm(2*||B_BS(x)|| (x_phi + iota * x_theta) ]^T * dB_dcoils
    lm^T dresidual_dcurrents = [G*lm - lm(2*||B_BS(x)|| (x_phi + iota * x_theta) ]^T * dB_dcurrents
    
    G is known for exact boozer surfaces, so if G=None is passed, then that
    value is used instead.
    """
    surface = booz_surf.surface
    user_provided_G = G is not None
    if not user_provided_G:
        G = 2. * np.pi * np.sum(np.abs(biotsavart.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))

    res, dres_dB = boozer_surface_residual_dB(surface, iota, G, biotsavart, derivatives=0)
    dres_dB = dres_dB.reshape((-1, 3, 3))

    lm_label = lm[-1]
    lmask = np.zeros(booz_surf.res["mask"].shape)
    lmask[booz_surf.res["mask"]] = lm[:-1]
    lm = lmask.reshape((-1, 3))
    
    lm_times_dres_dB = np.sum(lm[:, :, None] * dres_dB, axis=1).reshape((-1, 3))
    dres_dcoils = biotsavart.B_vjp(lm_times_dres_dB)
    dlabel_dcoils = booz_surf.label.dJ_by_dcoilcoefficients()
    dres_dcoils = [t1+lm_label*t2 for t1, t2 in zip(dres_dcoils, dlabel_dcoils)]

    dB_by_dcoilcurrents = biotsavart.dB_by_dcoilcurrents()
    dres_dcurrents = [np.sum(lm_times_dres_dB*dB_dcurr) for dB_dcurr in dB_by_dcoilcurrents]
    dlabel_dcoilcurrents = booz_surf.label.dJ_by_dcoilcurrents()
    dres_dcurrents = [t1+lm_label*t2 for t1, t2 in zip(dres_dcurrents, dlabel_dcoilcurrents)] 

    return dres_dcoils, dres_dcurrents
   
    #start = time.time()
    #resd2B_dcdc = sopp.res_dot_d2B_dcdc(d2B_by_dXdX, dx_dc, residual)
    #end = time.time()
    #print(end-start)
 
#@profile
def boozer_surface_residual_accumulate(surface, iota, G, biotsavart, derivatives=0):
    user_provided_G = G is not None
    if not user_provided_G:
        G = 2. * np.pi * np.sum(np.abs(biotsavart.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))

    x = surface.gamma()
    xphi = surface.gammadash1()
    xtheta = surface.gammadash2()
    nphi = x.shape[0]
    ntheta = x.shape[1]

    xsemiflat = x.reshape((x.size//3, 3)).copy()

    biotsavart.set_points(xsemiflat)

    biotsavart.compute(derivatives)
    B = biotsavart.B().reshape((nphi, ntheta, 3))

    tang = xphi + iota * xtheta
    B2 = np.sum(B**2, axis=2)
    residual = G*B - B2[..., None] * tang

    residual_flattened = residual.reshape((nphi*ntheta*3, ))
    r = residual_flattened
    val = 0.5 * np.sum(r**2)
    if derivatives == 0:
        return val,

    dx_dc = surface.dgamma_by_dcoeff()
    dxphi_dc = surface.dgammadash1_by_dcoeff()
    dxtheta_dc = surface.dgammadash2_by_dcoeff()
    nsurfdofs = dx_dc.shape[-1]

    dB_by_dX = biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
    dB_dc = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_dc, optimize=True)

    dresidual_dc = sopp.boozer_dresidual_dc(G, dB_dc, B, tang, B2, dxphi_dc, iota, dxtheta_dc)
    dresidual_diota = -B2[..., None] * xtheta

    dresidual_dc_flattened = dresidual_dc.reshape((nphi*ntheta*3, nsurfdofs))
    dresidual_diota_flattened = dresidual_diota.reshape((nphi*ntheta*3, 1))

    if user_provided_G:
        dresidual_dG = B
        dresidual_dG_flattened = dresidual_dG.reshape((nphi*ntheta*3, 1))
        J = np.concatenate((dresidual_dc_flattened, dresidual_diota_flattened, dresidual_dG_flattened), axis=1)
    else:
        J = np.concatenate((dresidual_dc_flattened, dresidual_diota_flattened), axis=1)
    
    dval = np.sum(r[:, None]*J, axis=0)
    if derivatives == 1:
        return val, dval
    
    #import ipdb;ipdb.set_trace()
    d2B_by_dXdX = biotsavart.d2B_by_dXdX().reshape((nphi, ntheta, 3, 3, 3))
    resd2B_dcdc = np.einsum('ijkpl,ijpn,ijkm,ijl->mn', d2B_by_dXdX, dx_dc, dx_dc, residual, optimize=True)
    dB2_dc = 2. * np.einsum('ijl,ijlm->ijm', B, dB_dc, optimize=True)
    resterm1 = np.einsum('ijlm,ijln,ijq->mn', dB_dc, dB_dc, residual*tang, optimize=True)
    resterm2 = np.einsum('ijkpl,ijpn,ijkm,ijl,ijq->mn', d2B_by_dXdX, dx_dc, dx_dc, B, residual*tang, optimize=['einsum_path', (3, 4), (0, 3), (0, 2), (0, 1)])
    resd2B2_dcdc = -2*(resterm1 + resterm2)
    
    tang_dc = dxphi_dc + iota * dxtheta_dc
    resterm1 = -np.einsum('ijm,ijkn,ijk->mn', dB2_dc, tang_dc, residual, optimize=True)
    resterm2 = -np.einsum('ijn,ijkm,ijk->mn', dB2_dc, tang_dc, residual, optimize=True)
    resd2residual_by_dcdc = G * resd2B_dcdc + resterm1 + resterm2 + resd2B2_dcdc
    
    d2residual_by_dcdiota = -(dB2_dc[..., None, :] * xtheta[..., :, None] + B2[..., None, None] * dxtheta_dc)
    resd2residual_by_dcdiota = np.sum(residual[...,None]*d2residual_by_dcdiota, axis=(0,1,2) )
    resd2residual_by_diotadiota = 0.
    
    d2val = J.T@J 
    if user_provided_G:
        resd2residual_by_dcdG = np.sum(residual[...,None]*dB_dc, axis=(0,1,2))
        resd2residual_by_diotadG = 0.
        resd2residual_by_dGdG = 0.
        # noqa turns out linting so that we can align everything neatly
        d2val[:nsurfdofs, :nsurfdofs]   += resd2residual_by_dcdc        # noqa (0, 0) dcdc
        d2val[:nsurfdofs, nsurfdofs]    += resd2residual_by_dcdiota    # noqa (0, 1) dcdiota
        d2val[:nsurfdofs, nsurfdofs+1]  += resd2residual_by_dcdG        # noqa (0, 2) dcdG
        d2val[nsurfdofs, :nsurfdofs]    += resd2residual_by_dcdiota     # noqa (1, 0) diotadc
        d2val[nsurfdofs, nsurfdofs]     += resd2residual_by_diotadiota  # noqa (1, 1) diotadiota
        d2val[nsurfdofs, nsurfdofs+1]   += resd2residual_by_diotadiota  # noqa (1, 2) diotadG
        d2val[nsurfdofs+1, :nsurfdofs]  += resd2residual_by_dcdG        # noqa (2, 0) dGdc
        d2val[nsurfdofs+1, nsurfdofs]   += resd2residual_by_diotadG     # noqa (2, 1) dGdiota
        d2val[nsurfdofs+1, nsurfdofs+1] += resd2residual_by_dGdG        # noqa (2, 2) dGdG
    else:
        d2val[:nsurfdofs, :nsurfdofs] += resd2residual_by_dcdc        # noqa (0, 0) dcdc
        d2val[:nsurfdofs, nsurfdofs]  += resd2residual_by_dcdiota     # noqa (0, 1) dcdiota
        d2val[nsurfdofs, :nsurfdofs]  += resd2residual_by_dcdiota     # noqa (1, 0) diotadc
        d2val[nsurfdofs, nsurfdofs]   += resd2residual_by_diotadiota  # noqa (1, 1) diotadiota

    return val, dval, d2val




#@profile
def boozer_surface_residual(surface, iota, G, biotsavart, derivatives=0):
    r"""
    For a given surface, this function computes the
    residual

    .. math::
        G\mathbf B_\text{BS}(\mathbf x) - ||\mathbf B_\text{BS}(\mathbf x)||^2  (\mathbf x_\varphi + \iota  \mathbf x_\theta)

    as well as the derivatives of this residual with respect to surface dofs,
    iota, and G.  In the above, :math:`\mathbf x` are points on the surface, :math:`\iota` is the
    rotational transform on that surface, and :math:`\mathbf B_{\text{BS}}` is the magnetic field
    computed using the Biot-Savart law.

    :math:`G` is known for exact boozer surfaces, so if ``G=None`` is passed, then that
    value is used instead.
    """

    user_provided_G = G is not None
    if not user_provided_G:
        G = 2. * np.pi * np.sum(np.abs(biotsavart.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))

    x = surface.gamma()
    xphi = surface.gammadash1()
    xtheta = surface.gammadash2()
    nphi = x.shape[0]
    ntheta = x.shape[1]

    xsemiflat = x.reshape((x.size//3, 3)).copy()

    biotsavart.set_points(xsemiflat)

    biotsavart.compute(derivatives)
    B = biotsavart.B().reshape((nphi, ntheta, 3))

    tang = xphi + iota * xtheta
    B2 = np.sum(B**2, axis=2)
    residual = G*B - B2[..., None] * tang

    residual_flattened = residual.reshape((nphi*ntheta*3, ))
    r = residual_flattened
    if derivatives == 0:
        return r,

    dx_dc = surface.dgamma_by_dcoeff()
    dxphi_dc = surface.dgammadash1_by_dcoeff()
    dxtheta_dc = surface.dgammadash2_by_dcoeff()
    nsurfdofs = dx_dc.shape[-1]

    dB_by_dX = biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
    dB_dc = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_dc, optimize=True)

    dresidual_dc = sopp.boozer_dresidual_dc(G, dB_dc, B, tang, B2, dxphi_dc, iota, dxtheta_dc)
    dresidual_diota = -B2[..., None] * xtheta

    dresidual_dc_flattened = dresidual_dc.reshape((nphi*ntheta*3, nsurfdofs))
    dresidual_diota_flattened = dresidual_diota.reshape((nphi*ntheta*3, 1))

    if user_provided_G:
        dresidual_dG = B
        dresidual_dG_flattened = dresidual_dG.reshape((nphi*ntheta*3, 1))
        J = np.concatenate((dresidual_dc_flattened, dresidual_diota_flattened, dresidual_dG_flattened), axis=1)
    else:
        J = np.concatenate((dresidual_dc_flattened, dresidual_diota_flattened), axis=1)
    if derivatives == 1:
        return r, J
    
    d2B_by_dXdX = biotsavart.d2B_by_dXdX().reshape((nphi, ntheta, 3, 3, 3))
    d2B_dcdc = np.einsum('ijkpl,ijpn,ijkm->ijlmn', d2B_by_dXdX, dx_dc, dx_dc, optimize=True)
    dB2_dc = 2. * np.einsum('ijl,ijlm->ijm', B, dB_dc, optimize=True)
    term1 = np.einsum('ijlm,ijln->ijmn', dB_dc, dB_dc, optimize=True)
    term2 = np.einsum('ijlmn,ijl->ijmn', d2B_dcdc, B, optimize=True)
    d2B2_dcdc = 2*(term1 + term2)

    term1 = -(dxphi_dc[..., None, :] + iota * dxtheta_dc[..., None, :]) * dB2_dc[..., None, :, None]
    term2 = -(dxphi_dc[..., :, None] + iota * dxtheta_dc[..., :, None]) * dB2_dc[..., None, None, :]
    term3 = -(xphi[..., None, None] + iota * xtheta[..., None, None]) * d2B2_dcdc[..., None, :, :]
    d2residual_by_dcdc = G * d2B_dcdc + term1 + term2 + term3
    d2residual_by_dcdiota = -(dB2_dc[..., None, :] * xtheta[..., :, None] + B2[..., None, None] * dxtheta_dc)
    d2residual_by_diotadiota = np.zeros(dresidual_diota.shape)

    d2residual_by_dcdc_flattened = d2residual_by_dcdc.reshape((nphi*ntheta*3, nsurfdofs, nsurfdofs))
    d2residual_by_dcdiota_flattened = d2residual_by_dcdiota.reshape((nphi*ntheta*3, nsurfdofs))
    d2residual_by_diotadiota_flattened = d2residual_by_diotadiota.reshape((nphi*ntheta*3,))

    if user_provided_G:
        d2residual_by_dcdG = dB_dc
        d2residual_by_diotadG = np.zeros(dresidual_diota.shape)
        d2residual_by_dGdG = np.zeros(dresidual_dG.shape)
        d2residual_by_dcdG_flattened = d2residual_by_dcdG.reshape((nphi*ntheta*3, nsurfdofs))
        d2residual_by_diotadG_flattened = d2residual_by_diotadG.reshape((nphi*ntheta*3,))
        d2residual_by_dGdG_flattened = d2residual_by_dGdG.reshape((nphi*ntheta*3,))
        H = np.zeros((nphi*ntheta*3, nsurfdofs + 2, nsurfdofs + 2))
        # noqa turns out linting so that we can align everything neatly
        H[:, :nsurfdofs, :nsurfdofs] = d2residual_by_dcdc_flattened        # noqa (0, 0) dcdc
        H[:, :nsurfdofs, nsurfdofs] = d2residual_by_dcdiota_flattened     # noqa (0, 1) dcdiota
        H[:, :nsurfdofs, nsurfdofs+1] = d2residual_by_dcdG_flattened        # noqa (0, 2) dcdG
        H[:, nsurfdofs, :nsurfdofs] = d2residual_by_dcdiota_flattened     # noqa (1, 0) diotadc
        H[:, nsurfdofs, nsurfdofs] = d2residual_by_diotadiota_flattened  # noqa (1, 1) diotadiota
        H[:, nsurfdofs, nsurfdofs+1] = d2residual_by_diotadiota_flattened  # noqa (1, 2) diotadG
        H[:, nsurfdofs+1, :nsurfdofs] = d2residual_by_dcdG_flattened        # noqa (2, 0) dGdc
        H[:, nsurfdofs+1, nsurfdofs] = d2residual_by_diotadG_flattened     # noqa (2, 1) dGdiota
        H[:, nsurfdofs+1, nsurfdofs+1] = d2residual_by_dGdG_flattened        # noqa (2, 2) dGdG
    else:
        H = np.zeros((nphi*ntheta*3, nsurfdofs + 1, nsurfdofs + 1))

        H[:, :nsurfdofs, :nsurfdofs] = d2residual_by_dcdc_flattened        # noqa (0, 0) dcdc
        H[:, :nsurfdofs, nsurfdofs] = d2residual_by_dcdiota_flattened     # noqa (0, 1) dcdiota
        H[:, nsurfdofs, :nsurfdofs] = d2residual_by_dcdiota_flattened     # noqa (1, 0) diotadc
        H[:, nsurfdofs, nsurfdofs] = d2residual_by_diotadiota_flattened  # noqa (1, 1) diotadiota

    return r, J, H


def boozer_surface_dlsqgrad_dcoils_vjp(lm, booz_surf, iota, G, biotsavart):
    """
    For a given surface with points x on it, this function computes the
    vector-Jacobian product of \lm^T * dlsqgrad_dcoils, \lm^T * dlsqgrad_dcurrents:

    lm^T dresidual_dcoils    = lm^T [dr_dsurface]^T[dr_dcoils]    + sum r_i lm^T d2ri_dsdc
    lm^T dresidual_dcurrents = lm^T [dr_dsurface]^T[dr_dcurrents] + sum r_i lm^T d2ri_dsdcurrents
    
    G is known for exact boozer surfaces, so if G=None is passed, then that
    value is used instead.
    """

    #lm_label = lm[-1]
    surface = booz_surf.surface
    # r, dr_dB, J, d2residual_dsurfacedB, d2residual_dsurfacedgradB
    boozer = boozer_surface_residual_dB(surface, iota, G, biotsavart, derivatives=1)
    r = boozer[0]
    dr_dB = boozer[1].reshape((-1, 3, 3))
    dr_ds = boozer[2]
    d2r_dsdB = boozer[3]
    d2r_dsdgradB = boozer[4]

    v1 = np.sum(np.sum(lm[:, None]*dr_ds.T, axis=0).reshape((-1, 3, 1)) * dr_dB, axis=1)
    v2 = np.sum(r.reshape((-1, 3, 1))*np.sum(lm[None, None, :]*d2r_dsdB, axis=-1).reshape((-1, 3, 3)), axis=1)
    v3 = np.sum(r.reshape((-1, 3, 1, 1))*np.sum(lm[None, None, None, :]*d2r_dsdgradB, axis=-1).reshape((-1, 3, 3, 3)), axis=1)
    dres_dcoils = biotsavart.B_and_dB_vjp(v1+v2, v3)
    dres_dcoils = [a + b for a, b in zip(dres_dcoils[0], dres_dcoils[1])]

    lm_times_dres_dB = v1 + v2
    lm_times_dres_dgradB = v3
    dB_by_dcoilcurrents = biotsavart.dB_by_dcoilcurrents()
    d2B_by_dXdcoilcurrents = biotsavart.d2B_by_dXdcoilcurrents()
    dres_dcurrents = [np.sum(lm_times_dres_dB*dB_dcurr) + np.sum(lm_times_dres_dgradB * dgradB_dcurr) for dB_dcurr, dgradB_dcurr in zip(dB_by_dcoilcurrents, d2B_by_dXdcoilcurrents)]

    return dres_dcoils, dres_dcurrents


def boozer_surface_residual_dB(surface, iota, G, biotsavart, derivatives=0):
    """
    For a given surface with points x on it, this function computes the
    differentiated residual

       d/dB[ G*B_BS(x) - ||B_BS(x)||^2 * (x_phi + iota * x_theta) ]

    as well as the derivatives of this residual with respect to surface dofs,
    iota, and G.

    G is known for exact boozer surfaces, so if G=None is passed, then that
    value is used instead.
    """

    user_provided_G = G is not None
    if not user_provided_G:
        G = 2. * np.pi * np.sum(np.abs(biotsavart.coil_currents)) * (4 * np.pi * 10**(-7) / (2 * np.pi))

    x = surface.gamma()
    xphi = surface.gammadash1()
    xtheta = surface.gammadash2()
    nphi = x.shape[0]
    ntheta = x.shape[1]
    
    xsemiflat = x.reshape((x.size//3, 3)).copy()

    biotsavart.set_points(xsemiflat)

    B = biotsavart.B().reshape((nphi, ntheta, 3))

    tang = xphi + iota * xtheta
    residual = G*B - np.sum(B**2, axis=2)[..., None] * tang
    
    GI = np.eye(3, 3) * G
    dresidual_dB = GI[None, None, :, :] - 2. * tang[:, :, :, None] * B[:, :, None, :]

    residual_flattened = residual.reshape((nphi*ntheta*3, ))
    dresidual_dB_flattened = dresidual_dB.reshape((nphi*ntheta*3, 3))
    r = residual_flattened
    dr_dB = dresidual_dB_flattened
    if derivatives == 0:
        return r, dr_dB

    dx_dc = surface.dgamma_by_dcoeff()
    dxphi_dc = surface.dgammadash1_by_dcoeff()
    dxtheta_dc = surface.dgammadash2_by_dcoeff()
    nsurfdofs = dx_dc.shape[-1]

    dB_by_dX = biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
    dB_dc = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_dc, optimize=True)
    dtang_dc = dxphi_dc + iota * dxtheta_dc
    dresidual_dc = G*dB_dc \
        - 2*np.sum(B[..., None]*dB_dc, axis=2)[:, :, None, :] * tang[..., None] \
        - np.sum(B**2, axis=2)[..., None, None] * dtang_dc 
    dresidual_diota = -np.sum(B**2, axis=2)[..., None] * xtheta

    d2residual_dcdB = -2*dB_dc[:, :, None, :, :] * tang[:, :, :, None, None] - 2*B[:, :, None, :, None] * dtang_dc[:, :, :, None, :]
    d2residual_diotadB = -2.*B[:, :, None, :] * xtheta[:, :, :, None]
    d2residual_dcdgradB = -2.*B[:, :, None, None, :, None]*dx_dc[:, :, None, :, None, :]*tang[:, :, :, None, None, None]
    idx = np.arange(3)
    d2residual_dcdgradB[:, :, idx, :, idx, :] += dx_dc * G

    dresidual_dc_flattened = dresidual_dc.reshape((nphi*ntheta*3, nsurfdofs))
    dresidual_diota_flattened = dresidual_diota.reshape((nphi*ntheta*3, 1))
    d2residual_dcdB_flattened = d2residual_dcdB.reshape((nphi*ntheta*3, 3, nsurfdofs))
    d2residual_diotadB_flattened = d2residual_diotadB.reshape((nphi*ntheta*3, 3, 1))
    d2residual_dcdgradB_flattened = d2residual_dcdgradB.reshape((nphi*ntheta*3, 3, 3, nsurfdofs))
    d2residual_diotadgradB_flattened = np.zeros((nphi*ntheta*3, 3, 3, 1)) 

    if user_provided_G:
        dresidual_dG = B
        dresidual_dG_flattened = dresidual_dG.reshape((nphi*ntheta*3, 1))
        J = np.concatenate((dresidual_dc_flattened, dresidual_diota_flattened, dresidual_dG_flattened), axis=1)
        
        d2residual_dGdB = np.ones((nphi*ntheta, 3, 3))
        d2residual_dGdB[:, :, :] = np.eye(3)[None, :, :]
        d2residual_dGdB = d2residual_dGdB.reshape((3*nphi*ntheta, 3, 1))
        d2residual_dGdgradB = np.zeros((3*nphi*ntheta, 3, 3, 1))

        d2residual_dsurfacedB = np.concatenate((d2residual_dcdB_flattened, 
                                                d2residual_diotadB_flattened, 
                                                d2residual_dGdB), axis=-1)
        d2residual_dsurfacedgradB = np.concatenate((d2residual_dcdgradB_flattened, 
                                                    d2residual_diotadgradB_flattened, 
                                                    d2residual_dGdgradB), axis=-1)
    else:
        J = np.concatenate((dresidual_dc_flattened, dresidual_diota_flattened), axis=1)
        d2residual_dsurfacedB = np.concatenate((d2residual_dcdB_flattened, d2residual_diotadB_flattened), axis=-1)
        d2residual_dsurfacedgradB = np.concatenate((d2residual_dcdgradB_flattened, 
                                                    d2residual_diotadgradB_flattened), axis=-1)
    
    if derivatives == 1:
        return r, dr_dB, J, d2residual_dsurfacedB, d2residual_dsurfacedgradB
    

#class NonQuasiAxisymmetricComponent(object):
#    r"""
#    This objective decomposes the field magnitude :math:`B(\varphi,\theta)` into quasiaxisymmetric and
#    non-quasiaxisymmetric components, then returns a penalty on the latter component.
#    
#    .. math::
#        J &= \frac{1}{2}\int_{\Gamma_{s}} (B-B_{\text{QS}})^2~dS
#          &= \frac{1}{2}\int_0^1 \int_0^1 (B - B_{\text{QS}})^2 \|\mathbf n\| ~d\varphi~\d\theta 
#    where
#    
#    .. math::
#        B &= \| \mathbf B(\varphi,\theta) \|_2
#        B_{\text{QS}} &= \frac{\int_0^1 \int_0^1 B \| n\| ~d\varphi ~d\theta}{\int_0^1 \int_0^1 \|\mathbf n\| ~d\varphi ~d\theta}
#    
#    """
#    def __init__(self, boozer_surface):
#        self.boozer_surface = boozer_surface
#        self.biotsavart = boozer_surface.bs
#        self.clear_cached_properties()
#
#    def clear_cached_properties(self):
#        self._J = None
#        self._dJ_by_dcoefficients = None
#        self._dJ_by_dcoilcurrents = None
#
#    def compute(self, compute_derivatives=0):
#        
#        # compute J
#        surface = self.boozer_surface.surface
#        nphi = surface.quadpoints_phi.size
#        ntheta = surface.quadpoints_theta.size
#
#        B = self.biotsavart.B()
#        B = B.reshape((nphi, ntheta, 3))
#        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
#        
#        nor = surface.normal()
#        dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)
#
#        B_QS = np.mean(modB * dS, axis=0) / np.mean(dS, axis=0)
#        B_nonQS = modB - B_QS[None, :]
#        self._J = 0.5 * np.mean(dS * B_nonQS**2) 
#        
#        if compute_derivatives > 0:
#            bs = self.biotsavart
#            booz_surf = self.boozer_surface
#            iota = booz_surf.res['iota']
#            G = booz_surf.res['G']
#            P, L, U = booz_surf.res['PLU']
#            dconstraint_dcoils_vjp = booz_surf.res['dconstraint_dcoils_vjp']
#
#            dJ_by_dB = self.dJ_by_dB().reshape((-1, 3))
#            dJ_by_dcoils = self.biotsavart.B_vjp(dJ_by_dB)
#
#            # tack on dJ_diota = dJ_dG = 0 to the end of dJ_ds
#            dJ_ds = np.concatenate((self.dJ_by_dsurfacecoefficients(), [0., 0.]))
#            adj = forward_backward(P, L, U, dJ_ds)
#            
#            adj_times_dg_dcoil, adj_times_dg_dcurr = dconstraint_dcoils_vjp(adj, booz_surf, iota, G, bs)
#            self._dJ_by_dcoefficients = [dj_dc - adj_dg_dc for dj_dc, adj_dg_dc in zip(dJ_by_dcoils, adj_times_dg_dcoil)]
#
#            dB_by_dcoilcurrents = self.biotsavart.dB_by_dcoilcurrents()
#            dJ_by_dcoilcurrents = [np.sum(dJ_by_dB*dB_dcurr) for dB_dcurr in dB_by_dcoilcurrents]
#            self._dJ_by_dcoilcurrents = [dj_dc - adj_dg_dc for dj_dc, adj_dg_dc in zip(dJ_by_dcoilcurrents, adj_times_dg_dcurr)]
#    
#    def J(self, compute_derivatives=0):
#        assert compute_derivatives >= 0
#        if self._J is None:
#            self.compute(compute_derivatives=compute_derivatives)
#        return self._J
#
#    def dJ_by_dcoefficients(self, compute_derivatives=1):
#        assert compute_derivatives >= 1
#        if self._dJ_by_dcoefficients is None:
#            self.compute(compute_derivatives=compute_derivatives)
#        return self._dJ_by_dcoefficients
#
#    def dJ_by_dcoilcurrents(self, compute_derivatives=1):
#        assert compute_derivatives >= 1
#        if self._dJ_by_dcoilcurrents is None:
#            self.compute(compute_derivatives=compute_derivatives)
#        return self._dJ_by_dcoilcurrents
#
#    def dJ_by_dB(self):
#        """
#        Return the partial derivative of the objective with respect to the magnetic field
#        """
#        surface = self.boozer_surface.surface
#        nphi = surface.quadpoints_phi.size
#        ntheta = surface.quadpoints_theta.size
#
#        B = self.biotsavart.B()
#        B = B.reshape((nphi, ntheta, 3))
#        
#        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
#        nor = surface.normal()
#        dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)
#
#        denom = np.mean(dS, axis=0)
#        B_QS = np.mean(modB * dS, axis=0) / denom
#        B_nonQS = modB - B_QS[None, :]
#        
#        dmodB_dB = B / modB[..., None]
#        dJ_by_dB = B_nonQS[..., None] * dmodB_dB * dS[:, :, None] / (nphi * ntheta)
#        return dJ_by_dB
#    
#    def dJ_by_dsurfacecoefficients(self):
#        """
#        Return the partial derivative of the objective with respect to the surface coefficients
#        """
#        surface = self.boozer_surface.surface
#        nphi = surface.quadpoints_phi.size
#        ntheta = surface.quadpoints_theta.size
#
#        B = self.biotsavart.B()
#        B = B.reshape((nphi, ntheta, 3))
#        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
#        
#        nor = surface.normal()
#        dnor_dc = surface.dnormal_by_dcoeff()
#        dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)
#        dS_dc = (nor[:, :, 0, None]*dnor_dc[:, :, 0, :] + nor[:, :, 1, None]*dnor_dc[:, :, 1, :] + nor[:, :, 2, None]*dnor_dc[:, :, 2, :])/dS[:, :, None]
#
#        B_QS = np.mean(modB * dS, axis=0) / np.mean(dS, axis=0)
#        B_nonQS = modB - B_QS[None, :]
#        
#        dB_by_dX = self.biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
#        dx_dc = surface.dgamma_by_dcoeff()
#        dB_dc = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_dc, optimize=True)
#        
#        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
#        dmodB_dc = (B[:, :, 0, None] * dB_dc[:, :, 0, :] + B[:, :, 1, None] * dB_dc[:, :, 1, :] + B[:, :, 2, None] * dB_dc[:, :, 2, :])/modB[:, :, None]
#        
#        num = np.mean(modB * dS, axis=0)
#        denom = np.mean(dS, axis=0)
#        dnum_dc = np.mean(dmodB_dc * dS[..., None] + modB[..., None] * dS_dc, axis=0) 
#        ddenom_dc = np.mean(dS_dc, axis=0)
#        B_QS_dc = (dnum_dc * denom[:, None] - ddenom_dc * num[:, None])/denom[:, None]**2
#        B_nonQS_dc = dmodB_dc - B_QS_dc[None, :]
#        
#        dJ_by_dc = np.mean(0.5 * dS_dc * B_nonQS[..., None]**2 + dS[..., None] * B_nonQS[..., None] * B_nonQS_dc, axis=(0, 1)) 
#        return dJ_by_dc
#

class Iotas(object):
    def __init__(self, boozer_surface):
        self.boozer_surface = boozer_surface
        self.biotsavart = boozer_surface.bs 
        self.clear_cached_properties()
    
    def clear_cached_properties(self):
        self._J = None
        self._dJ_by_dcoefficients = None
        self._dJ_by_dcoilcurrents = None
    
    def compute(self, compute_derivatives=0):
        self._J = self.boozer_surface.res['iota']
        
        if compute_derivatives > 0:
            bs = self.biotsavart
            booz_surf = self.boozer_surface
            iota = booz_surf.res['iota']
            G = booz_surf.res['G']
            P, L, U = booz_surf.res['PLU']
            dconstraint_dcoils_vjp = booz_surf.res['dconstraint_dcoils_vjp']

            # tack on dJ_diota = dJ_dG = 0 to the end of dJ_ds
            dJ_ds = np.zeros(L.shape[0])
            dJ_ds[-2] = 1.
            adj = forward_backward(P, L, U, dJ_ds)
            
            adj_times_dg_dcoil, adj_times_dg_dcurr = dconstraint_dcoils_vjp(adj, booz_surf, iota, G, bs)
            self._dJ_by_dcoefficients = [- adj_dg_dc for adj_dg_dc in adj_times_dg_dcoil]
            self._dJ_by_dcoilcurrents = [- adj_dg_dc for adj_dg_dc in adj_times_dg_dcurr]

    def J(self, compute_derivatives=0):
        assert compute_derivatives >= 0
        if self._J is None:
            self.compute(compute_derivatives=compute_derivatives)
        return self._J

    def dJ_by_dcoefficients(self, compute_derivatives=1):
        assert compute_derivatives >= 1
        if self._dJ_by_dcoefficients is None:
            self.compute(compute_derivatives=compute_derivatives)
        return self._dJ_by_dcoefficients

    def dJ_by_dcoilcurrents(self, compute_derivatives=1):
        assert compute_derivatives >= 1
        if self._dJ_by_dcoilcurrents is None:
            self.compute(compute_derivatives=compute_derivatives)
        return self._dJ_by_dcoilcurrents


#class MajorRadius(object): 
#    r"""
#    This objective computes the major radius of a toroidal surface with the following formula:
#    
#    R_major = (1/2*pi) * (V / mean_cross_sectional_area)
#    
#    where mean_cross_sectional_area = \int^1_0 \int^1_0 z_\theta(xy_\varphi -yx_\varphi) - z_\varphi(xy_\theta-yx_\theta) ~d\theta d\varphi
#    
#    """
#    def __init__(self, boozer_surface):
#        self.boozer_surface = boozer_surface
#        self.surface = boozer_surface.surface
#        self.biotsavart = boozer_surface.bs
#        self.clear_cached_properties()
#
#    def clear_cached_properties(self):
#        self._J = None
#        self._dJ_by_dcoefficients = None
#        self._dJ_by_dcoilcurrents = None
# 
#    def compute(self, compute_derivatives=0):
#
#        surface = self.surface
#    
#        g = surface.gamma()
#        g1 = surface.gammadash1()
#        g2 = surface.gammadash2()
#        
#        x = g[:, :, 0]
#        y = g[:, :, 1]
#        
#        r = np.sqrt(x**2+y**2)
#    
#        xvarphi = g1[:, :, 0]
#        yvarphi = g1[:, :, 1]
#        zvarphi = g1[:, :, 2]
#    
#        xtheta = g2[:, :, 0]
#        ytheta = g2[:, :, 1]
#        ztheta = g2[:, :, 2]
#    
#        mean_area = np.mean((ztheta*(x*yvarphi-y*xvarphi)-zvarphi*(x*ytheta-y*xtheta))/r)/(2*np.pi)
#        R_major = surface.volume() / (2. * np.pi * mean_area)
#        self._J = R_major
#        
#        if compute_derivatives > 0:
#            bs = self.biotsavart
#            booz_surf = self.boozer_surface
#            iota = booz_surf.res['iota']
#            G = booz_surf.res['G']
#            P, L, U = booz_surf.res['PLU']
#            dconstraint_dcoils_vjp = booz_surf.res['dconstraint_dcoils_vjp']
#            
#            # tack on dJ_diota = dJ_dG = 0 to the end of dJ_ds
#            dJ_ds = np.zeros(L.shape[0])
#            dj_ds = self.dJ_dsurfacecoefficients()
#            dJ_ds[:dj_ds.size] = dj_ds
#            adj = forward_backward(P, L, U, dJ_ds)
#            
#            adj_times_dg_dcoil, adj_times_dg_dcurr = dconstraint_dcoils_vjp(adj, booz_surf, iota, G, bs)
#            self._dJ_by_dcoefficients = [- adj_dg_dc for adj_dg_dc in adj_times_dg_dcoil]
#            self._dJ_by_dcoilcurrents = [- adj_dg_dc for adj_dg_dc in adj_times_dg_dcurr]
#
#    def dJ_dsurfacecoefficients(self):
#        """
#        Return the objective value
#        """
#    
#        surface = self.surface
#    
#        g = surface.gamma()
#        g1 = surface.gammadash1()
#        g2 = surface.gammadash2()
#    
#        dg_ds = surface.dgamma_by_dcoeff()
#        dg1_ds = surface.dgammadash1_by_dcoeff()
#        dg2_ds = surface.dgammadash2_by_dcoeff()
#    
#        x = g[:, :, 0, None]
#        y = g[:, :, 1, None]
#    
#        dx_ds = dg_ds[:, :, 0, :]
#        dy_ds = dg_ds[:, :, 1, :]
#    
#        r = np.sqrt(x**2+y**2)
#        dr_ds = (x*dx_ds+y*dy_ds)/r
#    
#        xvarphi = g1[:, :, 0, None]
#        yvarphi = g1[:, :, 1, None]
#        zvarphi = g1[:, :, 2, None]
#    
#        xtheta = g2[:, :, 0, None]
#        ytheta = g2[:, :, 1, None]
#        ztheta = g2[:, :, 2, None]
#    
#        dxvarphi_ds = dg1_ds[:, :, 0, :]
#        dyvarphi_ds = dg1_ds[:, :, 1, :]
#        dzvarphi_ds = dg1_ds[:, :, 2, :]
#    
#        dxtheta_ds = dg2_ds[:, :, 0, :]
#        dytheta_ds = dg2_ds[:, :, 1, :]
#        dztheta_ds = dg2_ds[:, :, 2, :]
#    
#        mean_area = np.mean((1/r) * (ztheta*(x*yvarphi-y*xvarphi)-zvarphi*(x*ytheta-y*xtheta)))
#        dmean_area_ds = np.mean((1/(r**2))*((xvarphi * y * ztheta - xtheta * y * zvarphi + x * (-yvarphi * ztheta + ytheta * zvarphi)) * dr_ds + r * (-zvarphi * (ytheta * dx_ds - y * dxtheta_ds - xtheta * dy_ds + x * dytheta_ds) + ztheta * (yvarphi * dx_ds - y * dxvarphi_ds - xvarphi * dy_ds + x * dyvarphi_ds) + (-xvarphi * y + x * yvarphi) * dztheta_ds + (xtheta * y - x * ytheta) * dzvarphi_ds)), axis=(0, 1))
#    
#        dR_major_ds = (-surface.volume() * dmean_area_ds + surface.dvolume_by_dcoeff() * mean_area) / mean_area**2
#        return dR_major_ds
#
#    def J(self, compute_derivatives=0):
#        assert compute_derivatives >= 0
#        if self._J is None:
#            self.compute(compute_derivatives=compute_derivatives)
#        return self._J
#
#    def dJ_by_dcoefficients(self, compute_derivatives=1):
#        assert compute_derivatives >= 1
#        if self._dJ_by_dcoefficients is None:
#            self.compute(compute_derivatives=compute_derivatives)
#        return self._dJ_by_dcoefficients
#
#    def dJ_by_dcoilcurrents(self, compute_derivatives=1):
#        assert compute_derivatives >= 1
#        if self._dJ_by_dcoilcurrents is None:
#            self.compute(compute_derivatives=compute_derivatives)
#        return self._dJ_by_dcoilcurrents



class AreaElement(object): 
    def __init__(self, boozer_surface, target_area=None):
        self.boozer_surface = boozer_surface
        self.biotsavart = boozer_surface.bs
        self.surface = boozer_surface.surface
        
        if target_area is None:
            N = np.linalg.norm(self.surface.normal(), axis=2)
            self.target_area = np.mean(N)
        else:
            self.target_area = target_area

        self.clear_cached_properties()

    def clear_cached_properties(self):
        self._J = None
        self._dJ_by_dcoefficients = None
        self._dJ_by_dcoilcurrents = None
 
    def dJ_dsurfacecoefficients(self):
        surface = self.surface
        nor = surface.normal()
        dnor_dc = surface.dnormal_by_dcoeff()
        dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)
        dS_dc = (nor[:, :, 0, None]*dnor_dc[:, :, 0, :] + nor[:, :, 1, None]*dnor_dc[:, :, 1, :] + nor[:, :, 2, None]*dnor_dc[:, :, 2, :])/dS[:, :, None]
        dJ_ds = -np.mean( np.maximum(self.target_area - dS, 0.)[:, :, None] * dS_dc, axis=(0,1))/self.target_area**2
        return dJ_ds

    def compute(self, compute_derivatives=0):
        N = np.linalg.norm(self.surface.normal(), axis=2)
        self._J = 0.5* np.mean( np.maximum(self.target_area - N, 0.)**2. )/self.target_area**2
        
        if compute_derivatives > 0:
            bs = self.biotsavart
            booz_surf = self.boozer_surface
            iota = booz_surf.res['iota']
            G = booz_surf.res['G']
            P, L, U = booz_surf.res['PLU']
            dconstraint_dcoils_vjp = booz_surf.res['dconstraint_dcoils_vjp']
            
            # tack on dJ_diota = dJ_dG = 0 to the end of dJ_ds
            dJ_ds = np.zeros(L.shape[0])
            dj_ds = self.dJ_dsurfacecoefficients()
            dJ_ds[:dj_ds.size] = dj_ds
            adj = forward_backward(P, L, U, dJ_ds)
            
            adj_times_dg_dcoil, adj_times_dg_dcurr = dconstraint_dcoils_vjp(adj, booz_surf, iota, G, bs)
            self._dJ_by_dcoefficients = [- adj_dg_dc for adj_dg_dc in adj_times_dg_dcoil]
            self._dJ_by_dcoilcurrents = [- adj_dg_dc for adj_dg_dc in adj_times_dg_dcurr]

    def J(self, compute_derivatives=0):
        assert compute_derivatives >= 0
        if self._J is None:
            self.compute(compute_derivatives=compute_derivatives)
        return self._J

    def dJ_by_dcoefficients(self, compute_derivatives=1):
        assert compute_derivatives >= 1
        if self._dJ_by_dcoefficients is None:
            self.compute(compute_derivatives=compute_derivatives)
        return self._dJ_by_dcoefficients

    def dJ_by_dcoilcurrents(self, compute_derivatives=1):
        assert compute_derivatives >= 1
        if self._dJ_by_dcoilcurrents is None:
            self.compute(compute_derivatives=compute_derivatives)
        return self._dJ_by_dcoilcurrents


class AreaPenalty(object): 
    def __init__(self, boozer_surface, max_area=None):
        self.boozer_surface = boozer_surface
        self.biotsavart = boozer_surface.bs
        self.surface = boozer_surface.surface
        
        if max_area is None:
            self.max_area = self.surface.area()
        else:
            self.max_area = max_area

        self.clear_cached_properties()

    def clear_cached_properties(self):
        self._J = None
        self._dJ_by_dcoefficients = None
        self._dJ_by_dcoilcurrents = None
 
    def dJ_dsurfacecoefficients(self):
        surface = self.surface
        dJ_ds = np.max([self.surface.area() - self.max_area, 0.]) * surface.darea_by_dcoeff() / self.max_area**2
        return dJ_ds

    def compute(self, compute_derivatives=0):
        self._J = 0.5*np.max([self.surface.area() - self.max_area, 0.])**2 / self.max_area**2
        
        if compute_derivatives > 0:
            bs = self.biotsavart
            booz_surf = self.boozer_surface
            iota = booz_surf.res['iota']
            G = booz_surf.res['G']
            P, L, U = booz_surf.res['PLU']
            dconstraint_dcoils_vjp = booz_surf.res['dconstraint_dcoils_vjp']
            
            # tack on dJ_diota = dJ_dG = 0 to the end of dJ_ds
            dJ_ds = np.zeros(L.shape[0])
            dj_ds = self.dJ_dsurfacecoefficients()
            dJ_ds[:dj_ds.size] = dj_ds
            adj = forward_backward(P, L, U, dJ_ds)
            
            adj_times_dg_dcoil, adj_times_dg_dcurr = dconstraint_dcoils_vjp(adj, booz_surf, iota, G, bs)
            self._dJ_by_dcoefficients = [- adj_dg_dc for adj_dg_dc in adj_times_dg_dcoil]
            self._dJ_by_dcoilcurrents = [- adj_dg_dc for adj_dg_dc in adj_times_dg_dcurr]

    def J(self, compute_derivatives=0):
        assert compute_derivatives >= 0
        if self._J is None:
            self.compute(compute_derivatives=compute_derivatives)
        return self._J

    def dJ_by_dcoefficients(self, compute_derivatives=1):
        assert compute_derivatives >= 1
        if self._dJ_by_dcoefficients is None:
            self.compute(compute_derivatives=compute_derivatives)
        return self._dJ_by_dcoefficients

    def dJ_by_dcoilcurrents(self, compute_derivatives=1):
        assert compute_derivatives >= 1
        if self._dJ_by_dcoilcurrents is None:
            self.compute(compute_derivatives=compute_derivatives)
        return self._dJ_by_dcoilcurrents


