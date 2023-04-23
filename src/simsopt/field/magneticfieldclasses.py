class SolovevField(MagneticField):
    """
    Axisymmetric (d_phi=0) equilibrium (d_t = 0) magnetic field corresponding to Solov'ev's 
    set of exact solutions to the Grad-Shafranov equation. For reference, see Gurnett & 
    Bhattacharjee's textbook Ch 7 Sect 1. However the solution here has the free parameters 
    re-scaled such that one can easily tune basic properties of the field. For reference, 
    see Jardin's textbook Ch 4. Sect 4.4.2. 
    
    To Do:
    This solution is valid only in the range 0<psi<psi_B, as stated just above Jardin Eqn. 4.39
    however this constraint is not included in this code. I simply do not simulate particle motion
    outside that boundary. 
    
    Args:
    r0: Radius of minor axis [Note: this argument is included but not used]
    R0: Radius of major axis
    B0: Toroidal field magnitude at R0
    q0: Safety factor at R0
    k0: Ellipticity/elongation. Roughly k0*r0 is the height of the vessel.
    """
    
    def __init__(self,r0,R0,B0,q0,k0):
        MagneticField.__init__(self)
        self.r0 = r0
        self.R0 = R0
        self.B0 = B0
        self.q0 = q0
        self.k0 = k0
        
    def _B_impl(self, B):
        points = self.get_points_cart_ref()
        
        x = points[:,0]
        y = points[:,1]
        z = points[:,2]

        rho = np.sqrt(np.square(x) + np.square(y))
        phi = np.arctan2(y,x)

        rhoUnitVector = np.vstack(( np.cos(phi), np.sin(phi), np.zeros(len(rho)) )).T
        phiUnitVector = np.vstack(( -np.sin(phi), np.cos(phi), np.zeros(len(phi)) )).T
        zUnitVector = np.vstack(( np.zeros(len(z)), np.zeros(len(z)), np.ones(len(z)) )).T
        
        B[:] = (
            rhoUnitVector * np.vstack( -self.B0*rho*z/(np.square(self.R0)*self.k0*self.q0 ))
            + phiUnitVector * np.vstack( self.B0*self.R0/rho )
            + zUnitVector * np.vstack( self.B0/(2*np.square(self.R0)*self.k0*self.q0) * ( 2*np.square(z)+np.square(self.k0)*(np.square(rho)-np.square(self.R0)) ) )
        )
        
    def _dB_by_dX_impl(self,dB):
        points = self.get_points_cart_ref()
        
        x = points[:,0]
        y = points[:,1]
        z = points[:,2]
        
        rho_4th = np.square(np.square(x) + np.square(y))

        A0 = np.divide(self.B0*self.k0, self.q0*np.square(self.R0))
        
        dBdx = np.vstack([2*self.B0*self.R0*x*y/rho_4th-A0*z/self.k0**2,
                         self.B0*self.R0*(y**2-x**2)/rho_4th,
                         A0*x])
        dBdy = np.vstack([self.B0*self.R0*(y**2-x**2)/rho_4th,
                         -2*self.B0*self.R0*x*y/rho_4th-A0*z/self.k0**2,
                         A0*y])
        dBdz = np.vstack([-A0*x/self.k0**2,
                         -A0*y/self.k0**2,
                         2*A0*z/self.k0**2])
        dB[:] = np.array([dBdx, dBdy, dBdz]).T
