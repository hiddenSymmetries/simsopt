from simsopt.geo.magneticfield import MagneticField
from scipy.special import ellipk, ellipe
from math import factorial
import jax.numpy as jnp
from jax import grad, jit
import numpy as np
try:
    from sympy.parsing.sympy_parser import parse_expr
    import sympy as sp
    sympy_found = True
except:
    sympy_found = False
    pass

class ToroidalField(MagneticField):
    '''Magnetic field purely in the toroidal direction, that is, in the phi direction with (R,phi,Z) the standard cylindrical coordinates.
       Its modulus is given by B = B0*R0/R where R0 is the first input and B0 the second input to the function.    

    Args:
        B0:  modulus of the magnetic field at R0
        R0:  radius of normalization
    '''
    def __init__(self, R0, B0):
        self.R0=R0
        self.B0=B0

    def compute(self, points, compute_derivatives=0):
        assert compute_derivatives <= 2

        phi = np.arctan2(points[:,1],points[:,0])
        R   = np.sqrt(np.power(points[:,0],2) + np.power(points[:,1],2))
        phiUnitVectorOverR = np.vstack((np.divide(-np.sin(phi),R),np.divide(np.cos(phi),R),np.zeros(len(phi)))).T
        self._B = np.multiply(self.B0*self.R0,phiUnitVectorOverR)

        x=points[:,0]
        y=points[:,1]

        if compute_derivatives >= 1:

            dB_by_dX1=np.vstack((
                np.multiply(np.divide(self.B0*self.R0,R**4),2*np.multiply(x,y)),
                np.multiply(np.divide(self.B0*self.R0,R**4),y**2-x**2),
                0*R))
            dB_by_dX2=np.vstack((
                np.multiply(np.divide(self.B0*self.R0,R**4),y**2-x**2),
                np.multiply(np.divide(self.B0*self.R0,R**4),-2*np.multiply(x,y)),
                0*R))
            dB_by_dX3=np.vstack((0*R,0*R,0*R))

            dToroidal_by_dX=np.array([dB_by_dX1,dB_by_dX2,dB_by_dX3]).T

            self._dB_by_dX = dToroidal_by_dX

        if compute_derivatives >= 2:
            self._d2B_by_dXdX = 2*self.B0*self.R0*np.array([(1/(point[0]**2+point[1]**2)**3)*np.array([
                    [ [ 3*point[0]**2+point[1]**3         , point[0]**3-3*point[0]*point[1]**2, 0] , [ point[0]**3-3*point[0]*point[1]**2, 3*point[0]**2*point[1]-point[1]**3 , 0] , [ 0, 0, 0] ],
                    [ [ point[0]**3-3*point[0]*point[1]**2, 3*point[0]**2*point[1]-point[1]**3, 0] , [ 3*point[0]**2*point[1]-point[1]**3, -point[0]**3+3*point[0]*point[1]**2, 0] , [ 0, 0, 0] ],
                    [ [ 0                                 , 0                                 , 0] , [ 0                                 , 0                                  , 0] , [ 0, 0, 0] ] ])
            for point in points])

        return self

    def compute_A(self, points, compute_derivatives=0):
        assert compute_derivatives <= 2

        self._A = self.B0*self.R0*np.array([
            [point[2]*point[0]/(point[0]**2+point[1]**2),
             point[2]*point[1]/(point[0]**2+point[1]**2),
            0] for point in points])

        if compute_derivatives >= 1:
            self._dA_by_dX = self.B0*self.R0*np.array([(point[2]/(point[0]**2+point[1]**2)**2)*np.array(
                [[-point[0]**2+point[1]**2,-2*point[0]*point[1],0],
                [-2*point[0]*point[1],point[0]**2-point[1]**2,0],
                [point[0]*(point[0]**2+point[1]**2)/point[2],point[1]*(point[0]**2+point[1]**2)/point[2],0]])
                for point in points])

        if compute_derivatives >= 2:
            self._d2A_by_dXdX = 2*self.B0*self.R0*np.array([(point[2]/(point[0]**2+point[1]**2)**3)*np.array([
                    [ [ point[0]**3-3*point[0]*point[1]**2, 3*point[0]**2*point[1]-point[1]**3, (-point[0]**4+point[1]**4)/(2*point[2])                ] , [ 3*point[0]**2*point[1]-point[1]**3,-point[0]**3+3*point[0]*point[1]**2, -point[0]*point[1]*(point[0]**2+point[1]**2)/point[2]] , [ (-point[0]**4+point[1]**4)/(2*point[2])               , -point[0]*point[1]*(point[0]**2+point[1]**2)/point[2], 0 ] ],
                    [ [ 3*point[0]**2*point[1]-point[1]**3,-point[0]**3+3*point[0]*point[1]**2,  -point[0]*point[1]*(point[0]**2+point[1]**2)/point[2] ] , [-point[0]**3+3*point[0]*point[1]**2,-3*point[0]**2*point[1]+point[1]**3, (point[0]**4-point[1]**4)/(2*point[2])]                 , [ -point[0]*point[1]*(point[0]**2+point[1]**2)/point[2], (point[0]**4-point[1]**4)/(2*point[2])               , 0 ] ],
                    [ [ 0                                 , 0                                 , 0                                                      ] , [ 0                                 , 0                                 , 0                                     ]                 , [ 0                                                    , 0                                                    , 0 ] ]])
            for point in points])

class ScalarPotentialRZMagneticField(MagneticField):
    '''Vacuum magnetic field as a solution of B = grad(Phi) where Phi is the magnetic field scalar potential.
       It takes Phi as an input string, which should contain an expression involving the standard cylindrical coordinates (R, phi, Z)
       Example: ScalarPotentialRZMagneticField("2*phi") yields a magnetic field B = grad(2*phi) = (0,2/R,0).
       Note: this function needs sympy.    

    Args:
        PhiStr:  string containing vacuum scalar potential expression as a function of R, Z and phi
    '''
    def __init__(self, PhiStr):
        if not sympy_found:
            raise RuntimeError("Sympy is required for the ScalarPotentialRZMagneticField class")
        self.PhiStr = PhiStr
        self.Phiparsed = parse_expr(PhiStr)
        R,Z,Phi = sp.symbols('R Z phi')
        self.Blambdify = sp.lambdify((R, Z, Phi), [self.Phiparsed.diff(R), self.Phiparsed.diff(Phi)/R, self.Phiparsed.diff(Z)])
        self.dBlambdify_by_dX = sp.lambdify((R, Z, Phi), 
                            [[sp.cos(Phi)*self.Phiparsed.diff(R).diff(R)-(sp.sin(Phi)/R)*self.Phiparsed.diff(R).diff(Phi),sp.cos(Phi)*(self.Phiparsed.diff(Phi)/R).diff(R)-(sp.sin(Phi)/R)*(self.Phiparsed.diff(Phi)/R).diff(Phi),sp.cos(Phi)*self.Phiparsed.diff(Z).diff(R)-(sp.sin(Phi)/R)*self.Phiparsed.diff(Z).diff(Phi)],
                             [sp.sin(Phi)*self.Phiparsed.diff(R).diff(R)+(sp.cos(Phi)/R)*self.Phiparsed.diff(R).diff(Phi),sp.sin(Phi)*(self.Phiparsed.diff(Phi)/R).diff(R)+(sp.cos(Phi)/R)*(self.Phiparsed.diff(Phi)/R).diff(Phi),sp.sin(Phi)*self.Phiparsed.diff(Z).diff(R)+(sp.cos(Phi)/R)*self.Phiparsed.diff(Z).diff(Phi)],
                             [self.Phiparsed.diff(R).diff(Z),(self.Phiparsed.diff(Phi)/R).diff(Z),self.Phiparsed.diff(Z).diff(Z)]])
        
    def compute(self, points, compute_derivatives=0):
        assert compute_derivatives <= 2

        r   = np.sqrt(np.power(points[:,0],2) + np.power(points[:,1],2))
        z   = points[:,2]
        phi = np.arctan2(points[:,1],points[:,0])
        self._B = [self.Blambdify(r[i],z[i],phi[i]) for i in range(len(r))]

        if compute_derivatives >= 1:
            self._dB_by_dX = [self.dBlambdify_by_dX(r[i],z[i],phi[i]) for i in range(len(r))]

        if compute_derivatives >= 2:
            self._d2B_by_dXdX = None
            raise RuntimeError("Second derivative of scalar potential magnetic field not implemented yet")


class CircularCoil(MagneticField):
    '''Magnetic field created by a single circular coil evaluated using analytical functions, including complete elliptic integrals of the first and second kind.
    As inputs, it takes the radius of the coil (r0), its center, current (I) and its normal vector spherical angle components (normal=[theta,phi]).

    Args:
        r0: radius of the coil
        center: point at the coil center
        I: current of the coil in Ampere's
        normal: spherical angles theta and phi of the normal vector to the plane of the coil centered at the coil center
    '''
    def __init__(self, r0=0.1, center=[0,0,0], I=5e5/np.pi, normal=[0,0]):
        self.r0     = r0
        self.Inorm  = I*4e-7
        self.center = center
        self.normal = normal
        self.rotMatrix = np.array([[np.cos(normal[1]), np.sin(normal[0])*np.sin(normal[1]),  np.cos(normal[0])*np.sin(normal[1])],
                                   [0,                 np.cos(normal[0])                  , -np.sin(normal[0])],
                                   [np.sin(normal[1]), np.sin(normal[0])*np.cos(normal[1]),  np.cos(normal[0])*np.cos(normal[1])]])
        self.rotMatrixInv = np.array(self.rotMatrix.T)

    def compute(self, points, compute_derivatives=0):
        assert compute_derivatives <= 2

        points = np.array(np.dot(self.rotMatrix,np.array([np.subtract(point,self.center) for point in points]).T).T)
        rho    = np.sqrt(np.power(points[:,0],2) + np.power(points[:,1],2))
        r      = np.sqrt(np.power(points[:,0],2) + np.power(points[:,1],2) + np.power(points[:,2],2))
        alpha  = np.sqrt(self.r0**2 + np.power(r,2) - 2*self.r0*rho)
        beta   = np.sqrt(self.r0**2 + np.power(r,2) + 2*self.r0*rho)
        k      = np.sqrt(1-np.divide(np.power(alpha,2),np.power(beta,2)))
        gamma  = np.power(points[:,0],2) - np.power(points[:,1],2)
        self._B = np.dot(self.rotMatrixInv,np.array([
            [self.Inorm*point[0]*point[2]/(2*alpha[i]**2*beta[i]*rho[i]**2)*((self.r0**2+r[i]**2)*ellipe(k[i]**2)-alpha[i]**2*ellipk(k[i]**2)),
             self.Inorm*point[1]*point[2]/(2*alpha[i]**2*beta[i]*rho[i]**2)*((self.r0**2+r[i]**2)*ellipe(k[i]**2)-alpha[i]**2*ellipk(k[i]**2)),
             self.Inorm/(2*alpha[i]**2*beta[i])*((self.r0**2-r[i]**2)*ellipe(k[i]**2)+alpha[i]**2*ellipk(k[i]**2))]
            for i,point in enumerate(points)]).T).T

        if compute_derivatives >= 1:

            dBxdx = np.array([
                    (self.Inorm*point[2]*(ellipk(k[i]**2)*alpha[i]**2*((2*point[0]**4 + gamma[i]*(point[1]**2 + 
                    point[2]**2))*r[i]**2 + self.r0**2*(gamma[i]*(self.r0**2 + 2*point[2]**2) - 
                    (3*point[0]**2 - 2*point[1]**2)*rho[i]**2)) + ellipe(k[i]**2)*(-((2*point[0]**4 + gamma[i]*(point[1]**2 + 
                    point[2]**2))*r[i]**4) + self.r0**4*(-(gamma[i]*(self.r0**2 + 3*point[2]**2)) + (8*point[0]**2 - point[1]**2)*rho[i]**2) - 
                    self.r0**2*(3*gamma[i]*point[2]**4 - 2*(2*point[0]**2 + point[1]**2)*point[2]**2*
                    rho[i]**2 + (5*point[0]**2 + point[1]**2)*rho[i]**4))))/(2*alpha[i]**4*beta[i]**3*rho[i]**4)
                    for i,point in enumerate(points)])
            
            dBydx = np.array([
                    (self.Inorm*point[0]*point[1]*point[2]*(ellipk(k[i]**2)*alpha[i]**2*(2*self.r0**4 + r[i]**2*(2*r[i]**2 + rho[i]**2) - 
                    self.r0**2*(-4*point[2]**2 + 5*rho[i]**2)) + ellipe(k[i]**2)*(-2*self.r0**6 - r[i]**4*(2*r[i]**2 + rho[i]**2) + 
                    3*self.r0**4*(-2*point[2]**2 + 3*rho[i]**2) - 2*self.r0**2*(3*point[2]**4 - point[2]**2*rho[i]**2 + 
                    2*rho[i]**4))))/(2*alpha[i]**4*beta[i]**3*rho[i]**4)
                    for i,point in enumerate(points)])

            dBzdx = np.array([
                    (self.Inorm*point[0]*(-(ellipk(k[i]**2)*alpha[i]**2*((-self.r0**2 + rho[i]**2)**2 + 
                    point[2]**2*(self.r0**2 + rho[i]**2))) + ellipe(k[i]**2)*(point[2]**4*(self.r0**2 + rho[i]**2) + 
                    (-self.r0**2 + rho[i]**2)**2*(self.r0**2 + rho[i]**2) + 2*point[2]**2*(self.r0**4 - 6*self.r0**2*rho[i]**2 + 
                    rho[i]**4))))/(2*alpha[i]**4*beta[i]**3*rho[i]**2)
                    for i,point in enumerate(points)])

            dBxdy = dBydx

            dBydy = np.array([
                    (self.Inorm*point[2]*(ellipk(k[i]**2)*alpha[i]**2*((2*point[1]**4 - gamma[i]*(point[0]**2 + point[2]**2))*r[i]**2 + 
                    self.r0**2*(-(gamma[i]*(self.r0**2 + 2*point[2]**2)) - (-2*point[0]**2 + 3*point[1]**2)*rho[i]**2)) + 
                    ellipe(k[i]**2)*(-((2*point[1]**4 - gamma[i]*(point[0]**2 + point[2]**2))*r[i]**4) + 
                    self.r0**4*(gamma[i]*(self.r0**2 + 3*point[2]**2) + (-point[0]**2 + 8*point[1]**2)*rho[i]**2) - 
                    self.r0**2*(-3*gamma[i]*point[2]**4 - 2*(point[0]**2 + 2*point[1]**2)*point[2]**2*rho[i]**2 + 
                    (point[0]**2 + 5*point[1]**2)*rho[i]**4))))/(2*alpha[i]**4*beta[i]**3*rho[i]**4)
                    for i,point in enumerate(points)])

            dBzdy = np.array([dBzdx[i]*point[1]/point[0]for i,point in enumerate(points)])

            dBxdz = dBzdx

            dBydz = dBzdy

            dBzdz = np.array([
                    (self.Inorm*point[2]*(ellipk(k[i]**2)*alpha[i]**2*(self.r0**2 - r[i]**2) + 
                    ellipe(k[i]**2)*(-7*self.r0**4 + r[i]**4 + 6*self.r0**2*(-point[2]**2 + rho[i]**2))))/(2*alpha[i]**4*beta[i]**3)
                    for i,point in enumerate(points)])

            dB_by_dXm = np.array([[
                [dBxdx[i],dBydx[i],dBzdx[i]],
                [dBxdy[i],dBydy[i],dBzdy[i]],
                [dBxdz[i],dBydz[i],dBzdz[i]]
                ] for i in range(len(points))])

            self._dB_by_dX = np.array([[
                [np.dot(self.rotMatrix[:,0],np.dot(self.rotMatrixInv[0,:],dB_by_dXm[i])),np.dot(self.rotMatrix[:,1],np.dot(self.rotMatrixInv[0,:],dB_by_dXm[i])),np.dot(self.rotMatrix[:,2],np.dot(self.rotMatrixInv[0,:],dB_by_dXm[i]))],
                [np.dot(self.rotMatrix[:,0],np.dot(self.rotMatrixInv[1,:],dB_by_dXm[i])),np.dot(self.rotMatrix[:,1],np.dot(self.rotMatrixInv[1,:],dB_by_dXm[i])),np.dot(self.rotMatrix[:,2],np.dot(self.rotMatrixInv[1,:],dB_by_dXm[i]))],
                [np.dot(self.rotMatrix[:,0],np.dot(self.rotMatrixInv[2,:],dB_by_dXm[i])),np.dot(self.rotMatrix[:,1],np.dot(self.rotMatrixInv[2,:],dB_by_dXm[i])),np.dot(self.rotMatrix[:,2],np.dot(self.rotMatrixInv[2,:],dB_by_dXm[i]))]
                ] for i in range(len(points))])

        if compute_derivatives >= 2:
            self._d2B_by_dXdX = None
            raise RuntimeError("Second derivative of scalar potential magnetic field not implemented yet")

    def compute_A(self, points, compute_derivatives=0):
        assert compute_derivatives <= 2

        points = np.array(np.dot(self.rotMatrix,np.array([np.subtract(point,self.center) for point in points]).T).T)
        rho    = np.sqrt(np.power(points[:,0],2) + np.power(points[:,1],2))
        r      = np.sqrt(np.power(points[:,0],2) + np.power(points[:,1],2) + np.power(points[:,2],2))
        alpha  = np.sqrt(self.r0**2 + np.power(r,2) - 2*self.r0*rho)
        beta   = np.sqrt(self.r0**2 + np.power(r,2) + 2*self.r0*rho)
        k = np.sqrt(1-np.divide(np.power(alpha,2),np.power(beta,2)))

        self._A = -self.Inorm/2*np.dot(self.rotMatrixInv,np.array([
            (2*self.r0*+np.sqrt(point[0]**2+point[1]**2)*ellipe(k[i]**2)+(self.r0**2+point[0]**2+point[1]**2+point[2]**2)*(ellipe(k[i]**2)-ellipk(k[i]**2)))/
            ((point[0]**2+point[1]**2)*np.sqrt(self.r0**2+point[0]**2+point[1]**2+2*self.r0*np.sqrt(point[0]**2+point[1]**2)+point[2]**2))*
            np.array([-point[1],point[0],0])
             for i,point in enumerate(points)]).T)
        
        if compute_derivatives >= 1:
            self._dA_by_dX = None
            raise RuntimeError("First derivative of magnetic vector potential of the circular coil magnetic field not implemented yet")

        if compute_derivatives >= 2:
            self._d2A_by_dXdX = None
            raise RuntimeError("Second derivative of magnetic vector potential of the circular coil magnetic field not implemented yet")

class Dommaschk(MagneticField):
    '''Vacuum magnetic field created by an explicit representation of the magnetic field scalar potential as proposed by W. Dommaschk (1986), Computer Physics Communications 40, 203-218
    As inputs, the scalar coefficients a, b, c, d

    Args:
        a: Cos coefficients of the Dirichlet solution
        b: Sin coefficients of the Dirichlet solution
        c: Cos coefficients of the Neumann solution
        d: Sin coefficients of the Neumann solution
    '''
    def __init__(self, a=[[0]], b=[[0]], c=[[0]], d=[[0]]):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.maxm = int(np.array(a).shape[0])
        self.maxl = int(np.array(a).shape[1])
        self.gradV     = jit(grad(self.V,(0,1,2)))
        self.Bcylx     = jit(self.Bcylindricalx)
        self.Bcyly     = jit(self.Bcylindricaly)
        self.Bcylz     = jit(self.Bcylindricalz)
        self.dBxdRphiz = jit(grad(self.Bcylx,(0,1,2)))
        self.dBydRphiz = jit(grad(self.Bcyly,(0,1,2)))
        self.dBzdRphiz = jit(grad(self.Bcylz,(0,1,2)))
        # initialize the compilation
        # self.set_points([[0.9,1.1,0.1]])
        # self.B()
        # self.clear_cached_properties()

    def alpha(self,n,m):
        if n<0:
            return 0
        else:
            return (-1)**n/(factorial(m+n)*factorial(n)*2**(2*n+m))

    def alphas(self,n,m):
        return (2*n+m)*self.alpha(n,m)

    def beta(self,n,m):
        if n<0:
            return 0
        elif n>=m:
            return 0
        else:
            return factorial(m-n-1)/(factorial(n)*2**(2*n-m+1))

    def betas(self,n,m):
        return (2*n-m)*self.beta(n,m)

    def gamma(self,n,m):
        if n<=0:
            return 0
        else:
            return self.alpha(n,m)/2*(np.sum(jnp.array([1/i for i in range(1,n+1)]))+np.sum(jnp.array([1/i for i in range(1,n+m+1)])))
    
    def gammas(self,n,m):
        return (2*n+m)*self.gamma(n,m)

    def CD(self,R,m,k):
        CDtemp = jnp.array([-(self.alpha(j,m)*(self.alphas(k-m-j,m)*jnp.log(R)+self.gammas(k-m-j,m)-self.alpha(k-m-j,m))-self.gamma(j,m)*self.alphas(k-m-j,m)+self.alpha(j,m)*self.betas(k-j,m))*R**(2*j+m)+self.beta(j,m)*self.alphas(k-j,m)*R**(2*j-m) for j in range(k+1)])
        return jnp.sum(CDtemp)

    def CN(self,R,m,k):
        CNtemp = jnp.array([+(self.alpha(j,m)*(self.alpha(k-m-j,m)*jnp.log(R)+self.gamma(k-m-j,m))-self.gamma(j,m)*self.alpha(k-m-j,m)+self.alpha(j,m)*self.beta(k-j,m))*R**(2*j+m)-self.beta(j,m)*self.alpha(k-j,m)*R**(2*j-m) for j in range(k+1)])
        return jnp.sum(CNtemp)

    def ImnD(self,m,n,R,Z):
        ImnDtemp = jnp.array([Z**(n-2*k)/factorial(n-2*k)*self.CD(R,m,k) for k in range(int(np.floor(n/2)+1))])
        return jnp.sum(ImnDtemp)

    def ImnN(self,m,n,R,Z):
        ImnNtemp = jnp.array([Z**(n-2*k)/factorial(n-2*k)*self.CN(R,m,k) for k in range(int(np.floor(n/2)+1))])
        return jnp.sum(ImnNtemp)

    def Vml(self,m,l,R,phi,Z):
        return (self.a[m,l]*jnp.cos(m*phi)+self.b[m,l]*jnp.sin(m*phi))*self.ImnD(m,l,R,Z)+(self.c[m,l]*jnp.cos(m*phi)+self.d[m,l]*jnp.sin(m*phi))*self.ImnN(m,l-1,R,Z)

    def V(self,R,phi,Z):
        return phi+jnp.sum(jnp.array([[self.Vml(m,l,R,phi,Z) for m in range(self.maxm)] for l in range(self.maxl)]))

    def Bcylindricalx(self,R,phi,Z):
        return self.gradV(R,phi,Z)[0]*jnp.cos(phi)-self.gradV(R,phi,Z)[1]*jnp.sin(phi)/R

    def Bcylindricaly(self,R,phi,Z):
        return self.gradV(R,phi,Z)[0]*jnp.sin(phi)+self.gradV(R,phi,Z)[1]*jnp.cos(phi)/R

    def Bcylindricalz(self,R,phi,Z):
        return self.gradV(R,phi,Z)[2]

    def compute(self, points, compute_derivatives=0):
        assert compute_derivatives <= 2

        R   = np.sqrt(np.power(points[:,0],2) + np.power(points[:,1],2))
        phi = np.arctan2(points[:,1],points[:,0])
        Z   = points[:,2]

        self._B = np.array([[self.Bcylx(R[i],phi[i],Z[i]),self.Bcyly(R[i],phi[i],Z[i]),self.Bcylz(R[i],phi[i],Z[i])] for i in range(len(points))])

        if compute_derivatives >= 1:
            dBxdRphizvec = np.array([self.dBxdRphiz(R[i],phi[i],Z[i]) for i in range(len(points))])
            dBydRphizvec = np.array([self.dBydRphiz(R[i],phi[i],Z[i]) for i in range(len(points))])
            dBzdRphizvec = np.array([self.dBzdRphiz(R[i],phi[i],Z[i]) for i in range(len(points))])

            self._dB_by_dX = np.array([
                [[ dBxdRphizvec[i,0]*np.cos(phi[i])-dBxdRphizvec[i,1]*np.sin(phi[i])/R[i], dBydRphizvec[i,0]*np.cos(phi[i])-dBydRphizvec[i,1]*np.sin(phi[i])/R[i], dBzdRphizvec[i,0]*np.cos(phi[i])-dBzdRphizvec[i,1]*np.sin(phi[i])/R[i]],
                 [ dBxdRphizvec[i,0]*np.sin(phi[i])+dBxdRphizvec[i,1]*np.cos(phi[i])/R[i], dBydRphizvec[i,0]*np.sin(phi[i])+dBydRphizvec[i,1]*np.cos(phi[i])/R[i], dBzdRphizvec[i,0]*np.sin(phi[i])+dBzdRphizvec[i,1]*np.cos(phi[i])/R[i]],
                 [ dBxdRphizvec[i,2]                                                     , dBydRphizvec[i,2]                                                     , dBzdRphizvec[i,2]                                                     ]]
                for i in range(len(points))])

        if compute_derivatives >= 2:
            self._d2B_by_dX = None