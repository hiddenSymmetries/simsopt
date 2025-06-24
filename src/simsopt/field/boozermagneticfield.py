import simsoptpp as sopp
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import logging
from booz_xform import Booz_xform
from .._core.util import parallel_loop_bounds, align_and_pad, allocate_aligned_and_padded_array
import os.path
import warnings

logger = logging.getLogger(__name__)

__all__ = [
    "BoozerMagneticField",
    "BoozerAnalytic",
    "BoozerRadialInterpolant",
    "InterpolatedBoozerField",
    "ShearAlfvenWave",
    "ShearAlfvenHarmonic",
    "ShearAlfvenWavesSuperposition"
]

try:
    from mpi4py import MPI
except ImportError as e:
    MPI = None
    logger.debug(str(e))

class BoozerMetric:
    r'''
     A generic class representing the metric tensor in normalized Boozer coordinates
     :math:`(s, \theta, \zeta)`, where:

     - :math:`s` is the normalized toroidal flux, :math:`s = \psi / \psi_0`.
     - :math:`\theta` is the poloidal Boozer angle.
     - :math:`\zeta` is the toroidal Boozer angle.

     The metric tensor defines the local geometry of the magnetic field in these coordinates.
     Given the symmetry of the metric tensor, the components stored are:

     .. math::
         g_{ij} = \begin{pmatrix}
         g_{ss} & g_{s\theta} & g_{s\zeta} \\
         g_{s\theta} & g_{\theta\theta} & g_{\theta\zeta} \\
         g_{s\zeta} & g_{\theta\zeta} & g_{\zeta\zeta}
         \end{pmatrix},

     where the metric components :math:`g_{ij}` are functions of :math:`(s, \theta, \zeta)`.

     **Attributes:**

     - `gss` : Co-(Counter-)variant metric component :math:`g_{ss}` (`g^{ss}`)).
     - `gst` : Co-(Counter-)variant metric component :math:`g_{s\theta}` (`g^{s\theta}`).
     - `gsz` : Co-(Counter-)variant metric component :math:`g_{s\zeta}` (`g^{s\zeta}`).
     - `gtt` : Co-(Counter-)variant metric component :math:`g_{\theta\theta}` (`g^{\theta\theta}`).
     - `gtz` : Co-(Counter-)variant metric component :math:`g_{\theta\zeta}` (`g^{\theta\theta}`).
     - `gzz` : Co-(Counter-)variant metric component :math:`g_{\zeta\zeta}` (`g^{\zeta\zeta}`).

     **Usage Example:**

     .. code-block:: python

         # Use BoozerMagneticField object (named `bfield` here)
         # to obtain the covariant metric tensor
         covariant_metric = bfield.get_covariant_metric()

         # Convert covariant to contravariant metric tensor
         contravariant_metric = covariant_metric.to_contravariant()

         # Access specific metric components
         gss_component = covariant_metric.ss
         gst_component = covariant_metric.st

         # Convert to matrix form for a single point
         matrix_form = covariant_metric[0].as_matrix

         # Compute determinant
         determinant = covariant_metric.det
    '''
    
    def __init__(self, gss, gst, gsz, gtt, gtz, gzz):
        self.ss = np.asarray(gss)
        self.st = np.asarray(gst)
        self.sz = np.asarray(gsz)
        self.tt = np.asarray(gtt)
        self.tz = np.asarray(gtz)
        self.zz = np.asarray(gzz)
        
        shape = self.ss.shape
        for g in [self.st, self.sz, self.tt, self.tz, self.zz]:
            if g.shape != shape:
                raise ValueError(
                    "All metric components must have the same shape"
                )
        for g in [self.ss, self.tt, self.zz]:
            if not (g > 0).all():
                raise ValueError(
                    "All diagonal metric components must be positive"
                )
    
    def as_matrix(self, idx=None):
        """
        Return the metric tensor as a 3x3 matrix for a given point.
                
        Parameters
        ----------
        idx : int, optional
            Index of the point to get the matrix for.
            If None and there's only one point, return that point's matrix.
                    
        Returns
        -------
            numpy.ndarray
            3x3 matrix representing the metric tensor
        """
        if idx is None:
            if self.ss.size == 1:
                idx = 0
            else:
                raise ValueError("Must specify idx for multi-point metric")
                        
        return np.array([
            [self.ss[idx], self.st[idx], self.sz[idx]],
            [self.st[idx], self.tt[idx], self.tz[idx]],
            [self.sz[idx], self.tz[idx], self.zz[idx]]
        ])
        
    def det(self):
        """
        Compute the determinant of the metric tensor at each point.
        
        Returns
        -------
        numpy.ndarray
            Array of determinant values
        """
        return (self.ss * (self.tt * self.zz - self.tz**2) -
                self.st * (self.st * self.zz - self.tz * self.sz) +
                self.sz * (self.st * self.tz - self.sz * self.tt))

class CovariantBoozerMetric(BoozerMetric):
    r"""
    Represents the covariant metric tensor for normalized Boozer coordinates
    :math:`(s, \theta, \zeta)` in a magnetic field. The covariant metric defines the
    local geometry of the space with respect to the basis vectors
    :math:`(\nabla s, \nabla \theta, \nabla \zeta)`.

    The covariant metric tensor :math:`g_{ij}` in Boozer coordinates is given by:

    .. math::
        g_{ij} = \begin{pmatrix}
        g_{ss} & g_{s\theta} & g_{s\zeta} \\
        g_{s\theta} & g_{\theta\theta} & g_{\theta\zeta} \\
        g_{s\zeta} & g_{\theta\zeta} & g_{\zeta\zeta}
        \end{pmatrix},

    where each component :math:`g_{ij}` is a function of the Boozer coordinates
    :math:`(s, \theta, \zeta)`.

    **Methods:**

    - `to_contravariant()`: Converts the covariant metric to its contravariant form by inverting
        the metric tensor. This operation is mathematically equivalent to computing the inverse
        of the covariant metric matrix.

    **Usage Example:**

    .. code-block:: python

        # Given a BoozerMagneticField instance named `bfield`
        covariant_metric = bfield.get_covariant_metric()

        # Convert to contravariant metric
        contravariant_metric = covariant_metric.to_contravariant()

        # Access specific metric components
        gss_component = covariant_metric.ss

        # Compute the determinant of the metric tensor
        determinant = covariant_metric.det

    **Returns:**

    An instance of :class:`ContravariantBoozerMetric` representing the contravariant form of the metric.
    
    """
    def to_contravariant(self):
        """
        Converts the covariant metric to its contravariant form by inverting
        the metric tensor.
        
        Returns
        -------
        ContravariantBoozerMetric
            The contravariant form of the metric.
        
        Raises
        -------
        `LinAlgError`: If the matrix inversion fails, due to the matrix being singular.
        """
        inv_matrices = np.zeros((len(self.ss), 6))
        for k in range(len(self.ss)):
            matrix = np.array([
                [self.ss[k], self.st[k], self.sz[k]],
                [self.st[k], self.tt[k], self.tz[k]],
                [self.sz[k], self.tz[k], self.zz[k]]
            ])
            inv_matrix = np.linalg.inv(matrix)
            inv_matrices[k] = [ # gss, gst, gsz, gtt, gtz, gzz
                inv_matrix[0,0], inv_matrix[0,1], inv_matrix[0,2],
                inv_matrix[1,1], inv_matrix[1,2], inv_matrix[2,2]
            ]
        return ContravariantBoozerMetric(
                inv_matrices[:,0], inv_matrices[:,1], inv_matrices[:,2],
                inv_matrices[:,3], inv_matrices[:,4], inv_matrices[:,5]
            )
        
class ContravariantBoozerMetric(BoozerMetric):
    r"""
    Represents the contravariant metric tensor for normalized Boozer coordinates
    :math:`(s, \theta, \zeta)` in a magnetic field. The contravariant metric is
    associated with the basis vectors :math:`(\partial / \partial s, \partial / \partial \theta, \partial / \partial \zeta)`.

    The contravariant metric tensor :math:`g^{ij}` in Boozer coordinates is given by:

    .. math::
        g^{ij} = \begin{pmatrix}
        g^{ss} & g^{s\theta} & g^{s\zeta} \\
        g^{s\theta} & g^{\theta\theta} & g^{\theta\zeta} \\
        g^{s\zeta} & g^{\theta\zeta} & g^{\zeta\zeta}
        \end{pmatrix},

    where each component :math:`g^{ij}` is a function of the Boozer coordinates
    :math:`(s, \theta, \zeta)`.

    **Methods:**

    - `to_covariant()`: Converts the contravariant metric to its covariant form by inverting
        the metric tensor. This operation is mathematically equivalent to computing the inverse
        of the contravariant metric matrix.

    **Usage Example:**

    .. code-block:: python

        # Given a BoozerMagneticField instance named `bfield`
        contravariant_metric = bfield.get_contravariant_metric()

        # Convert to covariant metric
        covariant_metric = contravariant_metric.to_covariant()

        # Access specific metric components
        gss_component = contravariant_metric.ss

        # Compute the determinant of the metric tensor
        determinant = contravariant_metric.det

    **Returns:**

    An instance of :class:`CovariantBoozerMetric` representing the covariant form of the metric.

    """
    def to_covariant(self):
        """
        Converts the contravariant metric to its covariant form by inverting
        the metric tensor.
                
        Returns
        -------
        CovariantBoozerMetric
            The covariant form of the metric.
        
        Raises
        -------
        `LinAlgError`: If the matrix inversion fails, due to the matrix being singular.
        """
        inv_matrices = np.zeros((len(self.ss), 6))
        for k in range(len(self.ss)):
            matrix = np.array([
                [self.ss[k], self.st[k], self.sz[k]],
                [self.st[k], self.tt[k], self.tz[k]],
                [self.sz[k], self.tz[k], self.zz[k]]
            ])
            inv_matrix = np.linalg.inv(matrix)
            inv_matrices[k] = [ # gss, gst, gsz, gtt, gtz, gzz
                inv_matrix[0,0], inv_matrix[0,1], inv_matrix[0,2],
                inv_matrix[1,1], inv_matrix[1,2], inv_matrix[2,2]
            ]
        return CovariantBoozerMetric(
            inv_matrices[:,0], inv_matrices[:,1], inv_matrices[:,2],
            inv_matrices[:,3], inv_matrices[:,4], inv_matrices[:,5]
        )
        
class BoozerMagneticField(sopp.BoozerMagneticField):
    r"""
    Generic class that represents a magnetic field in Boozer coordinates
    :math:`(s,\theta,\zeta)`. Here :math:`s = \psi/\psi_0` is the normalized
    toroidal flux where :math:`2\pi\psi_0` is the toroidal flux at the boundary.
    The magnetic field in the covariant form is,

    .. math::
        \textbf B(s,\theta,\zeta) = G(s) \nabla \zeta + I(s) \nabla \theta + K(s,\theta,\zeta) \nabla \psi,

    and the contravariant form is,

    .. math::
        \textbf B(s,\theta,\zeta) = \frac{1}{\sqrt{g}} \left(\frac{\partial \mathbf r}{\partial \zeta} + \iota(s)\frac{\partial \mathbf r}{\partial \theta}\right),

    where,

    .. math::
        \sqrt{g}(s,\theta,\zeta) = \frac{G(s) + \iota(s)I(s)}{B^2}.

    Here :math:`\iota(s) = \psi_P'(\psi)` where :math:`2\pi\psi_P` is the
    poloidal flux and :math:`2\pi\psi` is the toroidal flux. Each subclass of
    :class:`BoozerMagneticField` implements functions to compute
    :math:`B`, :math:`G`, :math:`I`, :math:`\iota`, :math:`\psi_P`, and their
    derivatives. The cylindrical coordinates :math:`R(s,\theta,\zeta)` and
    :math:`Z(s,\theta,\zeta)` in addition to :math:`K(s,\theta,\zeta)` and
    :math:`\nu` where :math:`\zeta = \phi + \nu(s,\theta,\zeta)` and :math:`\phi`
    is the cylindrical azimuthal angle are also implemented by
    :class:`BoozerRadialInterpolant` and :class:`InterpolatedBoozerField`.
    The usage is similar to the :class:`MagneticField` class.

    The usage of :class:`BoozerMagneticField`` is as follows:

    .. code-block::

        booz = BoozerAnalytic(etabar,B0,N,G0,psi0,iota0) # An instance of BoozerMagneticField
        points = ... # points is a (n, 3) numpy array defining :math:`(s,\theta,\zeta)`
        booz.set_points(points)
        modB = bfield.modB() # returns the magnetic field strength at `points`
    """

    def __init__(self, psi0):
        self.psi0 = psi0
        sopp.BoozerMagneticField.__init__(self, psi0)

    def _modB_derivs_impl(self, modB_derivs):
        self._dmodBds_impl(np.reshape(modB_derivs[:, 0], (len(modB_derivs[:, 0]), 1)))
        self._dmodBdtheta_impl(np.reshape(modB_derivs[:, 1], (len(modB_derivs[:, 0]), 1)))
        self._dmodBdzeta_impl(np.reshape(modB_derivs[:, 2], (len(modB_derivs[:, 0]), 1)))

    def _K_derivs_impl(self, K_derivs):
        self._dKdtheta_impl(np.reshape(K_derivs[:, 0], (len(K_derivs[:, 0]), 1)))
        self._dKdzeta_impl(np.reshape(K_derivs[:, 1], (len(K_derivs[:, 0]), 1)))

    def _nu_derivs_impl(self, nu_derivs):
        self._dnuds_impl(np.reshape(nu_derivs[:, 0], (len(nu_derivs[:, 0]), 1)))
        self._dnudtheta_impl(np.reshape(nu_derivs[:, 1], (len(nu_derivs[:, 0]), 1)))
        self._dnudzeta_impl(np.reshape(nu_derivs[:, 2], (len(nu_derivs[:, 0]), 1)))

    def _R_derivs_impl(self, R_derivs):
        self._dRds_impl(np.reshape(R_derivs[:, 0], (len(R_derivs[:, 0]), 1)))
        self._dRdtheta_impl(np.reshape(R_derivs[:, 1], (len(R_derivs[:, 0]), 1)))
        self._dRdzeta_impl(np.reshape(R_derivs[:, 2], (len(R_derivs[:, 0]), 1)))

    def _Z_derivs_impl(self, Z_derivs):
        self._dZds_impl(np.reshape(Z_derivs[:, 0], (len(Z_derivs[:, 0]), 1)))
        self._dZdtheta_impl(np.reshape(Z_derivs[:, 1], (len(Z_derivs[:, 0]), 1)))
        self._dZdzeta_impl(np.reshape(Z_derivs[:, 2], (len(Z_derivs[:, 0]), 1)))
        
    def get_covariant_metric(self):
        r'''
        Computes and returns the covariant metric tensor for normalized Boozer coordinates
        :math:`(s, \theta, \zeta)`.

        In normalized Boozer coordinates, the metric tensor defines the local geometry of space
        with respect to the covariant basis vectors :math:`(\nabla s, \nabla \theta, \nabla \zeta)`.


        The metric components are computed by evaluating the derivatives of the cylindrical coordinates
        and the Boozer angle with respect to :math:`s`, :math:`\theta`, and :math:`\zeta`.
        The determinant of the metric tensor is computed and compared to the inverse Jacobian
        for consistency. If the discrepancy exceeds 0.1%, a warning is issued.

        Returns
        -------
        CovariantBoozerMetric
            The covariant metric tensor.
    
        Raises
        ------
        AssertionError
            If the metric is singular on the magnetic axis s=0.
        RuntimeWarning
            If there is a large discrepancy (>0.1%) between the computed determinant
            of the covariant metric and the inverse Jacobian.

        **Usage Example:**

        .. code-block:: python

            # Given a BoozerMagneticField instance named `bfield`
            covariant_metric = bfield.get_covariant_metric()

            # Access specific metric components
            gss_component = covariant_metric.ss
            gst_component = covariant_metric.st

            # Convert to matrix form for a single point
            matrix_form = covariant_metric[0].as_matrix
        '''
        points = self.get_points_ref()
        s = points[:, 0]
        assert np.all(s>0), 'Metric is singular on magnetic axis s=0, can not compute. Choose different point.'
        zetas = points[:, 2]
        R = self.R()[:, 0]
        dRdtheta = self.dRdtheta()[:, 0]
        dRdzeta = self.dRdzeta()[:, 0]
        dRds = self.dRds()[:, 0]
        dZdtheta = self.dZdtheta()[:, 0]
        dZdzeta = self.dZdzeta()[:, 0]
        dZds = self.dZds()[:, 0]
        nu = self.nu()[:, 0]
        dnudtheta = self.dnudtheta()[:, 0]
        dnudzeta = self.dnudzeta()[:, 0]
        dnuds = self.dnuds()[:, 0]
    
        phi = zetas - nu
        dphids = -dnuds
        dphidtheta = -dnudtheta
        dphidzeta = 1 - dnudzeta
    
        dXdtheta = dRdtheta * np.cos(phi) - R * np.sin(phi) * dphidtheta
        dYdtheta = dRdtheta * np.sin(phi) + R * np.cos(phi) * dphidtheta
        dXds = dRds * np.cos(phi) - R * np.sin(phi) * dphids
        dYds = dRds * np.sin(phi) + R * np.cos(phi) * dphids
        dXdzeta = dRdzeta * np.cos(phi) - R * np.sin(phi) * dphidzeta
        dYdzeta = dRdzeta * np.sin(phi) + R * np.cos(phi) * dphidzeta
    
        gss = dXds**2 + dYds**2 + dZds**2
        gstheta = dXds*dXdtheta + dYds*dYdtheta + dZds*dZdtheta
        gszeta = dXds*dXdzeta + dYds*dYdzeta + dZds*dZdzeta
        gthetatheta = dXdtheta**2 + dYdtheta**2 + dZdtheta**2
        gthetazeta = dXdtheta*dXdzeta + dYdtheta*dYdzeta + dZdtheta*dZdzeta
        gzetazeta = dXdzeta**2 + dYdzeta**2 + dZdzeta**2
    
        # Test that determinant of covariant Boozer metric matches inverse Jacobian
        detg = (
            gss * (gthetatheta*gzetazeta - gthetazeta**2)
          - gstheta * (gstheta*gzetazeta - gthetazeta*gszeta)
          + gszeta * (gstheta*gthetazeta - gszeta*gthetatheta)
        )
    
        G = self.G()[:, 0]
        I = self.I()[:, 0]
        iota = self.iota()[:, 0]
        B = self.modB()[:, 0]
        sqrtg = (G + iota * I)*self.psi0/(B*B)
        assert np.all(detg > 0), 'Metric determinant must be positive'
        assert np.all(sqrtg > 0), 'Jacobian must be positive'
        
        relative_error = np.abs(np.sqrt(detg) - np.abs(sqrtg)) / np.abs(sqrtg)
        max_relative_error_percent = np.max(relative_error) * 100
        if max_relative_error_percent > 0.1:
            # Find the location of maximum error
            max_error_idx = np.argmax(relative_error)
            s_error = s[max_error_idx]
            theta_error = points[max_error_idx, 1]
            zeta_error = points[max_error_idx, 2]
            
            # Get metric values at error location
            metric_at_error = np.array([
                [gss[max_error_idx], gstheta[max_error_idx], gszeta[max_error_idx]],
                [gstheta[max_error_idx], gthetatheta[max_error_idx], gthetazeta[max_error_idx]],
                [gszeta[max_error_idx], gthetazeta[max_error_idx], gzetazeta[max_error_idx]]
            ])
            
            warnings.warn(
                f"\nLarge maximum relative error ({max_relative_error_percent:.2f}%) between "
                f"metric determinant and Jacobian at:\n"
                f"  (s, theta, zeta) = ({s_error:.3f}, {theta_error:.3f}, {zeta_error:.3f})\n"
                f"  sqrt(detg) = {np.sqrt(detg[max_error_idx]):.6e}\n"
                f"  sqrtg     = {sqrtg[max_error_idx]:.6e}\n"
                f"Metric tensor at this point:\n"
                f"  [[ {metric_at_error[0,0]:.6e}  {metric_at_error[0,1]:.6e}  {metric_at_error[0,2]:.6e} ]\n"
                f"   [ {metric_at_error[1,0]:.6e}  {metric_at_error[1,1]:.6e}  {metric_at_error[1,2]:.6e} ]\n"
                f"   [ {metric_at_error[2,0]:.6e}  {metric_at_error[2,1]:.6e}  {metric_at_error[2,2]:.6e} ]]\n"
                "exceeds 0.1% tolerance.",
                RuntimeWarning
            )

        return CovariantBoozerMetric(
            gss=gss,
            gst=gstheta,
            gsz=gszeta,
            gtt=gthetatheta,
            gtz=gthetazeta,
            gzz=gzetazeta
            )

    def get_contravariant_metric(self):
        r'''
        Computes and returns the contravariant metric tensor for normalized Boozer coordinates
        :math:`(s, \theta, \zeta)`.
    
        In normalized Boozer coordinates, the contravariant metric tensor defines the local geometry 
        of space with respect to the contravariant basis vectors 
        :math:`(\partial / \partial s, \partial / \partial \theta, \partial / \partial \zeta)`.
        
        The contravariant metric is computed by inverting the covariant metric tensor.
    
        Returns
        -------
        ContravariantBoozerMetric
            The contravariant metric tensor.
    
        Raises
        ------
        AssertionError
            If the metric is singular on the magnetic axis s=0.
        LinAlgError
            If the covariant metric tensor cannot be inverted.
    
        **Usage Example:**
    
        .. code-block:: python
    
            # Given a BoozerMagneticField instance named `bfield`
            contravariant_metric = bfield.get_contravariant_metric()
    
            # Access specific metric components
            gss_component = contravariant_metric.ss
            gst_component = contravariant_metric.st
    
            # Convert to matrix form for a single point
            matrix_form = contravariant_metric[0].as_matrix
        '''
        return self.get_covariant_metric().to_contravariant()

class BoozerAnalytic(BoozerMagneticField):
    r"""
    Computes a :class:`BoozerMagneticField` based on a first-order expansion in
    distance from the magnetic axis (Landreman & Sengupta, Journal of Plasma
    Physics 2018). A possibility to include QS-breakign perturbation is added, so the magnetic field strength is expressed as,

    .. math::
        B(s,\theta,\zeta) = B_0 \left(1 + \overline{\eta} \sqrt{2s\psi_0/\overline{B}}\cos(\theta - N \zeta)\right) + B_{0z}\cos{m\theta-n\zeta},

    the covariant components of equilibrium field are,

    .. math::
        G(s) = G_0 + \sqrt{2s\psi_0/\overline{B}} G_1

        I(s) = I_0 + \sqrt{2s\psi_0/\overline{B}} I_1

        K(s,\theta,\zeta) = \sqrt{2s\psi_0/\overline{B}} K_1 \sin(\theta - N \zeta),

    and the rotational transform is,

    .. math::
        \iota(s) = \iota_0.

    While formally :math:`I_0 = I_1 = G_1 = K_1 = 0`, these terms have been included
    in order to test the guiding center equations at finite beta.

    Args:
        etabar: magnitude of first order correction to magnetic field strength
        B0: magnetic field strength on the axis
        N: helicity of symmetry (integer)
        G0: lowest order toroidal covariant component
        psi0: (toroidal flux)/ (2*pi) on the boundary
        iota0: lowest order rotational transform
        Bbar: normalizing magnetic field strength (defaults to 1)
        I0: lowest order poloidal covariant component (defaults to 0)
        G1: first order correction to toroidal covariant component (defaults to 0)
        I1: first order correction to poloidal covariant component (defaults to 0)
        K1: first order correction to radial covariant component (defaults to 0)
        B0z: amplitude of symmetry-breaking perturbation mode
        n: toroidal mode number for the perturbation
        m: poloidal mode bumber for the perturbation
    """

    def __init__(self, etabar, B0, N, G0, psi0, iota0, Bbar=1., I0=0., G1=0.,
                 I1=0., K1=0., iota1=0., B0z=[0.], n=[1], m=[2]):
        assert(len(B0z)==len(n))
        assert(len(m)==len(n))
        self.etabar = etabar
        self.B0 = B0
        self.B0z = np.array(B0z)
        self.m = np.array(m,dtype='float')
        self.n = np.array(n,dtype='float')
        self.Bbar = Bbar
        self.N = N
        self.G0 = G0
        self.I0 = I0
        self.I1 = I1
        self.G1 = G1
        self.K1 = K1
        self.iota0 = iota0
        self.psi0 = psi0
        self.iota1 = iota1
        BoozerMagneticField.__init__(self, psi0)

    def set_etabar(self, etabar):
        self.etabar = etabar

    def set_B0(self, B0):
        self.B0 = B0

    def set_B0z(self, B0z):
        self.B0z = B0z

    def set_Bbar(self, Bbar):
        self.Bbar = Bbar

    def set_N(self, N):
        self.N = N

    def set_G0(self, G0):
        self.G0 = G0

    def set_I0(self, I0):
        self.I0 = I0

    def set_G1(self, G1):
        self.G1 = G1

    def set_I1(self, I1):
        self.I1 = I1

    def set_K1(self, K1):
        self.K1 = K1

    def set_iota0(self, iota0):
        self.iota0 = iota0

    def set_iota1(self, iota1):
        self.iota1 = iota1

    def set_psi0(self, psi0):
        self.psi0 = psi0

    def _psip_impl(self, psip):
        points = self.get_points_ref()
        s = points[:, 0]
        psip[:, 0] = self.psi0*(s*self.iota0 + s**2 * self.iota1/2)

    def _iota_impl(self, iota):
        points = self.get_points_ref()
        s = points[:, 0]
        iota[:, 0] = self.iota0 + self.iota1*s

    def _diotads_impl(self, diotads):
        diotads[:, 0] = self.iota1

    def _G_impl(self, G):
        points = self.get_points_ref()
        s = points[:, 0]
        G[:, 0] = self.G0 + s*self.G1

    def _dGds_impl(self, dGds):
        dGds[:, 0] = self.G1

    def _I_impl(self, I):
        points = self.get_points_ref()
        s = points[:, 0]
        I[:, 0] = self.I0 + s*self.I1

    def _dIds_impl(self, dIds):
        dIds[:, 0] = self.I1

    def _modB_impl(self, modB):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s*self.psi0
        r = np.sqrt(np.abs(2*psi/self.Bbar))
        modB[:, 0] = self.B0*(1 + self.etabar*r*np.cos(thetas-self.N*zetas)) + np.sum(self.B0z[:,None]*np.cos(self.m[:,None]*thetas[None,:] - self.n[:,None]*self.N*zetas[None,:]))

    def _dmodBds_impl(self, dmodBds):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s*self.psi0
        # drds = np.zeros_like(s)
        r = np.sqrt(np.abs(2*psi/self.Bbar))
        # drds[s!=0] = 0.5*r[s!=0]*self.psi0/psi[s!=0]
        if self.etabar!=0:
            drds = 0.5*r*self.psi0/psi
            dmodBds[:, 0] = self.B0*self.etabar*drds*np.cos(thetas-self.N*zetas)
        else:
            dmodBds[:, 0] = 0

    def _dmodBdtheta_impl(self, dmodBdtheta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s*self.psi0
        r = np.sqrt(np.abs(2*psi/self.Bbar))
        dmodBdtheta[:, 0] = -self.B0*self.etabar*r*np.sin(thetas-self.N*zetas) - np.sum(self.B0z[:,None]*self.m[:,None]*np.sin(self.m[:,None]*thetas[None,:] - self.n[:,None]*self.N*zetas[None,:]))

    def _dmodBdzeta_impl(self, dmodBdzeta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s*self.psi0
        r = np.sqrt(np.abs(2*psi/self.Bbar))
        dmodBdzeta[:, 0] = self.N*self.B0*self.etabar*r*np.sin(thetas-self.N*zetas) + np.sum(self.B0z[:,None]*self.n[:,None]*self.N*np.sin(self.m[:,None]*thetas[None,:] - self.n[:,None]*self.N*zetas[None,:]))

    def _K_impl(self, K):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s*self.psi0
        r = np.sqrt(np.abs(2*psi/self.Bbar))
        K[:, 0] = self.K1*r*np.sin(thetas-self.N*zetas)

    def _dKdtheta_impl(self, dKdtheta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s*self.psi0
        r = np.sqrt(np.abs(2*psi/self.Bbar))
        dKdtheta[:, 0] = self.K1*r*np.cos(thetas-self.N*zetas)

    def _dKdzeta_impl(self, dKdzeta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s*self.psi0
        r = np.sqrt(np.abs(2*psi/self.Bbar))
        dKdzeta[:, 0] = -self.N*self.K1*r*np.cos(thetas-self.N*zetas)

class BoozerRadialInterpolant(BoozerMagneticField):
    r"""
     The magnetic field can be computed at any point in Boozer coordinates using radial spline interpolation (``scipy.interpolate.InterpolatedUnivariateSpline``) and an inverse Fourier transform in the two angles. If given a `VMEC` output file, performs a Boozer coordinate transformation using ``BOOZXFORM``. If given a ``BOOZXFORM`` output file, 
     the Boozer transformation must be performed with all surfaces on the VMEC
     half grid, and with `phip`, `chi`, `pres`, and `phi` saved in the file.
     
    Args:
        equil: instance of :class:`Booz_xform` or string containing the         
            filename of a boozmn_*.nc file (produced with booz_xform) or 
            wout_*.nc file (produced with VMEC). If a :class:`Booz_xform` 
            instance or boozmn_*.nc file is passed, the `compute_surfs` needs to include all of the grid points in the half-radius grid of the corresponding Vmec equilibrium. Otherwise, a ValueError is raised.
        order: (int) order for radial interpolation. Must satisfy 1 <= order <= 
            5.
        mpol: (int) number of poloidal mode numbers for BOOZXFORM (defaults to 
            32). Only used if a wout_*.nc file is passed. 
        ntor: (int) number of toroidal mode numbers for BOOZXFORM (defaults to 
            32). Only used if a wout_*.nc file is passed. 
        N: Helicity of quasisymmetry to enforce. If specified, then the 
            non-symmetric Fourier harmonics of :math:`B` and :math:`K` are filtered out. Infinite (``float('inf')``) helicity specifies quasi-poloidal field. If helicity is unspecified, all harmonics are kept.
            (defaults to ``None``)
        enforce_vacuum: If True, a vacuum field is assumed, :math:`G` is
            set to its mean value, :math:`I = 0`, and :math:`K = 0`.
        rescale: If True, use the interpolation method in the DELTA5D code. 
            Here, a few of the first radial grid points or (``bmnc``, ``rmnc``, ``zmns``, ``numns``, ``kmns``) are deleted (determined by ``ns_delete``). The Fourier harmonics are then rescaled as:
                bmnc(s)/s^(1/2) for m = 1

                bmnc(s)/s for m even and >= 2

                bmnc(s)/s^(3/2) for m odd and >=3

            before performing interpolation and spline differentiation to
            obtain ``dbmncds``. If ``False``, interpolation of the unscaled Fourier harmonics and its finite-difference derivative wrt ``s`` is performed instead (defaults to ``False``)
        ns_delete: (see ``rescale``) (defaults to 0)
        no_K: (bool) If ``True``, the Boozer :math:`K` will not be computed or
            interpolated.
        write_boozmn: (bool) If ``True``, save the booz_xform transformation in
            a filename specified by ``boozmn_name``. (defaults to ``True``)
        comm: A MPI communicator to parallelize over, from which
          the worker groups will be used for spline calculations. If ``comm`` is
          ``None``, each MPI process will compute splines independently.
        boozmn_name: (string) Filename to save booz_xform transformation if
            ``write_boozmn`` is ``True``.
        verbose: If True, additional output is written by booz_xform. (defaults 
            to False). 
        no_shear: If True, the shear in the rotational transform will be 
            eliminated, and iota will be taken to be the mean value. (defaults to False).
    """
    def __init__(self, equil, order, mpol=32, ntor=32, N=None, 
                 enforce_vacuum=False, rescale=False, ns_delete=0, no_K=False, write_boozmn=True, comm=None, boozmn_name="boozmn.nc", verbose=0, no_shear=False):

        self.comm = comm

        if self.comm is not None:
            self.proc0 = False
            if self.comm.rank==0:
                self.proc0 = True
        else:
            self.proc0 = True

        if isinstance(equil, str):
            if self.proc0:
                basename = os.path.basename(equil)
                if basename[:4] == 'wout':
                    booz = Booz_xform()
                    booz.read_wout(equil,True)
                    booz.verbose = verbose
                    booz.mboz = mpol
                    booz.nboz = ntor
                    booz.run()
                    if write_boozmn:
                        booz.write_boozmn(boozmn_name)
                    self.bx = booz
                elif basename[:4] == 'booz':
                    booz = Booz_xform()
                    booz.read_boozmn(equil)
                    self.bx = booz
                    # Check if grid does not have correct size
                    if (self.bx.ns_in != len(self.bx.s_b)):
                        raise ValueError('booz filename has incorrect s grid!')
                    # Check if grid does not match Vmec half grid
                    s_in_full = np.linspace(0,1,self.bx.ns_in+1)
                    s_in = 0.5*(s_in_full[1::]+s_in_full[0:-1])
                    if (not np.allclose(s_in, self.bx.s_b)):
                        raise ValueError('booz filename has incorrect s grid!')
                else:
                    raise ValueError('Invalid filename')
        elif (isinstance(equil, Booz_xform)):
            if self.proc0:
                self.bx = equil
        else:
            raise ValueError("Incorrect equil type passed to BoozerRadialInterpolant.")

        self.no_shear = no_shear
        self.order = order
        self.enforce_qs = False
        self.enforce_vacuum = enforce_vacuum
        self.no_K = no_K
        if (self.enforce_vacuum):
            self.no_K = True
        self.ns_delete = ns_delete
        self.rescale = rescale
        if (N is not None):
            self.N = N
            self.enforce_qs = True

        if self.proc0:
            self.asym = self.bx.asym  # Bool for stellarator asymmetry
            self.psi0 = self.bx.phi[-1]/(2*np.pi)
            self.nfp = self.bx.nfp
            self.mpol = self.bx.mboz
            self.ntor = self.bx.nboz
            self.s_half_ext = np.zeros((self.bx.ns_b+2))
            self.s_half_ext[1:-1] = self.bx.s_b
            self.s_half_ext[-1] = 1
            self.init_splines()
        else:
            self.psip_spline = None
            self.G_spline = None
            self.I_spline = None
            self.dGds_spline = None
            self.dIds_spline = None
            self.iota_spline = None
            self.diotads_spline = None
            self.numns_splines = None
            self.rmnc_splines = None
            self.zmns_splines = None
            self.dnumnsds_splines = None
            self.drmncds_splines = None
            self.dzmnsds_splines = None
            self.bmnc_splines = None
            self.dbmncds_splines = None
            self.d_mn_factor_splines = None
            self.mn_factor_splines = None
            self.xm_b = None
            self.xn_b = None
            self.numnc_splines = None
            self.rmns_splines = None
            self.zmnc_splines = None
            self.dnumncds_splines = None
            self.drmnsds_splines = None
            self.dzmncds_splines = None
            self.bmns_splines = None
            self.dbmnsds_splines = None
            self.kmns_splines = None
            self.kmnc_splines = None
            self.asym = None
            self.psi0 = None
            self.nfp = None
            self.mpol = None
            self.ntor = None
            self.s_half_ext = None
        if self.comm is not None:
            self.psi0 = self.comm.bcast(self.psi0, root=0)
            self.nfp = self.comm.bcast(self.nfp, root=0)
            self.mpol = self.comm.bcast(self.mpol, root=0)
            self.ntor = self.comm.bcast(self.ntor, root=0)
            self.asym = self.comm.bcast(self.asym, root=0)
            self.psip_spline = self.comm.bcast(self.psip_spline, root=0)
            self.G_spline = self.comm.bcast(self.G_spline, root=0)
            self.I_spline = self.comm.bcast(self.I_spline, root=0)
            self.dGds_spline = self.comm.bcast(self.dGds_spline, root=0)
            self.dIds_spline = self.comm.bcast(self.dIds_spline, root=0)
            self.iota_spline = self.comm.bcast(self.iota_spline, root=0)
            self.diotads_spline = self.comm.bcast(self.diotads_spline, root=0)
            self.numns_splines = self.comm.bcast(self.numns_splines, root=0)
            self.rmnc_splines = self.comm.bcast(self.rmnc_splines, root=0)
            self.zmns_splines = self.comm.bcast(self.zmns_splines, root=0)
            self.dnumnsds_splines = self.comm.bcast(self.dnumnsds_splines, root=0)
            self.drmncds_splines = self.comm.bcast(self.drmncds_splines, root=0)
            self.dzmnsds_splines = self.comm.bcast(self.dzmnsds_splines, root=0)
            self.bmnc_splines = self.comm.bcast(self.bmnc_splines, root=0)
            self.dbmncds_splines = self.comm.bcast(self.dbmncds_splines, root=0)
            self.d_mn_factor_splines = self.comm.bcast(self.d_mn_factor_splines, root=0)
            self.mn_factor_splines = self.comm.bcast(self.mn_factor_splines, root=0)
            self.xm_b = self.comm.bcast(self.xm_b, root=0)
            self.xn_b = self.comm.bcast(self.xn_b, root=0)
            self.s_half_ext = self.comm.bcast(self.s_half_ext, root=0)
            if self.asym:
                self.numnc_splines = self.comm.bcast(self.numnc_splines, root=0)
                self.rmns_splines = self.comm.bcast(self.rmns_splines, root=0)
                self.zmnc_splines = self.comm.bcast(self.zmnc_splines, root=0)
                self.dnumncds_splines = self.comm.bcast(self.dnumncds_splines, root=0)
                self.drmnsds_splines = self.comm.bcast(self.drmnsds_splines, root=0)
                self.dzmncds_splines = self.comm.bcast(self.dzmncds_splines, root=0)
                self.bmns_splines = self.comm.bcast(self.bmns_splines, root=0)
                self.dbmnsds_splines = self.comm.bcast(self.dbmnsds_splines, root=0)

        if (not self.no_K):
            self.compute_K()

        BoozerMagneticField.__init__(self, self.psi0)

    def init_splines(self):
        self.xm_b = self.bx.xm_b
        self.xn_b = self.bx.xn_b

        # Define quantities on extended half grid
        iota = np.zeros((self.bx.ns_b+2))
        G = np.zeros((self.bx.ns_b+2))
        I = np.zeros((self.bx.ns_b+2))

        ds = self.bx.s_b[1]-self.bx.s_b[0]

        s_full = np.linspace(0, 1, self.bx.ns_b+1)

        psip = self.bx.chi/(2*np.pi)
        iota[1:-1] = self.bx.iota
        sign_psip = np.sign(((psip[1]-psip[0])/self.psi0)/np.sign(iota[1]))
        psip *= sign_psip
        G[1:-1] = self.bx.Boozer_G_all
        I[1:-1] = self.bx.Boozer_I_all
        if self.rescale:
            s_half_mn = self.bx.s_b[self.ns_delete::]
            bmnc = np.zeros((len(self.xm_b), self.bx.ns_b-self.ns_delete))
            rmnc = np.zeros((len(self.xm_b), self.bx.ns_b-self.ns_delete))
            zmns = np.zeros((len(self.xm_b), self.bx.ns_b-self.ns_delete))
            numns = np.zeros((len(self.xm_b), self.bx.ns_b-self.ns_delete))

            bmnc = self.bx.bmnc_b[:, self.ns_delete::]
            rmnc = self.bx.rmnc_b[:, self.ns_delete::]
            zmns = self.bx.zmns_b[:, self.ns_delete::]
            numns = self.bx.numns_b[:, self.ns_delete::]

            if self.asym:
                bmns = np.zeros((len(self.xm_b), self.bx.ns_b-self.ns_delete))
                rmns = np.zeros((len(self.xm_b), self.bx.ns_b-self.ns_delete))
                zmnc = np.zeros((len(self.xm_b), self.bx.ns_b-self.ns_delete))
                numnc = np.zeros((len(self.xm_b), self.bx.ns_b-self.ns_delete))

                bmns = self.bx.bmns_b[:, self.ns_delete::]
                rmns = self.bx.rmns_b[:, self.ns_delete::]
                zmnc = self.bx.zmnc_b[:, self.ns_delete::]
                numnc = self.bx.numnc_b[:, self.ns_delete::]

            mn_factor = np.ones_like(bmnc)
            d_mn_factor = np.zeros_like(bmnc)
            mn_factor[self.xm_b == 1, :] = s_half_mn[None, :]**(-0.5)
            d_mn_factor[self.xm_b == 1, :] = -0.5*s_half_mn[None, :]**(-1.5)
            mn_factor[(self.xm_b % 2 == 1)*(self.xm_b > 1), :] = s_half_mn[None, :]**(-1.5)
            d_mn_factor[(self.xm_b % 2 == 1)*(self.xm_b > 1), :] = -1.5*s_half_mn[None, :]**(-2.5)
            mn_factor[(self.xm_b % 2 == 0)*(self.xm_b > 1), :] = s_half_mn[None, :]**(-1.)
            d_mn_factor[(self.xm_b % 2 == 0)*(self.xm_b > 1), :] = -s_half_mn[None, :]**(-2.)
        else:
            s_half_mn = self.s_half_ext
            bmnc = np.zeros((len(self.xm_b), self.bx.ns_b+2))
            bmnc[:, 1:-1] = self.bx.bmnc_b
            bmnc[:, 0] = 1.5*bmnc[:, 1] - 0.5*bmnc[:, 2]
            bmnc[:, -1] = 1.5*bmnc[:, -2] - 0.5*bmnc[:, -3]
            dbmncds = (bmnc[:, 2:-1] - bmnc[:, 1:-2])/ds
            mn_factor = np.ones_like(bmnc)
            d_mn_factor = np.zeros_like(bmnc)

            numns = np.zeros((len(self.xm_b), self.bx.ns_b+2))
            rmnc = np.zeros((len(self.xm_b), self.bx.ns_b+2))
            zmns = np.zeros((len(self.xm_b), self.bx.ns_b+2))
            numns[:, 1:-1] = self.bx.numns_b
            numns[:, 0] = 1.5*numns[:, 1] - 0.5*numns[:, 2]
            numns[:, -1] = 1.5*numns[:, -2] - 0.5*numns[:, -3]
            rmnc[:, 1:-1] = self.bx.rmnc_b
            rmnc[:, 0] = 1.5*rmnc[:, 1] - 0.5*rmnc[:, 2]
            rmnc[:, -1] = 1.5*rmnc[:, -2] - 0.5*rmnc[:, -3]
            zmns[:, 1:-1] = self.bx.zmns_b
            zmns[:, 0] = 1.5*zmns[:, 1] - 0.5*zmns[:, 2]
            zmns[:, -1] = 1.5*zmns[:, -2] - 0.5*zmns[:, -3]

            drmncds = (rmnc[:, 2:-1] - rmnc[:, 1:-2])/ds
            dzmnsds = (zmns[:, 2:-1] - zmns[:, 1:-2])/ds
            dnumnsds = (numns[:, 2:-1] - numns[:, 1:-2])/ds

            if self.asym:
                bmns = np.zeros((len(self.xm_b), self.bx.ns_b+2))
                bmns[:, 1:-1] = self.bx.bmns_b
                bmns[:, 0] = 1.5*bmns[:, 1] - 0.5*bmns[:, 2]
                bmns[:, -1] = 1.5*bmns[:, -2] - 0.5*bmns[:, -3]
                dbmnsds = (bmns[:, 2:-1] - bmns[:, 1:-2])/ds

                numnc = np.zeros((len(self.xm_b), self.bx.ns_b+2))
                rmns = np.zeros((len(self.xm_b), self.bx.ns_b+2))
                zmnc = np.zeros((len(self.xm_b), self.bx.ns_b+2))
                numnc[:, 1:-1] = self.bx.numnc_b
                numnc[:, 0] = 1.5*numnc[:, 1] - 0.5*numnc[:, 2]
                numnc[:, -1] = 1.5*numnc[:, -2] - 0.5*numnc[:, -3]
                rmns[:, 1:-1] = self.bx.rmns_b
                rmns[:, 0] = 1.5*rmns[:, 1] - 0.5*rmns[:, 2]
                rmns[:, -1] = 1.5*rmns[:, -2] - 0.5*rmns[:, -3]
                zmnc[:, 1:-1] = self.bx.zmnc_b
                zmnc[:, 0] = 1.5*zmnc[:, 1] - 0.5*zmnc[:, 2]
                zmnc[:, -1] = 1.5*zmnc[:, -2] - 0.5*zmnc[:, -3]

                drmnsds = (rmns[:, 2:-1] - rmns[:, 1:-2])/ds
                dzmncds = (zmnc[:, 2:-1] - zmnc[:, 1:-2])/ds
                dnumncds = (numnc[:, 2:-1] - numnc[:, 1:-2])/ds

        # Extrapolate to get points at s = 0 and s = 1
        iota[0] = 1.5*iota[1] - 0.5*iota[2]
        G[0] = 1.5*G[1] - 0.5*G[2]
        I[0] = 1.5*I[1] - 0.5*I[2]
        iota[-1] = 1.5*iota[-2] - 0.5*iota[-3]
        G[-1] = 1.5*G[-2] - 0.5*G[-3]
        I[-1] = 1.5*I[-2] - 0.5*I[-3]
        # Compute first derivatives - on full grid points in [1,ns-1]
        dGds = (G[2:-1] - G[1:-2])/ds
        dIds = (I[2:-1] - I[1:-2])/ds
        diotads = (iota[2:-1] - iota[1:-2])/ds

        self.psip_spline = InterpolatedUnivariateSpline(s_full, psip, k=self.order)
        if not self.enforce_vacuum:
            self.G_spline = InterpolatedUnivariateSpline(self.s_half_ext, G, k=self.order)
            self.I_spline = InterpolatedUnivariateSpline(self.s_half_ext, I, k=self.order)
            self.dGds_spline = InterpolatedUnivariateSpline(s_full[1:-1], dGds, k=self.order)
            self.dIds_spline = InterpolatedUnivariateSpline(s_full[1:-1], dIds, k=self.order)
        else:
            self.G_spline = InterpolatedUnivariateSpline(self.s_half_ext, np.mean(G)*np.ones_like(self.s_half_ext), k=self.order)
            self.I_spline = InterpolatedUnivariateSpline(self.s_half_ext, np.zeros_like(self.s_half_ext), k=self.order)
            self.dGds_spline = InterpolatedUnivariateSpline(s_full[1:-1], np.zeros_like(s_full[1:-1]), k=self.order)
            self.dIds_spline = InterpolatedUnivariateSpline(s_full[1:-1], np.zeros_like(s_full[1:-1]), k=self.order)
        if not self.no_shear:
            self.iota_spline = InterpolatedUnivariateSpline(self.s_half_ext, iota, k=self.order)
            self.diotads_spline = InterpolatedUnivariateSpline(s_full[1:-1], diotads, k=self.order)
        else:
            self.iota_spline = InterpolatedUnivariateSpline(self.s_half_ext, np.mean(iota)*np.ones_like(self.s_half_ext), k=self.order)
            self.diotads_spline = InterpolatedUnivariateSpline(s_full[1:-1], np.zeros_like(s_full[1:-1]), k=self.order)

        self.numns_splines = []
        self.rmnc_splines = []
        self.zmns_splines = []
        self.dnumnsds_splines = []
        self.drmncds_splines = []
        self.dzmnsds_splines = []
        self.bmnc_splines = []
        self.dbmncds_splines = []
        self.d_mn_factor_splines = []
        self.mn_factor_splines = []
        for im in range(len(self.xm_b)):
            self.numns_splines.append(InterpolatedUnivariateSpline(s_half_mn, mn_factor[im, :]*numns[im, :], k=self.order))
            self.rmnc_splines.append(InterpolatedUnivariateSpline(s_half_mn, mn_factor[im, :]*rmnc[im, :], k=self.order))
            self.zmns_splines.append(InterpolatedUnivariateSpline(s_half_mn, mn_factor[im, :]*zmns[im, :], k=self.order))
            self.mn_factor_splines.append(InterpolatedUnivariateSpline(s_half_mn, mn_factor[im, :], k=self.order))
            self.d_mn_factor_splines.append(InterpolatedUnivariateSpline(s_half_mn, d_mn_factor[im, :], k=self.order))
            if self.enforce_qs:
                is_QP = np.isinf(self.N)
                is_QS = not is_QP
                if is_QP:
                    keep_mode = (self.xm_b[im] == 0)
                if is_QS:
                    keep_mode = (self.xn_b[im] == self.N * self.xm_b[im])
            else:
                keep_mode = True
            if not keep_mode:
                self.bmnc_splines.append(InterpolatedUnivariateSpline(s_half_mn, 0*bmnc[im, :], k=self.order))
                self.dbmncds_splines.append(InterpolatedUnivariateSpline(s_full[1:-1], 0*dbmncds[im, :], k=self.order))
            else:
                self.bmnc_splines.append(InterpolatedUnivariateSpline(s_half_mn, mn_factor[im, :]*bmnc[im, :], k=self.order))
                if self.rescale:
                    self.dbmncds_splines.append(self.bmnc_splines[-1].derivative())
                else:
                    self.dbmncds_splines.append(InterpolatedUnivariateSpline(s_full[1:-1], dbmncds[im, :], k=self.order))

            if self.rescale:
                self.dnumnsds_splines.append(self.numns_splines[-1].derivative())
                self.drmncds_splines.append(self.rmnc_splines[-1].derivative())
                self.dzmnsds_splines.append(self.zmns_splines[-1].derivative())
            else:
                self.dnumnsds_splines.append(InterpolatedUnivariateSpline(s_full[1:-1], dnumnsds[im, :], k=self.order))
                self.drmncds_splines.append(InterpolatedUnivariateSpline(s_full[1:-1], drmncds[im, :], k=self.order))
                self.dzmnsds_splines.append(InterpolatedUnivariateSpline(s_full[1:-1], dzmnsds[im, :], k=self.order))

        if self.asym:
            self.numnc_splines = []
            self.rmns_splines = []
            self.zmnc_splines = []
            self.dnumncds_splines = []
            self.drmnsds_splines = []
            self.dzmncds_splines = []
            self.bmns_splines = []
            self.dbmnsds_splines = []
            for im in range(len(self.xm_b)):
                self.numnc_splines.append(InterpolatedUnivariateSpline(s_half_mn, mn_factor[im, :]*numnc[im, :], k=self.order))
                self.rmns_splines.append(InterpolatedUnivariateSpline(s_half_mn, mn_factor[im, :]*rmns[im, :], k=self.order))
                self.zmnc_splines.append(InterpolatedUnivariateSpline(s_half_mn, mn_factor[im, :]*zmnc[im, :], k=self.order))
                if self.enforce_qs:
                    is_QP = np.isinf(self.N)
                    is_QS = not is_QP
                    if is_QP:
                        keep_mode = (self.xm_b[im] == 0)
                    if is_QS:
                        keep_mode = (self.xn_b[im] == self.N * self.xm_b[im])
                else:
                    keep_mode = True
                if not keep_mode:
                    self.bmns_splines.append(InterpolatedUnivariateSpline(s_half_mn, 0*bmns[im, :], k=self.order))
                    self.dbmnsds_splines.append(InterpolatedUnivariateSpline(s_full[1:-1], 0*dbmnsds[im, :], k=self.order))
                else:
                    self.bmns_splines.append(InterpolatedUnivariateSpline(s_half_mn, mn_factor[im, :]*bmns[im, :], k=self.order))
                    if self.rescale:
                        self.dbmnsds_splines.append(self.bmns_splines[-1].derivative())
                    else:
                        self.dbmnsds_splines.append(InterpolatedUnivariateSpline(s_full[1:-1], dbmnsds[im, :], k=self.order))

                if self.rescale:
                    self.dnumncds_splines.append(self.numnc_splines[-1].derivative())
                    self.drmnsds_splines.append(self.rmns_splines[-1].derivative())
                    self.dzmncds_splines.append(self.zmnc_splines[-1].derivative())
                else:
                    self.dnumncds_splines.append(InterpolatedUnivariateSpline(s_full[1:-1], dnumncds[im, :], k=self.order))
                    self.drmnsds_splines.append(InterpolatedUnivariateSpline(s_full[1:-1], drmnsds[im, :], k=self.order))
                    self.dzmncds_splines.append(InterpolatedUnivariateSpline(s_full[1:-1], dzmncds[im, :], k=self.order))

    def compute_K(self):
        ntheta = 2 * (2 * self.mpol + 1)
        nzeta = 2 * (2 * self.ntor + 1)
        thetas = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
        dtheta = thetas[1]-thetas[0]
        zetas = np.linspace(0, 2*np.pi/self.nfp, nzeta, endpoint=False)
        dzeta = zetas[1]-zetas[0]
        thetas, zetas = np.meshgrid(thetas, zetas)
        thetas = thetas.flatten()
        zetas = zetas.flatten()

        if (self.comm is not None):
            size = self.comm.size
            rank = self.comm.rank

            angle_idxs = np.array([i * len(thetas) // size for i in range(size + 1)])
            first, last = angle_idxs[rank], angle_idxs[rank + 1]
            
            if (self.asym):
                kmnc_buffer = allocate_aligned_and_padded_array((len(self.s_half_ext), len(self.xm_b)))
            kmns_buffer = allocate_aligned_and_padded_array((len(self.s_half_ext), len(self.xm_b)))
            thetas = thetas[first:last]
            zetas = zetas[first:last]

        dzmnsds_half = allocate_aligned_and_padded_array((len(self.s_half_ext), len(self.xm_b)))
        drmncds_half = allocate_aligned_and_padded_array((len(self.s_half_ext), len(self.xm_b)))
        dnumnsds_half = allocate_aligned_and_padded_array((len(self.s_half_ext), len(self.xm_b)))
        bmnc_half = allocate_aligned_and_padded_array((len(self.s_half_ext), len(self.xm_b)))
        rmnc_half = allocate_aligned_and_padded_array((len(self.s_half_ext), len(self.xm_b)))
        zmns_half = allocate_aligned_and_padded_array((len(self.s_half_ext), len(self.xm_b)))
        numns_half = allocate_aligned_and_padded_array((len(self.s_half_ext), len(self.xm_b)))
        kmns = allocate_aligned_and_padded_array((len(self.s_half_ext), len(self.xm_b)))
        if self.asym:
            dzmncds_half = allocate_aligned_and_padded_array((len(self.s_half_ext), len(self.xm_b)))
            drmnsds_half = allocate_aligned_and_padded_array((len(self.s_half_ext), len(self.xm_b)))
            dnumncds_half = allocate_aligned_and_padded_array((len(self.s_half_ext), len(self.xm_b)))
            bmns_half = allocate_aligned_and_padded_array((len(self.s_half_ext), len(self.xm_b)))
            rmns_half = allocate_aligned_and_padded_array((len(self.s_half_ext), len(self.xm_b)))
            zmnc_half = allocate_aligned_and_padded_array((len(self.s_half_ext), len(self.xm_b)))
            numnc_half = allocate_aligned_and_padded_array((len(self.s_half_ext), len(self.xm_b)))
            kmnc = allocate_aligned_and_padded_array((len(self.s_half_ext), len(self.xm_b)))

        for im in range(len(self.xm_b)):
            mn_factor = self.mn_factor_splines[im](self.s_half_ext)
            d_mn_factor = self.d_mn_factor_splines[im](self.s_half_ext)
            bmnc_half[:, im] = self.bmnc_splines[im](self.s_half_ext)/mn_factor
            rmnc_half[:, im] = self.rmnc_splines[im](self.s_half_ext)/mn_factor
            zmns_half[:, im] = self.zmns_splines[im](self.s_half_ext)/mn_factor
            numns_half[:, im] = self.numns_splines[im](self.s_half_ext)/mn_factor
            dnumnsds_half[:, im] = ((self.dnumnsds_splines[im](self.s_half_ext) - numns_half[:, im]*d_mn_factor)/mn_factor)
            drmncds_half[:, im] = ((self.drmncds_splines[im](self.s_half_ext) - rmnc_half[:, im]*d_mn_factor)/mn_factor)
            dzmnsds_half[:, im] = ((self.dzmnsds_splines[im](self.s_half_ext) - zmns_half[:, im]*d_mn_factor)/mn_factor)
            if self.asym:
                bmns_half[:, im] = self.bmns_splines[im](self.s_half_ext)/mn_factor
                rmns_half[:, im] = self.rmns_splines[im](self.s_half_ext)/mn_factor
                zmnc_half[:, im] = self.zmnc_splines[im](self.s_half_ext)/mn_factor
                numnc_half[:, im] = self.numnc_splines[im](self.s_half_ext)/mn_factor
                dnumncds_half[:, im] = ((self.dnumncds_splines[im](self.s_half_ext) - numnc_half[:, im]*d_mn_factor)/mn_factor)
                drmnsds_half[:, im] = ((self.drmnsds_splines[im](self.s_half_ext) - rmns_half[:, im]*d_mn_factor)/mn_factor)
                dzmncds_half[:, im] = ((self.dzmncds_splines[im](self.s_half_ext) - zmnc_half[:, im]*d_mn_factor)/mn_factor)

        G_half = self.G_spline(self.s_half_ext)
        I_half = self.I_spline(self.s_half_ext)
        iota_half = self.iota_spline(self.s_half_ext)

        xm_b = align_and_pad(self.xm_b)
        xn_b = align_and_pad(self.xn_b)

        if (self.asym):
            sopp.compute_kmnc_kmns(kmnc, kmns, rmnc_half, drmncds_half, zmns_half, dzmnsds_half,
                                   numns_half, dnumnsds_half, bmnc_half,
                                   rmns_half, drmnsds_half, zmnc_half, dzmncds_half,
                                   numnc_half, dnumncds_half, bmns_half,
                                   iota_half, G_half, I_half, xm_b, xn_b, thetas, zetas)

            kmnc = kmnc*dtheta*dzeta*self.nfp/self.psi0
        else:
            sopp.compute_kmns(kmns, rmnc_half, drmncds_half, zmns_half, dzmnsds_half,
                             numns_half, dnumnsds_half, bmnc_half, iota_half, G_half, I_half,
                             xm_b, xn_b, thetas, zetas)
        kmns = kmns*dtheta*dzeta*self.nfp/self.psi0
        if self.comm is not None:
            if (self.asym):
                self.comm.Allreduce([kmnc, MPI.DOUBLE], kmnc_buffer, op=MPI.SUM)
                kmnc = kmnc_buffer
            self.comm.Allreduce([kmns, MPI.DOUBLE], kmns_buffer, op=MPI.SUM)
            kmns = kmns_buffer
        if self.proc0:
            self.kmns_splines = []
            for im in range(len(self.xm_b)):
                if self.enforce_qs:
                    is_QP = np.isinf(self.N)
                    is_QS = not is_QP
                    if is_QP:
                        keep_mode = (self.xm_b[im] == 0)
                    if is_QS:
                        keep_mode = (self.xn_b[im] == self.N * self.xm_b[im])
                else:
                    keep_mode = True
                if not keep_mode:
                    self.kmns_splines.append(InterpolatedUnivariateSpline(self.s_half_ext, 0*kmns[:, im], k=self.order))
                else:
                    self.kmns_splines.append(InterpolatedUnivariateSpline(self.s_half_ext, self.mn_factor_splines[im](self.s_half_ext)*kmns[:, im], k=self.order))

            if self.asym:
                self.kmnc_splines = []
                for im in range(len(self.xm_b)):
                    if self.enforce_qs:
                        is_QP = np.isinf(self.N)
                        is_QS = not is_QP
                        if is_QP:
                            keep_mode = (self.xm_b[im] == 0)
                        if is_QS:
                            keep_mode = (self.xn_b[im] == self.N * self.xm_b[im])
                    else:
                        keep_mode = True
                    if not keep_mode:
                        self.kmnc_splines.append(InterpolatedUnivariateSpline(self.s_half_ext, 0*kmnc[:, im], k=self.order))
                    else:
                        self.kmnc_splines.append(InterpolatedUnivariateSpline(self.s_half_ext, self.mn_factor_splines[im](self.s_half_ext)*kmnc[:, im], k=self.order))

        if self.comm is not None:
            self.kmns_splines = self.comm.bcast(self.kmns_splines, root=0)
            if self.asym:
                self.kmnc_splines = self.comm.bcast(self.kmnc_splines, root=0)

    def _K_impl(self, K):
        K[:, 0] = 0.
        if self.no_K:
            return

        @self.iterate_and_invert
        def _harmonics(im, s):
            return self.kmns_splines[im](s)/self.mn_factor_splines[im](s)
        inverse_fourier = sopp.inverse_fourier_transform_odd

        self._compute_impl(K[:, 0], _harmonics, inverse_fourier)

        if (self.asym):
            @self.iterate_and_invert
            def _harmonics(im, s):
                return self.kmnc_splines[im](s)/self.mn_factor_splines[im](s)
            inverse_fourier = sopp.inverse_fourier_transform_even

            self._compute_impl(K[:, 0], _harmonics, inverse_fourier)

    def _dKdtheta_impl(self, dKdtheta):
        dKdtheta[:, 0] = 0.
        if self.no_K:
            return

        @self.iterate_and_invert
        def _harmonics(im, s):
            return self.kmns_splines[im](s) * self.xm_b[im]/self.mn_factor_splines[im](s)
        inverse_fourier = sopp.inverse_fourier_transform_even

        self._compute_impl(dKdtheta[:, 0], _harmonics, inverse_fourier)

        if (self.asym):
            @self.iterate_and_invert
            def _harmonics(im, s):
                return -self.kmnc_splines[im](s) * self.xm_b[im]/self.mn_factor_splines[im](s)
            inverse_fourier = sopp.inverse_fourier_transform_odd

            self._compute_impl(dKdtheta[:, 0], _harmonics, inverse_fourier)

    def _dKdzeta_impl(self, dKdzeta):
        dKdzeta[:, 0] = 0.
        if self.no_K:
            return

        @self.iterate_and_invert
        def _harmonics(im, s):
            return -self.kmns_splines[im](s) * self.xn_b[im]/self.mn_factor_splines[im](s)
        inverse_fourier = sopp.inverse_fourier_transform_even

        self._compute_impl(dKdzeta[:, 0], _harmonics, inverse_fourier)

        if (self.asym):
            @self.iterate_and_invert
            def _harmonics(im, s):
                return self.kmnc_splines[im](s) * self.xn_b[im]/self.mn_factor_splines[im](s)
            inverse_fourier = sopp.inverse_fourier_transform_odd

            self._compute_impl(dKdzeta[:, 0], _harmonics, inverse_fourier)

    def _nu_impl(self, nu):
        nu[:, 0] = 0.
        @self.iterate_and_invert
        def _harmonics(im, s):
            return self.numns_splines[im](s)/self.mn_factor_splines[im](s)
        inverse_fourier = sopp.inverse_fourier_transform_odd

        self._compute_impl(nu[:, 0], _harmonics, inverse_fourier)

        if (self.asym):
            @self.iterate_and_invert
            def _harmonics(im, s):
                return self.numnc_splines[im](s)/self.mn_factor_splines[im](s)
            inverse_fourier = sopp.inverse_fourier_transform_even

            self._compute_impl(nu[:, 0], _harmonics, inverse_fourier)

    def _dnudtheta_impl(self, dnudtheta):
        dnudtheta[:, 0] = 0.
        @self.iterate_and_invert
        def _harmonics(im, s):
            return self.numns_splines[im](s)*self.xm_b[im]/self.mn_factor_splines[im](s)
        inverse_fourier = sopp.inverse_fourier_transform_even

        self._compute_impl(dnudtheta[:, 0], _harmonics, inverse_fourier)

        if (self.asym):
            @self.iterate_and_invert
            def _harmonics(im, s):
                return -self.numnc_splines[im](s)*self.xm_b[im]/self.mn_factor_splines[im](s)
            inverse_fourier = sopp.inverse_fourier_transform_odd

            self._compute_impl(dnudtheta[:, 0], _harmonics, inverse_fourier)

    def _dnudzeta_impl(self, dnudzeta):
        dnudzeta[:, 0] = 0.
        @self.iterate_and_invert
        def _harmonics(im, s):
            return -self.numns_splines[im](s)*self.xn_b[im]/self.mn_factor_splines[im](s)
        inverse_fourier = sopp.inverse_fourier_transform_even

        self._compute_impl(dnudzeta[:, 0], _harmonics, inverse_fourier)

        if (self.asym):
            @self.iterate_and_invert
            def _harmonics(im, s):
                return self.numnc_splines[im](s)*self.xn_b[im]/self.mn_factor_splines[im](s)
            inverse_fourier = sopp.inverse_fourier_transform_odd

            self._compute_impl(dnudzeta[:, 0], _harmonics, inverse_fourier)

    def _dnuds_impl(self, dnuds):
        dnuds[:, 0] = 0.
        @self.iterate_and_invert
        def _harmonics(im, s):
            d_mn_factor = self.d_mn_factor_splines[im](s)
            mn_factor = self.mn_factor_splines[im](s)
            return ((self.dnumnsds_splines[im](s) - self.numns_splines[im](s)*d_mn_factor/mn_factor)/mn_factor)
        inverse_fourier = sopp.inverse_fourier_transform_odd

        self._compute_impl(dnuds[:, 0], _harmonics, inverse_fourier)

        if (self.asym):
            @self.iterate_and_invert
            def _harmonics(im, s):
                d_mn_factor = self.d_mn_factor_splines[im](s)
                mn_factor = self.mn_factor_splines[im](s)
                return ((self.dnumncds_splines[im](s) - self.numnc_splines[im](s)*d_mn_factor/mn_factor)/mn_factor)
            inverse_fourier = sopp.inverse_fourier_transform_even

            self._compute_impl(dnuds[:, 0], _harmonics, inverse_fourier)

    def _dRdtheta_impl(self, dRdtheta):
        dRdtheta[:, 0] = 0.
        @self.iterate_and_invert
        def _harmonics(im, s):
            return -self.rmnc_splines[im](s)*self.xm_b[im]/self.mn_factor_splines[im](s)
        inverse_fourier = sopp.inverse_fourier_transform_odd

        self._compute_impl(dRdtheta[:, 0], _harmonics, inverse_fourier)

        if (self.asym):
            @self.iterate_and_invert
            def _harmonics(im, s):
                return self.rmns_splines[im](s)*self.xm_b[im]/self.mn_factor_splines[im](s)
            inverse_fourier = sopp.inverse_fourier_transform_even

            self._compute_impl(dRdtheta[:, 0], _harmonics, inverse_fourier)

    def _dRdzeta_impl(self, dRdzeta):
        dRdzeta[:, 0] = 0.
        @self.iterate_and_invert
        def _harmonics(im, s):
            return self.rmnc_splines[im](s)*self.xn_b[im]/self.mn_factor_splines[im](s)
        inverse_fourier = sopp.inverse_fourier_transform_odd

        self._compute_impl(dRdzeta[:, 0], _harmonics, inverse_fourier)

        if (self.asym):
            @self.iterate_and_invert
            def _harmonics(im, s):
                return -self.rmns_splines[im](s)*self.xn_b[im]/self.mn_factor_splines[im](s)
            inverse_fourier = sopp.inverse_fourier_transform_even

            self._compute_impl(dRdzeta[:, 0], _harmonics, inverse_fourier)

    def _dRds_impl(self, dRds):
        dRds[:, 0] = 0.
        @self.iterate_and_invert
        def _harmonics(im, s):
            d_mn_factor = self.d_mn_factor_splines[im](s)
            mn_factor = self.mn_factor_splines[im](s)
            return ((self.drmncds_splines[im](s) - self.rmnc_splines[im](s)*d_mn_factor/mn_factor)/mn_factor)
        inverse_fourier = sopp.inverse_fourier_transform_even

        self._compute_impl(dRds[:, 0], _harmonics, inverse_fourier)

        if (self.asym):
            @self.iterate_and_invert
            def _harmonics(im, s):
                d_mn_factor = self.d_mn_factor_splines[im](s)
                mn_factor = self.mn_factor_splines[im](s)
                return ((self.drmnsds_splines[im](s) - self.rmns_splines[im](s)*d_mn_factor/mn_factor)/mn_factor)
            inverse_fourier = sopp.inverse_fourier_transform_odd

            self._compute_impl(dRds[:, 0], _harmonics, inverse_fourier)

    def _R_impl(self, R):
        R[:, 0] = 0.
        @self.iterate_and_invert
        def _harmonics(im, s):
            return self.rmnc_splines[im](s)/self.mn_factor_splines[im](s)
        inverse_fourier = sopp.inverse_fourier_transform_even

        self._compute_impl(R[:, 0], _harmonics, inverse_fourier)

        if (self.asym):
            @self.iterate_and_invert
            def _harmonics(im, s):
                return self.rmns_splines[im](s)/self.mn_factor_splines[im](s)
            inverse_fourier = sopp.inverse_fourier_transform_odd

            self._compute_impl(R[:, 0], _harmonics, inverse_fourier)

    def _dZdtheta_impl(self, dZdtheta):
        dZdtheta[:, 0] = 0.
        @self.iterate_and_invert
        def _harmonics(im, s):
            return self.zmns_splines[im](s)*self.xm_b[im]/self.mn_factor_splines[im](s)
        inverse_fourier = sopp.inverse_fourier_transform_even

        self._compute_impl(dZdtheta[:, 0], _harmonics, inverse_fourier)

        if (self.asym):
            @self.iterate_and_invert
            def _harmonics(im, s):
                return -self.zmnc_splines[im](s)*self.xm_b[im]/self.mn_factor_splines[im](s)
            inverse_fourier = sopp.inverse_fourier_transform_odd

            self._compute_impl(dZdtheta[:, 0], _harmonics, inverse_fourier)

    def _dZdzeta_impl(self, dZdzeta):
        dZdzeta[:, 0] = 0.
        @self.iterate_and_invert
        def _harmonics(im, s):
            return -self.zmns_splines[im](s)*self.xn_b[im]/self.mn_factor_splines[im](s)
        inverse_fourier = sopp.inverse_fourier_transform_even

        self._compute_impl(dZdzeta[:, 0], _harmonics, inverse_fourier)

        if (self.asym):
            @self.iterate_and_invert
            def _harmonics(im, s):
                return self.zmnc_splines[im](s)*self.xn_b[im]/self.mn_factor_splines[im](s)
            inverse_fourier = sopp.inverse_fourier_transform_odd

            self._compute_impl(dZdzeta[:, 0], _harmonics, inverse_fourier)

    def _dZds_impl(self, dZds):
        dZds[:, 0] = 0.
        @self.iterate_and_invert
        def _harmonics(im, s):
            d_mn_factor = self.d_mn_factor_splines[im](s)
            mn_factor = self.mn_factor_splines[im](s)
            return ((self.dzmnsds_splines[im](s) - self.zmns_splines[im](s)*d_mn_factor/mn_factor)/mn_factor)
        inverse_fourier = sopp.inverse_fourier_transform_odd

        self._compute_impl(dZds[:, 0], _harmonics, inverse_fourier)

        if (self.asym):
            @self.iterate_and_invert
            def _harmonics(im, s):
                d_mn_factor = self.d_mn_factor_splines[im](s)
                mn_factor = self.mn_factor_splines[im](s)
                return ((self.dzmncds_splines[im](s) - self.zmnc_splines[im](s)*d_mn_factor/mn_factor)/mn_factor)
            inverse_fourier = sopp.inverse_fourier_transform_even

            self._compute_impl(dZds[:, 0], _harmonics, inverse_fourier)

    def _Z_impl(self, Z):
        Z[:, 0] = 0.
        @self.iterate_and_invert
        def _harmonics(im, s):
            return self.zmns_splines[im](s)/self.mn_factor_splines[im](s)
        inverse_fourier = sopp.inverse_fourier_transform_odd

        self._compute_impl(Z[:, 0], _harmonics, inverse_fourier)

        if (self.asym):
            @self.iterate_and_invert
            def _harmonics(im, s):
                return self.zmnc_splines[im](s)/self.mn_factor_splines[im](s)
            inverse_fourier = sopp.inverse_fourier_transform_even

            self._compute_impl(Z[:, 0], _harmonics, inverse_fourier)

    def _psip_impl(self, psip):
        points = self.get_points_ref()
        s = points[:, 0]
        us, inv = np.unique(s, return_inverse=True)
        psip[:] = self.psip_spline(us)[inv][:, None]

    def _G_impl(self, G):
        points = self.get_points_ref()
        s = points[:, 0]
        us, inv = np.unique(s, return_inverse=True)
        G[:] = self.G_spline(us)[inv][:, None]

    def _I_impl(self, I):
        points = self.get_points_ref()
        s = points[:, 0]
        us, inv = np.unique(s, return_inverse=True)
        I[:] = self.I_spline(us)[inv][:, None]

    def _iota_impl(self, iota):
        points = self.get_points_ref()
        s = points[:, 0]
        us, inv = np.unique(s, return_inverse=True)
        iota[:] = self.iota_spline(us)[inv][:, None]

    def _dGds_impl(self, dGds):
        points = self.get_points_ref()
        s = points[:, 0]
        us, inv = np.unique(s, return_inverse=True)
        dGds[:] = self.dGds_spline(us)[inv][:, None]

    def _dIds_impl(self, dIds):
        points = self.get_points_ref()
        s = points[:, 0]
        us, inv = np.unique(s, return_inverse=True)
        dIds[:] = self.dIds_spline(us)[inv][:, None]

    def _diotads_impl(self, diotads):
        points = self.get_points_ref()
        s = points[:, 0]
        us, inv = np.unique(s, return_inverse=True)
        diotads[:] = self.diotads_spline(us)[inv][:, None]

    def _modB_impl(self, modB):
        modB[:, 0] = 0.
        @self.iterate_and_invert
        def _harmonics(im, s):
            return self.bmnc_splines[im](s)/self.mn_factor_splines[im](s)
        inverse_fourier = sopp.inverse_fourier_transform_even

        self._compute_impl(modB[:, 0], _harmonics, inverse_fourier)

        if (self.asym):
            @self.iterate_and_invert
            def _harmonics(im, s):
                return self.bmns_splines[im](s)/self.mn_factor_splines[im](s)
            inverse_fourier = sopp.inverse_fourier_transform_odd

            self._compute_impl(modB[:, 0], _harmonics, inverse_fourier)

    def _dmodBdtheta_impl(self, dmodBdtheta):
        dmodBdtheta[:, 0] = 0.
        @self.iterate_and_invert
        def _harmonics(im, s):
            return -self.xm_b[im]*self.bmnc_splines[im](s)/self.mn_factor_splines[im](s)
        inverse_fourier = sopp.inverse_fourier_transform_odd

        self._compute_impl(dmodBdtheta[:, 0], _harmonics, inverse_fourier)

        if (self.asym):
            @self.iterate_and_invert
            def _harmonics(im, s):
                return self.xm_b[im]*self.bmns_splines[im](s)/self.mn_factor_splines[im](s)
            inverse_fourier = sopp.inverse_fourier_transform_even

            self._compute_impl(dmodBdtheta[:, 0], _harmonics, inverse_fourier)

    def _dmodBdzeta_impl(self, dmodBdzeta):
        dmodBdzeta[:, 0] = 0.
        @self.iterate_and_invert
        def _harmonics(im, s):
            return self.xn_b[im]*self.bmnc_splines[im](s)/self.mn_factor_splines[im](s)
        inverse_fourier = sopp.inverse_fourier_transform_odd

        self._compute_impl(dmodBdzeta[:, 0], _harmonics, inverse_fourier)

        if (self.asym):
            @self.iterate_and_invert
            def _harmonics(im, s):
                return -self.xn_b[im]*self.bmns_splines[im](s)/self.mn_factor_splines[im](s)
            inverse_fourier = sopp.inverse_fourier_transform_even

            self._compute_impl(dmodBdzeta[:, 0], _harmonics, inverse_fourier)

    def _dmodBds_impl(self, dmodBds):
        dmodBds[:, 0] = 0.
        @self.iterate_and_invert
        def _harmonics(im, s):
            mn_factor = self.mn_factor_splines[im](s)
            d_mn_factor = self.d_mn_factor_splines[im](s)
            return ((self.dbmncds_splines[im](s) - self.bmnc_splines[im](s)*d_mn_factor/mn_factor)/mn_factor)
        inverse_fourier = sopp.inverse_fourier_transform_even

        self._compute_impl(dmodBds[:, 0], _harmonics, inverse_fourier)

        if (self.asym):
            @self.iterate_and_invert
            def _harmonics(im, s):
                mn_factor = self.mn_factor_splines[im](s)
                d_mn_factor = self.d_mn_factor_splines[im](s)
                return ((self.dbmnsds_splines[im](s) - self.bmns_splines[im](s)*d_mn_factor/mn_factor)/mn_factor)
            inverse_fourier = sopp.inverse_fourier_transform_odd

            self._compute_impl(dmodBds[:, 0], _harmonics, inverse_fourier)

    def _compute_impl(self, output, harmonics, inverse_fourier):
        if (self.comm is not None):
            size = self.comm.size
            rank = self.comm.rank

            mn_idxs = np.array([i * len(self.xm_b) // size for i in range(size + 1)])
            first_mn, last_mn = mn_idxs[rank], mn_idxs[rank + 1]

            recv_buffer = np.zeros(output.shape)
        else:
            first_mn, last_mn = 0, len(self.xm_b)

        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        us, inv = np.unique(s, return_inverse=True)
        if len(s) > 1:
            padded_thetas = align_and_pad(thetas)
            padded_zetas = align_and_pad(zetas)
            padded_buffer = allocate_aligned_and_padded_array(output.shape)
            chunk_mn = allocate_aligned_and_padded_array((last_mn-first_mn, len(inv)))
            # release memory manually. maybe not be needed anymore
            s, thetas, zetas = None, None, None
            harmonics(us, chunk_mn, inv, 0, last_mn-first_mn, first_mn)
            xm = self.xm_b[first_mn:last_mn]
            xn = self.xn_b[first_mn:last_mn]
        else:
            padded_thetas = thetas
            padded_zetas = zetas
            padded_buffer = np.zeros(output.shape)
            chunk_mn = allocate_aligned_and_padded_array((last_mn-first_mn,))
            harmonics(us, chunk_mn, inv, 0, last_mn-first_mn, first_mn)
            xm = align_and_pad(self.xm_b[first_mn:last_mn])
            xn =  align_and_pad(self.xn_b[first_mn:last_mn])
        
        
        inverse_fourier(padded_buffer, chunk_mn, xm, xn, padded_thetas, padded_zetas, self.ntor, self.nfp)
        chunk_mn, padded_thetas, padded_zetas =  None, None, None

        if (self.comm is not None):
            # In place reduce is slightly slower
            # self.mpi.comm_world.Allreduce(MPI.IN_PLACE, [padded_buffer[:len(inv)], MPI.DOUBLE], op=MPI.SUM)
            self.comm.Allreduce([padded_buffer[:len(inv)], MPI.DOUBLE], recv_buffer, op=MPI.SUM)
            output += recv_buffer
        else:
            output += padded_buffer[:len(inv)]

    def iterate_and_invert(self, func):
        def _f(us, output, inv, start, end, offset):
            length = len(inv)
            if length > 1:
                for im in range(start, end):
                    output[im, :length] = func(im + offset, us)[inv]
            else:
                for im in range(start, end):
                    output[im] = func(im + offset, us)[inv]
        return _f

class InterpolatedBoozerField(sopp.InterpolatedBoozerField, BoozerMagneticField):
    r"""
    This field takes an existing :class:`BoozerMagneticField` and interpolates it on a
    regular grid in :math:`s,\theta,\zeta`. This resulting interpolant can then
    be evaluated very quickly. This is modeled after :class:`InterpolatedField`.
    """

    def __init__(self, field, degree, srange, thetarange, zetarange, extrapolate=True, nfp=1, stellsym=True, initialize=[]):
        r"""
        Args:
            field: the underlying :class:`simsopt.field.boozermagneticfield.
                BoozerMagneticField` to be interpolated.
            degree: the degree of the piecewise polynomial interpolant.
            srange: a 3-tuple of the form ``(smin, smax, ns)``. This mean that
                the interval ``[smin, smax]`` is split into ``ns`` many subintervals.
            thetarange: a 3-tuple of the form ``(thetamin, thetamax, ntheta)``.
                thetamin must be >= 0, and thetamax must be <=2*pi.
            zetarange: a 3-tuple of the form ``(zetamin, zetamax, nzeta)``.
                zetamin must be >= 0, and thetamax must be <=2*pi.
            extrapolate: whether to extrapolate the field when evaluate outside
                         the integration domain or to throw an error.
            nfp: Whether to exploit rotational symmetry. In this case any toroidal angle
                 is always mapped into the interval :math:`[0, 2\pi/\mathrm{nfp})`,
                 hence it makes sense to use ``zetamin=0`` and
                 ``zetamax=2*np.pi/nfp``.
            stellsym: Whether to exploit stellarator symmetry. In this case
                      ``theta`` is always mapped to the interval :math:`[0, \pi]`,
                      hence it makes sense to use ``thetamin=0`` and ``thetamax=np.pi``.
            initialize: A list of strings, each of which is the name of a 
                field quantitty, e.g., `modB`, to be initialized when the interpolant is created.
        """
        BoozerMagneticField.__init__(self, field.psi0)
        if (np.any(np.asarray(thetarange[0:2]) < 0) or np.any(np.asarray(thetarange[0:2]) > 2*np.pi)):
            raise ValueError("thetamin and thetamax must be in [0,2*pi]")
        if (np.any(np.asarray(zetarange[0:2]) < 0) or np.any(np.asarray(zetarange[0:2]) > 2*np.pi)):
            raise ValueError("zetamin and zetamax must be in [0,2*pi]")
        if stellsym and (np.any(np.asarray(thetarange[0:2]) < 0) or np.any(np.asarray(thetarange[0:2]) > np.pi)):
            logger.warning(fr"Sure about thetarange=[{thetarange[0]},{thetarange[1]}]? When exploiting stellarator symmetry, the interpolant is only evaluated for theta in [0,pi].")
        if nfp > 1 and (np.any(np.asarray(zetarange[0:2]) < 0) or np.any(np.asarray(zetarange[0:2]) > 2*np.pi/nfp)):
            logger.warning(fr"Sure about zetarange=[{zetarange[0]},{zetarange[1]}]? When exploiting rotational symmetry, the interpolant is only evaluated for zeta in [0,2\pi/nfp].")

        sopp.InterpolatedBoozerField.__init__(self, field, degree, srange, thetarange, zetarange, extrapolate, nfp, stellsym)

        if initialize:
            for item in initialize:
                getattr(self, item)()


class ShearAlfvenWave(sopp.ShearAlfvenWave):
    r"""
        Class representing a generic Shear Alfvén Wave (SAW).

        The Shear Alfvén Wave (SAW) propagates in an equilibrium magnetic field `B0` and is represented by
        the scalar potential `Phi` and vector potential parameter `alpha`. The SAW magnetic field is defined
        as the curl of `(alpha * B0)`.

        This class provides a framework for representing SAWs in Boozer coordinates with attributes for computing
        the scalar and vector potentials and their derivatives: `Phi`, `dPhidpsi`, `Phidot`, etc.

        This class is designed to be a base class that can be extended to implement specific behaviors or
        variations of Shear Alfvén Waves.

        The usage of :class:`ShearAlfvenWave` is as follows:

        .. code-block:: python

            # Create an instance of a Boozer magnetic field
            B0 = sopp.BoozerAnalytic(etabar, B0, N, G0, psi0, iota0)

            # Create an instance of ShearAlfvenWave using the equilibrium field B0
            saw = ShearAlfvenWave(B0)

            # Points is a (n, 4) numpy array defining :math:`(s, \theta, \zeta, \text{time})`
            points = ...
            saw.set_points(points)

            # Compute scalar potential Phi at the specified points
            Phi = saw.Phi()

        Attributes:
        ----------
        Phi : function
            Computes the scalar potential `Phi` of the shear Alfvén wave perturbation.
        dPhidpsi : function
            Computes the derivative of the scalar potential `Phi` with respect to `psi`.
        Phidot : function
            Computes the time derivative of the scalar potential `Phi`.
        dPhidtheta : function
            Computes the derivative of the scalar potential `Phi` with respect to `theta`.
        dPhidzeta : function
            Computes the derivative of the scalar potential `Phi` with respect to `zeta`.
        alpha : function
            Computes the vector potential parameter `alpha`.
        alphadot : function
            Computes the time derivative of the vector potential parameter `alpha`.
        dalphadtheta : function
            Computes the derivative of the vector potential parameter `alpha` with respect to `theta`.
        dalphadpsi : function
            Computes the derivative of the vector potential parameter `alpha` with respect to `psi`.
        dalphadzeta : function
            Computes the derivative of the vector potential parameter `alpha` with respect to `zeta`.

        For further details, see Paul et al., JPP (2023; 89(5): 905890515. doi:10.1017/S0022377823001095)
        and references therein.

        Parameters
        ----------
        B0 : BoozerMagneticField
            Instance of a magnetic field in Boozer coordinates that provides the equilibrium field `B0`.

        Raises
        ------
        TypeError
            If `B0` is not an instance of `BoozerMagneticField`.

        """
    def __init__(self, B0):
        if not isinstance(B0, sopp.BoozerMagneticField):
            raise TypeError("B0 must be an instance of BoozerMagneticField.")

        # Call the constructor of the base C++ class
        super().__init__(B0)

class ShearAlfvenHarmonic(sopp.ShearAlfvenHarmonic):
    r"""
    Class representing a single harmonic Shear Alfvén Wave (SAW) in a given equilibrium magnetic field.

    This class initializes a Shear Alfvén Wave with a scalar potential of the form:

    .. math::
        \Phi(s, \theta, \zeta, t) = \hat{\Phi}(s) \sin(m \theta - n \zeta + \omega t + \text{phase}),

    where :math:`\hat{\Phi}(s)` is a radial profile, :math:`m` is the poloidal mode number, :math:`n` is the toroidal mode number,
    :math:`\omega` is the frequency, and `phase` is the phase shift. The vector potential `\alpha` is determined by the ideal
    Ohm's law (i.e., zero electric field along the field line). This representation is used to describe SAWs propagating in
    an equilibrium magnetic field :math:`B_0`.

    Attributes
    ----------
    Phihat_value_or_tuple : Union[float, Tuple[List[float], List[float]]]
        The radial profile of the scalar potential `\hat{\Phi}(s)`. It can be either:
        - A constant value (float) that represents a uniform `\hat{\Phi}`.
        - A tuple of two lists: `s_values` and `Phihat_values`, defining a varying profile.
    Phim : int
        Poloidal mode number `m`.
    Phin : int
        Toroidal mode number `n`.
    omega : float
        Frequency of the harmonic wave.
    phase : float
        Phase of the harmonic wave.
    """
    def __init__(self, Phihat_value_or_tuple, Phim: int, Phin: int, omega: float,
                 phase: float, B0: sopp.BoozerMagneticField):
        """
        Initialize a single harmonic Shear Alfvén Wave (SAW) in a given equilibrium magnetic field.

        Parameters
        ----------
        Phihat_value_or_tuple : Union[float, int, Tuple[List[float], List[float]]]
            The radial profile of the scalar potential `\hat{\Phi}(s)`.
            It can be either:
            - A constant value (float or int) that represents a uniform `\hat{\Phi}`.
            - A tuple of two lists: `s_values` and `Phihat_values`, defining a varying profile.
        Phim : int
            Poloidal mode number `m`.
        Phin : int
            Toroidal mode number `n`.
        omega : float
            Frequency of the harmonic wave.
        phase : float
            Phase of the harmonic wave.
        B0 : BoozerMagneticField
            Instance of a magnetic field in Boozer coordinates that provides the equilibrium field `B_0`.

        Raises
        ------
        TypeError
            If `B0` is not an instance of `BoozerMagneticField`.
            If `Phihat_value_or_tuple` is not a float, int, or a tuple of lists.
            If the tuple does not contain lists of floats.
        """

        # Validate B0 type
        if not isinstance(B0, sopp.BoozerMagneticField):
            raise TypeError("B0 must be an instance of BoozerMagneticField.")

        # Determine how to initialize Phihat
        if isinstance(Phihat_value_or_tuple, tuple):
            if len(Phihat_value_or_tuple) != 2:
                raise TypeError("Phihat_value_or_tuple must be a tuple of two lists: (s_values, Phihat_values).")

            s_vals, Phihat_vals = Phihat_value_or_tuple

            # Ensure both s_vals and Phihat_vals are lists of floats
            if not (all(isinstance(x, float) for x in s_vals) and
                    all(isinstance(x, float) for x in Phihat_vals)):
                raise TypeError("s_values and Phihat_values must be lists of floats.")

            phihat_object = sopp.Phihat(s_vals, Phihat_vals)
        else:
            # Try to convert Phihat_value_or_tuple to a float if possible
            try:
                Phihat_value = float(Phihat_value_or_tuple)
                # If Phihat_value_or_tuple can be converted to a float, use it as a constant value
                phihat_object = sopp.Phihat([0, 1], [Phihat_value, Phihat_value])
            except (TypeError, ValueError):
                raise TypeError("Phihat_value_or_tuple must be either a float, an int, or a tuple of (s_values, Phihat_values).")

        # Call the constructor of the base C++ class
        super().__init__(phihat_object, Phim, Phin, omega, phase, B0)

class ShearAlfvenWavesSuperposition(sopp.ShearAlfvenWavesSuperposition):
    r"""
    Class representing a superposition of multiple Shear Alfvén Waves (SAWs).

    This class models the superposition of multiple Shear Alfvén waves, combining their scalar
    potential `Phi`, vector potential `alpha`, and their respective derivatives to represent a more
    complex wave structure in the equilibrium field `B0`.

    The superposition of waves is initialized with a base wave, which defines the reference
    equilibrium field `B0` for all subsequent waves added to the superposition. All added waves
    must have the same `B0` field.

    See Paul et al., JPP (2023; 89(5):905890515. doi:10.1017/S0022377823001095) for more details.

    Parameters
    ----------
    SAWs : list of ShearAlfvenWave
        A list of ShearAlfvenWave objects to be superposed. The first wave in the list is used
        as the base wave and defines the reference `B0` field for the superposition. All other
        waves in the list must have the same `B0`.

    Raises
    ------
    TypeError
        If `SAWs` is not a list of `ShearAlfvenWave` objects.
        If the base wave is not provided or if the waves have different `B0` fields.

    Examples
    --------
    .. code-block:: python

        # Create a list of ShearAlfvenWave objects
        wave1 = ShearAlfvenHarmonic(...)  # Initialize a harmonic wave
        wave2 = ShearAlfvenHarmonic(...)  # Initialize another harmonic wave

        # Create a superposition of these waves
        superposition = ShearAlfvenWavesSuperposition([wave1, wave2])

        # Set points for evaluation
        points = ...  # Define points (s, theta, zeta, time)
        superposition.set_points(points)

    """
    def __init__(self, SAWs: list):
        if not isinstance(SAWs, list) or not all(isinstance(SAW, sopp.ShearAlfvenWave) for SAW in SAWs):
            raise TypeError("SAWs must be a list of ShearAlfvenWave objects.")

        if len(SAWs) == 0:
            raise ValueError("At least one ShearAlfvenWave object must be provided.")

        # Initialize the base C++ class with the first wave as the base wave
        super().__init__(SAWs[0])

        # Add subsequent waves to the superposition
        for SAW in SAWs[1:]:
            self.add_wave(SAW)
