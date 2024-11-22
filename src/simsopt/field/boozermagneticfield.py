import simsoptpp as sopp
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import logging
from booz_xform import Booz_xform
from .._core.util import parallel_loop_bounds, align_and_pad, allocate_aligned_and_padded_array

logger = logging.getLogger(__name__)

__all__ = ["BoozerMagneticField","BoozerAnalytic","BoozerRadialInterpolant","InterpolatedBoozerField"]

try:
    from mpi4py import MPI
except ImportError as e:
    MPI = None
    logger.debug(str(e))

from simsopt.mhd.vmec import Vmec
from simsopt.mhd.boozer import Boozer

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

    :class:`BoozerMagneticField` has a cache to avoid repeated calculations.
    To clear this cache manually, call the :func:`clear_cached_properties()` function.
    The cache is automatically cleared when :func:`set_points` is called or one of the dependencies
    changes.
    """

    def __init__(self, psi0):
        self.psi0 = psi0
        sopp.BoozerMagneticField.__init__(self, psi0)

    def clear_cached_properties(self):
        """Clear the cache."""
        sopp.BoozerMagneticField.invalidate_cache(self)

    def recompute_bell(self, parent=None):
        if np.any(self.dofs_free_status):
            self.clear_cached_properties()

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


class BoozerAnalytic(BoozerMagneticField):
    r"""
    Computes a :class:`BoozerMagneticField` based on a first-order expansion in
    distance from the magnetic axis (Landreman & Sengupta, Journal of Plasma
    Physics 2018). Here the magnetic field strength is expressed as,

    .. math::
        B(s,\theta,\zeta) = B_0 \left(1 + \overline{\eta} \sqrt{2s\psi_0/\overline{B}}\cos(\theta - N \zeta)\right),

    the covariant components are,

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
    """

    def __init__(self, etabar, B0, N, G0, psi0, iota0, Bbar=1., I0=0., G1=0.,
                 I1=0., K1=0., iota1=0., B0z=0.):
        self.etabar = etabar
        self.B0 = B0
        self.B0z = B0z
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
        self.invalidate_cache()
        self.etabar = etabar

    def set_B0(self, B0):
        self.invalidate_cache()
        self.B0 = B0

    def set_B0z(self, B0z):
        self.invalidate_cache()
        self.B0z = B0z

    def set_Bbar(self, Bbar):
        self.invalidate_cache()
        self.Bbar = Bbar

    def set_N(self, N):
        self.invalidate_cache()
        self.N = N

    def set_G0(self, G0):
        self.invalidate_cache()
        self.G0 = G0

    def set_I0(self, I0):
        self.invalidate_cache()
        self.I0 = I0

    def set_G1(self, G1):
        self.invalidate_cache()
        self.G1 = G1

    def set_I1(self, I1):
        self.invalidate_cache()
        self.I1 = I1

    def set_K1(self, K1):
        self.invalidate_cache()
        self.K1 = K1

    def set_iota0(self, iota0):
        self.invalidate_cache()
        self.iota0 = iota0

    def set_iota1(self, iota1):
        self.invalidate_cache()
        self.iota1 = iota1

    def set_psi0(self, psi0):
        self.invalidate_cache()
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
        modB[:, 0] = self.B0*(1 + self.etabar*r*np.cos(thetas-self.N*zetas)) + self.B0z*np.cos(zetas)

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
        dmodBdtheta[:, 0] = -self.B0*self.etabar*r*np.sin(thetas-self.N*zetas) 

    def _dmodBdzeta_impl(self, dmodBdzeta):
        points = self.get_points_ref()
        s = points[:, 0]
        thetas = points[:, 1]
        zetas = points[:, 2]
        psi = s*self.psi0
        r = np.sqrt(np.abs(2*psi/self.Bbar))
        dmodBdzeta[:, 0] = self.N*self.B0*self.etabar*r*np.sin(thetas-self.N*zetas) - self.B0z*np.sin(zetas)

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
    Given a :class:`Vmec` instance, performs a Boozer coordinate transformation using
    ``BOOZXFORM``.
    The magnetic field can be computed at any point in Boozer
    coordinates using radial spline interpolation (``scipy.interpolate.InterpolatedUnivariateSpline``)
    and an inverse Fourier transform in the two angles.
    Throughout stellarator symmetry is assumed.

    Args:
        equil: instance of :class:`simsopt.mhd.vmec.Vmec`, :class:`simsopt.mhd.boozer.Boozer`,
            or :class:`Booz_xform`. If it is an instance of :class:`simsopt.mhd.boozer.Boozer`, the
            `compute_surfs` needs to include all of the grid points in the
            half-radius grid of the corresponding Vmec equilibrium.
        order: (int) order for radial interpolation. Must satisfy 1 <= order <= 5.
        mpol: (int) number of poloidal mode numbers for BOOZXFORM (defaults to 32)
        ntor: (int) number of toroidal mode numbers for BOOZXFORM (defaults to 32)
        N: Helicity of quasisymmetry to enforce. If specified, then the non-symmetric Fourier
            harmonics of :math:`B` and :math:`K` are filtered out. Otherwise, all harmonics are kept.
            (defaults to ``None``)
        enforce_vacuum: If True, a vacuum field is assumed, :math:`G` is
            set to its mean value, :math:`I = 0`, and :math:`K = 0`.
        rescale: If True, use the interpolation method in the DELTA5D code. Here, a few
            of the first radial grid points or (``bmnc``, ``rmnc``, ``zmns``, ``numns``, ``kmns``)
            are deleted (determined by ``ns_delete``). The Fourier harmonics are then rescaled as:
                bmnc(s)/s^(1/2) for m = 1

                bmnc(s)/s for m even and >= 2

                bmnc(s)/s^(3/2) for m odd and >=3

            before performing interpolation and spline differentiation to
            obtain ``dbmncds``. If ``False``, interpolation of the unscaled Fourier
            harmonics and its finite-difference derivative wrt ``s`` is performed
            instead (defaults to ``False``)
        ns_delete: (see ``rescale``) (defaults to 0)
        no_K: (bool) If ``True``, the Boozer :math:`K` will not be computed or
            interpolated.
        write_boozmn: (bool) If ``True``, save the booz_xform transformation in
            a filename specified by ``boozmn_name``. (defaults to ``True``)
        boozmn_name: (string) Filename to save booz_xform transformation if
            ``write_boozmn`` is ``True``.
        mpi: A :obj:`simsopt.util.mpi.MpiPartition` instance, from which
          the worker groups will be used for spline calculations. If ``None``,
          defaults to ``equil.mpi``. If ``equil`` is a ``Booz_xform`` instance,
          and ``mpi`` is ``None``, each MPI process will compute splines
          independently.
         verbose: If True, additional output is written.
    """

    def __init__(self, equil, order, mpol=32, ntor=32, N=None, enforce_vacuum=False,
                 rescale=False, ns_delete=0, no_K=False, write_boozmn=True, mpi=None,
                 boozmn_name="boozmn.nc", verbose=0, no_shear=False):
        if (mpi is None and not isinstance(equil, Booz_xform)):
            self.mpi = equil.mpi
        else:
            self.mpi = mpi

        if self.mpi is not None:
            self.proc0 = False
            if self.mpi.comm_world.rank==0:
                self.proc0 = True
        else:
            self.proc0 = True

        if isinstance(equil, Vmec):
            if self.proc0:
                equil.run()
                booz = Booz_xform()
                booz.read_wout(equil.output_file,True)
                booz.verbose = verbose
                booz.mboz = mpol
                booz.nboz = ntor
                booz.run()
                if write_boozmn:
                    booz.write_boozmn(boozmn_name)
                self.bx = booz
        elif isinstance(equil, Boozer):
            if self.proc0:
                booz = equil
                # Determine if radial grid for Boozer needs to be updated

                # Grid not initialized
                if len(booz.bx.s_b) == 0:
                    booz.register(booz.equil.s_half_grid)
                # Grid does not have correct size
                elif (len(booz.bx.s_b) != len(booz.bx.s_b)):
                    booz.register(booz.equil.s_half_grid)
                # Grid does not match Vmec half grid
                elif (np.any(booz.bx.s_b != booz.bx.s_b)):
                    pass
                elif (len(booz.bx.s_b) != len(booz.equil.s_half_grid)):
                    booz.register(booz.equil.s_half_grid)
                # Grid does not match Vmec half grid
                elif (np.any(booz.bx.s_b != booz.equil.s_half_grid)):
                    booz.register(booz.equil.s_half_grid)

                # Run booz_xform if needed
                if booz.need_to_run_code:
                    booz.run()
                    if (write_boozmn):
                        if self.proc0:
                            booz.bx.write_boozmn(boozmn_name)
                self.bx = booz.bx
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
        if self.mpi is not None:
            self.psi0 = self.mpi.comm_world.bcast(self.psi0, root=0)
            self.nfp = self.mpi.comm_world.bcast(self.nfp, root=0)
            self.mpol = self.mpi.comm_world.bcast(self.mpol, root=0)
            self.ntor = self.mpi.comm_world.bcast(self.ntor, root=0)
            self.asym = self.mpi.comm_world.bcast(self.asym, root=0)
            self.psip_spline = self.mpi.comm_world.bcast(self.psip_spline, root=0)
            self.G_spline = self.mpi.comm_world.bcast(self.G_spline, root=0)
            self.I_spline = self.mpi.comm_world.bcast(self.I_spline, root=0)
            self.dGds_spline = self.mpi.comm_world.bcast(self.dGds_spline, root=0)
            self.dIds_spline = self.mpi.comm_world.bcast(self.dIds_spline, root=0)
            self.iota_spline = self.mpi.comm_world.bcast(self.iota_spline, root=0)
            self.diotads_spline = self.mpi.comm_world.bcast(self.diotads_spline, root=0)
            self.numns_splines = self.mpi.comm_world.bcast(self.numns_splines, root=0)
            self.rmnc_splines = self.mpi.comm_world.bcast(self.rmnc_splines, root=0)
            self.zmns_splines = self.mpi.comm_world.bcast(self.zmns_splines, root=0)
            self.dnumnsds_splines = self.mpi.comm_world.bcast(self.dnumnsds_splines, root=0)
            self.drmncds_splines = self.mpi.comm_world.bcast(self.drmncds_splines, root=0)
            self.dzmnsds_splines = self.mpi.comm_world.bcast(self.dzmnsds_splines, root=0)
            self.bmnc_splines = self.mpi.comm_world.bcast(self.bmnc_splines, root=0)
            self.dbmncds_splines = self.mpi.comm_world.bcast(self.dbmncds_splines, root=0)
            self.d_mn_factor_splines = self.mpi.comm_world.bcast(self.d_mn_factor_splines, root=0)
            self.mn_factor_splines = self.mpi.comm_world.bcast(self.mn_factor_splines, root=0)
            self.xm_b = self.mpi.comm_world.bcast(self.xm_b, root=0)
            self.xn_b = self.mpi.comm_world.bcast(self.xn_b, root=0)
            self.s_half_ext = self.mpi.comm_world.bcast(self.s_half_ext, root=0)
            if self.asym:
                self.numnc_splines = self.mpi.comm_world.bcast(self.numnc_splines, root=0)
                self.rmns_splines = self.mpi.comm_world.bcast(self.rmns_splines, root=0)
                self.zmnc_splines = self.mpi.comm_world.bcast(self.zmnc_splines, root=0)
                self.dnumncds_splines = self.mpi.comm_world.bcast(self.dnumncds_splines, root=0)
                self.drmnsds_splines = self.mpi.comm_world.bcast(self.drmnsds_splines, root=0)
                self.dzmncds_splines = self.mpi.comm_world.bcast(self.dzmncds_splines, root=0)
                self.bmns_splines = self.mpi.comm_world.bcast(self.bmns_splines, root=0)
                self.dbmnsds_splines = self.mpi.comm_world.bcast(self.dbmnsds_splines, root=0)

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
            if (self.enforce_qs and (self.xn_b[im] != self.N * self.xm_b[im])):
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
                if (self.enforce_qs and (self.xn_b[im] != self.N * self.xm_b[im])):
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

        if (self.mpi is not None):
            size = self.mpi.comm_world.size
            rank = self.mpi.comm_world.rank

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
        if self.mpi is not None:
            if (self.asym):
                self.mpi.comm_world.Allreduce([kmnc, MPI.DOUBLE], kmnc_buffer, op=MPI.SUM)
                kmnc = kmnc_buffer
            self.mpi.comm_world.Allreduce([kmns, MPI.DOUBLE], kmns_buffer, op=MPI.SUM)
            kmns = kmns_buffer
        if self.proc0:
            self.kmns_splines = []
            for im in range(len(self.xm_b)):
                if (self.enforce_qs and (self.xn_b[im] != self.N * self.xm_b[im])):
                    self.kmns_splines.append(InterpolatedUnivariateSpline(self.s_half_ext, 0*kmns[:, im], k=self.order))
                else:
                    self.kmns_splines.append(InterpolatedUnivariateSpline(self.s_half_ext, self.mn_factor_splines[im](self.s_half_ext)*kmns[:, im], k=self.order))

            if self.asym:
                self.kmnc_splines = []
                for im in range(len(self.xm_b)):
                    if (self.enforce_qs and (self.xn_b[im] != self.N * self.xm_b[im])):
                        self.kmnc_splines.append(InterpolatedUnivariateSpline(self.s_half_ext, 0*kmnc[:, im], k=self.order))
                    else:
                        self.kmnc_splines.append(InterpolatedUnivariateSpline(self.s_half_ext, self.mn_factor_splines[im](self.s_half_ext)*kmnc[:, im], k=self.order))

        if self.mpi is not None:
            self.kmns_splines = self.mpi.comm_world.bcast(self.kmns_splines, root=0)
            if self.asym:
                self.kmnc_splines = self.mpi.comm_world.bcast(self.kmnc_splines, root=0)

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
        if (self.mpi is not None):
            size = self.mpi.comm_world.size
            rank = self.mpi.comm_world.rank

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

        if (self.mpi is not None):
            # In place reduce is slightly slower
            # self.mpi.comm_world.Allreduce(MPI.IN_PLACE, [padded_buffer[:len(inv)], MPI.DOUBLE], op=MPI.SUM)
            self.mpi.comm_world.Allreduce([padded_buffer[:len(inv)], MPI.DOUBLE], recv_buffer, op=MPI.SUM)
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
            field: the underlying :class:`simsopt.field.boozermagneticfield.BoozerMagneticField` to be interpolated.
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
