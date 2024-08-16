import logging

import numpy as np

from .._core.optimizable import DOFs, Optimizable
from simsopt.geo import SurfaceRZFourier

logger = logging.getLogger(__name__)

try:
    import py_spec
except ImportError as e:
    py_spec = None
    logger.debug(str(e))

__all__ = ['NormalField']


class NormalField(Optimizable):
    r"""
    ``NormalField`` represents the magnetic field normal to a toroidal surface, for example the
    computational boundary of SPEC free-boundary.

    It consists a surface (the computational boundary), and a set of Fourier harmonics that describe
    the normal component of an externally provided field. 

    The Fourier harmonics are the degees of freedom, the computational boundary is kept fixed. 
    The Fourier harmonics are stored in the SPEC convention, 
    i.e. it is the 
    fourier components of B.(\partial\vec{r}/ \partial\theta \times \partial\vec{r}/ \partial\zeta) on the surface with a 1/(2\pi)^2
    Fourier normalization factor.

    Args:
        nfp: The number of field period
        stellsym: Whether (=True) or not (=False) stellarator symmetry is enforced.
        mpol: Poloidal Fourier resolution
        ntor: Toroidal Fourier resolution
        vns: Odd fourier modes of :math:`\mathbf{B}\cdot\mathbf{\hat{n}}`. 2D array of size
          (mpol+1)x(2ntor+1). Set to None to fill with zeros

            vns( mm, self.ntor+nn ) is the mode (mm,nn)

        vnc: Even fourier modes of :math:`\mathbf{B}\cdot\mathbf{\hat{n}}`. 2D array of size
          (mpol+1)x(2ntor+1). Ignored if stellsym if True. Set to None to fill with zeros

            vnc( mm, self.ntor+nn ) is the mode (mm,nn)
    """

    def __init__(self, nfp=1, stellsym=True, mpol=1, ntor=0,
                 vns=None, vnc=None, surface=None):

        self.nfp = nfp
        self.stellsym = stellsym
        self.mpol = mpol
        self.ntor = ntor

        if vns is None:
            vns = np.zeros((self.mpol + 1, 2 * self.ntor + 1))

        if not self.stellsym and vnc is None:
            vnc = np.zeros((self.mpol + 1, 2 * self.ntor + 1))
        
        if surface is None:
            surface = SurfaceRZFourier(nfp=nfp, stellsym=stellsym, mpol=mpol, ntor=ntor)
            surface.fix_all()
        self.surface = surface

        if self.stellsym:
            self.ndof = self.ntor + self.mpol * (2 * self.ntor + 1)
        else:
            self.ndof = 2 * (self.ntor + self.mpol * (2 * self.ntor + 1)) + 1
        
        self._vns = vns
        self._vnc = vnc
        
        dofs = self.get_dofs()

        Optimizable.__init__(
            self,
            x0=dofs,
            names=self._make_names())
    
    @property
    def vns(self):
        return self._vns
    
    @vns.setter
    def vns(self, value):
        raise AttributeError('Change Vns using set_vns() or set_vns_asarray()')
    
    @property
    def vnc(self):
        return self._vnc
    
    @vnc.setter
    def vnc(self, value):
        raise AttributeError('Change Vnc using set_vnc() or set_vnc_asarray()')

    @classmethod
    def from_spec(cls, filename):
        """
        Initialize using the harmonics in SPEC input file
        """

        # Test if py_spec is available
        if py_spec is None:
            raise RuntimeError(
                "Initialization from Spec requires py_spec to be installed.")

        # Read Namelist
        nm = py_spec.SPECNamelist(filename)
        ph = nm['physicslist']

        # Read modes from SPEC input file
        vns = np.asarray(ph['vns'])
        if ph['istellsym']:
            vnc = None
        else:
            vnc = np.asarray(ph['vnc'])
        mpol = ph['Mpol']
        ntor = ph['Ntor']
        surface = SurfaceRZFourier(nfp=ph['nfp'], stellsym=bool(ph['istellsym']), mpol=mpol, ntor=ntor)
        old_ntor = np.array(ph['rbc']).shape[1]//2
        surface.rc[:] = np.array(ph['rbc'])[:mpol+1, old_ntor-ntor:old_ntor+ntor+1]
        surface.zs[:] = np.array(ph['zbs'])[:mpol+1, old_ntor-ntor:old_ntor+ntor+1]
        if not ph['istellsym']:
            surface.zc[:] = np.array(ph['zbc'])[:mpol+1, old_ntor-ntor:old_ntor+ntor+1]
            surface.rs[:] = np.array(ph['rbs'])[:mpol+1, old_ntor-ntor:old_ntor+ntor+1]

        normal_field = cls(
            nfp=ph['nfp'],
            stellsym=bool(ph['istellsym']),
            mpol=ph['Mpol'],
            ntor=ph['Ntor'],
            vns=vns,
            vnc=vnc,
            surface=surface
        )

        return normal_field

    @classmethod
    def from_spec_object(cls, spec):
        """
        Initialize using the simsopt SPEC object's attributes
        """
        if not spec.freebound:
            raise ValueError('The given SPEC object is not free-boundary')
        
        surface = spec.computational_boundary
        
        # Grab all the attributes from the SPEC object into an input dictionary
        input_dict = {'nfp': spec.nfp,
                      'stellsym': spec.stellsym,
                      'mpol': spec.mpol,
                      'ntor': spec.ntor,
                      'vns': spec.array_translator(spec.inputlist.vns).as_simsopt,
                      'surface': surface}
        if not spec.stellsym:
            input_dict.append({'vnc': spec.array_translator(spec.inputlist.vnc).as_simsopt})

        normal_field = cls(**input_dict)

        return normal_field
    
    def get_dofs(self):
        """
        get DOFs from vns and vnc
        """
        # Pack in a single array
        dofs = np.zeros((self.ndof,))

        # Populate dofs array
        vns_shape = self.vns.shape
        input_mpol = int(vns_shape[0]-1)
        input_ntor = int((vns_shape[1]-1)/2)
        for mm in range(0, self.mpol+1):
            for nn in range(-self.ntor, self.ntor+1):
                if mm == 0 and nn < 0: continue
                if mm > input_mpol: continue
                if nn > input_ntor: continue

                if not (mm == 0 and nn == 0):
                    ii = self.get_index_in_dofs(mm, nn, even=False)
                    dofs[ii] = self.vns[mm, input_ntor+nn]

                if not self.stellsym:
                    ii = self.get_index_in_dofs(mm, nn, even=True)
                    dofs[ii] = self.vnc[mm, input_ntor+nn]
        return dofs

    def get_index_in_array(self, m, n, mpol=None, ntor=None, even=False):
        """
        Returns position of mode (m,n) in vns or vnc array

        Args:
        - m: poloidal mode number
        - n: toroidal mode number (normalized by Nfp)
        - mpol: resolution of dofs array. If None (by default), use self.mpol
        - ntor: resolution of dofs array. If None (by default), use self.ntor
        - even: set to True to get vnc. Default is False
        """

        if mpol is None:
            mpol = self.mpol
        if ntor is None:
            ntor = self.ntor

        if m < 0 or m > mpol:
            raise ValueError('m out of bound')
        if abs(n) > ntor:
            raise ValueError('n out of bound')
        if m == 0 and n < 0:
            raise ValueError('n has to be positive if m==0')
        if not even and m == 0 and n == 0:
            raise ValueError('m=n=0 not supported for odd series')
        
        return [m, n+ntor]


    def get_index_in_dofs(self, m, n, mpol=None, ntor=None, even=False):
        """
        Returns position of mode (m,n) in dofs array

        Args:
        - m: poloidal mode number
        - n: toroidal mode number (normalized by Nfp)
        - mpol: resolution of dofs array. If None (by default), use self.mpol
        - ntor: resolution of dofs array. If None (by default), use self.ntor
        - even: set to True to get vnc. Default is False
        """

        if mpol is None:
            mpol = self.mpol
        if ntor is None:
            ntor = self.ntor

        if m < 0 or m > mpol:
            raise ValueError('m out of bound')
        if abs(n) > ntor:
            raise ValueError('n out of bound')
        if m == 0 and n < 0:
            raise ValueError('n has to be positive if m==0')
        if not even and m == 0 and n == 0:
            raise ValueError('m=n=0 not supported for odd series')

        ii = -1
        if m == 0:
            ii = n
        else:
            ii = m * (2*ntor+1) + n

        nvns = ntor + mpol * (ntor * 2 + 1)
        if not even:  # Vns
            ii = ii - 1  # remove (0,0) element
        else:  # Vnc
            ii = ii + nvns

        return ii

    def get_vns(self, m, n):
        self.check_mn(m, n)
        ii = self.get_index_in_dofs(m, n)
        return self.local_full_x[ii]

    def set_vns(self, m, n, value):
        self.check_mn(m, n)
        i,j = self.get_index_in_array(m, n)
        self._vns[i,j] = value
        dofs = self.get_dofs()
        self.local_full_x = dofs

    def get_vnc(self, m, n):
        self.check_mn(m, n)
        ii = self.get_index_in_dofs(m, n, even=True)
        if self.stellsym:
            return 0.0
        else:
            return self.local_full_x[ii]

    def set_vnc(self, m, n, value):
        self.check_mn(m, n)
        i,j = self.get_index_in_array(m, n)
        if self.stellsym:
            raise ValueError('Stellarator symmetric has no vnc')
        else:
            self._vnc[i,j] = value
            dofs = self.get_dofs()
            self.local_full_x = dofs

    def check_mn(self, m, n):
        if m < 0 or m > self.mpol:
            raise ValueError('m out of bound')
        if n < -self.ntor or n > self.ntor:
            raise ValueError('n out of bound')
        if m == 0 and n < 0:
            raise ValueError('n has to be positive if m==0')

    def _make_names(self):
        """
        Form a list of names of the ``vns``, ``vnc``
        """
        if self.stellsym:
            names = self._make_names_helper(even=False)
        else:
            names = np.append(self._make_names_helper(even=False),
                              self._make_names_helper(even=True))

        return names

    def _make_names_helper(self, even=False):
        names = []
        indices = []

        if even:
            prefix = 'vnc'
        else:
            prefix = 'vns'

        for mm in range(0, self.mpol+1):
            for nn in range(-self.ntor, self.ntor+1):
                if mm == 0 and nn < 0:
                    continue
                if not even and mm == 0 and nn == 0:
                    continue

                ind = self.get_index_in_dofs(mm, nn, even=even)
                names.append(prefix + '({m},{n})'.format(m=mm, n=nn))
                indices.append(ind)

        # Sort names
        ind = np.argsort(np.asarray(indices))
        sorted_names = [names[ii] for ii in ind]

        return sorted_names

    def change_resolution(self, mpol, ntor):
        """
        Change the values of `mpol` and `ntor`. Any new Fourier amplitudes
        will have a magnitude of zero.  Any previous nonzero Fourier
        amplitudes that are not within the new range will be
        discarded.
        """

        # Set new number of dofs
        if self.stellsym:
            ndof = ntor + mpol * (2 * ntor + 1)  # Only Vns - odd series
        else:
            ndof = 2 * (ntor + mpol * (2 * ntor + 1)) + 1  # Vns and Vns

        # Fill relevant modes
        min_mpol = np.min((mpol, self.mpol))
        min_ntor = np.min((ntor, self.ntor))

        dofs = np.zeros((ndof,))
        for m in range(min_mpol + 1):
            for n in range(-min_ntor, min_ntor + 1):
                if m == 0 and n < 0: continue

                if m > 0 or n > 0:
                    ind = self.get_index_in_dofs(m, n, mpol=mpol, ntor=ntor, even=False)
                    dofs[ind] = self.get_vns(m, n)

                if not self.stellsym:
                    ind = self.get_index_in_dofs(m, n, mpol=mpol, ntor=ntor, even=True)
                    dofs[ind] = self.get_vnc(m, n)

        # Update attributes
        self.mpol = mpol
        self.ntor = ntor
        self.ndof = ndof
        self._dofs = DOFs(dofs, self._make_names())

    def fixed_range(self, mmin, mmax, nmin, nmax, fixed=True):
        """
        Set the 'fixed' property for a range of `m` and `n` values.

        All modes with `m` in the interval [`mmin`, `mmax`] and `n` in the
        interval [`nmin`, `nmax`] will have their fixed property set to
        the value of the `fixed` parameter. Note that `mmax` and `nmax`
        are included (unlike the upper bound in python's range(min,
        max).)

        In case of non stellarator symmetric field, both vns and vnc will be
        fixed (or unfixed)
        """

        fn = self.fix if fixed else self.unfix
        for m in range(mmin, mmax + 1):
            this_nmin = nmin
            if m == 0 and nmin < 0:
                this_nmin = 0
            for n in range(this_nmin, nmax + 1):
                if m > 0 or n != 0:
                    fn(f'vns({m},{n})')
                if not self.stellsym:
                    fn(f'vnc({m},{n})')

    def get_vns_asarray(self, mpol=None, ntor=None):
        """
        Return the vns as a single array
        """
        if mpol is None:
            mpol = self.mpol
        elif mpol > self.mpol:
            raise ValueError('mpol out of bound')

        if ntor is None: 
            ntor = self.ntor
        elif ntor > self.ntor:
            raise ValueError('ntor out of bound')

        vns = self.vns

        return vns[0:mpol, self.ntor-ntor:self.ntor+ntor+1]
    
    def get_vnc_asarray(self, mpol=None, ntor=None):
        """
        Return the vnc as a single array
        """
        if mpol is None:
            mpol = self.mpol
        elif mpol > self.mpol:
            raise ValueError('mpol out of bound')

        if ntor is None: 
            ntor = self.ntor
        elif ntor > self.ntor:
            raise ValueError('ntor out of bound')

        vnc = self.vns

        return vnc[0:mpol, self.ntor-ntor:self.ntor+ntor+1]
    
    def get_vns_vnc_asarray(self, mpol=None, ntor=None):
        """
        Return the vns and vnc as two arrays single array
        """
        if mpol is None:
            mpol = self.mpol
        elif mpol > self.mpol:
            raise ValueError('mpol out of bound')

        if ntor is None: 
            ntor = self.ntor
        elif ntor > self.ntor:
            raise ValueError('ntor out of bound')
        
        vns = self.get_vns_asarray(mpol, ntor)
        vnc = self.get_vnc_asarray(mpol, ntor)
        return vns, vnc
    
    def set_vns_asarray(self, vns, mpol=None, ntor=None):
        """
        Set the vns from a single array
        """
        if mpol is None:
            mpol = self.mpol
        elif mpol > self.mpol:
            raise ValueError('mpol out of bound')

        if ntor is None: 
            ntor = self.ntor
        elif ntor > self.ntor:
            raise ValueError('ntor out of bound')
        
        self._vns = vns[0:mpol, self.ntor-ntor:self.ntor+ntor+1]
        dofs = self.get_dofs()
        self.local_full_x = dofs

    def set_vnc_asarray(self, vnc, mpol=None, ntor=None):
        """
        Set the vnc from a single array
        """
        if mpol is None:
            mpol = self.mpol
        elif mpol > self.mpol:
            raise ValueError('mpol out of bound')

        if ntor is None: 
            ntor = self.ntor
        elif ntor > self.ntor:
            raise ValueError('ntor out of bound')
        
        self._vnc = vnc[0:mpol, self.ntor-ntor:self.ntor+ntor+1]
        dofs = self.get_dofs()
        self.local_full_x = dofs

    def set_vns_vnc_asarray(self, vns, vnc, mpol=None, ntor=None):
        """
        Set the vns and vnc from two single arrays
        """
        if mpol is None:
            mpol = self.mpol
        elif mpol > self.mpol:
            raise ValueError('mpol out of bound')

        if ntor is None: 
            ntor = self.ntor
        elif ntor > self.ntor:
            raise ValueError('ntor out of bound')

        self.set_vns_asarray(vns, mpol, ntor)
        self.set_vnc_asarray(vnc, mpol, ntor)
 
    def get_real_space_field(self):
        """
        Fourier transform the field and get the real-space values of the normal component of the externally
        provided field on the computational boundary. 
        The returned array will be of size specified by the surfaces'  quadpoints and located on the quadpoints. 
        """
        vns, vnc = self.get_vns_vnc_asarray(mpol=self.mpol, ntor=self.ntor)
        BdotN_unnormalized = self.surface.inverse_fourier_transform_scalar(vns, vnc, normalization=(2*np.pi)**2, stellsym=self.stellsym)
        normal_field_real_space = -1 * BdotN_unnormalized / np.linalg.norm(self.surface.normal(), axis=-1)
        return normal_field_real_space

