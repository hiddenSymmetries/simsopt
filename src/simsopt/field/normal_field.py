import logging

import numpy as np

from .._core.optimizable import DOFs, Optimizable

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
                 vns=None, vnc=None):

        self.nfp = nfp
        self.stellsym = stellsym
        self.mpol = mpol
        self.ntor = ntor

        if vns is None:
            vns = np.zeros((self.mpol + 1, 2 * self.ntor + 1))

        if vnc is None and not stellsym:
            vnc = np.zeros((self.mpol + 1, 2 * self.ntor + 1))

        if self.stellsym:
            self.ndof = self.ntor + self.mpol * (2 * self.ntor + 1)
        else:
            self.ndof = 2 * (self.ntor + self.mpol * (2 * self.ntor + 1)) + 1

        # Pack in a single array
        dofs = np.zeros((self.ndof,))

        # Populate dofs array
        vns_shape = vns.shape
        input_mpol = int(vns_shape[0]-1)
        input_ntor = int((vns_shape[1]-1)/2)
        for mm in range(0, self.mpol+1):
            for nn in range(-self.ntor, self.ntor+1):
                if mm == 0 and nn < 0: continue
                if mm > input_mpol: continue
                if nn > input_ntor: continue

                if not (mm == 0 and nn == 0):
                    ii = self.get_index_in_dofs(mm, nn, even=False)
                    dofs[ii] = vns[mm, input_ntor+nn]

                if not self.stellsym:
                    ii = self.get_index_in_dofs(mm, nn, even=True)
                    dofs[ii] = vnc[mm, input_ntor+nn]

        Optimizable.__init__(
            self,
            x0=dofs,
            names=self._make_names())

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
            vnc = np.asarray(ph['vnc'][1:])

        nf = cls(
            nfp=ph['nfp'], 
            stellsym=ph['istellsym'], 
            mpol=ph['Mpol'], 
            ntor=ph['Ntor'],
            vns=vns,
            vnc=vnc
        )

        return nf

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
        ii = self.get_index_in_dofs(m, n)
        self.local_full_x[ii] = value

    def get_vnc(self, m, n):
        self.check_mn(m, n)
        ii = self.get_index_in_dofs(m, n, even=True)
        if self.stellsym:
            return 0.0
        else:
            return self.local_full_x[ii]

    def set_vnc(self, m, n, value):
        self.check_mn(m, n)
        ii = self.get_index_in_dofs(m, n, even=True)
        if self.stellsym:
            raise ValueError('Stellarator symmetric has no vnc')
        else:
            self.local_full_x[ii] = value

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
            names = self._make_names_helper(False)
        else:
            names = np.append(self._make_names_helper(False),
                              self._make_names_helper(True))

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
