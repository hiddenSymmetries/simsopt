import numpy as np
from scipy.io import netcdf_file

__all__ = ["MGrid"]


def _pad_string(string): 
    '''
    Pads a string with 30 underscores (for writing coil group names).
    '''
    return '{:^30}'.format(string).replace(' ', '_')


def _unpack(binary_array):
    '''
    Decrypt binary char array into a string.
    This function is used for reading coil group names.
    '''
    return "".join(np.char.decode(binary_array)).strip()


class MGrid():

    '''
    This class reads and writes mgrid files for use in free boundary VMEC and other codes.

    The mgrid representation consists of the cylindrical components of B on a
    tensor product grid in cylindrical coordinates. Mgrid files are saved in
    NetCDF format.

    Args:
        nr: number of radial points
        nz: number of axial points
        nphi: number of azimuthal points, in one field period
        nfp: number of field periods
        rmin: minimum r grid point
        rmax: maximum r grid point
        zmin: minimum z grid point
        zmax: maximum z grid point

    Note the (r,z) dimensions include both end points. The (phi) dimension includes the start point and excludes the end point.
    '''

    def __init__(self,  # fname='temp', #binary=False,
                 nr: int = 51, 
                 nz: int = 51, 
                 nphi: int = 24, 
                 nfp: int = 2,
                 rmin: float = 0.20, 
                 rmax: float = 0.40, 
                 zmin: float = -0.10, 
                 zmax: float = 0.10,
                 ):

        self.nr = nr
        self.nz = nz
        self.nphi = nphi
        self.nfp = nfp

        self.rmin = rmin
        self.rmax = rmax
        self.zmin = zmin
        self.zmax = zmax

        self.n_ext_cur = 0
        self.coil_names = []

        self.br_arr = [] 
        self.bz_arr = [] 
        self.bp_arr = [] 

    def add_field_cylindrical(self, br, bp, bz, name=None):
        '''
        This function saves the vector field B.
        B is defined by cylindrical components.

        The Mgrid array assumes B is sampled linearly first in r, then z, and last phi.
        Python arrays use the opposite convention such that B[0] gives a (r,z) square at const phi
        and B[0,0] gives a radial line and const phi and z.

        It is assumed that the (br,bp,bz) inputs for this function is already in a
        (nphi, nz, nr) shaped array.

        This function may be called once for each coil group, 
        to save sets of fields that can be scaled using EXTCUR in VMEC.

        Args:
            br: the radial component of B field
            bp: the azimuthal component of B field
            bz: the axial component of B field
            name: Name of the coil group
        '''

        # appending B field to an array for all coil groups.
        self.br_arr.append(br)
        self.bz_arr.append(bz)
        self.bp_arr.append(bp)

        # add coil label
        if (name is None):
            label = _pad_string('magnet_%i' % self.n_ext_cur)
        else:
            label = _pad_string(name)
        self.coil_names.append(label)
        self.n_ext_cur = self.n_ext_cur + 1

        # TO-DO: this function could check for size consistency, between different fields, and for the (nr,nphi,nz) settings of a given instance

    def write(self, filename):
        '''
        Export class data as a netCDF binary.

        The field data is represented as a single "current group". For
        free-boundary vmec, the "extcur" array should have a single nonzero
        element, set to 1.0.

        Args:
            filename: output file name
        '''

        with netcdf_file(filename, 'w', mmap=False) as ds:

            # set netcdf dimensions
            ds.createDimension('stringsize', 30)
            ds.createDimension('dim_00001', 1)
            ds.createDimension('external_coil_groups', self.n_ext_cur)
            ds.createDimension('external_coils', self.n_ext_cur)
            ds.createDimension('rad', self.nr)
            ds.createDimension('zee', self.nz)
            ds.createDimension('phi', self.nphi)

            # declare netcdf variables
            var_ir = ds.createVariable('ir', 'i4', tuple())
            var_jz = ds.createVariable('jz', 'i4', tuple())
            var_kp = ds.createVariable('kp', 'i4', tuple())
            var_nfp = ds.createVariable('nfp', 'i4', tuple())
            var_nextcur = ds.createVariable('nextcur', 'i4', tuple())

            var_rmin = ds.createVariable('rmin', 'f8', tuple())
            var_zmin = ds.createVariable('zmin', 'f8', tuple())
            var_rmax = ds.createVariable('rmax', 'f8', tuple())
            var_zmax = ds.createVariable('zmax', 'f8', tuple())

            if self.n_ext_cur == 1:
                var_coil_group = ds.createVariable('coil_group', 'c', ('stringsize',))
                var_coil_group[:] = self.coil_names[0]
            else:
                var_coil_group = ds.createVariable('coil_group', 'c', ('external_coil_groups', 'stringsize',))
                var_coil_group[:] = self.coil_names
            var_mgrid_mode = ds.createVariable('mgrid_mode', 'c', ('dim_00001',))
            var_raw_coil_cur = ds.createVariable('raw_coil_cur', 'f8', ('external_coils',))

            # assign values
            var_ir.assignValue(self.nr)
            var_jz.assignValue(self.nz)
            var_kp.assignValue(self.nphi)
            var_nfp.assignValue(self.nfp)
            var_nextcur.assignValue(self.n_ext_cur)

            var_rmin.assignValue(self.rmin)
            var_zmin.assignValue(self.zmin)
            var_rmax.assignValue(self.rmax)
            var_zmax.assignValue(self.zmax)

            var_mgrid_mode[:] = 'N'  # R - Raw, S - scaled, N - none (old version)
            var_raw_coil_cur[:] = np.ones(self.n_ext_cur)

            # add fields
            for j in np.arange(self.n_ext_cur):

                tag = '_%.3i' % (j+1)
                var_br_001 = ds.createVariable('br'+tag, 'f8', ('phi', 'zee', 'rad'))
                var_bp_001 = ds.createVariable('bp'+tag, 'f8', ('phi', 'zee', 'rad'))
                var_bz_001 = ds.createVariable('bz'+tag, 'f8', ('phi', 'zee', 'rad'))

                var_br_001[:, :, :] = self.br_arr[j]
                var_bz_001[:, :, :] = self.bz_arr[j]
                var_bp_001[:, :, :] = self.bp_arr[j]

    @classmethod 
    def from_file(cls, filename):
        '''
        This method reads MGrid data from file.

        Args:
            filename: mgrid netCDF input file name
        '''

        with netcdf_file(filename, 'r', mmap=False) as f:

            # load grid
            nr = f.variables['ir'].getValue()
            nphi = f.variables['kp'].getValue()
            nz = f.variables['jz'].getValue()
            rmin = f.variables['rmin'].getValue()
            rmax = f.variables['rmax'].getValue()
            zmin = f.variables['zmin'].getValue()
            zmax = f.variables['zmax'].getValue()
            kwargs = {"nr": nr, "nphi": nphi, "nz": nz, 
                      "rmin": rmin, "rmax": rmax, "zmin": zmin, "zmax": zmax}

            mgrid = cls(**kwargs)
            mgrid.filename = filename
            mgrid.n_ext_cur = int(f.variables['nextcur'].getValue())
            coil_data = f.variables['coil_group'][:]
            mgrid.coil_names = [_unpack(coil_data[j]) for j in range(mgrid.n_ext_cur)] 

            mgrid.mode = f.variables['mgrid_mode'][:][0].decode()
            mgrid.raw_coil_current = np.array(f.variables['raw_coil_cur'][:])

            br_arr = []
            bp_arr = []
            bz_arr = []

            nextcur = mgrid.n_ext_cur
            for j in range(nextcur):
                idx = '{:03d}'.format(j+1)
                br = f.variables['br_'+idx][:]  # phi z r
                bp = f.variables['bp_'+idx][:]
                bz = f.variables['bz_'+idx][:]

                if (mgrid.mode == 'S'):
                    br_arr.append(br * mgrid.raw_coil_current[j])
                    bp_arr.append(bp * mgrid.raw_coil_current[j])
                    bz_arr.append(bz * mgrid.raw_coil_current[j])
                else:
                    br_arr.append(br)
                    bp_arr.append(bp)
                    bz_arr.append(bz)

            mgrid.br_arr = np.array(br_arr)
            mgrid.bp_arr = np.array(bp_arr)
            mgrid.bz_arr = np.array(bz_arr)

            # sum over coil groups
            if nextcur > 1:
                br = np.sum(br_arr, axis=0)
                bp = np.sum(bp_arr, axis=0)
                bz = np.sum(bz_arr, axis=0)
            else:
                br = br_arr[0]
                bp = bp_arr[0]
                bz = bz_arr[0]

            mgrid.br = br
            mgrid.bp = bp
            mgrid.bz = bz

            mgrid.bvec = np.transpose([br, bp, bz])

        return mgrid

    def plot(self, jphi=0, bscale=0, show=True):
        '''
        Creates a plot of the mgrid data.
        Shows the three components (br,bphi,bz) in a 2D plot for a fixed toroidal plane.

        Args: 
            jphi: integer index for a toroidal slice.
            bscale: sets saturation scale for colorbar. (This is useful, because
                the mgrid domain often includes coil currents, and arbitrarily
                close to the current source the Bfield values become singular.)
            show: Whether to call matplotlib's ``show()`` function.

        Returns:
            2-element tuple ``(fig, axs)`` with the Matplotlib figure and axes handles.
        '''

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 1, figsize=(5, 10))

        rax = np.linspace(self.rmin, self.rmax, self.nr)
        zax = np.linspace(self.zmin, self.zmax, self.nz)

        def subplot_slice(s, data, rax, zax, bscale=0.3, tag=''):

            if bscale > 0:
                cf = axs[s].contourf(rax, zax, data, np.linspace(-bscale, bscale, 11), extend='both')
            else:
                cf = axs[s].contourf(rax, zax, data)
            axs[s].plot([], [], '.', label=tag)
            axs[s].legend()
            fig.colorbar(cf, ax=axs[s])

        subplot_slice(np.s_[0], self.br[jphi], rax, zax, tag='br', bscale=bscale)
        subplot_slice(np.s_[1], self.bp[jphi], rax, zax, tag='bp', bscale=bscale)
        subplot_slice(np.s_[2], self.bz[jphi], rax, zax, tag='bz', bscale=bscale)

        axs[1].set_title(f"nextcur = {self.n_ext_cur}, mode {self.mode}", fontsize=10)
        axs[2].set_title(f"nr,np,nz = ({self.nr},{self.nphi},{self.nz})", fontsize=10)

        if (show): plt.show()

        return fig, axs
