import numpy as np
from scipy.io import netcdf as nc
import sys


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
    This class writes Mgrid files for use in free boundary VMEC and other codes.

    The B vector field is given in cylindricial coordinates for each coil.
    It is written as a netCDF binary file.

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

        print(f"Initialized mgrid file: (nr,nphi,nz,nfp) = ({nr}, {nphi}, {nz}, {nfp})")

    def add_field_cylindrical(self, br, bp, bz, name='default'):
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
        '''

        # appending B field to an array for all coil groups.
        self.br_arr.append(br)
        self.bz_arr.append(bz)
        self.bp_arr.append(bp)

        # add coil label
        if (name == 'default'):
            label = _pad_string('magnet_%i' % self.n_ext_cur)
        else:
            label = _pad_string(name)
        self.coil_names.append(label)
        self.n_ext_cur = self.n_ext_cur + 1

        # TO-DO: this function could check for size consistency, between different fields, and for the (nr,nphi,nz) settings of a given instance

    def write(self, filename):
        '''
        Export class data as a netCDF binary.

        Args:
            filename: output file name
        '''

        print('Writing mgrid file')
        ds = nc.netcdf_file(filename, 'w')

        # set netcdf dimensions
        ds.createDimension('stringsize', 30)
        ds.createDimension('dim_00001', 1)
        ds.createDimension('external_coil_groups', self.n_ext_cur)
        ds.createDimension('external_coils', self.n_ext_cur)
        ds.createDimension('rad', self.nr)
        ds.createDimension('zee', self.nz)
        ds.createDimension('phi', self.nphi)

        # declare netcdf variables
        var_ir = ds.createVariable('ir', 'i4', ('dim_00001',))
        var_jz = ds.createVariable('jz', 'i4', ('dim_00001',))
        var_kp = ds.createVariable('kp', 'i4', ('dim_00001',))
        var_nfp = ds.createVariable('nfp', 'i4', ('dim_00001',))
        var_nextcur = ds.createVariable('nextcur', 'i4', ('dim_00001',))

        var_rmin = ds.createVariable('rmin', 'f8', ('dim_00001',))
        var_zmin = ds.createVariable('zmin', 'f8', ('dim_00001',))
        var_rmax = ds.createVariable('rmax', 'f8', ('dim_00001',))
        var_zmax = ds.createVariable('zmax', 'f8', ('dim_00001',))

        var_coil_group = ds.createVariable('coil_group', 'c', ('external_coil_groups', 'stringsize',))
        var_mgrid_mode = ds.createVariable('mgrid_mode', 'c', ('dim_00001',))
        var_raw_coil_cur = ds.createVariable('raw_coil_cur', 'f8', ('external_coils',))

        # assign values
        var_ir[:] = self.nr
        var_jz[:] = self.nz
        var_kp[:] = self.nphi
        var_nfp[:] = self.nfp
        var_nextcur[:] = self.n_ext_cur

        var_rmin[:] = self.rmin
        var_zmin[:] = self.zmin
        var_rmax[:] = self.rmax
        var_zmax[:] = self.zmax

        var_coil_group[:] = self.coil_names
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

        ds.close()

        print('  Wrote to file:', filename)

    def export_phi(self):
        phi = np.linspace(0, 2*np.pi/self.nfp, self.nphi)
        return phi

    def export_grid_spacing(self):
        return self.nr, self.nz, self.nphi

    @classmethod 
    def from_file(cls, filename):
        '''
        This method reads MGrid data from file.

        Args:
            filename: mgrid netCDF input file name
        '''

        f = nc.netcdf_file(filename, 'r') 

        # parse file name
        filename = "".join(filename.split('/')[-1].split('.')[1:-1])
        if (filename == ""):
            filename = "".join(filename.split('_')[1:])

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
        mgrid.load_field(f)
        mgrid.filename = filename

        return mgrid

    def load_field(self, f):

        self.n_ext_cur = int(f.variables['nextcur'].getValue())
        coil_data = f.variables['coil_group'][:]
        self.coil_names = [_unpack(coil_data[j]) for j in range(self.n_ext_cur)] 

        self.mode = f.variables['mgrid_mode'][:][0].decode()
        self.raw_coil_current = np.array(f.variables['raw_coil_cur'][:])

        br_arr = []
        bp_arr = []
        bz_arr = []

        nextcur = self.n_ext_cur
        for j in range(nextcur):
            idx = '{:03d}'.format(j+1)
            br = f.variables['br_'+idx][:]  # phi z r
            bp = f.variables['bp_'+idx][:]
            bz = f.variables['bz_'+idx][:]

            if (self.mode == 'S'):
                br_arr.append(br * self.raw_coil_current[j])
                bp_arr.append(bp * self.raw_coil_current[j])
                bz_arr.append(bz * self.raw_coil_current[j])
            else:
                br_arr.append(br)
                bp_arr.append(bp)
                bz_arr.append(bz)

        self.br_arr = np.array(br_arr)
        self.bp_arr = np.array(bp_arr)
        self.bz_arr = np.array(bz_arr)

        # sum over coil groups
        if nextcur > 1:
            br = np.sum(br_arr, axis=0)
            bp = np.sum(bp_arr, axis=0)
            bz = np.sum(bz_arr, axis=0)
        else:
            br = br_arr[0]
            bp = bp_arr[0]
            bz = bz_arr[0]

        self.br = br
        self.bp = bp
        self.bz = bz

        self.bvec = np.transpose([br, bp, bz])

    def plot(self, jphi=0, bscale=0, show=True):
        '''
        Creates a plot of the mgrid data for debugging.
        Shows the three components (br,bphi,bz) in a 2D plot for a fixed toroidal plane.

        Args: 
            jphi: integer index for a toroidal slice.
            bscale: sets saturation scale for colorbar. (This is useful, because the mgrid domain often includes coil currents, 
            and arbitrarily close to the current source the Bfield values become singular.) 
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

        axs[0].set_title(self.filename)
        axs[1].set_title(f"nextcur = {self.n_ext_cur}, mode {self.mode}", fontsize=10)
        axs[2].set_title(f"nr,np,nz = ({self.nr},{self.nphi},{self.nz})", fontsize=10)

        if (show): plt.show()


