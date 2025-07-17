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

    """This class reads and writes mgrid (NetCDF) files for use in free boundary VMEC and other codes.

    An mgrid file contains a grid in cylindrical coordinates ``(R, phi, z)`` upon which the
    magnetic field is evaluated. The grid is defined by the number of
    points in the radius (``nr``), the number of planes per field period 
    in the toroidal angle (``nphi``), and the number of points in the z-coordinate (``nz``). 
    A good rule here is to choose at least 4 times as many toroidal planes as the maximum toroidal 
    mode number (ex. if ``ntor=6`` then you should have at least 24 toroidal planes). 
    
    The grid boundary is defined by the minimum and maximum values of the radial coordinate
    (``rmin``, ``rmax``), the minimum and maximum values of the z-coordinate (``zmin``, ``zmax``).
    Note the ``R`` and ``z`` grids include both end points while the ``phi`` dimension includes 
    the start point and excludes the end point.
    For free boundary calculations, the grid should be large enough to contain the entire plasma. 
    For example, ``rmin`` should be smaller than ``min R(theta, phi)`` where ``R`` is the radial 
    coordinate of the plasma. The same applies for the z-coordinate.
    
    The choice of the number or radial and vertical gridpoints is not as straightforward as choosing ``nphi``.
    The VMEC code uses these grids to "deform" the plasma boundary in a iterative sense.
    The complication revolves around the spectral condensation VMEC performs in the poloidal direction.
    The code calculates the location of the poloidal grid points so that an optimized choice is made for 
    the form of the plasma.
    The user should decide how accurately the wish to know the plasma boundary.
    If centimeter precision is required then the number of grid points chosen should provide at least this resolution.
    Remember that the more datapoints, the slower VMEC will run.
    
    Args:
        nr (int): Number of grid points in the radial direction. Default is 51.
        nz (int): Number of grid points in the z coordinate. Default is 51.
        nphi (int): Number of planes in the toroidal angle. Default is 24.
        nfp (int): Number of field periods. Default is 2.
        rmin (float): Minimum value of major radius coordinate of the grid. Default is 0.2.
        rmax (float): Maximum value of major radius coordinate of the grid. Default is 0.4.
        zmin (float): Minimum value of z-coordinate of the grid. Default is -0.1.
        zmax (float): Maximum value of z-coordinate of the grid. Default is 0.1.
    """

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

        self.ar_arr = []
        self.az_arr = []
        self.ap_arr = []

    def add_field_cylindrical(self, br, bp, bz, ar=None, ap=None, az=None, name=None):
        '''
        This function saves the magnetic field :math:`B`, and (optionally) the vector potential :math:`A`, to the ``MGrid`` object.
        :math:`B` and :math:`A` are provided on a tensor product grid in cylindrical components :math:`(R, \phi, z)`.

        This function may be called once for each current group, to save groups of fields that can be scaled using the
        ``vmec.indata.extcur`` array. Current groups emmulate groups of coils that are connected to distinct power supplies,
        and allows for the current in each group can be controlled independently.
        For example, W7-X has 7 current groups, one for each base coil (5 nonplanar coils + 2 planar coils), which
        allows all planar coils to be turned on without changing the currents in the nonplanar coils.
        In free-boundary vmec, the currents in each current group are scaled using the "extcur" array in the input file.

        Args:
            br (ndarray): (nphi, nz, nr) array of the radial component of B-field. 
            bp (ndarray): (nphi, nz, nr) array of the azimuthal component of B-field.
            bz (ndarray): (nphi, nz, nr) array of the z-component of B-field.
            ar (ndarray, Optional): (nphi, nz, nr) array of the radial component of the vector potential A. Default is None.
            ap (ndarray, Optional): (nphi, nz, nr) array of the azimuthal component of the vector potential A. Default is None.
            az (ndarray, Optional): (nphi, nz, nr) array of the axial component of the vector potential A. Default is None.
            name (str, Optional): Name of the coil group. Default is None.
        '''

        # appending B field to an array for all coil groups.
        self.br_arr.append(br)
        self.bz_arr.append(bz)
        self.bp_arr.append(bp)

        # add potential
        if ar is not None:
            self.ar_arr.append(ar)
        if ap is not None:
            self.az_arr.append(az)
        if az is not None:
            self.ap_arr.append(ap)

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

        Args:
            filename (str): output file name.
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
            var_ir.data[()] = self.nr
            var_jz.data[()] = self.nz
            var_kp.data[()] = self.nphi
            var_nfp.data[()] = self.nfp
            var_nextcur.data[()] = self.n_ext_cur

            var_rmin.data[()] = self.rmin
            var_zmin.data[()] = self.zmin
            var_rmax.data[()] = self.rmax
            var_zmax.data[()] = self.zmax

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

                #If the potential value is not an empty cell, then include it
                if len(self.ar_arr) > 0:
                    var_ar_001 = ds.createVariable('ar'+tag, 'f8', ('phi', 'zee', 'rad'))
                    var_ar_001[:, :, :] = self.ar_arr[j]
                if len(self.ap_arr) > 0:
                    var_ap_001 = ds.createVariable('ap'+tag, 'f8', ('phi', 'zee', 'rad'))
                    var_ap_001[:, :, :] = self.ap_arr[j]
                if len(self.az_arr) > 0:
                    var_az_001 = ds.createVariable('az'+tag, 'f8', ('phi', 'zee', 'rad'))
                    var_az_001[:, :, :] = self.az_arr[j]

    @classmethod
    def from_file(cls, filename):
        '''
        This method reads ``MGrid`` data from file.

        Args:
            filename (str): mgrid netCDF input file name.

        Returns:
            MGrid: ``MGrid`` object with data from file.
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
            if len(f.variables['coil_group'].dimensions) == 2:
                mgrid.coil_names = [_unpack(coil_data[j]) for j in range(mgrid.n_ext_cur)]
            else:
                mgrid.coil_names = [_unpack(coil_data)]

            mgrid.mode = f.variables['mgrid_mode'][:][0].decode()
            mgrid.raw_coil_current = np.array(f.variables['raw_coil_cur'][:])

            br_arr = []
            bp_arr = []
            bz_arr = []

            ar_arr = []
            ap_arr = []
            az_arr = []

            ar = []
            ap = []
            az = []

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

                #check for potential in mgrid file
                if 'ar_'+idx in f.variables:
                    ar = f.variables['ar_'+idx][:]
                    if mgrid.mode == 'S':
                        ar_arr.append(ar * mgrid.raw_coil_current[j])
                    else:
                        ar_arr.append(ar)

                if 'ap_'+idx in f.variables:
                    ap = f.variables['ap_'+idx][:]
                    if mgrid.mode == 'S':
                        ap_arr.append(ap * mgrid.raw_coil_current[j])
                    else:
                        ap_arr.append(ap)

                if 'az_'+idx in f.variables:
                    az = f.variables['az_'+idx][:]
                    if mgrid.mode == 'S':
                        az_arr.append(az * mgrid.raw_coil_current[j])
                    else:
                        az_arr.append(az)

            mgrid.br_arr = np.array(br_arr)
            mgrid.bp_arr = np.array(bp_arr)
            mgrid.bz_arr = np.array(bz_arr)

            mgrid.ar_arr = np.array(ar_arr)
            mgrid.ap_arr = np.array(ap_arr)
            mgrid.az_arr = np.array(az_arr)

            # sum over coil groups
            if nextcur > 1:
                br = np.sum(br_arr, axis=0)
                bp = np.sum(bp_arr, axis=0)
                bz = np.sum(bz_arr, axis=0)

                if len(mgrid.ar_arr) > 0:
                    ar = np.sum(ar_arr, axis=0)
                if len(mgrid.ap_arr) > 0:
                    ap = np.sum(ap_arr, axis=0)
                if len(mgrid.az_arr) > 0:
                    az = np.sum(az_arr, axis=0)
            else:
                br = br_arr[0]
                bp = bp_arr[0]
                bz = bz_arr[0]
                if len(mgrid.ar_arr) > 0:
                    ar = ar_arr[0]
                if len(mgrid.ap_arr) > 0:
                    ap = ap_arr[0]
                if len(mgrid.az_arr) > 0:
                    az = az_arr[0]

            mgrid.br = br
            mgrid.bp = bp
            mgrid.bz = bz

            mgrid.ar = ar
            mgrid.ap = ap
            mgrid.az = az

            mgrid.bvec = np.transpose([br, bp, bz])

        return mgrid

    def plot(self, jphi=0, bscale=0, show=True):
        '''
        Creates a plot of the mgrid data.
        Shows the three components ``(br,bphi,bz)`` in a 2D plot for a fixed toroidal plane.

        Args: 
            jphi (int): integer index for a toroidal slice. Default is 0.
            bscale (float): sets saturation scale for colorbar. (This is useful, because
                the mgrid domain often includes coil currents, and arbitrarily
                close to the current source the Bfield values become singular.). Default is 0.
            show (bool): If True, will call matplotlib's ``show()`` function. Default is True.

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

        if (show):
            plt.show()

        return fig, axs
