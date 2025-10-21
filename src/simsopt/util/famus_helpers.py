"""
This file contains a class for reading in FOCUS-style data files. 

This file was copied over from the MAGPIE code used for generating 
permanent magnet structures. All credit for this code is thanks 
to the PM4Stell team and Ken Hammond for his consent to use this file
and work with the permanent magnet branch of SIMSOPT.
"""
__all__ = ['FocusData', 'FocusPlasmaBnormal', 'stell_point_transform', 'stell_vector_transform']
import numpy as np
from simsopt.geo import Surface

FOCUS_PLASMAFILE_NHEADER_TOP = 1
FOCUS_PLASMAFILE_NHEADER_BDRY = 2
FOCUS_PLASMAFILE_NHEADER_BNORM = 2


class FocusPlasmaBnormal(object):

    def __init__(self, fname):

        with open(fname, 'r') as f:
            lines = f.readlines()

        # Read numbers of modes and nfp from the header
        position = FOCUS_PLASMAFILE_NHEADER_TOP
        splitline = lines[position].split()
        errmsg = "This does not appear to be a FOCUS-format " \
                 + "plasma boundary file"
        assert len(splitline) == 3, errmsg

        Nfou = int(splitline[0])
        nfp = int(splitline[1])
        NfouBn = int(splitline[2])

        # Read Fourier harmonics of the normal field on the plasma boundary
        position = position + 1 + FOCUS_PLASMAFILE_NHEADER_BDRY + Nfou + \
            FOCUS_PLASMAFILE_NHEADER_BNORM

        self.n = np.zeros(NfouBn)
        self.m = np.zeros(NfouBn)
        self.bnc = np.zeros(NfouBn)
        self.bns = np.zeros(NfouBn)
        for i in range(NfouBn):
            splitline = lines[position + i].split()
            self.n[i] = int(splitline[0])
            self.m[i] = int(splitline[1])
            self.bnc[i] = float(splitline[2])
            self.bns[i] = float(splitline[3])
        assert np.min(self.m) == 0

        self.nfp = nfp
        self.NfouBn = NfouBn
        self.stellsym = np.max(np.abs(self.bnc)) == 0

    def bnormal_grid(self, nphi, ntheta, range):
        '''
        Calculates the normal component of the magnetic field on a
        regularly-spaced 2D grid of points on the surface with a specified
        resolution.

        Parameters
        ----------
          nphi: int
            Number of grid points in the toroidal dimension per field period.
          ntheta: int
            Number of grid points in the poloidal dimension.
          range: str
            Extent of the grid in the toroidal dimension. Choices include
            `half period`, `field period`, or `full torus`.

        Returns
        -------
          bnormal: 2D array
            Normal component of the magnetic field evaluated at each of the
            grid points.
        '''
        # Determine the theta and phi points using Simsopt Surface class methods
        phi = Surface.get_phi_quadpoints(nphi=nphi, range=range, nfp=self.nfp)
        theta = Surface.get_theta_quadpoints(ntheta=ntheta)
        # Rescale the angles
        phi = 2.*np.pi*np.array(phi)
        theta = 2.*np.pi*np.array(theta)

        # Prepare 3D arrays with mode numbers and coefficients
        Theta, Phi, Mode = np.meshgrid(theta, phi, np.arange(0, self.NfouBn))
        M = self.m[Mode]
        N = self.n[Mode]
        BNS = self.bns[Mode]
        if not self.stellsym:
            BNC = self.bnc[Mode]

        # Calculate the values of each mode at each grid point
        SinGrid = BNS*np.sin(M*Theta - self.nfp*N*Phi)
        if not self.stellsym:
            CosGrid = BNC*np.cos(M*Theta - self.nfp*N*Phi)

        # Add the contributions of each mode
        bnormal = np.sum(SinGrid, axis=2)
        if not self.stellsym:
            bnormal = bnormal + np.sum(CosGrid, axis=2)

        return bnormal


class FocusData(object):
    """
    Class object for reading in FOCUS-style data files.

    Args:
        filename: a FOCUS file
        keep_Ic_zeros: (optional) if True, all dipoles in the file will be 
            retained in the data structure irrespective of the value of their
            Ic parameter. If False, only magnets with Ic == 1 will be imported.
            Default is False.
        downsample: (optional) if set to integer value n, the initialization
            method will only load every nth magnet from the file. If 1, all
            magnets will be loaded. Default is 1.
    """
    propNames = ['type', 'symm', 'coilname', 'ox', 'oy', 'oz', 'Ic', 'M_0',
                 'pho', 'Lc', 'mp', 'mt', 'op']

    float_inds = [3, 4, 5, 7, 8, 10, 11]

    def __init__(self, filename, keep_Ic_zeros=False, downsample=1):

        self.nMagnets = 0
        self.nPol = 0

        # initialize to # of mandatory properties ('op' is currently optional)
        self.nProps = len(FocusData.propNames) - 1

        self.read_from_file(filename, keep_Ic_zeros, downsample)

    def read_from_file(self, filename, keep_Ic_zeros, downsample):

        with open(str(filename), 'r') as focusfile:
            # Ignore the first line in the file
            focusfile.readline()

            # Record the number of magnets and the momentq
            line2data = [int(number) for number in
                         focusfile.readline().strip().split()]
            self.nMagnets = line2data[0]
            if len(line2data) > 1:
                self.momentq = line2data[1]
                self.has_momentq = True
            else:
                self.has_momentq = False

            # Initialize the property data arrays
            self.magtype = np.zeros(self.nMagnets)
            self.symm = np.zeros(self.nMagnets)
            self.coilname = []
            self.ox = np.zeros(self.nMagnets)
            self.oy = np.zeros(self.nMagnets)
            self.oz = np.zeros(self.nMagnets)
            self.Ic = np.zeros(self.nMagnets)
            self.M_0 = np.zeros(self.nMagnets)
            self.pho = np.zeros(self.nMagnets)
            self.Lc = np.zeros(self.nMagnets)
            self.mp = np.zeros(self.nMagnets)
            self.mt = np.zeros(self.nMagnets)
            self.op = np.zeros(self.nMagnets)

            # Ignore the third line in the file
            focusfile.readline()

            # Read the data for each magnet from the file
            count = 0
            self.max_float_length = 0
            self.min_float_val = 0
            for i in range(self.nMagnets):

                linedata = focusfile.readline().strip().split(',')
                if len(linedata) < self.nProps:
                    raise Exception(('Problem accessing data for magnet %d in '
                                     + 'file ' + filename) % (i))

                self.magtype[i] = int(linedata[0])
                self.symm[i] = int(linedata[1])
                self.coilname.append(linedata[2].strip())
                self.ox[i] = np.double(linedata[3])
                self.oy[i] = np.double(linedata[4])
                self.oz[i] = np.double(linedata[5])
                self.Ic[i] = int(float(linedata[6].strip()))
                self.M_0[i] = np.double(linedata[7])
                self.pho[i] = np.double(linedata[8])
                self.Lc[i] = int(float(linedata[9].strip()))
                self.mp[i] = np.double(linedata[10])
                self.mt[i] = np.double(linedata[11])

                # Check for presence of op parameter
                if i == 0:
                    if len(linedata) > self.nProps:
                        try:
                            _ = np.double(linedata[12])
                            self.has_op = True
                            self.nProps = self.nProps + 1
                        except:
                            self.has_op = False
                    else:
                        self.has_op = False

                # Record op parameter if applicable
                if self.has_op:
                    self.op[i] = np.double(linedata[12])

                # Keep track of the longest and lowest-valued floats recorded
                max_float_length = max([len(linedata[i].strip()) for i
                                        in FocusData.float_inds])
                if max_float_length > self.max_float_length:
                    self.max_float_length = max_float_length
                min_float_val = min([np.double(linedata[i]) for i
                                     in FocusData.float_inds])
                if min_float_val < self.min_float_val:
                    self.min_float_val = min_float_val

                count += 1

        self.max_name_length = max([len(string) for string in self.coilname])

        # Add space for a negative sign in the max float length if necessary
        if self.min_float_val >= 0:
            self.max_float_length = self.max_float_length+1

        # Drop magnets from downsample and port locations
        inds_total = np.arange(self.nMagnets)
        inds_downsampled = inds_total[::downsample]

        # if desired, remove dipoles where Ic=0 (e.g. diagnostic port locations)
        if keep_Ic_zeros:
            nonzero_inds = inds_downsampled
        else:
            nonzero_inds = np.intersect1d(np.ravel(np.where(self.Ic == 1.0)), inds_downsampled)
        self.ox = self.ox[nonzero_inds]
        self.oy = self.oy[nonzero_inds]
        self.oz = self.oz[nonzero_inds]
        self.magtype = self.magtype[nonzero_inds]
        self.symm = self.symm[nonzero_inds]
        self.coilname = np.array(self.coilname)[nonzero_inds]
        self.coilname = self.coilname.tolist()
        self.M_0 = self.M_0[nonzero_inds]
        self.pho = self.pho[nonzero_inds]
        self.Lc = self.Lc[nonzero_inds]
        self.mp = self.mp[nonzero_inds]
        self.mt = self.mt[nonzero_inds]
        self.nMagnets = len(nonzero_inds)

    def unit_vector(self, inds):
        """
        Returns the x, y, and z components of a Cartesian unit vector in the 
        polarizaton direction of the nth magnet
        """
        if len(inds) > 0 and max(inds) > self.nMagnets-1:
            raise Exception('unit_vector: requested magnet %d does not exist'
                            % (max(inds)))

        return np.cos(self.mp[inds])*np.sin(self.mt[inds]), \
            np.sin(self.mp[inds])*np.sin(self.mt[inds]), \
            np.cos(self.mt[inds])

    def perp_vector(self, inds):
        """
        Returns the x, y, and z components of a Cartesian unit vector 
        perpendicular to the polarizaton direction of the nth magnet
        """
        if len(inds) > 0 and max(inds) > self.nMagnets-1:
            raise Exception('unit_vector: requested magnet %d does not exist'
                            % (max(inds)))

        mt_perp = self.mt[inds] + np.pi / 2.0
        return np.cos(self.mp[inds])*np.sin(mt_perp), \
            np.sin(self.mp[inds])*np.sin(mt_perp), \
            np.cos(mt_perp)

    def flip_negative_magnets(self):
        """
        For magnets with negative rho values, adjust the orientation angles
        (mt and mp) so that the rho value can be negated to a positive value
        """
        # Indices of magnets with negative rho values
        neg_inds = np.where(self.pho < 0)[0]

        if len(neg_inds) == 0:
            return

        # x, y, and z components of unit vectors for the negative magnets
        ux, uy, uz = self.unit_vector(neg_inds)

        # Calculate azimuthal and polar angles for negated unit vectors
        mp_neg = np.arctan2(-uy, -ux)
        mt_neg = np.arctan2(np.sqrt(ux**2 + uy**2), -uz)

        # Replace relevant values for the (initially) negative magnets
        self.pho[neg_inds] = -self.pho[neg_inds]
        self.mp[neg_inds] = mp_neg
        self.mt[neg_inds] = mt_neg

    def adjust_rho(self, q_new):
        """
        Adjust rho according to the desired momentq

        Args:
            q_new: New exponent to use for FAMUS representation
              of dipole magnitudes.
        """
        if min(self.pho) < 0:
            raise RuntimeError('adjust_rho: rho contains negative values')

        if self.has_momentq:
            self.pho = self.pho**(float(self.momentq)/float(q_new))
            self.momentq = q_new

        # If no momentq currently specified, assume to be one
        else:
            self.pho = self.pho**(1./float(q_new))
            self.momentq = q_new
            self.has_momentq = True

    def print_to_file(self, filename):
        """
        Write the FocusData class object information to a FOCUS
        style file.

        Args:
            filename: string denoting the name of the output file.
        """
        if self.nMagnets < 1:
            raise RuntimeError('print_to_file: no magnets to print')

        with open(str(filename), 'w') as focusfile:

            if self.has_momentq:
                focusfile.write('Total number of dipoles, momentq\n')
                focusfile.write('%10d %5g\n' % (self.nMagnets, self.momentq))
            else:
                focusfile.write('Total number of dipoles\n')
                focusfile.write('%10d\n' % (self.nMagnets))

            # String format specifiers: float length, decimal precision, name length
            lf = '%s' % (self.max_float_length)
            nd = '%s' % (self.max_float_length - 7)
            ln = '%s' % (self.max_name_length)

            # Write the header line for the individual magnet data
            focusfile.write(('%s, ' * 2 + '%' + ln + 's, ' + ('%' + lf + 's, ') * 3 +
                             '%s, ' + ('%' + lf + 's, ') * 2 + '%s, ' +
                             ('%' + lf + 's, ') * 2) %
                            tuple(FocusData.propNames[:12]))
            if self.has_op:
                focusfile.write(('%' + lf + 's, \n') % (FocusData.propNames[12]))
            else:
                focusfile.write('\n')

            # Write the data for each magnet to the file
            for i in range(self.nMagnets):

                lineStr = ('%4d, ' * 2 + '%' + ln + 's, ' +
                           ('%' + lf + '.' + nd + 'E, ') * 3 + '%2d, ' +
                           ('%' + lf + '.' + nd + 'E, ') * 2 + '%2d, ' +
                           ('%' + lf + '.' + nd + 'E, ') * 2) %  \
                    (self.magtype[i], self.symm[i], self.coilname[i],
                     self.ox[i], self.oy[i], self.oz[i], self.Ic[i],
                     self.M_0[i], self.pho[i], self.Lc[i],
                     self.mp[i], self.mt[i])

                if self.has_op:
                    lineStr = lineStr + ('%' + lf + '.' + nd + 'E, \n') % self.op[i]
                else:
                    lineStr = lineStr + '\n'

                focusfile.write(lineStr)

    def init_pol_vecs(self, n_pol):
        """
        Initializes arrays for the x, y, and z components of allowable
        polarization vectors, as an array giving the ID of the polarization
        vector actually used for the respective magnet (0 means magnet is off)

        Args:
            n_pol: int
                Number of allowed polarization vectors, typically 3 or 12.
        """
        self.nPol = n_pol
        self.pol_x = np.zeros((self.nMagnets, n_pol))
        self.pol_y = np.zeros((self.nMagnets, n_pol))
        self.pol_z = np.zeros((self.nMagnets, n_pol))
        self.pol_id = np.zeros(self.nMagnets, dtype=int)
        self.pol_type = np.zeros(self.nMagnets, dtype=int)
        self.pol_type_key = np.zeros(n_pol, dtype=int)
        self.cyl_r = np.zeros(self.nMagnets)
        self.cyl_p = np.zeros(self.nMagnets)
        self.cyl_z = np.zeros(self.nMagnets)

    def repeat_hp_to_fp(self, nfp, magnet_sector=1):
        '''
        Duplicates the magnets to the adjacent half-period, implementing the
        appropriate transformations to uphold stellarator symmetry.

        NOTES: 
            - At the present time, duplicate magnets will have the same pol_id
              as their corresponding originals
            - At the present time, duplicate magnets will have the same name
              their corresponding originals.

        Parameters
        ----------
            nfp: integer
                Number of field periods in the configuration
            magnet_sector: integer (optional)
                Sector (half-period) of the torus in which the magnets are 
                assumed to be located. Must be between 1 and 2*nfp, inclusive.
                Sector 1 starts at toroidal angle 0.
        '''
        symm_inds = np.where(self.symm == 2)[0]
        n_symm = len(symm_inds)
        if n_symm == 0:
            raise ValueError('repeat_hp_to_fp is only valid for magnets '
                             'that are stellarator symmetric (symm=2)')

        # Toroidal angle of the symmetry plane between the adjacent half-periods
        if magnet_sector > 2*nfp or magnet_sector < 1:
            raise ValueError('magnet_sector must be positive and less than '
                             '2*nfp')
        phi = magnet_sector * np.pi/nfp

        nx, ny, nz = self.unit_vector(symm_inds)

        # Reflect polarization vector origins and directions
        nx2, ny2, nz2 = stell_vector_transform('reflect', phi, nx, ny, nz)
        ox2, oy2, oz2 = \
            stell_point_transform('reflect', phi, self.ox, self.oy, self.oz)
        mp2 = np.arctan2(ny2, nx2)
        mt2 = np.arctan2(np.sqrt(nx2**2 + ny2**2), nz2)
        op2 = 2*phi - self.op[symm_inds]

        # Update the number of magnets
        self.nMagnets = self.nMagnets + n_symm

        # Update the symmetry coding for magnets that have been duplicated
        self.symm[symm_inds] = 1

        # Extend the dipole moment data to the new half-period
        self.magtype = np.concatenate((self.magtype, self.magtype[symm_inds]))
        self.symm = np.concatenate((self.symm, self.symm[symm_inds]))
        self.coilname = self.coilname + [self.coilname[i] for i in symm_inds]
        self.ox = np.concatenate((self.ox, ox2))
        self.oy = np.concatenate((self.oy, oy2))
        self.oz = np.concatenate((self.oz, oz2))
        self.Ic = np.concatenate((self.Ic, self.Ic[symm_inds]))
        self.M_0 = np.concatenate((self.M_0, self.M_0[symm_inds]))
        self.pho = np.concatenate((self.pho, self.pho[symm_inds]))
        self.Lc = np.concatenate((self.Lc, self.Lc[symm_inds]))
        self.mp = np.concatenate((self.mp, mp2))
        self.mt = np.concatenate((self.mt, mt2))
        self.op = np.concatenate((self.op, op2))

        # Update discrete polarization properties if they exist
        if self.nPol > 0:
            pol_x2, pol_y2, pol_z2 = \
                stell_vector_transform('reflect', phi,
                                       self.pol_x[symm_inds, :], self.pol_y[symm_inds, :],
                                       self.pol_z[symm_inds, :])
            pol_type2 = self.pol_type[symm_inds]
            pol_id2 = self.pol_id[symm_inds]
            cyl_r2 = -self.cyl_r[symm_inds]
            cyl_p2 = self.cyl_p[symm_inds]
            cyl_z2 = self.cyl_z[symm_inds]

            self.pol_x = np.concatenate((self.pol_x, pol_x2), axis=0)
            self.pol_y = np.concatenate((self.pol_y, pol_y2), axis=0)
            self.pol_z = np.concatenate((self.pol_z, pol_z2), axis=0)
            self.pol_type = np.concatenate((self.pol_type, pol_type2))
            self.pol_id = np.concatenate((self.pol_id, pol_id2))
            self.cyl_r = np.concatenate((self.cyl_r, cyl_r2))
            self.cyl_p = np.concatenate((self.cyl_p, cyl_p2))
            self.cyl_z = np.concatenate((self.cyl_z, cyl_z2))


def stell_vector_transform(mode, phi, vx_in, vy_in, vz_in):
    '''
    Transforms a vector in one of two ways, depending on the mode selected:
        reflect:   Reflects a vector, with a defined origin point, in a given 
                   poloidal plane that is presumed to form the boundary between
                   two stellarator-symmetryc half-periods. The output vector 
                   will have the equivalent origin point and direction 
                   associated with the target half-period. Vector magnitude
                   is preserved.
        translate: Translates a vector, with a defined origin point, in the 
                   toroidal direction. The origin thus moves along a circle
                   of fixed radius about the z axis by a given angle. The
                   azimuthal components of the vector change in order to 
                   preserve the radial and toroidal components. The vertical
                   component of the vector, as well as its length, are held
                   fixed.

    Parameters
    ----------
        mode: string
            'translate' or 'reflect' (see description above)
        phi: double
            'reflect' mode: toroidal angle (rad.) of the symmetry plane
            'translate' mode: toroidal interval (rad.) along which to translate
        vx_in, vy_in, vz_in: double (possibly arrays of equal dimensions)
            x, y, z components of the input vectors

    Returns
    -------
        vx_out, vy_out, vz_out: 
            x, y, z components of the output transformed vector(s)
    '''

    # Check input mode
    if mode == 'reflect':
        refl = True
    elif mode == 'translate':
        refl = False
    else:
        raise ValueError('Unrecognized mode for stell_vector_transform')

    if refl:
        vx_out = -np.cos(2*phi)*vx_in - np.sin(2*phi)*vy_in
        vy_out = -np.sin(2*phi)*vx_in + np.cos(2*phi)*vy_in
        vz_out = vz_in
    else:
        vx_out = np.cos(phi)*vx_in - np.sin(phi)*vy_in
        vy_out = np.sin(phi)*vx_in + np.cos(phi)*vy_in
        vz_out = vz_in

    return vx_out, vy_out, vz_out


def stell_point_transform(mode, phi, x_in, y_in, z_in):
    '''
    Transforms a point in one of two ways, depending on the mode selected:
        reflect:   Reflects a point in a given poloidal plane presumed to form 
                   the boundary between two stellarator-symmetryc half-periods. 
                   The output point will have the equivalent location associated
                   with the adjacent half-period. 
        translate: Translates a point in the toroidal direction (i.e., along
                   a circle with a fixed radius about the z axis) by a given
                   angle.

    Parameters
    ----------
        mode: string
            'translate' or 'reflect' (see description above)
        phi: double
            'reflect' mode: toroidal angle (rad.) of the symmetry plane
            'translate' mode: toroidal interval (rad.) along which to translate
        x_in, y_in, z_in: double (possibly arrays of equal dimensions)
            x, y, z coordinates of the point(s) to be transformed

    Returns
    -------
        x_out, y_out, z_out: double (possibly arrays)
            x, y, z coordinates of the transformed point(s)
    '''

    # Check input mode
    if mode == 'reflect':
        refl = True
    elif mode == 'translate':
        refl = False
    else:
        raise ValueError('Unrecognized mode for stell_vector_transform')

    if refl:
        x_out = np.cos(2*phi)*x_in + np.sin(2*phi)*y_in
        y_out = np.sin(2*phi)*x_in - np.cos(2*phi)*y_in
        z_out = -z_in
    else:
        x_out = np.cos(phi)*x_in - np.sin(phi)*y_in
        y_out = np.sin(phi)*x_in + np.cos(phi)*y_in
        z_out = z_in

    return x_out, y_out, z_out
