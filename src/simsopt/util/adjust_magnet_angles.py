"""
    This file contains a class for reading in FOCUS-style data files. 

    This file was copied over from the MAGPIE code used for generating 
    permanent magnet structures. All credit for this code is thanks 
    to the PM4Stell team and Ken Hammond for his consent to use this file
    and work with the permanent magnet branch of SIMSOPT.
"""

import numpy as np
import pandas as pd
import sys


class focus_data(object):
    """
        Class object for reading in FOCUS-style data files.

        Args:
            filename: a FOCUS file
    """

    propNames = ['type', 'symm', 'coilname', 'ox', 'oy', 'oz', 'Ic', 'M_0', \
                 'pho', 'Lc', 'mp', 'mt', 'op']

    float_inds = [3, 4, 5, 7, 8, 10, 11]

    def __init__(self, filename):

        self.nMagnets = 0
        self.nPol = 0

        # initialize to # of mandatory properties ('op' is currently optional)
        self.nProps = len(focus_data.propNames) - 1

        self.read_from_file(filename)

    def read_from_file(self, filename):

        focusfile = open(filename, 'r')

        # Ignore the first line in the file
        line1 = focusfile.readline()

        # Record the number of magnets and the momentq
        line2data = [int(number) for number in \
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
        line3 = focusfile.readline()

        # Read the data for each magnet from the file
        count = 0
        self.max_float_length = 0
        self.min_float_val = 0
        for i in range(self.nMagnets):

            linedata = focusfile.readline().strip().split(',')
            if len(linedata) < self.nProps:
                raise Exception(('Problem accessing data for magnet %d in ' \
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
                        testnum = np.double(linedata[12])
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
            max_float_length = max([len(linedata[i].strip()) for i \
                                    in focus_data.float_inds])
            if max_float_length > self.max_float_length:
                self.max_float_length = max_float_length
            min_float_val = min([np.double(linedata[i]) for i \
                                 in focus_data.float_inds])
            if min_float_val < self.min_float_val:
                self.min_float_val = min_float_val

            count += 1

        focusfile.close()

        self.max_name_length = max([len(string) for string in self.coilname])

        # Add space for a negative sign in the max float length if necessary
        if self.min_float_val >= 0:
            self.max_float_length = self.max_float_length+1

    def unit_vector(self, inds):
        """
            Returns the x, y, and z components of a Cartesian unit vector in the 
            polarizaton direction of the nth magnet
        """
        if len(inds) > 0 and max(inds) > self.nMagnets-1:
            raise Exception('unit_vector: requested magnet %d does not exist' \
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
            raise Exception('unit_vector: requested magnet %d does not exist' \
                            % (max(inds)))

        mt_perp = self.mt + np.pi/2.0
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

        if self.nMagnets < 1:
            raise RuntimeError('print_to_file: no magnets to print')

        focusfile = open(filename, 'w')

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
        focusfile.write(('%s, '*2 + '%'+ln+'s, ' + ('%'+lf+'s, ')*3 + \
                         '%s, ' + ('%'+lf+'s, ')*2 + '%s, ' + \
                         ('%'+lf+'s, ')*2) % \
                        tuple(focus_data.propNames[:12]))
        if self.has_op:
            focusfile.write(('%'+lf+'s, \n') % (focus_data.propNames[12]))
        else:
            focusfile.write('\n')

        # Write the data for each magnet to the file
        for i in range(self.nMagnets):

            lineStr = ('%4d, '*2 + '%'+ln+'s, ' + \
                       ('%'+lf+'.'+nd+'E, ')*3 + '%2d, ' + \
                       ('%'+lf+'.'+nd+'E, ')*2 + '%2d, ' + \
                       ('%'+lf+'.'+nd+'E, ')*2) %  \
                (self.magtype[i], self.symm[i], self.coilname[i], \
                 self.ox[i], self.oy[i], self.oz[i], self.Ic[i],  \
                 self.M_0[i], self.pho[i], self.Lc[i], \
                 self.mp[i], self.mt[i])

            if self.has_op:
                lineStr = lineStr + ('%'+lf+'.'+nd+'E, \n') % self.op[i]
            else:
                lineStr = lineStr + '\n'

            focusfile.write(lineStr)

        focusfile.close()

    def init_pol_vecs(self, n_pol):
        """
            Initializes arrays for the x, y, and z components of allowable
            polarization vectors, as an array giving the ID of the polarization
            vector actually used for the respective magnet (0 means magnet is off)
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
            raise ValueError('repeat_hp_to_fp is only valid for magnets ' \
                             'that are stellarator symmetric (symm=2)')

        # Toroidal angle of the symmetry plane between the adjacent half-periods
        if magnet_sector > 2*nfp or magnet_sector < 1:
            raise ValueError('magnet_sector must be positive and less than ' \
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
                stell_vector_transform('reflect', phi, \
                                       self.pol_x[symm_inds, :], self.pol_y[symm_inds, :], \
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

    def moments_file(self, blocks_fname, moments_fname, rev):
        '''
        Generates a "moments file" for a set of magnets with projected 
        polarizations. The moments file is a .csv with two header lines and
        the following columns:

        rid, pid, zid, wid, hid, did, ioid, ox, oy, oz, Mx, My, Mz, 
            px, py, pz, rho, type, cyl_r, cyl_p, cyl_z

        where:
            rid, pid, zid: 
                radial, toroidal, and vertical index labels for the magnets' 
                spatial block in the arrangement
            wid, hid, did: 
                toroidal, vertical, and radial index labels for the magnets' 
                spatial subblock
            ioid: 
                label indicating whether the magnet is inboard or outboard
            ox, oy, oz: 
                x, y, z (Cartesian) coordinates of the magnets' centers
            Mx, My, Mz: 
                x, y, z (Cartesian) components of the dipole moments of each 
                magnet in A*m^2
            px, py, pz: 
                x, y, z (Cartesian) components of unit vectors perpendicular to 
                (Mx, My, Mz)
            rho: 
                1 if the magnet exists; 0 if it was removed during optimization
            type: 
                ID of the magnet polarization type; 0 if magnet is nonexistent
            axid: 
                Unique identifier for each discrete polarization vector within
                a given phi sector (i.e. for each value of pid)
            cyl_r, cyl_p, cyl_z: 
                r, phi, z (cylindrical) components of the magnets' polarization
                in the frame of the magnets' respective support structures

        The first header line gives the arrangement ID from the blocks file,
        as well as the revision number supplied by the user. The second 
        header line contains the titles of each column.
        '''

        blocks_file = open(blocks_fname, 'r')
        argmt_id = blocks_file.readline().split(',')[0].strip()
        blocks_file.close()

        blocks_data = pd.read_csv(blocks_fname, header=1)
        blocks_data.rename(columns=lambda x: x.strip(), inplace=True)

        moments_data = \
            pd.DataFrame(data=blocks_data[['rid', 'pid', 'zid', \
                                           'wid', 'hid', 'did', 'ioid']])
        moments_data['ox'] = self.ox
        moments_data['oy'] = self.oy
        moments_data['oz'] = self.oz

        Mx, My, Mz = self.unit_vector(np.arange(self.nMagnets))*self.M_0
        moments_data['Mx'] = Mx
        moments_data['My'] = My
        moments_data['Mz'] = Mz

        px, py, pz = self.perp_vector(np.arange(self.nMagnets))
        moments_data['px'] = px
        moments_data['py'] = py
        moments_data['pz'] = pz

        moments_data['rho'] = self.pho

        moments_data['type'] = self.pol_type
        moments_data['cyl_r'] = self.cyl_r
        moments_data['cyl_p'] = self.cyl_p
        moments_data['cyl_z'] = self.cyl_z        
        moments_data['axid'] = self.pol_id

        data_str = moments_data.to_csv(index=False, float_format='%16.8e')

        moments_file = open(moments_fname, 'w')
        moments_file.write('%s, %s\n' % (argmt_id, rev))
        moments_file.write(data_str)
        moments_file.close()


def stell_point_transform_2(mode, phi, x_in, y_in, z_in):
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
        raise ValueError('Unrecognized mode for stell_point_transform')

    # Radial coordinate and toroidal angle of input point
    r_in = np.sqrt(x_in**2 + y_in**2)
    phi_in = np.arctan2(y_in, x_in)

    # Radial coordinate and toroidal angle of the output point
    r_out = r_in
    if refl:
        phi_out = phi + (phi - phi_in)
        z_out = -z_in
    else:
        phi_out = phi_in + phi
        z_out = z_in

    # Cartesian coordinates of the output point
    x_out = r_out*np.cos(phi_out)
    y_out = r_out*np.sin(phi_out)

    return x_out, y_out, z_out


def stell_vector_transform_2(mode, phi, \
                             vx_in, vy_in, vz_in, ox_in, oy_in, oz_in):
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
        ox_in, oy_in, oz_in: double (possibly arrays of equal dimensions)
            x, y, z coordinates of the origin point(s) of the input vectors

    Returns
    -------
        vx_out, vy_out, vz_out: 
            x, y, z components of the output transformed vector(s)
        ox_out, oy_out, oz_out:
            x, y, z coordinates of the origin point(s) of the vector(s)
    '''

    # Check input mode
    if mode == 'reflect':
        refl = True
    elif mode == 'translate':
        refl = False
    else:
        raise ValueError('Unrecognized mode for stell_vector_transform')

    # Radial coordinate and toroidal angle of input point
    or_in = np.sqrt(ox_in**2 + oy_in**2)
    ophi_in = np.arctan2(oy_in, ox_in)

    # Cartesian components of the radial and toroidal unit vectors (input)
    rhat_x_in = np.cos(ophi_in)
    rhat_y_in = np.sin(ophi_in)
    phat_x_in = -np.sin(ophi_in)
    phat_y_in = np.cos(ophi_in)

    # Radial and toroidal components of the input vector
    vr_in = vx_in*rhat_x_in + vy_in*rhat_y_in
    vphi_in = vx_in*phat_x_in + vy_in*phat_y_in

    # Radial and vertical components
    or_out = or_in
    vz_out = vz_in
    vphi_out = vphi_in
    if refl:
        ophi_out = phi + (phi - ophi_in)
        oz_out = -oz_in
        vr_out = -vr_in
    else:
        ophi_out = ophi_in + phi
        oz_out = oz_in
        vr_out = vr_in

    # Cartesian components of the radial and toroidal unit vectors (outnput)
    rhat_x_out = np.cos(ophi_out)
    rhat_y_out = np.sin(ophi_out)
    phat_x_out = -np.sin(ophi_out)
    phat_y_out = np.cos(ophi_out)

    # Cartesian x/y components of the output vector
    vx_out = vr_out*rhat_x_out + vphi_out*phat_x_out
    vy_out = vr_out*rhat_y_out + vphi_out*phat_y_out

    # Cartesian coordinates of the output origin point
    ox_out = or_out*np.cos(ophi_out)
    oy_out = or_out*np.sin(ophi_out)

    return vx_out, vy_out, vz_out, ox_out, oy_out, oz_out


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


if __name__ == '__main__':

    #dirname = '/u/khammond/focus/ncsx/magnets_trial24_single/'
    #filename = dirname + 'trial_24_single.focus'

    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print('usage: python adjust_magnet_angles.py infile outfile [q_new]')
        exit()

    infile = sys.argv[1]
    outfile = sys.argv[2]

    magdata = focus_data(infile)
    magdata.flip_negative_magnets()

    if len(sys.argv) == 4:
        q_new = float(sys.argv[3])
        magdata.adjust_rho(q_new)

    magdata.print_to_file(outfile)

