"""
This file contains definitions and functions for generating the 
PM4Stell magnet orientations and other magnet functionality.

This file was copied over from the MAGPIE code used for generating 
permanent magnet structures. All credit for this code is thanks 
to the PM4Stell team and Ken Hammond for his consent to use this file
and work with the permanent magnet branch of SIMSOPT.
"""
__all__ = ['orientation_phi', 'polarization_axes', 'discretize_polarizations',
           'face_triplet', 'edge_triplet',
           ]

import numpy as np


# Designated column indices from MAGPIE corners files
ind_xib = 3
ind_yib = 4
ind_xob = 6
ind_yob = 7

# Optimal offset angles for face-triplet and edge-triplet vector sets
theta_fe_ftri = 30.35 * np.pi / 180.0
theta_fc_ftri = 38.12 * np.pi / 180.0
theta_fe_etri = 18.42 * np.pi / 180.0
theta_fc_etri = 38.56 * np.pi / 180.0

# Face-centered polarizations
pol_f_pos = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])

# Edge-centered polarizations
pol_e_pos = np.array([[1.0, 1.0, 0.0],
                      [1.0, -1.0, 0.0],
                      [1.0, 0.0, 1.0],
                      [1.0, 0.0, -1.0],
                      [0.0, 1.0, 1.0],
                      [0.0, 1.0, -1.0]]
                     ) / np.sqrt(2.)

# Corner-centered polarizations
pol_c_pos = np.array([[1.0, 1.0, 1.0],
                      [1.0, 1.0, -1.0],
                      [1.0, -1.0, -1.0],
                      [1.0, -1.0, 1.0]]
                     ) / np.sqrt(3.)

# Face/edge-centered polarizations
pol_fe_pos = np.array([[1.0, 0.5, 0.0],
                       [1.0, -0.5, 0.0],
                       [1.0, 0.0, 0.5],
                       [1.0, 0.0, -0.5],
                       [0.5, 1.0, 0.0],
                       [0.5, -1.0, 0.0],
                       [0.5, 0.0, 1.0],
                       [0.5, 0.0, -1.0],
                       [0.0, 1.0, 0.5],
                       [0.0, 1.0, -0.5],
                       [0.0, 0.5, 1.0],
                       [0.0, 0.5, -1.0]]
                      ) * 2. / np.sqrt(5.)

# Face/edge polarization, 30 degrees from face-centered
fe30_1 = np.sin(np.pi / 6.0)
fe30_2 = np.cos(np.pi / 6.0)
pol_fe30_pos = np.array([[fe30_1, fe30_2, 0],
                         [fe30_1, -fe30_2, 0],
                         [fe30_1, 0, fe30_2],
                         [fe30_1, 0, -fe30_2],
                         [fe30_2, fe30_1, 0],
                         [-fe30_2, fe30_1, 0],
                         [0, fe30_1, fe30_2],
                         [0, fe30_1, -fe30_2],
                         [fe30_2, 0, fe30_1],
                         [-fe30_2, 0, fe30_1],
                         [0, fe30_2, fe30_1],
                         [0, -fe30_2, fe30_1]]
                        )

# Face/edge polarization, 22.5 degrees from face-centered
fe23_1 = np.sin(np.pi / 8.0)
fe23_2 = np.cos(np.pi / 8.0)
pol_fe23_pos = np.array([[fe23_1, fe23_2, 0],
                         [fe23_1, -fe23_2, 0],
                         [fe23_1, 0, fe23_2],
                         [fe23_1, 0, -fe23_2],
                         [fe23_2, fe23_1, 0],
                         [-fe23_2, fe23_1, 0],
                         [0, fe23_1, fe23_2],
                         [0, fe23_1, -fe23_2],
                         [fe23_2, 0, fe23_1],
                         [-fe23_2, 0, fe23_1],
                         [0, fe23_2, fe23_1],
                         [0, -fe23_2, fe23_1]]
                        )

# Face/edge polarization, 17 degrees from face-centered
fe17_1 = np.sin(17.0 * np.pi / 180.0)
fe17_2 = np.cos(17.0 * np.pi / 180.0)
pol_fe17_pos = np.array([[fe17_1, fe17_2, 0],
                         [fe17_1, -fe17_2, 0],
                         [fe17_1, 0, fe17_2],
                         [fe17_1, 0, -fe17_2],
                         [fe17_2, fe17_1, 0],
                         [-fe17_2, fe17_1, 0],
                         [0, fe17_1, fe17_2],
                         [0, fe17_1, -fe17_2],
                         [fe17_2, 0, fe17_1],
                         [-fe17_2, 0, fe17_1],
                         [0, fe17_2, fe17_1],
                         [0, -fe17_2, fe17_1]]
                        )

# Face/corner-centered polarizations
pol_fc_pos = np.array([[1.0, 0.5, 0.5],
                       [1.0, 0.5, -0.5],
                       [1.0, -0.5, 0.5],
                       [1.0, -0.5, -0.5],
                       [0.5, 1.0, 0.5],
                       [0.5, 1.0, -0.5],
                       [-0.5, 1.0, 0.5],
                       [-0.5, 1.0, -0.5],
                       [0.5, 0.5, 1.0],
                       [0.5, -0.5, 1.0],
                       [-0.5, 0.5, 1.0],
                       [-0.5, -0.5, 1.0]]
                      ) * np.sqrt(2. / 3.)

# Face/corner polarization, 27 degrees from face-centered
fc27_1 = np.sin(27.0 * np.pi / 180.0) / np.sqrt(2.0)
fc27_2 = np.cos(27.0 * np.pi / 180.0)
pol_fc27_pos = np.array([[fc27_1, fc27_1, fc27_2],
                         [fc27_1, -fc27_1, fc27_2],
                         [-fc27_1, -fc27_1, fc27_2],
                         [-fc27_1, fc27_1, fc27_2],
                         [fc27_1, fc27_2, fc27_1],
                         [fc27_1, fc27_2, -fc27_1],
                         [-fc27_1, fc27_2, -fc27_1],
                         [-fc27_1, fc27_2, fc27_1],
                         [fc27_2, fc27_1, fc27_1],
                         [fc27_2, fc27_1, -fc27_1],
                         [fc27_2, -fc27_1, -fc27_1],
                         [fc27_2, -fc27_1, fc27_1]]
                        )

# Face/corner polarization, 39 degrees from face-centered
fc39_1 = np.sin(39.0 * np.pi / 180.0) / np.sqrt(2.0)
fc39_2 = np.cos(39.0 * np.pi / 180.0)
pol_fc39_pos = np.array([[fc39_1, fc39_1, fc39_2],
                         [fc39_1, -fc39_1, fc39_2],
                         [-fc39_1, -fc39_1, fc39_2],
                         [-fc39_1, fc39_1, fc39_2],
                         [fc39_1, fc39_2, fc39_1],
                         [fc39_1, fc39_2, -fc39_1],
                         [-fc39_1, fc39_2, -fc39_1],
                         [-fc39_1, fc39_2, fc39_1],
                         [fc39_2, fc39_1, fc39_1],
                         [fc39_2, fc39_1, -fc39_1],
                         [fc39_2, -fc39_1, -fc39_1],
                         [fc39_2, -fc39_1, fc39_1]]
                        )

# Edge/corner-centered polarizations
pol_ec_pos = np.array([[0.5, 1.0, 1.0],
                       [0.5, 1.0, -1.0],
                       [0.5, -1.0, 1.0],
                       [0.5, -1.0, -1.0],
                       [1.0, 0.5, 1.0],
                       [1.0, 0.5, -1.0],
                       [-1.0, 0.5, 1.0],
                       [-1.0, 0.5, -1.0],
                       [1.0, 1.0, 0.5],
                       [1.0, -1.0, 0.5],
                       [-1.0, 1.0, 0.5],
                       [-1.0, -1.0, 0.5]]
                      ) * 2. / 3.

# Face/corner polarization, 27 degrees from face-centered
ec23_1 = 1.0 / np.sqrt(2.0 + np.tan(22.5 * np.pi / 180.0) ** 2)
ec23_2 = np.tan(22.5 * np.pi / 180.0) / np.sqrt(2.0 + np.tan(22.5 * np.pi / 180.0) ** 2)
pol_ec23_pos = np.array([[ec23_1, ec23_1, ec23_2],
                         [ec23_1, -ec23_1, ec23_2],
                         [-ec23_1, -ec23_1, ec23_2],
                         [-ec23_1, ec23_1, ec23_2],
                         [ec23_1, ec23_2, ec23_1],
                         [ec23_1, ec23_2, -ec23_1],
                         [-ec23_1, ec23_2, -ec23_1],
                         [-ec23_1, ec23_2, ec23_1],
                         [ec23_2, ec23_1, ec23_1],
                         [ec23_2, ec23_1, -ec23_1],
                         [ec23_2, -ec23_1, -ec23_1],
                         [ec23_2, -ec23_1, ec23_1]]
                        )

pol_f = np.concatenate((pol_f_pos, -pol_f_pos), axis=0)
pol_e = np.concatenate((pol_e_pos, -pol_e_pos), axis=0)
pol_c = np.concatenate((pol_c_pos, -pol_c_pos), axis=0)
pol_fe = np.concatenate((pol_fe_pos, -pol_fe_pos), axis=0)
pol_fc = np.concatenate((pol_fc_pos, -pol_fc_pos), axis=0)
pol_ec = np.concatenate((pol_ec_pos, -pol_ec_pos), axis=0)
pol_fe17 = np.concatenate((pol_fe17_pos, -pol_fe17_pos), axis=0)
pol_fe23 = np.concatenate((pol_fe23_pos, -pol_fe23_pos), axis=0)
pol_fe30 = np.concatenate((pol_fe30_pos, -pol_fe30_pos), axis=0)
pol_fc27 = np.concatenate((pol_fc27_pos, -pol_fc27_pos), axis=0)
pol_fc39 = np.concatenate((pol_fc39_pos, -pol_fc39_pos), axis=0)
pol_ec23 = np.concatenate((pol_ec23_pos, -pol_ec23_pos), axis=0)


def faceedge_vectors(theta):
    """
    Generates an array of unit vectors for a set of face-edge polarizations
    defined by the polar angle theta with respect to the nearest face-normal
    vector.

    Args:
        theta: double
            Polar angle theta between polarizations and nearest face-normal vector
    """

    # Unique vector component magnitudes
    comp_1 = np.sin(theta)
    comp_2 = np.cos(theta)

    vectors_pos = np.array([[comp_1, comp_2, 0],
                            [comp_1, -comp_2, 0],
                            [comp_1, 0, comp_2],
                            [comp_1, 0, -comp_2],
                            [comp_2, comp_1, 0],
                            [-comp_2, comp_1, 0],
                            [0, comp_1, comp_2],
                            [0, comp_1, -comp_2],
                            [comp_2, 0, comp_1],
                            [-comp_2, 0, comp_1],
                            [0, comp_2, comp_1],
                            [0, -comp_2, comp_1]]
                           )

    return np.concatenate((vectors_pos, -vectors_pos), axis=0)


def facecorner_vectors(theta):
    """
    Generates an array of unit vectors for a set of face-corner polarizations
    defined by the polar angle theta with respect to the nearest face-normal
    vector.

    Args:
        theta: polar angle theta with respect to the nearest face-normal vector.
    """

    # Unique vector component magnitudes
    comp_1 = np.sin(theta) / np.sqrt(2.0)
    comp_2 = np.cos(theta)

    vectors_pos = np.array([[comp_1, comp_1, comp_2],
                            [comp_1, -comp_1, comp_2],
                            [-comp_1, -comp_1, comp_2],
                            [-comp_1, comp_1, comp_2],
                            [comp_1, comp_2, comp_1],
                            [comp_1, comp_2, -comp_1],
                            [-comp_1, comp_2, -comp_1],
                            [-comp_1, comp_2, comp_1],
                            [comp_2, comp_1, comp_1],
                            [comp_2, comp_1, -comp_1],
                            [comp_2, -comp_1, -comp_1],
                            [comp_2, -comp_1, comp_1]]
                           )

    return np.concatenate((vectors_pos, -vectors_pos), axis=0)


def face_triplet(theta_fe, theta_fc):
    """
    Generates a set of vectors corresponding to the "face-triplet" set of 
    polarization types: face, face-edge, and face-corner.

    Args: 
        theta_fe: double
            Angle with respect to the nearest face-normal for the face-edge type
        theta_fc: double
            Angle with respect to the nearest face-normal for face-corner type

    Returns:
        vectors: double array
            3-column array in which each row is one unit vector in the full set
    """
    return np.concatenate((pol_f, faceedge_vectors(theta_fe),
                           facecorner_vectors(theta_fc)),
                          axis=0
                          )


def edge_triplet(theta_fe, theta_fc):
    """
    Generates a set of vectors corresponding to the "edge-triplet" set of 
    polarization types: edge, face-edge, and face-corner.

    Args:
        theta_fe: double
            Angle with respect to the nearest face-normal for the face-edge type
        theta_fc: double
            Angle with respect to the nearest face-normal for face-corner type

    Returns:
        vectors: double array
            3-column array in which each row is one unit vector in the full set
    """
    return np.concatenate((pol_e, faceedge_vectors(theta_fe),
                           facecorner_vectors(theta_fc)),
                          axis=0
                          )


def orientation_phi(corners_fname):
    """
    Determines the azimuthal (phi) angle that sets the orientations of
    rectangular prisms from an arrangement based on data from the corners file

    Phi is the azimuthal angle of the normal vector of the face whose normal
    vector is in the "radial-like" direction with a polar angle of pi/2 radians
    (i.e., parallel to the x-y plane). The remaining faces will be assumed to
    have normal vectors in either:

        #. the z-direction, or
        #. parallel to the x-y plane with an azimuthal angle equal to the 
           orientation phi plus pi/2 radians (the "phi-like" direction).

    Args:
        corners_fname (string): Corners file

    Returns:
        orientation_phi (double array): Orientation phi angle (radians) of each prism in the arrangement
    """

    corners_data = np.loadtxt(corners_fname, delimiter=',')

    xib = corners_data[:, ind_xib]
    yib = corners_data[:, ind_yib]
    xob = corners_data[:, ind_xob]
    yob = corners_data[:, ind_yob]

    dx = xob-xib
    dy = yob-yib

    return np.arctan2(dy, dx)


def polarization_axes(polarizations):
    """
    Populates an array of allowable polarization vectors based on an input
    list of names. Also outputs a 1D array with an integer corresponding
    to the polarization type in the order it appears in the list of names.

    Args:
        polarizations: list of strings
            List of names of polarization types. Each type should not be listed
            more than once

    Returns:
        pol_axes: 2d double array
            Allowable axes for the magnets expressed as unit vectors in local
            cylindrical coordinates. The first, second, and third columns 
            contain the r, phi, and z components, respectively.
        pol_type: 1d integer array
            ID number for the polarization type associated with each row
            in pol_axes. The ID corresponds to the order in which the type
            appears in the input list (polarizations). The first type is given
            an ID of 1.
    """

    pol_axes = np.zeros((0, 3))
    pol_type = np.zeros(0, dtype=int)
    i = 0
    if not isinstance(polarizations, list):
        polarizations = [polarizations]
    for pol_name in polarizations:
        if pol_name.lower() == 'face':
            pol_axes = np.concatenate((pol_axes, pol_f), axis=0)
            n_polarizations = np.shape(pol_f)[0]
        elif pol_name.lower() == 'edge':
            pol_axes = np.concatenate((pol_axes, pol_e), axis=0)
            n_polarizations = np.shape(pol_e)[0]
        elif pol_name.lower() == 'corner':
            pol_axes = np.concatenate((pol_axes, pol_c), axis=0)
            n_polarizations = np.shape(pol_c)[0]
        elif pol_name.lower() == 'faceedge':
            pol_axes = np.concatenate((pol_axes, pol_fe), axis=0)
            n_polarizations = np.shape(pol_fe)[0]
        elif pol_name.lower() == 'facecorner':
            pol_axes = np.concatenate((pol_axes, pol_fc), axis=0)
            n_polarizations = np.shape(pol_fc)[0]
        elif pol_name.lower() == 'edgecorner':
            pol_axes = np.concatenate((pol_axes, pol_ec), axis=0)
            n_polarizations = np.shape(pol_ec)[0]
        elif pol_name.lower() == 'fe17':
            pol_axes = np.concatenate((pol_axes, pol_fe17), axis=0)
            n_polarizations = np.shape(pol_fe17)[0]
        elif pol_name.lower() == 'fe23':
            pol_axes = np.concatenate((pol_axes, pol_fe23), axis=0)
            n_polarizations = np.shape(pol_fe23)[0]
        elif pol_name.lower() == 'fe30':
            pol_axes = np.concatenate((pol_axes, pol_fe30), axis=0)
            n_polarizations = np.shape(pol_fe30)[0]
        elif pol_name.lower() == 'fc27':
            pol_axes = np.concatenate((pol_axes, pol_fc27), axis=0)
            n_polarizations = np.shape(pol_fc27)[0]
        elif pol_name.lower() == 'fc39':
            pol_axes = np.concatenate((pol_axes, pol_fc39), axis=0)
            n_polarizations = np.shape(pol_fc39)[0]
        elif pol_name.lower() == 'ec23':
            pol_axes = np.concatenate((pol_axes, pol_ec23), axis=0)
            n_polarizations = np.shape(pol_ec23)[0]
        elif pol_name.lower() == 'fe_ftri':
            vectors = faceedge_vectors(theta_fe_ftri)
            pol_axes = np.concatenate((pol_axes, vectors), axis=0)
            n_polarizations = np.shape(vectors)[0]
        elif pol_name.lower() == 'fc_ftri':
            vectors = facecorner_vectors(theta_fc_ftri)
            pol_axes = np.concatenate((pol_axes, vectors), axis=0)
            n_polarizations = np.shape(vectors)[0]
        elif pol_name.lower() == 'fe_etri':
            vectors = faceedge_vectors(theta_fe_etri)
            pol_axes = np.concatenate((pol_axes, vectors), axis=0)
            n_polarizations = np.shape(vectors)[0]
        elif pol_name.lower() == 'fc_etri':
            vectors = facecorner_vectors(theta_fc_etri)
            pol_axes = np.concatenate((pol_axes, vectors), axis=0)
            n_polarizations = np.shape(vectors)[0]
        else:
            raise ValueError('polarization_axes: Polarization type ' +
                             pol_name + ' is not recognized or supported.')

        i = i + 1
        pol_type = np.concatenate((pol_type, i * np.ones(n_polarizations)))

    return pol_axes, pol_type


def discretize_polarizations(mag_data, orientation_phi, pol_axes, pol_type):
    """
    Adjusts the polarization axes of a set of dipole moments to the closest
    from a set of allowable axes. 

    Also populates the pol_x, pol_y, pol_z, and pol_id arrays within the
    input instance of the mag_data class, providing all of the allowable 
    polarization vectors for each magnet in the lab frame.

    Args:
        mag_data: mag_data object
            Instance of the mag_data class characterizing the dipole moments
            of a set of magnets, whose orientations are to be updated.
        orientation_phi: double array
            Orientation angle defining a reference axes for the allowable axes
            for each respective magnet (as described in the documentation for 
            orientation_phi().) Number of elements must agree with the number
            of magnets in mag_data.
        pol_axes: double array with 3 columns 
            Set of Cartesian unit vectors defining the allowable polarization
            vectors for each magnet. The first, second, and third columns 
            contain the x, y, and z components, respectively.
        pol_type: integer array
            Type IDs corresponding to each row in pol_axes.
    """

    nMagnets = mag_data.nMagnets
    nPol = pol_axes.shape[0]
    mag_data.init_pol_vecs(nPol)

    # Unit vector components for the "r" and "phi" faces of each magnet
    rhat_x = np.cos(orientation_phi).reshape((nMagnets, 1))
    rhat_y = np.sin(orientation_phi).reshape((nMagnets, 1))
    phat_x = -np.sin(orientation_phi).reshape((nMagnets, 1))
    phat_y = np.cos(orientation_phi).reshape((nMagnets, 1))

    # Allowable axes for each magnet in the magnet frame
    pol_r = np.repeat(np.reshape(pol_axes[:, 0], (1, nPol)), mag_data.nMagnets, axis=0)
    pol_p = np.repeat(np.reshape(pol_axes[:, 1], (1, nPol)), mag_data.nMagnets, axis=0)
    pol_z = np.repeat(np.reshape(pol_axes[:, 2], (1, nPol)), mag_data.nMagnets, axis=0)

    # Populate allowable axes for each magnet in the lab frame
    mag_data.pol_x[:, :] = pol_r*rhat_x + pol_p*phat_x
    mag_data.pol_y[:, :] = pol_r*rhat_y + pol_p*phat_y
    mag_data.pol_z[:, :] = pol_z
    mag_data.pol_type_key[:] = pol_type[:]

    # Concatenate with column zeros to the left, representing a null magnet
    pol_x = np.concatenate((np.zeros((nMagnets, 1)), mag_data.pol_x), axis=1)
    pol_y = np.concatenate((np.zeros((nMagnets, 1)), mag_data.pol_y), axis=1)
    pol_z = np.concatenate((np.zeros((nMagnets, 1)), mag_data.pol_z), axis=1)
    pol_r = np.concatenate((np.zeros((nMagnets, 1)), pol_r), axis=1)
    pol_p = np.concatenate((np.zeros((nMagnets, 1)), pol_p), axis=1)
    pol_type0 = np.concatenate(([0], pol_type))

    # Original polarization axes for each magnet
    nx_orig = np.cos(mag_data.mp)*np.sin(mag_data.mt)
    ny_orig = np.sin(mag_data.mp)*np.sin(mag_data.mt)
    nz_orig = np.cos(mag_data.mt)

    # Scale orig axes by magnet density and copy for each allowable polarization
    rho = mag_data.pho**mag_data.momentq
    rmx_orig = np.repeat(np.reshape(rho*nx_orig, (nMagnets, 1)), nPol+1, axis=1)
    rmy_orig = np.repeat(np.reshape(rho*ny_orig, (nMagnets, 1)), nPol+1, axis=1)
    rmz_orig = np.repeat(np.reshape(rho*nz_orig, (nMagnets, 1)), nPol+1, axis=1)

    # 2-norm of differences between allowable vectors and scaled orig vectors
    resid2 = (pol_x-rmx_orig)**2 + (pol_y-rmy_orig)**2 + (pol_z-rmz_orig)**2

    # Choose allowable polarization vectors that minimize the residual 2-norm
    min_ind = np.argmin(resid2, axis=1)

    # Record indices of the residual-minimizing polarizations
    mag_data.pol_id[:] = min_ind

    # Arrays of the residual-minimizing polarizations for each magnet
    pol_x_min = pol_x[np.arange(nMagnets), min_ind]
    pol_y_min = pol_y[np.arange(nMagnets), min_ind]
    pol_z_min = pol_z[np.arange(nMagnets), min_ind]
    pol_r_min = pol_r[np.arange(nMagnets), min_ind]
    pol_p_min = pol_p[np.arange(nMagnets), min_ind]

    # Determine lab-frame polar angles for the resid-minimizing polarizations
    mag_data.mp = np.arctan2(pol_y_min, pol_x_min)
    mag_data.mt = np.arctan2(np.sqrt(pol_x_min**2 + pol_y_min**2), pol_z_min)

    # Magnet-frame cylindrical coordinates for resid-minimizing polarizations
    mag_data.cyl_r[:] = pol_r_min
    mag_data.cyl_p[:] = pol_p_min
    mag_data.cyl_z[:] = pol_z_min

    # Type ID of the residual-minimizing polarization
    mag_data.pol_type[:] = pol_type0[min_ind]

    # Set pho to 0 if min_ind is 0; 1 otherwise
    pho_0_inds = np.where(min_ind == 0)
    pho_1_inds = np.where(min_ind != 0)
    mag_data.pho[pho_0_inds] = 0.0
    mag_data.pho[pho_1_inds] = 1.0
