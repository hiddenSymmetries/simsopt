import numpy as np
import unittest
import simsoptpp as sopp
from numpy.testing import assert_raises
import numpy as np
import os

from simsopt.field import (BoozerRadialInterpolant, InterpolatedBoozerField, trace_particles_boozer,
                           MinToroidalFluxStoppingCriterion, MaxToroidalFluxStoppingCriterion,
                           ToroidalTransitStoppingCriterion, compute_resonances)
from booz_xform import Booz_xform

def setup_interpolant(field, nfp, n_metagrid_pts):
    ### NEW INTERPOLANT

    srange = (0, 1.0, 3*n_metagrid_pts+1)
    trange = (0, np.pi, 3*n_metagrid_pts+1)
    zrange = (0, 2*np.pi/nfp, 3*n_metagrid_pts+1)

    s_grid = np.linspace(srange[0], srange[1], srange[2])
    theta_grid = np.linspace(trange[0], trange[1], trange[2])
    zeta_grid = np.linspace(zrange[0], zrange[1], zrange[2])

    quad_pts = np.empty((srange[2]*trange[2]*zrange[2], 3))
    for i in range(srange[2]):
        for j in range(trange[2]):
            for k in range(zrange[2]):
                quad_pts[trange[2]*zrange[2]*i + zrange[2]*j + k, :] = [s_grid[i], theta_grid[j], zeta_grid[k]]

    field.set_points(quad_pts)


    # Quantities to interpolate
    modB = field.modB()
    modB_derivs = field.modB_derivs()
    G = field.G()
    dGds = field.dGds()
    I = field.I()
    dIds = field.dIds()
    iota = field.iota()
    diotads = field.diotads()
    
    quad_info = np.hstack((modB, modB_derivs, G, dGds, I, dIds, iota, diotads))
    quad_info = np.ascontiguousarray(quad_info)

    return srange, trange, zrange, quad_info


def test_interpolant_bfield(field, nfp, n_metagrid_pts, n_test_pts):

    # generate test points
    s = np.random.uniform(low=0, high=1, size=(n_test_pts,1))
    t = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
    z = np.random.uniform(low=0, high=2*np.pi, size=(n_test_pts,1))
    stz = np.hstack((s,t,z))

    # SIMSOPT INTERPOLANT
    field.set_points(stz)
    modB = field.modB()
    modB_derivs = field.modB_derivs()
    G = field.G()
    dGds = field.dGds()
    I = field.I()
    dIds = field.dIds()
    iota = field.iota()
    diotads = field.diotads()
    
    simsopt_interpolation = np.hstack((modB, modB_derivs, G, dGds, I, dIds, iota, diotads))
    ## NEW INTERPOLANT
    srange, trange, zrange, quad_info = setup_interpolant(field, nfp, n_metagrid_pts)
    stz = np.ascontiguousarray(stz)

    # Calculate interpolation

    new_interpolation = sopp.test_gpu_interpolation(quad_info, srange, trange, zrange, stz, 10, stz.shape[0])
    new_interpolation = np.reshape(new_interpolation, (stz.shape[0], 10))

    # print(np.abs(simsopt_interpolation - new_interpolation) / simsopt_interpolation)
    rel_err = np.abs((simsopt_interpolation - new_interpolation) / simsopt_interpolation)
    diff = np.max(rel_err)
    print("Maximum relative error in interpolation values on {} points: {}".format(n_test_pts, diff))

    print("culprit particle:")
    row_index = np.argmax(rel_err) // rel_err.shape[1]
    print(stz[row_index, :])
    print("simsopt", simsopt_interpolation[row_index, :])
    print("new", new_interpolation[row_index, :])
    print(rel_err[row_index, :])


if __name__ == "__main__":
    ### CREATE A FIELD FOR TRACING

    # compute equlibrium
    filename = 'boozmn_qhb_100.nc'
    equil = Booz_xform()
    equil.verbose = 0
    equil.read_boozmn(filename)
    nfp = equil.nfp

    bri = BoozerRadialInterpolant(
        equil=equil,
        order=3,
        no_K=False
    )

    # compute harmonic information
    saw_filename = 'scaled_mode_32.935kHz.npy'
    saw_data = np.load(saw_filename, allow_pickle=True)
    saw_data = saw_data[()]
    saw_omega = 1000*np.sqrt(saw_data['eigenvalue'])
    print("omega=", saw_omega)
    s = saw_data['s_coords']
    saw_srange = (s[0], s[-1], len(s))
    saw_m = np.ascontiguousarray([x[0] for x in saw_data['harmonics']])
    saw_n = np.ascontiguousarray([x[1] for x in saw_data['harmonics']])
    saw_phihats = np.ascontiguousarray(np.column_stack([x[2].T for x in saw_data['harmonics']]))
    saw_nharmonics = len(saw_m) 

    degree = 3
    n_meta_grid_pts = 15
    srange = (0, 1, n_meta_grid_pts)
    thetarange = (0, np.pi, n_meta_grid_pts)
    zetarange = (0, 2*np.pi/nfp, n_meta_grid_pts)

    field = InterpolatedBoozerField(
        bri,
        degree=3,
        srange=(0, 1, 15),
        thetarange=(0, np.pi, 15),
        zetarange=(0, 2*np.pi/nfp, 15),
        extrapolate=True,
        nfp=nfp,
        stellsym=True,
        initialize=['modB','modB_derivs']
    )

    test_interpolant_bfield(field, nfp, n_meta_grid_pts, 100000)