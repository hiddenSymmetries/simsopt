'''
tr104b_045deg_nb200_na026.py

Uses the GPMO backtracking algorithm with arbitrarily-defined polarization 
vectors to optimize magnets in an arrangement around the NCSX plasma in the 
C09R00 configuration with an average magnetic field strength on axis of about 
0.5 T.

This test applies the backtracking approach to the PM4Stell magnet arrangement
with face-triplet polarizations. The threshold angle for removal of magnet 
pairs is 45 degrees.

'''

import os
import numpy as np
import time
from simsopt.geo import SurfaceRZFourier, PermanentMagnetGrid
from simsopt.field import BiotSavart
from simsopt.solve import GPMO
from simsopt.util.permanent_magnet_helper_functions \
    import initialize_default_kwargs, write_pm_optimizer_to_famus
from famus_io import FocusPlasmaBnormal, coils_from_focus
from adjust_magnet_angles import focus_data
from polarization_project import polarization_axes, orientation_phi, \
    discretize_polarizations
t_start = time.time()

# Set some parameters
N = 8
nphi = N
ntheta = N
nfp = 3
algorithm = 'ArbVec_backtracking'
nBacktracking = 200 
nAdjacent = 26
nIter_max = 100000
max_nMagnets = 5000  # 50000
thresh_angle = 0.25*np.pi
nHistory = 250  # must divide evenly into nIter_max
out_dir = 'soln_tr104b_' + str(int(thresh_angle * 180 / np.pi)) + 'deg_nb' + str(nBacktracking) + '_na' + str(nAdjacent) + '/' 

# Obtain the plasma boundary for the NCSX configuration
dir_pm4stell = ''
dir_ncsx_surf = dir_pm4stell + ''
fname_plasma = dir_ncsx_surf + 'c09r00_B_axis_half_tesla.plasma'
lcfs_ncsx = SurfaceRZFourier.from_focus(fname_plasma, range='half period', \
                                        nphi=nphi, ntheta=ntheta)

# Obtain the normal field on the plasma boundary arising from plasma currents
bnormal_obj_ncsx = FocusPlasmaBnormal(fname_plasma)
bn_plasma = bnormal_obj_ncsx.bnormal_grid(nphi, ntheta, 'half period')

# Obtain the NCSX TF coil data and calculate their normal field on the boundary
dir_ncsx_coils = dir_pm4stell + ''
fname_ncsx_coils = dir_ncsx_coils + 'tf_only_half_tesla_symmetry_baxis.focus'
ncsx_tfcoils = coils_from_focus(fname_ncsx_coils, nfp)
bs_tfcoils = BiotSavart(ncsx_tfcoils)
bs_tfcoils.set_points(lcfs_ncsx.gamma().reshape((-1, 3)))
bn_tfcoils = np.sum(bs_tfcoils.B().reshape((nphi, ntheta, 3)) \
                    * lcfs_ncsx.unitnormal(), axis=2)

bn_total = bn_plasma + bn_tfcoils

# Obtain data on the magnet arrangement
fname_argmt = 'magpie_trial104b.focus'
fname_corn = dir_pm4stell + 'magpie_trial104b_corners.csv'
mag_data = focus_data(fname_argmt)
nMagnets_tot = mag_data.nMagnets

# Determine the allowable polarization types and reject the negatives
pol_axes = np.zeros((0, 3))
pol_type = np.zeros(0, dtype=int)

pol_axes_f, pol_type_f = polarization_axes(['face'])
ntype_f = int(len(pol_type_f)/2)
pol_axes_f = pol_axes_f[:ntype_f, :]
pol_type_f = pol_type_f[:ntype_f]
pol_axes = np.concatenate((pol_axes, pol_axes_f), axis=0)
pol_type = np.concatenate((pol_type, pol_type_f))

pol_axes_fe_ftri, pol_type_fe_ftri = polarization_axes(['fe_ftri'])
ntype_fe_ftri = int(len(pol_type_fe_ftri)/2)
pol_axes_fe_ftri = pol_axes_fe_ftri[:ntype_fe_ftri, :]
pol_type_fe_ftri = pol_type_fe_ftri[:ntype_fe_ftri] + 1
pol_axes = np.concatenate((pol_axes, pol_axes_fe_ftri), axis=0)
pol_type = np.concatenate((pol_type, pol_type_fe_ftri))

pol_axes_fc_ftri, pol_type_fc_ftri = polarization_axes(['fc_ftri'])
ntype_fc_ftri = int(len(pol_type_fc_ftri)/2)
pol_axes_fc_ftri = pol_axes_fc_ftri[:ntype_fc_ftri, :]
pol_type_fc_ftri = pol_type_fc_ftri[:ntype_fc_ftri] + 2
pol_axes = np.concatenate((pol_axes, pol_axes_fc_ftri), axis=0)
pol_type = np.concatenate((pol_type, pol_type_fc_ftri))

ophi = orientation_phi(fname_corn)[:nMagnets_tot]
discretize_polarizations(mag_data, ophi, pol_axes, pol_type)
pol_vectors = np.zeros((nMagnets_tot, len(pol_type), 3))
pol_vectors[:, :, 0] = mag_data.pol_x
pol_vectors[:, :, 1] = mag_data.pol_y
pol_vectors[:, :, 2] = mag_data.pol_z

# Initialize the permanent magnet grid from the PM4Stell arrangement
pm_ncsx = PermanentMagnetGrid(lcfs_ncsx, famus_filename=fname_argmt, \
                              Bn=bn_total, coordinate_flag='cartesian', pol_vectors=pol_vectors)

# Optimize with the GPMO algorithm
kwargs = initialize_default_kwargs('GPMO')
kwargs['K'] = nIter_max
kwargs['nhistory'] = nHistory
if algorithm == 'backtracking' or algorithm == 'ArbVec_backtracking':
    kwargs['backtracking'] = nBacktracking
    kwargs['Nadjacent'] = nAdjacent
    kwargs['dipole_grid_xyz'] = np.ascontiguousarray(pm_ncsx.dipole_grid_xyz)
    if algorithm == 'ArbVec_backtracking':
        kwargs['thresh_angle'] = thresh_angle
        kwargs['max_nMagnets'] = max_nMagnets
t1 = time.time()
R2_history, Bn_history, m_history = GPMO(pm_ncsx, algorithm, **kwargs)
dt = time.time() - t1
print('GPMO took t = ', dt, ' s')

# Save files
os.makedirs(out_dir, exist_ok=True)
write_pm_optimizer_to_famus(out_dir, pm_ncsx)
np.savetxt(out_dir + 'R2_history.txt', R2_history)
np.savetxt(out_dir + 'absBn_history.txt', Bn_history)
nmags = m_history.shape[0]
nhist = m_history.shape[2]
m_history_2d = m_history.reshape((nmags*m_history.shape[1], nhist))
np.savetxt(out_dir + 'm_history_nmags=%d_nhist=%d.txt' % (nmags, nhist), \
           m_history_2d)
t_end = time.time()  
print('Script took in total t = ', t_end - t_start, ' s')
