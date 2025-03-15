"""
This module contains the a number of useful functions for  
optimizing dipole arrays in the SIMSOPT code.
"""
__all__ = ['remove_inboard_dipoles',
           'remove_interlinking_dipoles_and_TFs',
           'align_dipoles_with_plasma']

import numpy as np


def remove_inboard_dipoles(plasma_surf, base_curves, eps=-0.4):
    """
    Remove all the dipole coils on the inboard side. Useful if the desired plasma
    configuration is fairly compact, so no room for dipoles on inboard side.

    Args:
        base_curves:
            The curve objects for the dipoles in the array.

    Returns:
        base_curves:
            The same objects, minus any dipoles on the inboard side. 
    """
    import warnings
    keep_inds = []
    for ii in range(len(base_curves)):
        counter = 0
        for i in range(base_curves[0].gamma().shape[0]):
            dij = np.sqrt(np.sum((base_curves[ii].gamma()[i, :]) ** 2))
            conflict_bool = (dij < (1.0 + eps) * plasma_surf.get_rc(0, 0))
            if conflict_bool:
                print('bad index = ', i, dij, plasma_surf.get_rc(0, 0))
                warnings.warn(
                    'There is a PSC coil initialized such that it is within a radius'
                    'of a TF coil. Deleting these PSCs now.')
                counter += 1
                break
        if counter == 0:
            keep_inds.append(ii)
    return np.array(base_curves)[keep_inds]


def remove_interlinking_dipoles_and_TFs(base_curves, base_curves_TF):
    """
    Similar process to remove any dipole coils that are initialized
    intersecting with the TF coils, to avoid interlinking at the very beginning of optimization.

    Args:
        base_curves:
            The curve objects for the dipoles in the array.
        base_curves_TF:
            The curve objects for the TF (modular) coils in the array.

    Returns:
        base_curves:
            The same objects, minus any dipoles that were interlinking the TF coils.
    """
    import warnings
    keep_inds = []
    for ii in range(len(base_curves)):
        counter = 0
        for i in range(base_curves[0].gamma().shape[0]):
            eps = 0.05  # controls how close a TF-dipole coil distance can be before being removed
            for j in range(len(base_curves_TF)):
                for k in range(base_curves_TF[j].gamma().shape[0]):
                    dij = np.sqrt(np.sum((base_curves[ii].gamma()[i, :] - base_curves_TF[j].gamma()[k, :]) ** 2))
                    conflict_bool = (dij < (1.0 + eps) * base_curves[0].x[0])
                    if conflict_bool:
                        print('bad indices = ', i, j, dij, base_curves[0].x[0])
                        warnings.warn(
                            'There is a PSC coil initialized such that it is within a radius'
                            'of a TF coil. Deleting these PSCs now.')
                        counter += 1
                        break
        if counter == 0:
            keep_inds.append(ii)
    return np.array(base_curves)[keep_inds]

def align_dipoles_with_plasma(plasma_surf, base_curves):
    ncoils = len(base_curves)
    coil_normals = np.zeros((ncoils, 3))
    plasma_points = plasma_surf.gamma().reshape(-1, 3)
    plasma_unitnormals = plasma_surf.unitnormal().reshape(-1, 3)
    for i in range(ncoils):
        point = (base_curves[i].get_dofs()[-3:])
        dists = np.sum((point - plasma_points) ** 2, axis=-1)
        min_ind = np.argmin(dists)
        coil_normals[i, :] = plasma_unitnormals[min_ind, :]
    coil_normals = coil_normals / np.linalg.norm(coil_normals, axis=-1)[:, None]
    alphas = np.arcsin(
        -coil_normals[:, 1],
    )
    deltas = np.arctan2(coil_normals[:, 0], coil_normals[:, 2])
    return alphas, deltas
