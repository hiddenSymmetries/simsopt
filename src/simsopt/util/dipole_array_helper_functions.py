"""
This module contains the a number of useful functions for  
optimizing dipole arrays in the SIMSOPT code.
"""
__all__ = ['remove_inboard_dipoles',
           'remove_interlinking_dipoles_and_TFs',
           'align_dipoles_with_plasma',
           'initialize_coils',
           'dipole_array_optimization_function',
           'save_coil_sets']

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


def remove_interlinking_dipoles_and_TFs(base_curves, base_curves_TF, eps=0.05):
    """
    Similar process to remove any dipole coils that are initialized
    intersecting with the TF coils, to avoid interlinking at the very beginning of optimization.

    Args:
        base_curves:
            The curve objects for the dipoles in the array.
        base_curves_TF:
            The curve objects for the TF (modular) coils in the array.
        eps: float
            controls how close a TF-dipole coil distance can be before being removed

    Returns:
        base_curves:
            The same objects, minus any dipoles that were interlinking the TF coils.
    """
    import warnings
    keep_inds = []
    for ii in range(len(base_curves)):
        counter = 0
        for i in range(base_curves[0].gamma().shape[0]):
            for j in range(len(base_curves_TF)):
                for k in range(base_curves_TF[j].gamma().shape[0]):
                    dij = np.sqrt(np.sum((base_curves[ii].gamma()[i, :] - base_curves_TF[j].gamma()[k, :]) ** 2))
                    conflict_bool = (dij < (1.0 + eps) * base_curves[0].x[0])
                    if conflict_bool:
                        # print('bad indices = ', i, j, dij, base_curves[0].x[0])
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


def initialize_coils(s, TEST_DIR, configuration):
    """
    Initializes appropriate coils for the Schuett-Henneberg 2-field-period QA,
    Landreman-Paul QA, Landreman-Paul QH, etc., usually for purposes
    of finding a dipole coil array solution. 

    Args:
        TEST_DIR: String denoting where to find the input files.
        s: plasma boundary surface.
        configuration: String denoting the stellarator name.
    Returns:
        base_curves: List of CurveXYZ class objects.
        curves: List of Curve class objects.
        coils: List of Coil class objects.
    """
    from simsopt.geo import create_equally_spaced_curves
    from simsopt.field import Current, coils_via_symmetries

    # parameters for the TF coils, increase order for a better solution
    # Total current scaled to give B ~ 5.7 T on axis (actually averaged over the major radius)
    if configuration == 'LandremanPaulQA':
        ncoils = 3
        R0 = s.get_rc(0, 0) * 1
        R1 = s.get_rc(1, 0) * 3
        order = 4
        total_current = 66237606
    elif configuration == 'LandremanPaulQH':
        ncoils = 2
        R0 = s.get_rc(0, 0) * 1
        R1 = s.get_rc(1, 0) * 4
        order = 4
        total_current = 45642162
    elif configuration == 'SchuettHennebergQAnfp2':
        ncoils = 2
        R0 = s.get_rc(0, 0) * 1.4
        R1 = s.get_rc(1, 0) * 4
        order = 16
        total_current = 35000000
    else:
        raise ValueError("Stellarator configuration not recognized.")
    print('Total current = ', total_current)

    # Create the initial coils
    base_curves = create_equally_spaced_curves(
        ncoils, s.nfp, stellsym=True,
        R0=R0, R1=R1, order=order, numquadpoints=256,
    )
    base_currents = [(Current(total_current / ncoils * 1e-7) * 1e7) for _ in range(ncoils - 1)]
    total_current = Current(total_current)
    total_current.fix_all()
    base_currents += [total_current - sum(base_currents)]
    coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
    curves = [c.curve for c in coils]
    return base_curves, curves, coils, base_currents


def dipole_array_optimization_function(dofs, obj_dict, weight_dict):
    """
        Wrapper function for performing dipole array optimization.

    """
    # unpack all the dictionary objects
    btot = obj_dict["btot"]
    s = obj_dict["s"]
    base_curves_TF = obj_dict["base_curves_TF"]
    nphi = len(s.quadpoints_phi)
    ntheta = len(s.quadpoints_theta)
    JF = obj_dict["JF"]
    Jf = obj_dict["Jf"]
    Jlength = obj_dict["Jlength"]
    Jlength2 = obj_dict["Jlength2"]
    Jmscs = obj_dict["Jmscs"]
    Jls = obj_dict["Jls"]
    Jls_TF = obj_dict["Jls_TF"]
    Jccdist = obj_dict["Jccdist"]
    Jccdist2 = obj_dict["Jccdist2"]
    Jcsdist = obj_dict["Jcsdist"]
    linkNum = obj_dict["linkNum"]
    Jforce = obj_dict["Jforce"]
    Jforce2 = obj_dict["Jforce2"]
    Jtorque = obj_dict["Jtorque"]
    Jtorque2 = obj_dict["Jtorque2"]
    length_weight = weight_dict["length_weight"]
    cc_weight = weight_dict["cc_weight"]
    cs_weight = weight_dict["cs_weight"]
    link_weight = weight_dict["link_weight"]
    force_weight = weight_dict["force_weight"]
    net_force_weight = weight_dict["net_force_weight"]
    torque_weight = weight_dict["torque_weight"]
    net_torque_weight = weight_dict["net_torque_weight"]

    # Set the overall objective degrees of freedom
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()

    # Get all the individual objective values and print them
    jf = Jf.J()
    length_val = length_weight * Jlength.J()
    length_val2 = length_weight * Jlength2.J()
    cc_val = cc_weight * Jccdist.J()
    cc_val2 = cc_weight * Jccdist.J()
    cs_val = cs_weight * Jcsdist.J()
    link_val = link_weight * linkNum.J()
    forces_val = Jforce.J()
    forces_val2 = Jforce2.J()
    torques_val = Jtorque.J()
    torques_val2 = Jtorque2.J()
    BdotN = np.mean(np.abs(np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
    BdotN_over_B = np.mean(np.abs(np.sum(btot.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))
                           ) / np.mean(btot.AbsB())
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨B·n⟩={BdotN:.1e}, ⟨B·n⟩/⟨B⟩={BdotN_over_B:.1e}"
    valuestr = f"J={J:.2e}, Jf={jf:.2e}"
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls_TF])
    cl_string2 = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.2f}" for c in base_curves_TF)
    msc_string = ", ".join(f"{J.J():.2f}" for J in Jmscs)
    outstr += f", ϰ=[{kap_string}], ∫ϰ²/L=[{msc_string}]"
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls_TF):.2f}"
    outstr += f", LenDipoles=sum([{cl_string2}])={sum(J.J() for J in Jls):.2f}"
    valuestr += f", LenDipoles={length_val2:.2e}"
    valuestr += f", LenObj={length_val:.2e}"
    valuestr += f", ccObj={cc_val:.2e}"
    valuestr += f", ccObj2={cc_val2:.2e}"
    valuestr += f", csObj={cs_val:.2e}"
    valuestr += f", Lk1Obj={link_val:.2e}"
    valuestr += f", forceObj={force_weight * forces_val:.2e}"
    valuestr += f", forceObj2={net_force_weight * forces_val2:.2e}"
    valuestr += f", torqueObj={torque_weight * torques_val:.2e}"
    valuestr += f", torqueObj2={net_torque_weight * torques_val2:.2e}"
    outstr += f", F={forces_val:.2e}"
    outstr += f", Fnet={forces_val2:.2e}"
    outstr += f", T={torques_val:.2e}"
    outstr += f", Tnet={torques_val2:.2e}"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f}, C-C-Sep2={Jccdist2.shortest_distance():.2f}, C-S-Sep={Jcsdist.shortest_distance():.2f}"
    outstr += f", Link Number = {linkNum.J()}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    print(valuestr)
    return J, grad


def save_coil_sets(btot, OUT_DIR, file_suffix, a, b, nturns_TF, aa, bb, nturns):
    """
    Save separate TF and dipole array coil sets for a dipole array solution.

    """
    from simsopt.field.force import pointData_forces_torques, coil_net_torques, coil_net_forces
    from simsopt.field import regularization_rect
    from simsopt.geo import curves_to_vtk
    bs = btot.Bfields[0]
    bs_TF = btot.Bfields[1]
    coils = bs.coils
    coils_TF = bs_TF.coils
    allcoils = coils + coils_TF
    dipole_currents = [c.current.get_value() for c in coils]
    curves_to_vtk(
        [c.curve for c in bs.coils],
        OUT_DIR + "dipole_curves" + file_suffix,
        close=True,
        extra_point_data=pointData_forces_torques(coils, allcoils, np.ones(len(coils)) * aa, np.ones(len(coils)) * bb, np.ones(len(coils)) * nturns),
        I=dipole_currents,
        NetForces=coil_net_forces(coils, allcoils, regularization_rect(np.ones(len(coils)) * aa, np.ones(len(coils)) * bb), np.ones(len(coils)) * nturns),
        NetTorques=coil_net_torques(coils, allcoils, regularization_rect(np.ones(len(coils)) * aa, np.ones(len(coils)) * bb), np.ones(len(coils)) * nturns),
    )
    curves_to_vtk(
        [c.curve for c in bs_TF.coils],
        OUT_DIR + "TF_curves" + file_suffix,
        close=True,
        extra_point_data=pointData_forces_torques(coils_TF, allcoils, np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b, np.ones(len(coils_TF)) * nturns_TF),
        I=[c.current.get_value() for c in coils_TF],
        NetForces=coil_net_forces(coils_TF, allcoils, regularization_rect(np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b), np.ones(len(coils_TF)) * nturns_TF),
        NetTorques=coil_net_torques(coils_TF, allcoils, regularization_rect(np.ones(len(coils_TF)) * a, np.ones(len(coils_TF)) * b), np.ones(len(coils_TF)) * nturns_TF),
    )
    print('Max I = ', np.max(np.abs(dipole_currents)))
    print('Min I = ', np.min(np.abs(dipole_currents)))
