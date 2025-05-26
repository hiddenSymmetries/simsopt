.. role:: bash(code)
   :language: bash

.. role:: python(code)
   :language: python

Single stage optimization
============================

In this tutorial it is shown how to optimize the boundary shape of a
VMEC configuration to achieve quasisymmetry at the same time that the
shape of coils are optimized to reproduce the VMEC field. The approach
employed here follows the work in the paper `arXiv:2302.10622
<https://arxiv.org/pdf/2302.10622>`__ and includes both a vacuum and a
finite beta implementation. This generalizes the "stage 1" and "stage 2"
approaches in previous tutorials to a "single stage" approach. In this
case, instead of optimizing the :obj:`simsopt.mhd.QuasisymmetryRatioResidual`
or the coil optimization residual :obj:`simsopt.geo.SquaredFlux` individually,
they are both summed together to optimize a single objective function
(cost function) given by

.. math::
  
  J = \text{Quasisymmetry} + \text{coils_objective_weight}*\frac{1}{2} \int |\vec{B} \cdot \vec{n}|^2 ds
      \\+ \text{coil regularization terms}.

where the coil regularization terms are a sum of :obj:`simsopt.geo.CurveLength` for
the length of the coils, :obj:`simsopt.geo.CurveCurveDistance` for the distance
between coils, :obj:`simsopt.geo.LpCurveCurvature` for the coil curvature,
:obj:`simsopt.geo.MeanSquaredCurvature` for the coil mean squared curvature and
:obj:`simsopt.geo.ArclengthVariation` for the arclength variation of the coils.

In order to minimize the number of single stage iterations, in this script,
an initial stage 2 optimization is performed.
The finite beta optimization script requires the ``Virtual Casing`` module
from `the hiddenSymmetries repository <https://github.com/hiddenSymmetries/virtual-casing>`_.


Vacuum optimization
-------------------

This example is available at :simsopt_file:`examples/3_Advanced/single_stage_optimization.py`.
As usual, a driver script begins with imports of the classes and functions we will need

.. code-block::

    import os
    import numpy as np
    from mpi4py import MPI
    from pathlib import Path
    from scipy.optimize import minimize
    from simsopt.util import MpiPartition
    from simsopt._core.derivative import Derivative
    from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
    from simsopt._core.finite_difference import MPIFiniteDifference
    from simsopt.field import BiotSavart, Current, coils_via_symmetries
    from simsopt.objectives import SquaredFlux, QuadraticPenalty, LeastSquaresProblem
    from simsopt.geo import (CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                            LpCurveCurvature, ArclengthVariation, curves_to_vtk, create_equally_spaced_curves)


For a description of each class and function imported, please see the stage 1
script at :simsopt_file:`examples/2_Intermediate/QH_fixed_resolution.py` 
and the stage 2 script at :simsopt_file:`examples/1_Simple/stage_two_optimization_minimal.py`.
For convenience, a ``print`` function is defined that only prints once even when
the code is ran in parallel:

.. code-block::

    def pprint(*args, **kwargs):
        if comm.rank == 0:
            print(*args, **kwargs)

The input parameters are given by 

1. :python:`MAXITER_stage_2 = 10`, which selects the maximum number of 
   iterations for the initial stage 2 optimization.

2. :python:`MAXITER_single_stage = 10`, which selects the maximum number 
   of iterations for the single stage optimization.

3. :python:`max_mode = 1`, which selects the maximum poloidal and 
   toroidal modes on the surface being optimized with VMEC.

4. :python:`vmec_input_filename = os.path.join(parent_path, 'inputs', 'input.nfp4_QH_warm_start')`, which represents the initial VMEC input file.

5. :python:`ncoils = 3`:  the number of coils per field period.

6. :python:`aspect_ratio_target = 7.0`:  target aspect ratio for the VMEC surface.

7. :python:`coils_objective_weight = 1e+3`: the weight given to the coils 
   objective function with respect to the stage 1 optimization.

The remaining input parameters follow the convention of the
stage 2 optimization script.

Then, the results directory are created to hold the VMEC configurations
and the coils

.. code-block::

    directory = 'optimization_QH'
    vmec_verbose = False
    # Create output directories
    this_path = os.path.join(parent_path, directory)
    os.makedirs(this_path, exist_ok=True)
    os.chdir(this_path)
    vmec_results_path = os.path.join(this_path, "vmec")
    coils_results_path = os.path.join(this_path, "coils")
    if comm.rank == 0:
        os.makedirs(vmec_results_path, exist_ok=True)
        os.makedirs(coils_results_path, exist_ok=True)

The function ``fun_coils`` returns the objective function and gradients
used in the initial stage 2 optimization, while the ``fun`` function
returns the objective function and gradients used in the single stage
optimization. In this function, the derivatives with respect to the coils
and to the surface are computed separately. The derivatives with respect
to the coils are analytical, while the derivatives with respect to the surface
are a mix of analytical (defined as ``mixed_dJ``) and finite-diference
derivatives

.. code-block::

    def fun(dofs, prob_jacobian=None, info={'Nfeval': 0}):
        info['Nfeval'] += 1
        JF.x = dofs[:-number_vmec_dofs]
        prob.x = dofs[-number_vmec_dofs:]
        bs.set_points(surf.gamma().reshape((-1, 3)))
        os.chdir(vmec_results_path)
        J_stage_1 = prob.objective()
        J_stage_2 = coils_objective_weight * JF.J()
        J = J_stage_1 + J_stage_2
        if J > JACOBIAN_THRESHOLD or np.isnan(J):
            pprint(f"Exception caught during function evaluation with J={J}."
                   f" Returning J={JACOBIAN_THRESHOLD}")
            J = JACOBIAN_THRESHOLD
            grad_with_respect_to_surface = [0] * number_vmec_dofs
            grad_with_respect_to_coils = [0] * len(JF.x)
        else:
            pprint(f"fun#{info['Nfeval']}: Objective function = {J:.4f}")
            prob_dJ = prob_jacobian.jac(prob.x)
            ## Finite differences for the second-stage objective function
            coils_dJ = JF.dJ()
            ## Mixed term - derivative of squared flux with respect to the surface shape
            n = surf.normal()
            absn = np.linalg.norm(n, axis=2)
            B = bs.B().reshape((nphi_VMEC, ntheta_VMEC, 3))
            dB_by_dX = bs.dB_by_dX().reshape((nphi_VMEC, ntheta_VMEC, 3, 3))
            Bcoil = bs.B().reshape(n.shape)
            unitn = n * (1./absn)[:, :, None]
            Bcoil_n = np.sum(Bcoil*unitn, axis=2)
            mod_Bcoil = np.linalg.norm(Bcoil, axis=2)
            B_n = Bcoil_n
            B_diff = Bcoil
            B_N = np.sum(Bcoil * n, axis=2)
            assert Jf.definition == "local"
            dJdx = (B_n/mod_Bcoil**2)[:, :, None] * (np.sum(dB_by_dX*(n-B*(B_N/mod_Bcoil**2)[:, :, None])[:, :, None, :], axis=3))
            dJdN = (B_n/mod_Bcoil**2)[:, :, None] * B_diff - 0.5 * (B_N**2/absn**3/mod_Bcoil**2)[:, :, None] * n
            deriv = surf.dnormal_by_dcoeff_vjp(dJdN/(nphi_VMEC*ntheta_VMEC)) + surf.dgamma_by_dcoeff_vjp(dJdx/(nphi_VMEC*ntheta_VMEC))
            mixed_dJ = Derivative({surf: deriv})(surf)
            ## Put both gradients together
            grad_with_respect_to_coils = coils_objective_weight * coils_dJ
            grad_with_respect_to_surface = np.ravel(prob_dJ) + coils_objective_weight * mixed_dJ
        grad = np.concatenate((grad_with_respect_to_coils, grad_with_respect_to_surface))
        return J, grad


The initial stage 2 optimization is then performed at the line

.. code-block::

    res = minimize(fun_coils, dofs[:-number_vmec_dofs], jac=True,
                   args=({'Nfeval': 0}), method='L-BFGS-B',
                   options={'maxiter': MAXITER_stage_2, 'maxcor': 300},
                   tol=1e-12)

while the single stage optimization is performed at

.. code-block::

    with MPIFiniteDifference(prob.objective, mpi,
                             diff_method=diff_method,
                             abs_step=finite_difference_abs_step,
                             rel_step=finite_difference_rel_step) as prob_jacobian:
        if mpi.proc0_world:
            res = minimize(fun, dofs,
                           args=(prob_jacobian, {'Nfeval': 0}),
                           jac=True, method='BFGS',
                           options={'maxiter': MAXITER_single_stage},
                           tol=1e-15)

The results are then printed and stored in files.


Finite beta optimization
-------------------------

The finite beta generalization example is available at
:simsopt_file:`examples/3_Advanced/single_stage_optimization_finite_beta.py`.
In addition to the parameters in the
previous example, the finite beta script uses the Virtual Casing principle
to decouple the plasma magnetic field from the coil magnetic field.
The VirtualCasing module is imported in

.. code-block::

    from simsopt.mhd import Vmec, QuasisymmetryRatioResidual, VirtualCasing

and its resolution is set in :python:`vc_src_nphi = ntheta_VMEC`.

The initialization of the VirtualCasing is performed at the line

.. code-block::

    vc = VirtualCasing.from_vmec(
        vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC,
        trgt_ntheta=ntheta_VMEC)
    total_current_vmec = vmec.external_current() / (2 * surf.nfp)

Now the gradients of the objective function are computed using
finite differences instead of a mix of analytical and finite difference
derivatives. The objective function is then wrapped in the ``fun_J`` function

.. code-block::

    def fun_J(prob, coils_prob):
        global previous_surf_dofs
        J_stage_1 = prob.objective()
        if np.any(previous_surf_dofs != prob.x):  # Only run virtual casing if surface dofs have changed
            previous_surf_dofs = prob.x
            try:
                vc = VirtualCasing.from_vmec(
                    vmec, src_nphi=vc_src_nphi, trgt_nphi=nphi_VMEC,
                    trgt_ntheta=ntheta_VMEC)
                Jf.target = vc.B_external_normal
            except ObjectiveFailure as e:
                pass

        bs.set_points(surf.gamma().reshape((-1, 3)))
        J_stage_2 = coils_objective_weight * JF.J()
        J = J_stage_1 + J_stage_2
        return J


And the resulting objective function and gradients are computed using
the ``fun`` function

.. code-block::

    def fun(dofss, prob_jacobian, info={'Nfeval': 0}):
        info['Nfeval'] += 1
        os.chdir(vmec_results_path)
        prob.x = dofss[-number_vmec_dofs:]
        coil_dofs = dofss[:-number_vmec_dofs]
        # Un-fix the desired coil dofs so they can be updated:
        JF.full_unfix(free_coil_dofs)
        JF.x = coil_dofs
        J = fun_J(prob, JF)
        if J > JACOBIAN_THRESHOLD or isnan(J):
            pprint(f"fun#{info['Nfeval']}: Exception caught during function evaluation with J={J}. Returning J={JACOBIAN_THRESHOLD}")
            J = JACOBIAN_THRESHOLD
            grad_with_respect_to_surface = [0] * number_vmec_dofs
            grad_with_respect_to_coils = [0] * len(coil_dofs)
        else:
            pprint(f"fun#{info['Nfeval']}: Objective function = {J:.4f}")
            coils_dJ = JF.dJ()
            grad_with_respect_to_coils = coils_objective_weight * coils_dJ
            JF.fix_all()  # Must re-fix the coil dofs before beginning the finite differencing.
            grad_with_respect_to_surface = prob_jacobian.jac(prob.x)[0]

        JF.fix_all()
        grad = np.concatenate((grad_with_respect_to_coils,
                               grad_with_respect_to_surface))

        return J, grad


The initial stage 2 optimization and single stage optimization follow
the previous vacuum case, with the exception of the lines

.. code-block::

    dofs[:-number_vmec_dofs] = res.x
    JF.x = dofs[:-number_vmec_dofs]
    mpi.comm_world.Bcast(dofs, root=0)
    opt = make_optimizable(fun_J, prob, JF)
    free_coil_dofs = JF.dofs_free_status
    JF.fix_all()


where the coils and surface degrees of freedom are defined and MPI broadcasted
and

.. code-block::

    JF.full_unfix(free_coil_dofs)  # Needed to evaluate JF.dJ

where the coils degrees of freedom are unfixed to evaluate their Jacobian.
