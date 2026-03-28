.. _example_boozer_finitebeta:

Finite-Beta Boozer Surfaces
===========================

This tutorial documents the finite-beta Boozer-surface workflow introduced
alongside the existing vacuum :simsopt_file:`examples/2_Intermediate/boozerQA.py`
example. The corresponding script is
:simsopt_file:`examples/2_Intermediate/boozerQA_finitebeta.py`.

The current implementation is intentionally staged. It already supports a
finite-beta least-squares solve on a given surface for

- the rotational transform ``iota``,
- the Boozer-current parameter ``G``,
- the internal current ``I``, and
- a single-valued surface-current potential ``Phi``.

The example script is intentionally kept close to the structure of
:simsopt_file:`examples/2_Intermediate/boozerQA.py`: a flat top-level flow with
the prescribed-field QA path as the default branch, plus two explicit alternate
modes for fixed-field and VMEC-backed runs.

The residual blocks enforced in this solve are

.. math::

   (G + \iota I) B_{\mathrm{in}} - |B_{\mathrm{in}}|^2 (x_\varphi + \iota x_\theta) = 0,

.. math::

   B_{\mathrm{out}} \cdot \hat{n} = 0,

.. math::

   |B_{\mathrm{out}}|^2 - |B_{\mathrm{in}}|^2 - 2 \mu_0 \Delta p = 0,

.. math::

   \mu_0 K - \hat{n} \times (B_{\mathrm{out}} - B_{\mathrm{in}}) = 0,

with

.. math::

   K = \hat{n} \times \nabla \Phi.

Three modes are presently exposed in the example script.

Prescribed-field QA mode
------------------------

The default example path is a self-contained prescribed-field problem. In this
mode the magnetic fields are provided by callables that can be reevaluated when
the surface moves, so the script can perform an actual surface-optimization
loop around :obj:`~simsopt.geo.FiniteBetaBoozerSurface` without any ideal-MHD
solver in the optimization loop.

One validation command is

.. code-block:: bash

   SIMSOPT_FINITE_BETA_MODE=prescribed \
   SIMSOPT_FINITE_BETA_NPHI=8 \
   SIMSOPT_FINITE_BETA_NTHETA=8 \
   SIMSOPT_FINITE_BETA_MAX_NFEV=20 \
   SIMSOPT_FINITE_BETA_QA_MAXITER=5 \
   python examples/2_Intermediate/boozerQA_finitebeta.py

This prescribed-field branch is the current end-to-end moving-surface QA
demonstration.
The ``SIMSOPT_FINITE_BETA_QA_MAXITER`` variable controls the outer L-BFGS-B
iterations, while ``SIMSOPT_FINITE_BETA_MAX_NFEV`` controls the inner
finite-beta least-squares solves.

Prescribed fixed-field solve mode
---------------------------------

Set ``SIMSOPT_FINITE_BETA_MODE=prescribed-fixed`` to run the same prescribed
surface-field setup with explicit ``B_in`` and ``B_out`` arrays held fixed on
the surface quadrature grid. In this mode the example exercises the analytic
surface-dof Jacobian inside :obj:`~simsopt.geo.FiniteBetaBoozerSurface.run_code`.

One validation command is

.. code-block:: bash

   SIMSOPT_FINITE_BETA_MODE=prescribed-fixed \
   SIMSOPT_FINITE_BETA_NPHI=8 \
   SIMSOPT_FINITE_BETA_NTHETA=8 \
   SIMSOPT_FINITE_BETA_MAX_NFEV=20 \
   python examples/2_Intermediate/boozerQA_finitebeta.py

This mode is useful when the surface fields are the given optimization data and
no field reevaluation is needed when the surface moves.

Auxiliary VMEC-backed mode
--------------------------

Set ``SIMSOPT_FINITE_BETA_MODE=vmec`` to load a VMEC equilibrium and
virtual-casing data, fit a :obj:`~simsopt.geo.SurfaceXYZTensorFourier`
representation to the VMEC boundary, and solve the finite-beta residual system
on that fixed surface.

This VMEC-backed branch is an auxiliary validation and diagnostics workflow,
not the main optimization path. It is useful when an external equilibrium is
available and you want to generate or compare surface field data.

One VMEC-backed validation command is

.. code-block:: bash

   SIMSOPT_FINITE_BETA_MODE=vmec \
   SIMSOPT_FINITE_BETA_NPHI=8 \
   SIMSOPT_FINITE_BETA_NTHETA=8 \
   SIMSOPT_FINITE_BETA_MAX_NFEV=20 \
   python examples/2_Intermediate/boozerQA_finitebeta.py

With such reduced smoke-test settings, this branch may stop because the maximum
number of function evaluations is reached before convergence. That is expected
for a quick validation run. Increase ``SIMSOPT_FINITE_BETA_MAX_NFEV`` when you
want a tighter fixed-surface solve from VMEC-backed data.

You can override the equilibrium with ``SIMSOPT_FINITE_BETA_VMEC=/path/to/input.file``
and set a constant pressure jump using ``SIMSOPT_FINITE_BETA_PRESSURE_JUMP=value``.
If you already have cached virtual-casing data, set
``SIMSOPT_FINITE_BETA_VCASING=/path/to/vcasing_*.nc`` to use that file directly.
For backward compatibility, ``SIMSOPT_FINITE_BETA_SYNTHETIC=1`` still selects
the prescribed-field QA mode.

Relevant APIs
-------------

- :obj:`~simsopt.geo.FiniteBetaBoozerSurface`
- :obj:`~simsopt.geo.finite_beta_boozer_residual`
- :obj:`~simsopt.geo.finite_beta_sheet_current`
- :obj:`~simsopt.geo.surface_field_nonquasisymmetric_ratio`

For the public geometry and API reference, see :doc:`geo` and :doc:`simsopt.geo`.