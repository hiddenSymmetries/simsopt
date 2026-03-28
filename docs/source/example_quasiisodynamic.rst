.. _example_quasiisodynamic:

Optimizing for quasi-isodynamicity on a Boozer surface
======================================================

This tutorial introduces the milestone-1 quasi-isodynamic Boozer-surface
objective implemented in :obj:`simsopt.geo.NonQuasiIsodynamicRatio` and the
example script :simsopt_file:`examples/2_Intermediate/boozerQI.py`.

The workflow is intentionally close to :simsopt_file:`examples/2_Intermediate/boozerQA.py`:

- a coil-generated Biot-Savart field is used,
- a single :obj:`~simsopt.geo.BoozerSurface` is computed near the magnetic axis,
- the coils are optimized on that surface,
- penalty terms keep the rotational transform, major radius, and total coil length near their initial values.

The key difference is the symmetry objective. Instead of minimizing a
quasi-axisymmetric ratio based on averaging :math:`|B|` along one coordinate,
the QI objective compares each sampled magnetic well against a shuffled target
that equalizes weighted bounce distances across field lines.


Objective definition
--------------------

For one Boozer surface, let

.. math::

   \phi_B = 2\pi\varphi,
   \qquad
   \theta_B = 2\pi\theta,

and sample one field period

.. math::

   \phi_B \in [\phi_{\min}, \phi_{\min} + 2\pi / n_{fp}].

For field-line labels

.. math::

   \alpha_j = 2\pi j / n_{\alpha},

the line is sampled at

.. math::

   \theta_B(\phi_B; \alpha_j) = \alpha_j + \iota (\phi_B - \phi_{\min}),

and the field strength is

.. math::

   B_j(\phi_B) = |B(\phi_B, \theta_B(\phi_B; \alpha_j))|.

The sampled field strengths are normalized globally on the auxiliary Boozer
surface,

.. math::

   \widehat{B}_j(\phi_B) = \frac{B_j(\phi_B) - B_{\min}}{\max(B_{\max} - B_{\min}, \varepsilon_B)}.

For each field line, the magnetic well is split at its minimum, transformed with
the legacy left and right squash rules, stretched using cosine-power profiles,
and converted into bounce branches using the same crossing logic as the legacy
``GetBranches`` routine from the old omnigenity optimization code.

The branch locations are then shifted so the weighted average bounce distance is
shared across field lines, producing a shuffled target
:math:`\widehat{B}_{QI,j}`. The residual samples are

.. math::

   r_{jm} = \frac{\widehat{B}_{QI,j}(\phi_m) - \widehat{B}_j(\phi_m)}{\sqrt{n_{\alpha} n_{\phi,\mathrm{out}}}},

and the milestone-1 scalar objective is

.. math::

   J_{QI} = \sum_{j,m} r_{jm}^2.


How the example is structured
-----------------------------

The example performs the following sequence:

1. Load the NCSX coil set from :obj:`simsopt.configs.get_data`.
2. Build an initial :obj:`~simsopt.geo.SurfaceXYZTensorFourier` surface.
3. Solve for a Boozer surface using :obj:`~simsopt.geo.BoozerSurface`.
4. Build a total objective from:

   - :obj:`~simsopt.geo.NonQuasiIsodynamicRatio`
   - :obj:`~simsopt.geo.Iotas`
   - :obj:`~simsopt.geo.MajorRadius`
   - total coil-length penalties.

5. Run a BFGS optimization.

In normal interactive use the script writes VTK files for the initial and final
coils and surfaces. For testing or quick checks, runtime can be reduced with
environment variables.


Runtime controls
----------------

The example supports these environment variables:

- ``SIMSOPT_BOOZER_QI_MPOL`` and ``SIMSOPT_BOOZER_QI_NTOR`` to reduce the Boozer-surface representation.
- ``SIMSOPT_BOOZER_QI_SDIM`` to reduce the auxiliary QI quadrature surface.
- ``SIMSOPT_BOOZER_QI_NPHI``, ``SIMSOPT_BOOZER_QI_NALPHA``, ``SIMSOPT_BOOZER_QI_NBJ``, and ``SIMSOPT_BOOZER_QI_NPHI_OUT`` to reduce the QI residual resolution.
- ``SIMSOPT_BOOZER_QI_MAXITER`` to limit optimization iterations.
- ``SIMSOPT_BOOZER_QI_SKIP_TAYLOR=1`` to skip the expensive Taylor-test block.
- ``SIMSOPT_BOOZER_QI_WRITE_VTK=0`` to suppress output files.
- ``SIMSOPT_BOOZER_QI_OUT_DIR`` to choose an output directory.
- ``SIMSOPT_BOOZER_QI_VMEC`` to initialize the starting surface from a VMEC boundary shape.

An example reduced-runtime invocation is::

   SIMSOPT_BOOZER_QI_WRITE_VTK=0 \
   SIMSOPT_BOOZER_QI_SKIP_TAYLOR=1 \
   SIMSOPT_BOOZER_QI_MAXITER=0 \
   SIMSOPT_BOOZER_QI_MPOL=3 \
   SIMSOPT_BOOZER_QI_NTOR=3 \
   SIMSOPT_BOOZER_QI_SDIM=6 \
   SIMSOPT_BOOZER_QI_NPHI=21 \
   SIMSOPT_BOOZER_QI_NALPHA=3 \
   SIMSOPT_BOOZER_QI_NBJ=5 \
   SIMSOPT_BOOZER_QI_NPHI_OUT=21 \
   python examples/2_Intermediate/boozerQI.py


Current limitations
-------------------

This milestone-1 implementation is intentionally narrow:

- It operates on a single Boozer surface in a coil-generated Biot-Savart field.
- It does not use ``booz_xform`` or a VMEC quasisymmetry objective.
- The current derivative of :obj:`~simsopt.geo.NonQuasiIsodynamicRatio` is a finite-difference coil gradient, which is suitable for smoke tests and experimentation but slower than an adjoint implementation.
- The optional VMEC initializer is only a geometry initializer for the starting surface. The magnetic field still comes from the coils, so the VMEC configuration must have the same number of field periods as the coil set.

These limitations keep the new objective close to the existing Boozer-surface
optimization workflow while preserving parity with the legacy single-surface QI
residual.