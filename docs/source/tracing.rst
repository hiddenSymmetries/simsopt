Field line and particle tracing
===============================

The :obj:`simsopt.field.tracing` submodule provides tools for tracing
field lines, and for following particle motion in magnetic fields.
These tools could be accessed from the :obj:`simsopt.field` module.
For the latter, full orbits (including gyromotion) or guiding center
trajectories can be followed in cylindrical coordinates, or guiding
center motion can be followed in Boozer coordinates.  Examples of
these various tracing features can be found in
:simsopt_file:`examples/1_Simple/tracing_fieldline.py`,
:simsopt_file:`examples/1_Simple/tracing_particle.py`, and
:simsopt_file:`examples/2_Intermediate/tracing_boozer.py`,



Particle motion in cylindrical coordinates
------------------------------------------

The main function to use in this case is
:obj:`simsopt.field.trace_particles` (click the link for more
information on the input and output parameters) and it is able to use
two different sets of equations depending on the input parameter
``mode``:

- In the case of ``mode='full'`` it solves

.. math::

  [\ddot x, \ddot y, \ddot z] = \frac{q}{m}  [\dot x, \dot y, \dot z] \times \mathbf B

- In the case of ``mode='gc_vac'`` it solves the guiding center equations under the assumption :math:`\nabla p=0`, that is

.. math::

  [\dot x, \dot y, \dot z] &= v_{||}\frac{\mathbf B}{B} + \frac{m}{q|B|^3}  \left(\frac{v_\perp^2}{2} + v_{||}^2\right)  \mathbf B\times \nabla B\\
  \dot v_{||}    &= -\frac{\mu}{B}  \mathbf B \cdot \nabla B

where :math:`v_\perp^2 = 2\mu B`.
See equations (12) and (13) of `Guiding Center Motion, H.J. de Blank <https://doi.org/10.13182/FST04-A468>`_.

Below is an example of the vertical drift experienced by two particles in a simple toroidal magnetic field.

.. code-block::

    from simsopt.field import ToroidalField
    from simsopt.geo import CurveXYZFourier
    from simsopt.util.constants import PROTON_MASS, ELEMENTARY_CHARGE, ONE_EV
    from simsopt.field import trace_particles_starting_on_curve
    # NOTE: Most of the functions and classes implemented in the tracing
    # NOTE: submodule can be imported directly from the field module

    bfield = ToroidalField(B0=1., R0=1.)
    start_curve = CurveXYZFourier(300, 1)
    start_curve.set_dofs([0, 0, 1.01, 0, 1.01, 0., 0, 0., 0.])
    nparticles = 2
    m = PROTON_MASS
    q = ELEMENTARY_CHARGE
    tmax = 1e-6
    Ekin = 100*ONE_EV
    gc_tys, gc_phi_hits = trace_particles_starting_on_curve(
        start_curve, bfield, nparticles, tmax=tmax, seed=1, mass=m, charge=q,
        Ekin=Ekin, umin=0.2, umax=0.5, phis=[], mode='gc_vac', tol=1e-11)
    z_particle_1 = gc_tys[0][:][2]
    z_particle_2 = gc_tys[1][:][2]
    print(z_particle_1)
    print(z_particle_2)


We note that SIMSOPT draws initial data for particles consisting of
the guiding center position, the parallel velocity, and the
perpendicular speed.

* To compute the speed, the user specifies the total kinetic energy and an interval of pitch-angles :math:`[u_\min, u_\max]`. Given a pitch angle :math:`u` and total speed :math:`v` (computed from the kinetic energy and the particle mass), the parallel speed is given by :math:`v_{||} = u v` and, and :math:`v_\perp = \sqrt{v^2-v_{||}^2}`.
* In the case of full orbit simulations, we need velocity initial data and hence this only defines the initial data up to the phase. To specify the phase, one can pass the ``phase_angle`` variable to the tracing functions. A value in :math:`[0, 2\pi]` is expected.
