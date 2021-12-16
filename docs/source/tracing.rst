Field line and particle tracing
===============================

Simsopt is able to follow particles in a magnetic field. The main function to use in this case is :obj:`simsopt.field.tracing.trace_particles` (click the link for more information on the input and output parameters) and it is able to use two different sets of equations depending on the input parameter ``mode``:

- In the case of ``mode='full'`` it solves

.. math::

  [\ddot x, \ddot y, \ddot z] = \frac{q}{m}  [\dot x, \dot y, \dot z] \times \mathbf B

- In the case of ``mode='gc_vac'`` it solves the guiding center equations under the assumption :math:`\nabla p=0`, that is

.. math::

  [\dot x, \dot y, \dot z] &= v_{||}\frac{\mathbf B}{B} + \frac{m}{q|B|^3}  \left(\frac{v_\perp^2}{2} + v_{||}^2\right)  \mathbf B\times \nabla B\\
  \dot v_{||}    &= -\mu  \mathbf B \cdot \nabla B

where :math:`v_\perp^2 = 2\mu B`.
See equations (12) and (13) of `Guiding Center Motion, H.J. de Blank <https://doi.org/10.13182/FST04-A468>`_.

Below is an example of the vertical drift experienced by two particles in a simple toroidal magnetic field.

.. code-block::

    from simsopt.field.magneticfieldclasses import ToroidalField
    from simsopt.geo.curvexyzfourier import CurveXYZFourier
    from simsopt.util.constants import PROTON_MASS, ELEMENTARY_CHARGE, ONE_EV
    from simsopt.field.tracing import trace_particles_starting_on_curve

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


We note that SIMSOPT draws initial data for particles consisting of the guiding center position, the parallel, and the perpendicular speed.

* To compute the speed, the user specifies the total kinetic energy and an interval of pitch-angles :math:`[u_\min, u_\max]`. Given a pitch angle :math:`u` and total speed :math:`v` (computed from the kinetic energy and the particle mass), the parallel speed is given by :math:`v_{||} = u v` and, and :math:`v_\perp = \sqrt{v^2-v_{||}^2}`.
* In the case of full orbit simulations, we need velocity initial data and hence this only defines the initial data up to the phase. To specify the phase, one can pass the ``phase_angle`` variable to the tracing functions. A value in :math:`[0, 2\pi]` is expected.
