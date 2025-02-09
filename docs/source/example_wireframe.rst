Wireframe optimization
======================

This tutorial demonstrates several examples of optimizations on wireframes
using the Regularized Constrained Least Squaraes (RCLS) or the Greedy 
Stellarator Coil Optimization (GSCO) techniques. Further details about these
methods may be found in the paper `A framework for discrete optimization of
stellarator coils <https://arxiv.org/abs/2412.00267>`__.

All examples here make use of the ``ToroidalWireframe`` class to formulate
the coil current distribution as a set of interconnected straight segments
of wire, for which the optimizable parameters are the currents carried by
each segment. Each optimization technique upholds the requirement for
current continuity; i.e. charge will not accumulate at the nodes where 
segments intersect.

Regularized Constrained Least Squares (RCLS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The RCLS technique uses linear least-squares methods to optimize the segment
currents with Tikhonov regularization, subject to constraints that include
current continuity.

The RCLS objective function to be minimized, subject to the constraints,
is defined as follows:

.. math::  
  J = \frac{1}{2} 
      \int \left| \left(\vec{B}_{wireframe} 
                  - \vec{B}_{ext}\right) \cdot \vec{n}\right|^2 ds
          + \frac{1}{2}\left( \mathbf{W}\vec{x} \right)^2.

Here, :math:`\vec{B}_{wireframe}` is the magnetic field produced by the 
wireframe, :math:`\vec{B}_{ext}` is the field produced by all other sources 
including plasma currents and other coils, :math:`\mathbf{W}` is a 
regularization matrix, and :math:`\vec{x}` contains the currents in each
wireframe segment. The first term in :math:`J` is the typical field accuracy
objective, while the second term is a Tikhonov regularization objective
that penalizes large currents in the segments. Often, the regularization
matrix :math:`\mathbf{W}` is replaced with a constant factor (in which case all
segment currents are penalized by the same factor) or is a diagonal matrix
(in which case different segment currents may be penalized by different
amounts).

The constraints that may be applied to the wireframe are as follows:

- Continuity constraints. These enforce current continuity at each node
  of the wireframe where segments intersect. The constraints require the 
  current flowing into each node to equal the current flowing out of it
  such that charge does not accumulate at the nodes. These constraints are
  automatically incorporated any time a wireframe is constructed, so there
  is no need for the user to add them manually.
- Poloidal current constraint. This requires the net poloidal current in the
  wireframe to have a specified value. Specifying the net poloidal current in
  this way guarantees a certain average value of the toroidal magnetic field 
  within the wireframe at a certain major radius.
- Toroidal current constraint. This requires the net toroidal current in the
  wireframe to have a specified value.
- Segment constraints. These constraints require the currents in specified
  segments to be zero, effectively removing those segments from consideration
  in the optimization. These are helpful for blocking off sections of the 
  wireframe to reserve space for other components, or for preventing current
  from flowing across certain user-imposed boundaries.

The RCLS examples solutions with the following characteristics:

- A basic use case that can be made arbitrarily accurate by increasing
  the wireframe resolution 
- Certain segments of the wireframe are blocked off to 
  leave room for ports modeled by ``Port`` subclasses

Basic RCLS
----------

The first example includes the fundamental elements of a wireframe optimization
using RCLS: creating the wireframe, setting its constraints, setting the
optimization parameters, and running the optimization. It can also be found
in ``/examples/2_Intermediate/wireframe_opt_rcls.py``.

First import the necessary classes, including the ones for creating and
optimizing toroidal wireframes::

  from simsopt.geo import SurfaceRZFourier, ToroidalWireframe
  from simsopt.solve import optimize_wireframe

To construct a ``ToroidalWireframe`` class instance, one must specify its 
geometry and resolution. The geometry is defined accourding to a toroidal
surface, represented by an instance of ``SurfaceRZFourier``,  on which the 
wireframe's nodes are to be placed. In this example, the toroidal surface
is constructed to be a certain uniform distance away from the target
plasma boundary.

The nodes are positioned in a 
two-dimensional grid on the toroidal surface, spaced evenly in the toroidal 
and poloidal angles. The number of nodes is set by two integers, which specify
number of nodes per half-period in the toroidal dimension and the number of
nodes in the poloidal dimension, respectively. The geometry of the wireframe
segments is fully determined by the node positions, as the segments simply
lie on straight lines connecting adjacent pairs of nodes.

The wireframe in the example is constructed as follows::

  # Number of wireframe segments per half period in the toroidal dimension
  wf_nPhi = 8
  
  # Number of wireframe segments in the poloidal dimension
  wf_nTheta = 12
  
  # Distance between the plasma boundary and the wireframe
  wf_surf_dist = 0.3
  
  # File for the desired boundary magnetic surface:
  TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
  filename_equil = TEST_DIR / 'input.LandremanPaul2021_QA'
  
  # Construct the wireframe on a toroidal surface
  surf_wf = SurfaceRZFourier.from_vmec_input(filename_equil)
  surf_wf.extend_via_projected_normal(wf_surf_dist)
  wf = ToroidalWireframe(surf_wf, wf_nPhi, wf_nTheta)

Since target plasma in this case is a vacuum equilibrium with no internal
currents, and the wireframe is the sole magnetic field source, the quantity
:math:`\vec{B}_{ext}` is zero. Hence, one can trivially optimize the 
objective function simply by setting all segment currents to zero
(:math:`\vec{x} = 0`). In order to find a nontrivial solution that produces
a nonzero magnetic field with a minimal normal component on the target plasma
boundary, it will be necessary to set a constraint requiring the net poloidal
current in the wireframe to have a certain value. In the example script, the
value is determined by specifying the desired magnetic field near the magnetic
axis of the plasma equilibrium::

  # Average magnetic field on axis, in Teslas, to be produced by the wireframe.
  # This will be used for setting the poloidal current constraint. The radius
  # of the magnetic axis will be estimated from the plasma boundary geometry.
  field_on_axis = 1.0

  # Load the geometry of the target plasma boundary
  plas_nPhi = 32
  plas_nTheta = 32
  surf_plas = SurfaceRZFourier.from_vmec_input(filename_equil,
                  nphi=plas_nPhi, ntheta=plas_nTheta, range='half period')
  
  # Calculate the required net poloidal current and set it as a constraint
  mu0 = 4.0 * np.pi * 1e-7
  pol_cur = -2.0*np.pi*surf_plas.get_rc(0,0)*field_on_axis/mu0
  wf.set_poloidal_current(pol_cur)

Finally, the optimization parameters must be specified. For RCLS, this is just
the regularization matrix :math:`\mathbf{W}`::

  # Weighting factor for Tikhonov regularization (used instead of a matrix)
  regularization_w = 10**-10.
  
  # Set the optimization parameters
  opt_params = {'reg_W': regularization_w}

In general, :math:`\mathbf{W}` may be an arbitrary matrix. However, for 
simplicity in this example, a simple scalar value will be used. Effectively, 
:math:`\mathbf{W}` is the scalar times the identity matrix, although it is
sufficient to define ``'reg_W'`` as a scalar. Similarly, if one wishes to
supply a diagonal matrix for :math:`\mathbf{W}`, one can simply input the
vector of diagonal elements as a one-dimensional array rather than a full 
matrix.

With all necessary inputs specified, the RCLS procedure may then be run
using the ``optimize_wireframe`` function. With the wireframe parameters 
specified above, the optimization itself should take less than a second to 
perform on a personal computer::

  # Run the RCLS optimization
  res = optimize_wireframe(wf, 'rcls', opt_params, surf_plas=surf_plas,
                           verbose=False)

When the optimization is complete, the ``ToroidalWireframe`` class instance 
(``wf`` in this case) will be updated such that its ``currents`` attribute 
contains the segment currents found by the optimizer. One can verify that the 
solution satisfies all constraints using the ``check_constraints`` method of 
the ``ToroidalWireframe`` class::

  # Verify that the solution satisfies all constraints
  assert wf.check_constraints()

In addition to updating the wireframe class instance, the ``optimize_wireframe``
function returns a dictionary with some key data associated with the
optimization. This includes a ``'wframe_field'``, an instance of 
``WireframeField`` class that representing the magnetic field produced by the
optimized wireframe. The ``WireframeField`` class is a subclass of the 
``MagneticField`` class and can therefore be used for subsequent magnetic 
field calculations.

There are a number of ways to visualize the solution. One way is to generate
a two-dimensional plot of the segment currents using the ``make_plot_2d``
method of the ``ToroidalWireframe`` class::

  # Save plots and visualization data to files
  wf.make_plot_2d(coordinates='degrees')
  pl.savefig(OUT_DIR + 'rcls_wireframe_curr2d.png')

The figure created by this command is shown below. The figure essentially
shows the wireframe unwrapped and flattened, with the toroidal dimension
running horizontally and the poloidal dimension running vertically. Accordingly,
toroidally-oriented wireframe segments are shown as horizontal line segments
on the plot, and poloidally-oriented wireframe segments are shown as 
vertical line segments. By default, a single half-period is plotted; hoever,
the user may plot other amounts of the wireframe with the keyword argument
``'extent'``. The line segments on the plot are color-coded according to
the segment they carry. Red tones represent positive values, while blue
tones represent negative values. For toroidal segments, positive current flows
to the right; for poloidal segments, positive current flows upward.

.. image:: rcls_wireframe_curr2d.png
   :width: 500
	
There are also options for generating a three-dimensional rendering of the
wireframe. With the ``to_vtk`` method, a file will be generated that may
be loaded in ParaView::

  wf.to_vtk(OUT_DIR + 'rcls_wireframe')

Another option is to use the ``make_plot_3d`` method, which generates a 
three-dimensional rendering using the mayavi package (must be installed
separately).

Incorporating ports to avoid
----------------------------

Using the ``PortSet`` class, the wireframe can be constrained to have its
current distribution avoid overlap with an arbitrary set of ports. The ports
may represent actual ports for diagnostics or heating systems, or they could
be used more generally to block out spatial regions for other components.

This example will modify the above example to ensure that the RCLS optimizer
avoids placing currents in segments that overlap a set of ports placed on
the outboard side of the stellarator. The ports will be assumed to have
circular cross-sections and can thus be represented by the ``CircularPort``
class. This example may also be found in the 
file ``/examples/2_Intermediate/wireframe_opt_rcls_with_ports.py``.

First, the general ``PortSet`` and ``CircularPort`` classes must be imported::

  from simsopt.geo import PortSet, CircularPort  

The addition of constraints for avoiding ports at certain locations will, 
in general, reduce the attainable field accuracy at a given wireframe
resolution. Thus, the wireframe resolution will be higher than in the 
previous example in order to achieve a similar level of field accuracy::

  # Number of wireframe segments per half period in the toroidal dimension
  wf_nPhi = 12
  
  # Number of wireframe segments in the poloidal dimension
  wf_nTheta = 22

The location of each port is independently specified by an arbitrary origin 
point. For convenience in this example, it will be assumed that all ports
should have their origins on the same toroidal surface used to generate the
wireframe, at specified toroidal and poloidal angles::

  # Angular positions in each half-period where ports should be placed
  port_phis = [np.pi/8, 3*np.pi/8]  # toroidal angles
  port_thetas = [np.pi/4, 7*np.pi/4]  # poloidal angles
  
Further, for the sake of simplicity, the ports in this example will all have 
the same dimensions, although in general this need not be the case::

  # Dimensions of each port 
  port_ir = 0.1       # inner radius [m]
  port_thick = 0.005  # wall thickness [m]
  port_gap = 0.04     # minimum gap between port and wireframe segments [m]
  port_l0 = -0.15     # distance from origin to end, negative axis direction [m]
  port_l1 = 0.15      # distance from origin to end, positive axis direction [m]
  
The set of ports is initialized as an empty instance of the ``PortSet`` class::

  ports = PortSet()

Each port is then initialized as an instance of the ``CircularPort`` class and
then added to ``PortSet``::

  # Construct the port geometry
  for i in range(len(port_phis)):
      # For simplicity, adjust the angles to the positions of the nearest existing
      # quadrature points in the surf_wf class instance
      phi_nearest = np.argmin(np.abs((0.5/np.pi)*port_phis[i]
                                     - surf_wf.quadpoints_phi))
      for j in range(len(port_thetas)):
          theta_nearest = np.argmin(np.abs((0.5/np.pi)*port_thetas[j] \
                                           - surf_wf.quadpoints_theta))
          ox = surf_wf.gamma()[phi_nearest, theta_nearest, 0]
          oy = surf_wf.gamma()[phi_nearest, theta_nearest, 1]
          oz = surf_wf.gamma()[phi_nearest, theta_nearest, 2]
          ax = surf_wf.normal()[phi_nearest, theta_nearest, 0]
          ay = surf_wf.normal()[phi_nearest, theta_nearest, 1]
          az = surf_wf.normal()[phi_nearest, theta_nearest, 2]
          ports.add_ports([CircularPort(ox=ox, oy=oy, oz=oz, ax=ax, ay=ay, az=az,
              ir=port_ir, thick=port_thick, l0=port_l0, l1=port_l1)])


Remarks about the above code:

- In this example, for simplicity, the port origins (specified by ``ox``, 
  ``oy``, and ``oz``) are not necessarily placed exactly at the toroidal and 
  poloidal angles specified above by ``port_thetas`` and ``port_phis``; rather, 
  they are placed at existing quadrature points of ``surf_wf`` that are close 
  to the requested angles.
- Each port is set to be locally perpendicular to the surface represented by 
  ``surf_wf``; hence, the port axis (specified by ``ax``, ``ay``, and ``az``) 
  aligns with the local normal vector to the surface represented

Finally, the ``repeat_via_symmetries`` method is used to ensure that equivalent
ports originating in all half-periods are accounted for::

  ports = ports.repeat_via_symmetries(surf_wf.nfp, True)

With the port set fully specified, it can now be used to impart constraints
to the wireframe. Note that the ``PortSet`` class has a method ``collides``,
which takes as input arrays of ``x``, ``y``, and ``z`` coordinates of a set
of test points and returns a logical array that is ``True`` for each point
that collides with the port set. This function can be passed as an argument
to the ``constrain_colliding_segments`` method of a ``ToroidalWireframe``
class instance::

  # Constrain wireframe segments that collide with the ports
  wf.constrain_colliding_segments(ports.collides, gap=port_gap)

Internally, the ``ToroidalWireframe`` class instance uses this function to
determine which of its segments collide with the port set. Any segments found
to be colliding are constrained to carry zero current.

Once these constraints are set, the optimization proceeds in the same way as
with the previous example::

  # Run the RCLS optimization
  res = optimize_wireframe(wf, 'rcls', opt_params, surf_plas=surf_plas, 
                           verbose=False)

To generate a 2D plot of the solution, it may be helpful to omit the 
segments that were constrained to have zero current due to collisions with
the ports. This can be done with the ``quantity`` keyword parameter to
the ``make_plot_2d`` method::

  # Save plots and visualization data to files
  wf.make_plot_2d(quantity='nonzero currents', coordinates='degrees')

.. image:: rcls_ports_wireframe_curr2d.png
   :width: 500

While it is possible to output VTK files for the ports and wireframes in 
order to generate 3D renderings with external software, it is also possible
to create 3D images directly in SIMSOPT. For 3D visualizations of wireframes
and ports, the mayavi package must be installed. Below is some code that
creates a 3D rendering of the plasma boundary, wireframe (with the constrained
segments hidden via the keyword argument ``to_show``), and ports::

  # Generate a 3D plot if desired
  if make_mayavi_plot:
  
      from mayavi import mlab
      mlab.options.offscreen = True
  
      mlab.figure(size=(1050,800), bgcolor=(1,1,1))
      wf.make_plot_3d(to_show='active')
      ports.plot()
      surf_plas_plot = SurfaceRZFourier.from_vmec_input(filename_equil, 
          nphi=plas_nPhi, ntheta=plas_nTheta, range='full torus')
      surf_plas_plot.plot(engine='mayavi', show=False, close=True, 
          wireframe=False, color=(1, 0.75, 1))
      mlab.view(distance=5.5, focalpoint=(0, 0, -0.15))
      mlab.savefig(OUT_DIR + 'rcls_ports_wireframe_plot3d.png')

.. image:: rcls_ports_wireframe_plot3d.png
   :width: 500

Greedy Stellarator Coil Optimization (GSCO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The GSCO technique uses a greedy optimization algorithm that adds loops of 
current one by one to the wireframe, each time choosing the location and
polarization that brings about the greatest reduction of the objective 
function while upholding certain constraints and eligibility conditions.

The objective function for GSCO is

.. math::  
  J = \frac{1}{2} 
      \int \left| \left(\vec{B}_{wireframe} 
                  - \vec{B}_{ext}\right) \cdot \vec{n}\right|^2 ds
          + \frac{\lambda_S}{2} N_{active},

where :math:`N_{active}` is the number of active segments; that is, the number
of segments that carry nonzero current and :math:`\lambda_S` is a weighting
factor. The first term, which is the same as the first term in the 
`RCLS <#regularized-constrained-least-squares-rcls>`__
objective function. The first term incentivizes magnetic field accuracy,
whereas the second term incentivizes sparsity in the solution. The higher
the value of :math:`\lambda_S`, the more the optimizer will prioritize 
sparsity over field accuracy.

The eligibility rules that determine whether the optimizer may place a loop
at a given location (cell) in the wireframe are as follows:

.. list-table:: 
   :widths: 20 55 25
   :header-rows: 1
   :class: tight-table

   * - Rule
     - Description
     - Parameter in :obj:`~simsopt.solve.optimize_wireframe`
   * - wireframe constraints
     - Solution must satisfy all constraint equations (this rule is mandatory)
     - n/a (always applied)
   * - no crossing
     - At each node, at most two segments may carry current
     - ``no_crossing``
   * - no new coils
     - Loops may not be added to a cell around which all segments presently carry no current
     - ``no_new_coils``
   * - max current :math:`(I_{max})`
     - The absolute value of the current in any given segment may not exceed a given 
       :math:`I_{max}`
     - ``max_current``
   * - max loops per cell :math:`(N_{max})`
     - The net number of positive or negative loops of current added to a given cell may
       Not exceed a defined maximum, :math:`N_{max}`
     - ``max_loop_count``

The GSCO examples include solutions with various characteristics:

- Modular coils
- Saddle coils confined to toroidal sectors
- Saddle coils with different currents combined with external toroidal field 
  coils

Modular coils
-------------

Sector-confined saddle coils
----------------------------

Multi-step GSCO optimization
----------------------------


