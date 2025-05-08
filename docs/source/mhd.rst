Magnetohydrodynamic codes
-------------------------

the :obj:`simsopt.mhd` module contains a collection of interfaces to Magnetohydrodynamic codes. 
Currently it contains an interface to the Variational Moments Equilibrium Code (VMEC) through the :obj:`simsopt.mhd.vmec` module and an interface to the Stepped
Pressure Equilibrium Code (SPEC) through the :obj:`simsopt.mhd.spec` module.

The equilibrium codes must be installed separately and with python bindings provided by f90wrap. 
The :obj:`simsopt.mhd` module provides them :obj:`Optimizable` objects that can depend on other :obj:`Optimizable` objects or have other :obj:`Optimizable` objects depend on them.
This includes :obj:`simsopt.mhd.Profile` for the pressure, current and rotational transform profiles, as well as :obj:`simsopt.mhd.Surface` for boundary and internal surfaces. 

In fixed boundary mode the equilibrium depends on the boundary (accessed through :obj:`.boundary`), which is a :obj:`SurfaceRZFourier`. 
In free-boundary mode however, the equilibrium on an external field (which can be provided by any :obj:`simsopt.field.MagneticField`) and its :obj:`boundary` is a child, whose properties (like aspect ratio, volume, etc) can be optimized for, by varying the degrees-of-freedom of the :obj:`MagneticField`(s).  
One can even combine different solvers such that the equilibrium objects depend on the same :obj:`Surface`, or where one equilibrium solves for the boundary, and metrics related to it are computed using the fixed-boundary
soltions provided by the other solver. 

.. warning::
    ``f90wrapped`` code is not re-entrant. You cannot have more than 
    one :obj:`Vmec` or :obj:`Spec` object active at a time. Code like:

    :: 

        from simsopt.mhd import vmec
        myvmec1 = vmec.Vmec()
        myvmec2 = vmec.Vmec()

    will have both objects accessing the same ``Fortran`` memory locations
    and lead to crashes and undefined behavior.

.. note::
    The Evaluating an MHD code is done by their :obj:`.run()` methods. 
    This is often a computationally expensive operation, and will only 
    be performed if the :obj:`recompute_bell()` method has been called 
    (as is done if any of its parents' degrees-of-freedom have changed).
    If you manually change parameters this not always triggered, and you
    have to call the :obj:`recompute_bell()` method
    yourself. 

``VMEC`` and ``SPEC`` do not provide analytical derivatives. 
As such, optimization can be done using derivative-free methods, or 
using finite-difference. 

VMEC
~~~~
VMEC is one of the most widely used codes for calculating 3d MHD equilibria. 
As such, it provices a very large number of diagnostics and outputs and has 
couplings to other codes providing further metrics that can be used in 
optimisation. 
VMEC assumes nested flux surfaces. 
The :obj:`simsopt.mhd.vmec` module provides the interface, and can be instantiated from the same input file as is usually used for running VMEC (an ``input.<name`` or ``wout_<name>.nc`` file): 

Example::
    from simsopt.mhd import vmec
    from simsopt.objectives imoprt least_squares_problem
    from simsopt.solve import least_squares_serial_solve

    myvmec = vmec.Vmec() # reads the default file
    myvmec.boundary.aspect()  # evaluate attributes of the equilibrium, aspect=10 initially

    prob = least_squares_problem(11, 1, myvmec.aspect) # create an optimization problem
    least_squares_serial_solve(prob) # solve the least squares problem



Vmec diagnostics
^^^^^^^^^^^^^^^^

There are many useful diagnostics available that depend on a :obj:`Vmec` object which provide target functions for optimization. 
These include:
- :obj:`QuasisymmetryRatioResidual`: Deviation from quasisymmetry
- :obj:`IotaTargetMetric`: Difference between the rotational transform and a provided target
- :obj:`IotaWeighted`: Weighted average of the rotational transorm
- :obj:`WellWeighted`: Measure for the magnetic well. 
- :obj:`Quasisymmetry`: Measure of the quasisymmetry using the boozer spectrum.
- :obj:`VmecRedlBootstrapMismatch`: the mismatch between the Vmec bootstrap and that provided by a recent calculation by Redl (for obtaining self-consistent bootstrap current).



SPEC
~~~~~

The Stepped Pressure Equilibrium Code (SPEC) computes equilibria using the Multi-region relaxed MHD (MRxMHD) formulation. 
This models the plasma equilibrium as a finite number of ideal interfaces between which the magnetic field is relaxed to a force-free solution. 
The :obj:`simsopt.mhd.spec` module provides the interface, and can be instantiated from the same input file as is usually used for running SPEC (an ``<name>.sp`` file). 

SPEC equilibria can contain magnetic islands and regions of magnetic chaos,
making it possible to check for and optimize such fieatures. 

All ideal interfaces in spec are available as :obj:`SurfaceRZFourier` objects. 


Greenes residue
^^^^^^^^^^^^^^^

Islands in a SPEC equilibrium can be optimized for using Cary and Hansons' method of Greenes residue minimization. 
The fixed points of the islands are found, and their residue is calculated using
``pyoculus`` through the :obj:`simsopt.mhd.GreenesResidue` that depends on the :obj:`simsopt.mhd.spec.Spec` object, and needs the poloidal and toroidal mode number of the island provided. 


Profiles
~~~~~~~~

An equilibrium depends on a number of profiles, for example the pressure, current and rotational transform profiles. 
These are sparate :obj:`Optimizable` objects, on which the equilibrium can depend. 
Because SPEC and VMEC have very different representations, specialized classes
are provided for each code. 

If not explicitly set, most profiles are handled by the equilibrium code 
internally, and not exposed to the user.
