Magnetohydrodynamic codes
-------------------------

The :obj:`simsopt.mhd` module contains a collection of interfaces to Magnetohydrodynamic codes. 
Currently it contains an interface to the Variational Moments Equilibrium Code (VMEC) through the :obj:`simsopt.mhd.vmec` module and an interface to the Stepped
Pressure Equilibrium Code (SPEC) through the :obj:`simsopt.mhd.spec` module.

The equilibrium codes must be installed separately with python bindings provided by f90wrap. 
See the the instructions for [VMEC](https://github.com/hiddenSymmetries/vmec2000) and for [SPEC](https://github.com/PrincetonUniversity/SPEC/blob/master/compilation_instructions.md) page for instructions.

The ``Vmec`` and ``Spec`` classes instantiate the MHD solvers as ``Optimizable`` objects that can depend on other :obj:`Optimizable` objects, such as a :obj:`simsopt.mhd.Profile` for the pressure, current and rotational transform profiles, as well as :obj:`simsopt.mhd.Surface` for boundary and internal surfaces
This allows for direct handling of profiles parameters, surface degrees of freedom, and solver options through the python class.

MHD codes are often run in two different modes: fixed boundary and free boundary.
In fixed boundary mode the equilibrium depends on the boundary (accessed through ``.boundary``), which is a :obj:`simsopt.geo.SurfaceRZFourier``. 
In free-boundary mode however, the equilibrium on an external field (which can be provided by any :obj:`simsopt.field.MagneticField`) and its :obj:`boundary` is a child, whose properties (like aspect ratio, volume, etc) can be optimized for, by varying the degrees-of-freedom of the :obj:`MagneticField`.  
One can even combine different solvers such that the equilibrium objects depend on the same :obj:`Surface`, or where one equilibrium solves for the boundary, and metrics related to it are computed using the fixed-boundary
soltions provided by the other solver. 

.. warning::
    ``f90wrapped`` code creates singleton classes. You cannot create multiple 
    ``Vmec`` or ``Spec`` objects in the same kernel. Code like:

    :: 

        from simsopt.mhd import vmec
        myvmec1 = vmec.Vmec()
        myvmec2 = vmec.Vmec()

    will have both objects accessing the same ``Fortran`` memory locations
    and lead to crashes and undefined behavior.


The Evaluating an MHD code is done by their ``.run()`` methods. 
This is often a computationally expensive operation, and will only 
be performed if the :obj:`recompute_bell()` method has been called 
(as is done if any of its parents' degrees-of-freedom have changed).
If you manually change parameters this not always triggered, and you
have to call the :obj:`recompute_bell()` method
yourself. 

One can optimize the result of an MHD calculation, by changing
the degrees-of-freedom of the equilibrium object. 
``VMEC`` and ``SPEC`` do not provide analytical derivatives. 
As such, optimization can be done using derivative-free methods, or 
using finite-difference. 

Profiles
~~~~~~~~

An equilibrium depends on a number of profiles, for example the pressure, current and rotational transform profiles. 
These are sparate :obj:`Optimizable` objects, on which the equilibrium can depend. 
Because SPEC and VMEC have very different representations, specialized classes
are provided for each code. 

If not explicitly set, most profiles are handled by the equilibrium code 
internally, and not exposed to the user.

The :ref:`running-vmec` tutorial contains more detailed information about profiles and using them with ``VMEC``.


VMEC
~~~~
VMEC is one of the most widely used codes for calculating 3D MHD equilibria. 
As such, it provices a very large number of diagnostics and outputs and has 
couplings to other codes providing further metrics that can be used in 
optimisation. 
VMEC assumes nested flux surfaces. 
The :obj:`simsopt.mhd.vmec` module provides the interface, and can be instantiated from the same input file as is usually used for running VMEC (an ``input.<name`` or ``wout_<name>.nc`` file): 

See :ref:`running_vmec` for a more in-depth tutorial on running ``VMEC`` in ``simsopt``.


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
making it possible to check for and optimize such features. 

All ideal interfaces in spec are available as :obj:`SurfaceRZFourier` objects. 


Greenes residue
^^^^^^^^^^^^^^^

Islands in a SPEC equilibrium can be optimized for using Cary and Hansons' method of Greenes residue minimization. 
The fixed points of the islands are found, and their residue is calculated using
``pyoculus`` through the :obj:`simsopt.mhd.GreenesResidue` that depends on the :obj:`simsopt.mhd.spec.Spec` object, and needs the poloidal and toroidal mode number of the island provided. 

See :ref:`eliminating-islands` for a tutorial on eliminating islands using Greenes residue minimization.

