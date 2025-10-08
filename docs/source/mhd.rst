.. _mhd:


Magnetohydrodynamic codes
-------------------------

.. _SPEC: https://github.com/PrincetonUniversity/SPEC
.. _VMEC: https://github.com/hiddenSymmetries/vmec2000

The :obj:`simsopt.mhd` module contains a collection of interfaces to magnetohydrodynamic codes. 
Currently it contains an interface to the Variational Moments Equilibrium Code (`VMEC`_) through the :obj:`~simsopt.mhd.vmec` module and an interface to the Stepped
Pressure Equilibrium Code (`SPEC`_) through the :obj:`~simsopt.mhd.spec` modules.

The equilibrium codes must be installed separately with python bindings provided by f90wrap. 
See the installation instructions for `VMEC`_ and for `SPEC <https://github.com/PrincetonUniversity/SPEC/blob/master/compilation_instructions.md>`_ in their respective repositories.

The :obj:`~simsopt.mhd.Vmec` and :obj:`~simsopt.mhd.Spec` classes instantiate the MHD solvers as :obj:`~simsopt._core.Optimizable` objects that can depend on other :obj:`~simsopt._core.Optimizable` objects, such as a :obj:`~simsopt.mhd.Profile` for the pressure, current and rotational transform profiles, as well as :obj:`~simsopt.geo.Surface` for boundary and internal surfaces
This allows for direct handling of profiles parameters, surface degrees of freedom, and solver options through the python class.

MHD codes are often run in two different modes: fixed boundary and free boundary.
In fixed boundary mode the equilibrium depends on the boundary (accessed through ``.boundary``), which is a :obj:`~simsopt.geo.SurfaceRZFourier`. 
In free-boundary mode however, the equilibrium on an external field (which can be provided by any :obj:`~simsopt.field.MagneticField`) and its :obj:`boundary` is a child, whose properties (like aspect ratio, volume, etc) can be optimized for, by varying the degrees-of-freedom of the :obj:`~simsopt.field.MagneticField`.  
One can even combine different solvers such that the equilibrium objects depend on the same :obj:`~simsopt.geo.Surface`, or where one equilibrium solves for the boundary, and metrics related to it are computed using the fixed-boundary
solutions provided by the other solver. 

.. warning::
    ``f90wrapped`` code creates singleton classes. You cannot create multiple 
    :obj:`~simsopt.mhd.Vmec` or :obj:`~simsopt.mhd.Spec` objects in the same kernel. Code like:

    :: 

        from simsopt.mhd import Vmec
        myvmec1 = Vmec()
        myvmec2 = Vmec()

    will have both objects accessing the same ``Fortran`` memory locations
    and lead to crashes and undefined behavior.


Evaluating an MHD code is done by the ``.run()`` method of its interface class. 
This is often a computationally expensive operation, and will only 
be performed if the :obj:`recompute_bell()` method has been called 
(as is done if any of its parents' degrees-of-freedom have changed).
If you manually change parameters this not always triggered, and you
have to call the :obj:`recompute_bell()` method
yourself. 

One can optimize the result of an MHD calculation, by changing
the degrees-of-freedom of the equilibrium object. 
`VMEC`_ and `SPEC`_ do not provide analytical derivatives. 
As such, optimization can be done using derivative-free methods, or 
using finite-difference. 

Profiles
~~~~~~~~

An equilibrium depends on a number of profiles, for example the pressure, current and rotational transform profiles. 
These are separate :obj:`~simsopt._core.Optimizable` objects, on which the equilibrium can depend. 
Because SPEC and VMEC have very different representations, specialized classes
are provided for each code. 

If not explicitly set, most profiles are handled by the equilibrium code 
internally, and not exposed to the user.

The :ref:`example_vmec` tutorial contains more detailed information about profiles and using them with ``VMEC``.


VMEC
~~~~
`VMEC`_ is one of the most widely used codes for calculating 3D MHD equilibria. 
As such, it provides a very large number of diagnostics and outputs and has 
couplings to other codes providing further metrics that can be used in 
optimization. 
VMEC assumes nested flux surfaces. 
The :obj:`~simsopt.mhd.Vmec` class provides the interface, and can be instantiated from the same input file as is usually used for running VMEC (an ``input.<name`` or ``wout_<name>.nc`` file): 

See :ref:`example_vmec` for a more in-depth tutorial on running ``VMEC`` in ``simsopt``.


Vmec diagnostics
^^^^^^^^^^^^^^^^

There are many useful diagnostics available in :simsoptpy_file:`mhd/vmec_diagnostics.py` that depend on a :obj:`~simsopt.mhd.Vmec` object which provide target functions for optimization. 
These include:

* :obj:`~simsopt.mhd.QuasisymmetryRatioResidual`: Deviation from quasisymmetry
* :obj:`~simsopt.mhd.IotaTargetMetric`: Difference between the rotational transform and a provided target
* :obj:`~simsopt.mhd.IotaWeighted`: Weighted average of the rotational transform
* :obj:`~simsopt.mhd.WellWeighted`: Measure for the magnetic well. 
* :obj:`~simsopt.mhd.Quasisymmetry`: Measure of the quasisymmetry using the boozer spectrum.
* :obj:`~simsopt.mhd.VmecRedlBootstrapMismatch`: the mismatch between the VMEC bootstrap and that provided by a recent calculation by Redl (for obtaining self-consistent bootstrap current).



SPEC
~~~~~

The Stepped Pressure Equilibrium Code (`SPEC`_) computes equilibria using the Multi-region relaxed MHD (MRxMHD) formulation. 
This models the plasma equilibrium as a finite number of ideal interfaces between which the magnetic field is relaxed to a force-free solution. 
The :obj:`~simsopt.mhd.Spec` class provides the interface, and can be instantiated from the same input file as is usually used for running SPEC (an ``<name>.sp`` file). 

SPEC equilibria can contain magnetic islands and regions of magnetic chaos,
making it possible to check for and optimize such features. 

All ideal interfaces in spec are available as :obj:`~simsopt.geo.SurfaceRZFourier` objects. 


Greene's residue
^^^^^^^^^^^^^^^^

Islands in a SPEC equilibrium can be optimized for using Cary and Hansons' method of minimizing Greene's residue. 
The fixed points of the islands are found, and their residue is calculated using
``pyoculus`` through the :obj:`~simsopt.mhd.GreenesResidue` that depends on the :obj:`~simsopt.mhd.Spec` object, and needs the poloidal and toroidal mode number of the island provided. 

See :ref:`here <eliminating-islands>` for a tutorial on eliminating islands using Greene's residue minimization.

