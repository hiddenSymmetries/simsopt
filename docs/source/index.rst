Simsopt documentation
=====================

.. _simsopt: https://github.com/hiddenSymmetries/simsopt

`simsopt`_ is a framework for optimizing `stellarators
<https://en.wikipedia.org/wiki/Stellarator>`_.  The high-level
routines are in python, with calls to C++ or fortran where needed for
performance. Several types of components are included:

- Interfaces to physics codes, e.g. for MHD equilibrium.
- Tools for defining objective functions and parameter spaces for
  optimization.
- Geometric objects that are important for stellarators -- surfaces and
  curves -- with several available parameterizations.
- Efficient implementations of the Biot-Savart law and other magnetic
  field representations, including derivatives.
- Tools for parallelized finite-difference gradient calculations.

The design of `simsopt`_ is guided by several principles:

- Thorough unit testing, regression testing, and continuous
  integration.
- Extensibility: It should be possible to add new codes and terms to
  the objective function without editing modules that already work,
  i.e. the `open-closed principle
  <https://en.wikipedia.org/wiki/Open%E2%80%93closed_principle>`_
  . This is because any edits to working code can potentially introduce bugs.
- Modularity: Physics modules that are not needed for your
  optimization problem do not need to be installed. For instance, to
  optimize SPEC equilibria, the VMEC module need not be installed.
- Flexibility: The components used to define an objective function can
  be re-used for applications other than standard optimization. For
  instance, a `simsopt`_ objective function is a standard python
  function that can be plotted, passed to optimization packages
  outside of `simsopt`_, etc.

`simsopt`_ is fully open-source, and anyone is welcome to use it,
make suggestions, and contribute.

Some of the physics modules with compiled code reside in separate
repositories. These separate modules include

- `VMEC <https://github.com/hiddenSymmetries/VMEC2000>`_, for MHD
  equilibrium.
- `SPEC <https://github.com/PrincetonUniversity/SPEC>`_, for MHD
  equilibrium.
- `booz_xform <https://hiddensymmetries.github.io/booz_xform/>`_, for
  Boozer coordinates and quasisymmetry.
- `virtual-casing <https://github.com/hiddenSymmetries/virtual-casing>`_,
  needed for coil optimization in the case of finite-beta plasmas.
  
We gratefully acknowledge funding from the `Simons Foundation's Hidden
symmetries and fusion energy project
<https://hiddensymmetries.princeton.edu>`_.

`simsopt`_ is one of several available systems for stellarator
optimization.  Others include `STELLOPT
<https://github.com/PrincetonUniversity/STELLOPT>`_, `DESC
<https://github.com/PlasmaControl/DESC>`_, `ROSE
<https://doi.org/10.1088/1741-4326/aaed50>`_, and
`StellaratorOptimization.jl <https://gitlab.com/wistell/StellaratorOptimization.jl>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview
   installation
   containers
   optimizable
   geo
   fields
   mhd
   tracing
   mpi
   testing
   source
   publications
   contributing

.. toctree::
   :maxdepth: 3
   :caption: Tutorials
   
   example_vmec
   example_vmec_only
   example_quasisymmetry
   example_islands
   example_coils
   example_single_stage
   example_permanent_magnets
   example_wireframe


.. toctree::
   :maxdepth: 3
   :caption: API Reference

   simsopt
   simsoptpp_cpp_api

.. toctree::
   :maxdepth: 3
   :caption: Developer Reference

   example_derivative
   cpp

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
