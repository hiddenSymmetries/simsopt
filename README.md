# simsopt

![GitHub](https://img.shields.io/github/license/hiddensymmetries/simsopt)
[![codecov](https://codecov.io/gh/hiddenSymmetries/simsopt/branch/master/graph/badge.svg?token=ltN6qonZ5p)](https://codecov.io/gh/hiddenSymmetries/simsopt)
[![DOI](https://zenodo.org/badge/247710081.svg)](https://zenodo.org/badge/latestdoi/247710081)

![SIMSOPT](docs/source/logo.png)
![SIMSOPT](docs/source/coils_and_surfaces.png)

`simsopt` is a framework for optimizing
[stellarators](https://en.wikipedia.org/wiki/Stellarator).
The high-level routines of `simsopt` are in python, with calls to C++
or fortran where needed for performance. Several types of components
are included:

- Interfaces to physics codes, e.g. for MHD equilibrium.
- Tools for defining objective functions and parameter spaces for
  optimization.
- Geometric objects that are important for stellarators - surfaces and
  curves - with several available parameterizations.
- Efficient implementations of the Biot-Savart law and other magnetic
  field representations, including derivatives.
- Tools for parallelized finite-difference gradient calculations.

The design of `simsopt` is guided by several principles:

- Thorough unit testing, regression testing, and continuous
  integration.
- Extensibility: It should be possible to add new codes and terms to
  the objective function without editing modules that already work,
  i.e. the [open-closed principle](https://en.wikipedia.org/wiki/Open%E2%80%93closed_principle).
  This is because any edits to working code can potentially introduce bugs.
- Modularity: Physics modules that are not needed for your
  optimization problem do not need to be installed. For instance, to
  optimize SPEC equilibria, the VMEC module need not be installed.
- Flexibility: The components used to define an objective function can
  be re-used for applications other than standard optimization. For
  instance, a `simsopt` objective function is a standard python
  function that can be plotted, passed to optimization packages
  outside of `simsopt`, etc.

`simsopt` is fully open-source, and anyone is welcome to use it, make
suggestions, and contribute.

Several methods are available for installing `simsopt`. One
recommended approach is to use pip:

    pip install simsopt

For detailed installation instructions on some specific systems, see
[the wiki](https://github.com/hiddenSymmetries/simsopt/wiki).
Also, a Docker container is available with `simsopt` and its components pre-installed, which
can be started using

    docker run -it --rm hiddensymmetries/simsopt

More [installation
options](https://simsopt.readthedocs.io/en/latest/installation.html),
[instructions for the Docker
container](https://simsopt.readthedocs.io/en/latest/containers.html), and
other information can be found in the [main simsopt documentation
here.](https://simsopt.readthedocs.io)

Some of the physics modules with compiled code reside in separate
repositories. These separate modules include

- [VMEC](https://github.com/hiddenSymmetries/VMEC2000), for MHD
  equilibrium.
- [SPEC](https://github.com/PrincetonUniversity/SPEC), for MHD
  equilibrium.
- [booz_xform](https://hiddensymmetries.github.io/booz_xform), for
  Boozer coordinates.
  
If you use `simsopt` in your research, kindly cite the code using
[this reference](https://doi.org/10.21105/joss.03525):

[1] M Landreman, B Medasani, F Wechsung, A Giuliani, R Jorge, and C Zhu,
    "SIMSOPT: A flexible framework for stellarator optimization",
    *J. Open Source Software* **6**, 3525 (2021).

See also [the simsopt publications page](https://simsopt.readthedocs.io/en/latest/publications.html).

We gratefully acknowledge funding from the [Simons Foundation's Hidden
symmetries and fusion energy
project](https://hiddensymmetries.princeton.edu). 
