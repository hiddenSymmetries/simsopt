---
title: 'SIMSOPT: A flexible framework for stellarator optimization'
tags:
  - Python
  - plasma
  - plasma physics
  - magnetohydrodynamics
  - optimization
  - stellarator
  - fusion energy
authors:
  - name: Matt Landreman^[mattland@umd.edu]
    orcid: 0000-0003-0872-7098
    affiliation: 1
  - name: Florian Wechsung
    affiliation: 2
  - name: Bharat Medasani
    orcid: 0000-0002-2073-4162
    affiliation: 3
  - name: Andrew Guiliani
    affiliation: 2
  - name: Rogerio Jorge
    affiliation: 1
  - name: Caoxiang Zhu
    affiliation: 3
affiliations:
 - name: University of Maryland 
   index: 1
 - name: Courant Institute of Mathematical Sciences, New York University
   index: 2
 - name: Princeton Plasma Physics Laboratory
   index: 3
date: 1 June 2021
bibliography: paper.bib

---

# Summary

[//]: # (JOSS guidelines: A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience.)

A stellarator is a magnetic field configuration used to confine
plasma, and it is a candidate configuration for fusion energy, as well
as a general charged particle trap.  A stellarator's magnetic field is
typically produced using electromagnetic coils, and the shaping of the
field and coils must be optimized to achieve good confinement.
`SIMSOPT` is a collection of software components for carrying out these
optimizations.  These components include

- Interfaces to physics codes, e.g. for magnetohydrodynamic (MHD) equilibrium.
- Tools for defining objective functions and parameter spaces for optimization.
- Geometric objects that are important for stellarators – surfaces and curves – with several available parameterizations.
- Implementations of the Biot-Savart law and other magnetic fields, including derivatives.
- Tools for parallelized finite-difference gradient calculations.




# Statement of need

[//]: # (Should include references "to other software addressing related needs.")

To effectively confine plasmas for the goal of fusion energy, 
the three-dimensional magnetic field of a stellarator has to be carefully designed.
The design effort is essentially to vary the magnetohydrodynamic (MHD) 
equilibrium to meet multiple metrics, for example, MHD stability, 
neoclassical transport, fast-ion confinement, turbulence transport, and buildable coils. 
This process involves calling several physics codes and cannot be done manually. 
Therefore, we need a stellarator optimization code.

Although the idea of stellarator optimization has been proposed for decades, 
there are limited codes available to use. 
The two most commonly used codes are STELLOPT [@STELLOPT] and ROSE [@ROSE]. 
While ROSE is closed-sourced, STELLOPT is written in Fortran and couples all the codes explicitly. 
It requires non-trial effort to learn to use the code and write an interface for a new module. 
The goal of SIMSOPT is to flatten the learning curve, 
improve the flexibility of prototyping new problems, and enhance the extendibility & maintainability. 
To achieve the goals, SIMSOPT is written in objective-oriented Python and stays close to standards. 
The physics codes are interfaced using a Fortran/Python wrapper or a C++/Python binding tool. 
Modern tools are used in SIMSOPT to manage the documentation and unity tests.


# Structure

The components of `SIMSOPT` that are not performance bottlenecks are
written in python, for flexibility and ease of use and development.
In components where performance is critical, compiled C++ code is
interfaced to python using the `pybind11` package [@pybind].  As
examples, the infrastructure for defining objective functions and
optimization problems is written in python, whereas the Biot-Savart
law is implemented in C++.

Some of the physics modules with compiled code reside in separate
repositories. Two such modules are the VMEC [@VMEC1983] and SPEC
[@SPEC] codes, for MHD equilibrium. These Fortran codes are interfaced
using the `f90wrap` package [@f90wrap], so data can be passed directly
in memory to and from python.  This is particularly useful for passing
MPI communicators for parallelized evaluation of finite-difference
gradients.  Another module in a separate repository is booz_xform, for
calculation of Boozer coordinates.  This latter repository is a new
C++ re-implementation of an algorithm in an older fortran 77 code of
the same name.

A variety of geometric objects and magnetic field types are included
in `SIMSOPT`.  Several discretizations of curves and toroidal surfaces
are included, since curves are important both in the context of
electromagnetic coils and the magnetic axis, and flux surfaces are a
key concept for stellarators. One magnetic field type represents the
Biot-Savart law, defined by a set of curves and the electric current
they carry. Other available magnetic field types include Dommaschk
potentials and the analytic formula for the field of a circular coil,
and magnetic field instances can be scaled and summed. All the
geometric and magnetic field classes provide one or two derivatives,
either provided by explicit formulae, or by automatic differentiation
with the `jax` package [@jax].  Caching is available to avoid repeated
calculations.


Presently, MPI and OpenMP parallelism are used in different code
components.  The parallelized finite-difference gradient capability
uses MPI, to support use of multiple compute nodes, and to support
concurrent calculations with physics codes like VMEC and SPEC that
employ MPI. Biot-Savart calculations are presently parallelized using
OpenMP for simplicity.

`SIMSOPT` does not presently use input data files to define optimization
problems, in contrast to `STELLOPT`. Rather, problems are specified
using a python driver script, in which objects are defined and
configured. However, objects related to specific physics codes may use
their own input files. In particular, a `Vmec` object can be
initialized using a standard VMEC `input.*` input file, and a `Spec`
object can be initialized using a standard SPEC `*.sp` input file.


# Capabilities

Presently, `SIMSOPT` provides tools for each of the two optimization
stages used for the design of stellarators such as W7-X and HSX.  In
the first stage, the boundary of a toroidal magnetic surface is varied
to optimize the physics properties inside it.  In the second stage,
coil shapes are optimized to approximately produce the boundary
magnetic surface that resulted from the first stage.

For the first stage, MHD equilibria or vacuum fields can be
represented using the VMEC or SPEC code, or both at the same time.
VMEC, which makes the assumption that good nested magnetic surfaces
exist, is extremely robust and many other physics codes are able to
postprocess its output.  SPEC can provide added value because of its
ability to represent magnetic islands, which are undesirable since a
large temperature gradient cannot be supported across them.  Islands
can be eliminated using `SIMSOPT` by minimizing the magnitude of the
residues [@Greene], similar to the method in [@Hanson].  An example of
stage-1 optimization including both VMEC and SPEC simultaneously is
shown in \autoref{fig:xsections}-\autoref{fig:poincare}. Here, the
shape is optimized to both eliminate an internal island chain, as
computed from SPEC, and to achieve quasisymmetry, as computed from
VMEC and booz_xform. This example will be described in more detail in
a separate publication.

![An example of stage-1 optimization using `SIMSOPT`, in which the
 shape of a toroidal boundary is optimized to eliminate magnetic
 islands and improve
 quasisymmetry.\label{fig:xsections}](20210530-01-014-combinedVmecSpecOpt_xsections.pdf)

![VMEC flux surfaces (black lines) and Poincare plot computed from the
 SPEC solution (colored points) for the initial and optimized
 configurations in
 \autoref{fig:xsections}.\label{fig:poincare}](20210530-01-014-combinedVmecSpecOpt_poincare.png)

The curve and magnetic field classes in `SIMSOPT` can then be used for
the second optimization stage, in which coil shapes are designed.
Varying the shapes of the coils, derivative-based optimization can be
used to minimize the normal component of the magnetic field on the
target surface, similar to the FOCUS code [@focus].

One can also use `SIMSOPT` for other optimization problems that differ
from the above two-stage approach.  For instance, `SIMSOPT` is
presently being used for a single-stage derivative-based method in
which coil shapes are varied to optimize directly for quasisymmetry
near the magnetic axis [@Giuliani]. \autoref{fig:coils} shows an
example in which stochastic optimization is applied to find a
configuration in which the quasisymmetry is relatively insensitive to
errors in the coil shapes.  This example will be described in more
detail in a separate publication.


![A stellarator obtained using stochasic optimization with `Curve` and
 `BiotSavart` classes from `SIMSOPT`, with magnetic surfaces computed
 using `Surface` classes.\label{fig:coils}](rt_angle.png){width=65%}





# Acknowledgements

We gratefully acknowledge discussions and assistance from
Aaron Bader,
Antoine Baillod,
David Bindel,
Benjamin Faber,
Stuart Hudson,
Thomas Kruger,
Jonathan Schilling,
Georg Stadler,
and
Zhisong Qu.
This work was supported by a grant from the Simons Foundation (560651,
ML).  BM and CZ are supported by the U.S. Department of Energy under
Contract No. DE-AC02-09CH11466 through the Princeton Plasma Physics
Laboratory.

# References
