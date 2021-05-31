---
title: 'Simsopt: A flexible framework for stellarator optimization'
tags:
  - Python
  - plasma
  - plasma physics
  - magnetohydrodynamics
  - optimization
  - stellarator
  - fusion energy
authors:
  - name: Adrian M. Price-Whelan^[Custom footnotes for e.g. denoting who the corresponding author is can be included like this.]
    orcid: 0000-0003-0872-7098
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID
    affiliation: 2
  - name: Author with no affiliation
    affiliation: 3
affiliations:
 - name: Lyman Spitzer, Jr. Fellow, Princeton University
   index: 1
 - name: Institution Name
   index: 2
 - name: Independent Researcher
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
SIMSOPT is a collection of software components for carrying out these
optimizations.  These components include

- Interfaces to physics codes, e.g. for magnetohydrodynamic (MHD) equilibrium.
- Tools for defining objective functions and parameter spaces for optimization.
- Geometric objects that are important for stellarators – surfaces and curves – with several available parameterizations.
- Implementations of the Biot-Savart law and other magnetic fields, including derivatives.
- Tools for parallelized finite-difference gradient calculations.




# Statement of need

[//]: # (Should include references "to other software addressing related needs.")



# Discussion

Some of the physics modules with compiled code reside in separate
repositories. Two such modules are VMEC [@VMEC1983] and
SPEC [@SPEC], for MHD equilibrium.  Another module in a separate
repository is booz_xform, for calculation of Boozer coordinates.  This
latter repository is a new C++ re-implementation of an algorithm in an
older fortran 77 code of the same name.

Simsopt does not presently use input data files to define optimization
problems, in contrast to STELLOPT. Rather, problems are specified
using a python driver script, in which objects are defined and
configured. However, objects related to specific physics codes may use
their own input files. In particular, a `Vmec` object can be
initialized using a standard VMEC `input.*` input file, and a `Spec`
object can be initialized using a standard SPEC `*.sp` input file.


# Example

To illustrate some unique capabilities of `simsopt`, here we show an
example of optimizing for both quasisymmety and the elimination of
magnetic islands at the same time, with both the VMEC and SPEC codes
inside the optimization loop simultaneously. The objective function
minimized is

$$f=(A-6)^2 + (\iota_0-0.39)^2 + (\iota_a-0.42)^2 + 2\sum_{m,n} (B_{m,n}/B_{0,0})^2 + 2R_X^2 + 2R_0^2$$

where $A$ is the aspect ratio, $\iota_0$ and $\iota_a$ are the
rotational transform at the magnetic axis and edge, $B_{m,n}$ is the
amplitude of the $\cos(m\theta-n\varphi)$ Fourier mode of the field
strength in Boozer coordinates for the flux surface with normalized
toroidal flux $s=0.5$, only $n \ne 0$ modes are included in the sum,
and $R_X$ and $R_O$ are the residues [@Greene] for the X- and
O-points.  The notion of eliminating islands by minimizing residues is
taken from [@Hanson].  A single radial domain is used for SPEC
calculations. The initial configuration, shown in
\autoref{fig:xsections}-\autoref{fig:poincare}, is a two field period
vacuum field obtained by minimizing the same objective function with
`simsopt` but without the residue terms.  As shown in figure
\autoref{fig:poincare}, this initial configuration has a significant
island chain at the $\iota=2/5$ resonance. The parameter space for the
optimization consists of the Fourier modes of the boundary toroidal
surface

$$R(\theta,\phi) = \sum_{m,n}R_{m,n}\cos(m\theta-2n\phi), \;\;\; Z(\theta,\phi) = \sum_{m,n}Z_{m,n}\sin(m\theta-2n\phi)$$

![Initial and optimized stellarator shapes\label{fig:xsections}](20210530-01-014-combinedVmecSpecOpt_xsections.pdf)

![VMEC flux surfaces (black lines) and Poincare plot computed from the
 SPEC solution (colored points) for the initial and optimized
 configurations\label{fig:poincare}](20210530-01-014-combinedVmecSpecOpt_poincare.png)

The function $f$ can minimized with `simsopt`, using parallelized
evaluation of finite difference gradients, using the following python
driver script:

~~~python
import numpy as np
from simsopt.util.mpi import MpiPartition
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.boozer import Boozer, Quasisymmetry
from simsopt.mhd.spec import Spec, Residue
from simsopt.objectives.least_squares import LeastSquaresProblem
from simsopt.solve.mpi import least_squares_mpi_solve

# Create objects for the Vmec and Spec equilibrium
mpi = MpiPartition()
vmec = Vmec("input.nfp2_QA_iota0.4", mpi=mpi)
surf = vmec.boundary
spec = Spec("nfp2_QA_iota0.4.sp", mpi=mpi)
spec.boundary = surf  # Identify the Vmec and Spec boundaries

# Configure quasisymmetry objective:
boozer = Boozer(vmec)
qs = Quasisymmetry(boozer,
                   0.5, # Radius s to target
                   1, 0) # (M, N) you want in |B|
		   
# Specify resonant surface by iota = p / q
p = -2
q = 5
residue1 = Residue(spec, p, q)  # X-point
residue2 = Residue(spec, p, q, theta=np.pi)  # O-point

# Define objective function                                                                                                                      
prob = LeastSquaresProblem([(vmec.aspect, 6, 1.0),
                            (vmec.iota_axis, 0.39, 1),
                            (vmec.iota_edge, 0.42, 1),
                            (qs, 0, 2),
                            (residue1, 0, 2),
                            (residue2, 0, 2)])

for step in range(3):
    # Define parameter space for this step:
    surf.all_fixed()
    max_mode = step + 3
    surf.fixed_range(mmin=0, mmax=max_mode,
                     nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.set_fixed("rc(0,0)")  # Fix the major radius

    # Dynamically increase the resolution parameters each step:
    vmec.indata.mpol = 4 + step
    vmec.indata.ntor = vmec.indata.mpol
    boozer.mpol = 24 + step * 8
    boozer.ntor = boozer.mpol

    least_squares_mpi_solve(prob, mpi, grad=True)
~~~

Notably, the `Spec` object is configured to use the same boundary
`Surface` object as the `Vmec` instance, so when the shape of this
single surface is modified during the optimization, the outputs of
both Vmec and Spec change accordingly.  Also, since the optimization
problem is formulated with a script, any other desired scripting
elements can be included. Here this capability is used to define a
series of optimization stages, in which the size of the parameter
space is increased at each step, along with the numerical resolution
parameters of the codes. The former is valuable to avoid getting stuck
in a local minimum, and the latter improves computational efficiency.

It can be seen in figure \autoref{fig:poincare} that the optimization
has successfully eliminated the islands; indeed the residues have been
reduced from $\pm2\times 10^{-3}$ to $-2\times 10^{-6}$. Therefore the
VMEC and SPEC solutions agree on the surface shapes at the optimum,
and calculations based on the VMEC solution can be trusted.  The final
configuration also has extremely good quasiaxisymmetry, as shown by
the straight horizontal contours of $|B|$ in \autoref{fig:boozer}.

![Magnetic field strength for the optimized stellarator shape,
 computed from VMEC and booz_xform, showing good
 quasisymmetry.\label{fig:boozer}](20210530-01-014-combinedVmecSpecOpt_boozPlot.pdf){
 width=50% }


# Acknowledgements

This work was supported by a grant from the Simons Foundation (560651, ML).
?? PPPL grant number??

# References
