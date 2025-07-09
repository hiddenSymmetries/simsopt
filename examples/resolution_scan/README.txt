This example traces particles in the Landreman & Buller 2.5% beta QH configuration. 

Landreman, Matt, Stefan Buller, and Michael Drevlak. "Optimization of quasi-symmetric stellarators with self-consistent bootstrap current and energetic particle confinement." Physics of Plasmas 29.8 (2022).

A resolution scan is performed in the number of gridpoints in the field interpolant and the integration tolerance. 
The conservation of the canonical momentum, p_eta, is checked. Since tracing is performed in an equilibrium with no 
non-quasisymmetric modes of the field strength, (i.e., BoozerRadialInterpolant is initialized with helicity_M and helicity_N),
p_eta should be exactly conserved. 

With resolution = 64 and tolerance = 1e-10, we see the relative error converge to ~1e-8. 

On perlmutter (07.09.25), the wallclock time is about 14 minutes using the attached slurm script. 