This example computes the trapped particle Poincare map in the beta = 2.5% QA configuration from Landreman & Buller. 

Landreman, Matt, Stefan Buller, and Michael Drevlak. "Optimization of quasi-symmetric stellarators with self-consistent bootstrap current and energetic particle confinement." Physics of Plasmas 29.8 (2022).

The non-quasisymmetric modes are artificially suppressed for the computation of the trapped particle frequencies, 
allowing ther identification of the resonances visible in the trapped particle Poincare plot. 

The particle is assumed to mirror at (s,theta,zeta)=(0.5,pi/2,0) with the alpha particle birth energy. 

On macOS (06.17.25), the wallclock time is about 69 seconds, running with the command "mpiexec -n 8 python trapped_frequencies.py"