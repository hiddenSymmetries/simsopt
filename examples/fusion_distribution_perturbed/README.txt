This example traces 5000 particles in the Landreman & Buller 2.5% beta QA configuration. 

Landreman, Matt, Stefan Buller, and Michael Drevlak. "Optimization of quasi-symmetric stellarators with self-consistent bootstrap current and energetic particle confinement." Physics of Plasmas 29.8 (2022).

An m = 1, n = 1 shear Alfven wave is included, with parameters descried in 

Paul, Elizabeth J., Harry E. Mynick, and Amitava Bhattacharjee. "Fast-ion transport in quasisymmetric equilibria in the presence of a resonant Alfvénic perturbation." Journal of Plasma Physics 89.5 (2023): 905890515.

Particles are initialized proportional to the fusion reactivity profile and traced until they reach the boundary (s=1) or the elapsed time is 1e-2 seconds. 

On perlmutter (06.27.25), the wallclock time is about 77 seconds using the included slurm script. 
