This example demonstrates the selection of an eigenmode from AE3D and postprocessing of the mode
to perform guiding center tracing. The Wistell-A equilibrium is used. 

Stellgap: https://github.com/ORNL-Fusion/Stellgap
AE3D: https://github.com/ORNL-Fusion/AE3D

First, using the notebook ChooseAE.ipynb, the spectrum computed from AE3D and stellgap is visualized,
so that the position of global eigenmodes can be identified with respect to the continuum gap structure. 
The eigenmode is then saved in a file, ae.npy. The data obtained from stellgap and AE3D are provided
in the directories stellgap_output and ae3d_output, respectively. 

Then, the script tracing_with_AE.py is used to perform guiding center tracing using a fusion birth 
distribution function. The loss fraction is plotted as a function of time. 

On perlmutter (07.05.25), the wallclock time is about 21 minutes using the included slurm script.
