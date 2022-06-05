#from turbulence.GX_io import GX_Runner
#from turbulence.gx_fluxtube import VMEC_Geometry
from simsopt.turbulence.gx_fluxtube import VMEC_Geometry
#from simsopt.turbulence import GX_io

import matplotlib.pyplot as plt
import sys

f_list = sys.argv[1:]
fin = f_list[0]

fig, axs= plt.subplots( 3,5, figsize=(15,7) )


#geo = VMEC_Geometry(fin, axs=axs)
#geo.plot()
gx_geo = [ VMEC_Geometry(f, axs=axs) for f in f_list]
[g.plot() for g in gx_geo]

#gx_ins = [GX_Runner(f) for f in f_list]
#print( 'name', 'ntheta', 'nx', 'ny', 'nhermite', 'nlaguerre', 'dt', 'nstep')
#[ g.list_inputs() for g in gx_ins ]

plt.tight_layout()
plt.show()

import pdb
pdb.set_trace()
