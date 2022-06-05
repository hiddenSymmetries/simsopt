from turbulence.GX_io import GX_Runner
#from simsopt.turbulence import GX_io

import sys

f_list = sys.argv[1:]
fin = f_list[0]


gx_ins = [GX_Runner(f) for f in f_list]

print( 'name', 'ntheta', 'nx', 'ny', 'nhermite', 'nlaguerre', 'dt', 'nstep')
[ g.list_inputs() for g in gx_ins ]


import pdb
pdb.set_trace()
