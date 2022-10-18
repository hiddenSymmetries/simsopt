from turbulence.GX_io import GX_Runner
#from simsopt.turbulence import GX_io

import sys

f_list = sys.argv[1:]
fin = f_list[0]


gx_ins = [GX_Runner(f) for f in f_list]

print( "name         , ntheta ,  nx,    ny, nhermite, nlaguerre, dt, nstep")

data = [ g.list_inputs() for g in gx_ins ]

for line in data:
    tag, ntheta,nx,ny, nhermite, nlaguerre, dt, nstep = line
    print( f"{tag:16}, {ntheta:4}, {nx:4}, {ny:4}, {nhermite:4}, {nlaguerre:4}, {dt:8}, {nstep:6}")


