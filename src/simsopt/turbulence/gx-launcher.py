from simsopt.turbulence.GX_io import GX_Runner  

import numpy as np
import sys

'''
    This program automates GX runs.

    It takex a sample input file, makes N changes
    to create N new input files.

    It then takes a sample batch script, makes N copies
    and queues N batch jobs on slurm.

    Updated 6 June 2022
    t.qian@pppl.gov
'''

fin = sys.argv[1] # see mod.csv for sample input
group = fin.split('.')[0]

slurm_sample = 'batch-gx.sh'

#ToDo: read changes from CSV input file

with open(fin) as f:
    datain = f.readlines()

vals = []
j = 0
for line in datain:
   
   # skip blanks
   if (line.strip() == ""): continue
   
   # skip (!) anywhere
   if (line.find('!') > -1): continue

   # skip everything that follows #
   k = line.find('#')
   if (k>0): line = line[:k]

   # parse header
   if (j==0): 
       keys = line.strip().split(';')
       gx_input = keys[0].strip()

       args = [ k.strip().split(':') for k in keys[1:] ]

       j += 1
       continue

   # store value
   vals.append( line.strip().split(';') )
   j+= 1






#ToDo: collect children files in subdirectory

gx_in = GX_Runner(gx_input)
#gx_in.pretty_print()

gx_in.load_slurm( slurm_sample )

# fixed
gx_in.inputs['Diagnostics']['all_non_zonal'] = 'false'

slurm_list = []

N_gpus_per_node = 4
i_batch = 0
k = 0
for line in vals:

    # output name
    tag = line[0]

    j = 0
    for data in line[1:]:

        a = args[j]
        N = len(a)

        if (len(a) == 1):
            gx_in.inputs[ a[0] ] = data
        elif (len(a) == 2):
            gx_in.inputs[ a[0] ][ a[1] ] = data
        elif (len(a) == 3):
            gx_in.inputs[ a[0] ][ a[1] ][ a[2] ] = data
        elif (len(a) == 4):
            gx_in.inputs[ a[0] ][ a[1] ][ a[2] ][ a[3] ] = data
        else:
            print(' WARNING: GX INPUT contains more than 4 nested layers (!) Likely issue with I/O.')

        j += 1
            
    fname = f"{group}_{tag}"
    gx_in.write(fout=f"{fname}.in", skip_overwrite=False)

    f_slurm = f"{group}_{k:02d}.sh"
    gx_in.run_slurm( f_slurm, fname )
    slurm_list.append(f_slurm)

    k += 1

##    if (k % N_gpus_per_node == 0):
##        f_slurm = f"{group}_{i_batch:02d}.sh"
##        gx_in.batch_slurm_init( f_slurm, fname )
##        i_batch += 1
##    else:
##        gx_in.batch_slurm_append( f_slurm, fname )
##    k += 1
##
##
##for i in np.arange(i_batch):
##    f_slurm = f"{group}_{i:02d}.sh"
##    gx_in.batch_slurm_close( f_slurm )

import subprocess
#[ print(f) for f in slurm_list]
for f in slurm_list:

    cmd = ["sbatch", f]
    subprocess.run(cmd)
    print("ran:", cmd)
