import numpy as np
from netCDF4 import Dataset
#from scipy.io import netcdf as nc  # GX uses netcdf4, but scipy only has netcdf3

import subprocess
import os

class GX_Runner():

    # This class handles GX input files, and also execution

    def __init__(self,template):
        
        self.read_input(template)


    def read_input(self, fin):

        with open(fin) as f:
            data = f.readlines()

        obj = {}
        header = ''
        for line in data:

            # skip comments
            if line.find('#') > -1:
                continue

            # parse headers
            if line.find('[') == 0:
                header = line.split('[')[1].split(']')[0]
                obj[header] = {}
                continue

            # skip blanks
            if line.find('=') < 0:
                continue

            # store data
            key, value = line.split('=')
            key   = key.strip()
            value = value.strip()
            
            if header == '':
                obj[key] = value
            else:
                obj[header][key] = value

        self.inputs = obj
        self.filename = fin


    def write(self, fout='temp.in', skip_overwrite=True):

        # do not overwrite
        if (os.path.exists(fout) and skip_overwrite):
            print( '  input exists, skipping write', fout )
            return

        with open(fout,'w') as f:
        
            for item in self.inputs.items():
                 
                if ( type(item[1]) is not dict ):
                    print('  %s = %s ' % item, file=f)  
                    continue
    
                header, nest = item
                print('\n[%s]' % header, file=f)
    
                longest_key =  max( nest.keys(), key=len) 
                N_space = len(longest_key) 
                for pair in nest.items():
                    s = '  {:%i}  =  {}' % N_space
                    print(s.format(*pair), file=f)

        # print('  wrote input:', fout)


    def load_slurm(self,f_slurm):

        with open(f_slurm) as f:
            datain = f.readlines()

        self.slurm_header = datain


    def list_inputs(self):

        ntheta = int(self.inputs['Dimensions']['ntheta'])
        nx     = int(self.inputs['Dimensions']['nx'])
        ny     = int(self.inputs['Dimensions']['ny'])
        nhermite = int(self.inputs['Dimensions']['nhermite'])
        nlaguerre = int(self.inputs['Dimensions']['nlaguerre'])
        
        
        dt = float(self.inputs['Time']['dt'])
        nstep = int(self.inputs['Time']['nstep'])

        tag = self.filename.split('/')[-1]
        #print( tag, ntheta,nx,ny, nhermite, nlaguerre, dt, nstep)
        return tag, ntheta,nx,ny, nhermite, nlaguerre, dt, nstep


    # queues a slurm file
    def run_slurm(self, f_batch, gx_name):

        with open(f_batch, 'w') as f:
            
            for line in self.slurm_header:
                f.write(line)

            run_cmd = f"srun gx {gx_name}.in > log.{gx_name} \n"
            f.write(run_cmd)

        batch_cmd = 'sbatch {}'.format(f_batch)
        print('  running:', batch_cmd)
#        os.system( batch_cmd )  # halt for temporary stellar-traverse interface

    # this combines 4 gx runs into 1 batch job
    def batch_slurm_init(self, f_batch, gx_name):

        with open(f_batch, 'w') as f:
            
            for line in self.slurm_header:
                f.write(line)

            run_cmd = f"srun gx {gx_name}.in > log.{gx_name} &\n"
            f.write(run_cmd)

    def batch_slurm_append(self, f_batch, gx_name):

        with open(f_batch, 'a') as f:
            
            run_cmd = f"srun gx {gx_name}.in > log.{gx_name} &\n"
            f.write(run_cmd)


    def batch_slurm_close(self, f_batch):

        with open(f_batch, 'a') as f:
            
            run_cmd = "wait \n"
            f.write(run_cmd)

        batch_cmd = 'sbatch {}'.format(f_batch)
        print('  preparing:', batch_cmd)

    ### the above should be functions in a GX_Slurm Class
    # issue with slurm, if I ask for 4 tasks per node, 1 gpu per task
    # all gpus get started on the first job. Maybe the issue is deviceid hard coded in CUDA?


    def make_fluxtube(self, f_wout):

        
        gx = VMEC_GX_geometry_module(input_path='./', tag='scan')
        gx.set_vmec( f_wout )
        gx.init_radius(s=0.5, local_id=f"-gx-simsopt")

        self.flux_tube = gx
        # get fluxtube name and return


    def set_gx_wout(self, fname):

        self.inputs['Geometry']['geofile'] = f'"{fname}"'

        # next level, could read the .nc, and get ntheta

    def pretty_print(self, entry=''):
    # dumps out current input data, in GX input format
    #    if entry, where entry is one of the GX input headers
    #       only print the inputs nested under entry

        for item in self.inputs.items():
        
            # a catch for the debug input, which has no header
            if ( type(item[1]) is not dict ):
                if (entry == ''):
                    print('  %s = %s ' % item)
                    continue
     
            header, nest = item

            # special case
            if (entry != ''):
                header = entry
                nest   = self.inputs[entry]

            print('\n[%s]' % header)
     
            longest_key =  max( nest.keys(), key=len) 
            N_space = len(longest_key) 
            for pair in nest.items():
                s = '  {:%i}  =  {}' % N_space
                print(s.format(*pair))

            # special case
            if (entry != ''):
                break

class GX_Output():

    def __init__(self,fname):

        try:
            f = Dataset(fname, mode='r')
            self.data = f
            #f = nc.netcdf_file(fname, 'r') 
        except: 
            print('  read_GX_output: could not read', fname)
            return

    
        self.get_qflux() 
    
        self.time  = f.variables['time'][:]

        self.tprim  = f.groups['Inputs']['Species']['T0_prime'][:]
        self.fprim  = f.groups['Inputs']['Species']['n0_prime'][:]


    def get_qflux(self):

        try:

            qflux = self.data.groups['Fluxes'].variables['qflux'][:,0]
        except:
            print("no qflux found")
            return
    
        # check for NANs
        if ( np.isnan(qflux).any() ):
             print('  nans found in', fname)
             qflux = np.nan_to_num(qflux)

        self.qflux = qflux

    # can extend this to select modes
    def get_gamma(self):

        data = self.data.groups['Special']['omega_v_time'][:]

        # get all times, first ky mode, kx=0, gamma = Im(omega)
        gamma = data[:,1,0,1]  # t, ky, kx, ri

        self.gamma = gamma


    def median_estimator(self):

        N = len(self.qflux)
        med = np.median( [ np.median( self.qflux[::-1][:k] ) for k in np.arange(1,N)] )

        self.q_median = med
        return med

    def exponential_window_estimator(self, tau=100):

        t0 = 0
        qavg = 0
        var_qavg = 0
        
        Q_avg = []
        Var_Q_avg = []
        
        # loop through time
        N = len(self.qflux)
        for k in np.arange(N):
        
            # get q(t)
            q = self.qflux[k]
            t = self.time [k]
        
            # compute weights
            gamma = (t - t0)/tau
            alpha = np.e**( - gamma)
            delta = q - qavg
        
            # update averages
            qavg = alpha * qavg + q * (1 - alpha)
            var_qavg = alpha * ( var_qavg + (1-alpha)* delta**2)
            t0 = t
        
            # save
            Q_avg.append(qavg)
            Var_Q_avg.append(var_qavg)

        self.Q_avg = Q_avg
        self.Var_Q_avg = Var_Q_avg

        return Q_avg[-1], Var_Q_avg[-1]
        


class VMEC_GX_geometry_module():

    # this class handles VMEC-GX Geometry .ing input files

    def __init__(self, f_sample    = 'gx-geometry-sample.ing',
                       tag         = 'default',
                       input_path  = 'gx-files/',
                       output_path = './'
                       ):

        self.data = self.read(input_path + f_sample)

        self.output_path = output_path
        self.input_path  = input_path
        self.tag  = tag


    # this function is run at __init__
    #    it parses a sample GX_geometry.ing input file
    #    as a dictionary for future modifications
    def read(self,fin):

        with open(fin) as f:
            indata = f.readlines()

        data = {} # create dictionary
        for line in indata:

            # remove comments
            info = line.split('#')[0]

            # skip blanks
            if info.strip() == '':
                continue

            # parse
            key,val = info.split('=')
            key = key.strip()
            val = val.strip()

            # save
            data[key] = val

        return data


    def write(self, fname): # writes a .ing file

        # load
        data = self.data
        path = self.output_path
        
        # set spacing
        longest_key =  max( data.keys(), key=len) 
        N_space = len(longest_key) 

        # write
        fout = path + fname + '.ing'
        with open(fout,'w') as f:
            for pair in data.items():
                s = '  {:%i}  =  {}' % N_space
                #print(s.format(*pair))   # print to screen for debugging
                print(s.format(*pair), file=f)


    def set_vmec(self,wout, vmec_path='./', output_path='./'):

        # copy vmec output from vmec_path to output_path
        #cmd = 'cp {:}{:} {:}'.format(vmec_path, wout, output_path)
        #os.system(cmd)

        self.data['vmec_file'] = '"{:}"'.format(wout)
    #    self.data['out_path'] = '"{:}"'.format(output_path)
    #    self.data['vmec_path'] = '"{:}"'.format(vmec_path) 



    def init_radius(self,s=0.5, local_id=""):

        # set radius
        self.data['desired_normalized_toroidal_flux'] = s

        # write input
        in_path  = self.input_path
        out_path = self.output_path
        fname = self.tag + local_id + '-psi-{:.2f}'.format(s)
        self.write(fname)
        # print('  wrote .ing', out_path+fname)

        # run
        cmd = ['./convert_VMEC_to_GX',  out_path+fname]
        # cmd = ['./{:}convert_VMEC_to_GX'.format(in_path),  out_path+fname]

        f_log = out_path + fname + '.log'
        with open(f_log, 'w') as fp:
            subprocess.call(cmd,stdout=fp)


