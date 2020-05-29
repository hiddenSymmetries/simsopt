# VMEC Python Wrapper
This is the python wrapper for VMEC to be interfaced with stellarator optimization codes.

Contributor: Caoxiang Zhu

## Install
1. Download VMEC source code from STELLOPT.
	```
	git clone https://github.com/PrincetonUniversity/STELLOPT.git
	```

2. Download required numerical libraries used by VMEC.
	- NetCDF library: Availble from your operating system package management or from [Unidata](http://www.unidata.ucar.edu).
	- MPI library: Any mpi library will work. For local machines there is [OpenMPI](https://www.open-mpi.org).
	- LAPACK/BLAS library: Should be available with your operating system or compiler.
	- SCALAPACK library: Availble from your operating system package management or from [SCALAPACK](http://www.netlib.org/scalapack/).

	Details of STELLOPT compilation can be found at [STELLOPT Compilation](https://princetonuniversity.github.io/STELLOPT/STELLOPT%20Compilation).
	Remember to set `STELLOPT_PATH` and configure `make.inc`.

3. Download required Python packages.
	- [numpy](https://numpy.org/) (Basic math package, also contains [F2PY](https://numpy.org/devdocs/f2py/index.html) . Install via `pip install numpy`)
	- [MPI4PY](https://mpi4py.readthedocs.io/en/stable/install.html) (MPI packages for python. Install via `pip install mpi4py` with `MPICC` env)
	- [f90wrap](https://github.com/jameskermode/f90wrap) (Optional: enhanced Fortran wrapper for python. Install via `pip install git+https://github.com/jameskermode/f90wrap`)

4. Make
	Right now, there are two options available.
		- Native F2PY
				```
		make -f makefile.f2py vmec
		```
	  
		  - f90wrap
			  ```
	   make -f makefile.f90wrap all
	   ```

## How to use
The user is recommended to use a python class [vmec_class.py](vmec_class.py) for calling VMEC. A simple example is shown below.
```python
import sys
sys.path.append('simsopt/modules/VMEC') # change it if necessary
from mpi4py import MPI
import numpy as np
from vmec_class import VMEC

# initialize communicators
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
fcomm = comm.py2f()

# call and run VMEC
input_file = '../../examples/VMEC/input.QAS' # change it if necessary
QAS = VMEC(input_file=input_file, verbose=True, comm=fcomm)
QAS.run(iseq=rank)
```
There is an example at [examples/VMEC/](../../examples/VMEC). More documentation will be generated automatically using `sphinx`.
