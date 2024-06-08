.. _container_doc:

Containers
**********

.. _docker_doc:

Docker container
================

A Docker container for simsopt is available, allowing you to use
simsopt without having to compile any code yourself.  The container
includes VMEC, SPEC, and BOOZ_XFORM.

.. warning::

   Docker is not generally allowed to run on computers at HPC centers due to security issues.
   For those wishing to run simsopt on NERSC machines, please refer to :ref:`shifter_doc`.

Requirements
^^^^^^^^^^^^
Docker needs to be installed before running the docker container. Docker
software can be obtained from `docker website <https://docs.docker.com/get-docker/>`_.
Check the `docker get started webpage <https://docs.docker.com/get-started/>`_ for installation instructions 
as well as for tutorials to get a feel for docker containers. On linux, you may need to start the docker daemon
before proceeding further.

.. warning::

   On Mac, the default 2 GB memory per container assigned by Docker Desktop is not sufficient. Increase the memory of
   the container to at least 3 GB to run simsopt much faster.

Install From Docker Hub
^^^^^^^^^^^^^^^^^^^^^^^
The easiest way to get simsopt docker image which comes with simsopt and all of its dependencies such as
SPEC and VMEC pre-installed is to use Docker Hub. After 
`installing docker <https://docs.docker.com/get-started/>`_, you can run
the simsopt container directly from the simsopt docker image uploaded to
Docker Hub.

.. code-block::

   docker run -it --rm hiddensymmetries/simsopt python # Linux users, prefix the command with sudo

The above command should load the python shell that comes with the simsopt
docker container. When you run it first time, the image is downloaded
automatically, so be patient.  You should now be able to import the module from
python::

  >>> import simsopt

Ways to use simsopt docker container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

IPython Shell
-------------

Easiest way is to start ipython shell and import the simsopt
library. But this approach is only useful if a few commands need to be
executed or to load a python module and execute it.

.. code-block::

    docker run -it --rm hiddensymmetries/simsopt ipython

Bash Shell
----------

In this approach, you write a simsopt based driver script for your optimization problem. One
needs to mount the host directory in the container before running scripts. Use the ``-v`` flag 
to mount the current directory

.. code-block:: 

    docker run -it --rm -v $PWD:/my_mount hiddensymmetries/simsopt 
    <container ###> cd /my_mount
    <container ###> python <driver_script>

Jupyter notebook
----------------

The simsopt container comes with jupyter notebook preloaded. You can launch the jupyter from
the container using the command:

.. code-block::
   
    docker run -it --rm -v $PWD:/my_mount -p 8888:8888 hiddensymmetries/simsopt 
    <container ###> cd /my_mount
    <container ###> jupyter notebook --ip 0.0.0.0 --no-browser --allow-root 

Running the above command, a link will be printed that you can use to
open the jupyter console in a web browser. The link typically starts
with ``http://127.0.0.1:8888/?token=``. (Several links are printed,
but only the last one listed will work for browsers running outside
the container.) Copy the full link and paste it into any browser in
your computer.


Persistent containers
^^^^^^^^^^^^^^^^^^^^^

Using the intructions above will create a fresh container each time and delete the container after exiting.
If you would like to create a persistent container (e.g. because you are installing additional pip packages inside) that you can reuse at any time,
you can do so by removing the ``--rm`` command and specifying a container name via ``--name=``

.. code-block::

    docker run --name=mycontainer -it -v $PWD:/my_mount hiddensymmetries/simsopt
    <container ###> cd /my_mount
    <container ###> python <driver_script>

And to restart and rejoin the container:

.. code-block::

    docker start mycontainer
    docker exec -it mycontainer /bin/bash
    <container ###> source /venv/bin/activate



.. _shifter_doc:

Shifter container
=================

`Shifter <https://docs.nersc.gov/development/shifter/>`_ is the
container technology deployed at NERSC to circumvent the security
issues associated with Docker containers. Shifter allows to you use
the simsopt Docker image files hosted on Docker Hub.  Detailed
instructions for using Shifter can be found at the `NERSC page on the
simsopt wiki
<https://github.com/hiddenSymmetries/simsopt/wiki/NERSC-Perlmutter>`_.


.. _singularity_doc:

Singularity container
=====================

`Singularity <https://en.wikipedia.org/wiki/Singularity_(software)>`_ is the
container technology developed at Lawrence Berkeley National Lab to run containers on HPC centers.
There are two versions of Singularity, the community version now renamed as Apptainer, and the commercial
version from sylabs.io. Singularity has its own image format and Singularity images are given ``.sif`` extension. 
Singularity also allows one to use the Docker image files hosted on Docker Hub or other registries. 
For simsopt, we developed a native Singularity image, which is hosted as a Github package.
This sections
explains on how to take advantage of the simsopt Singularity container so
you can use simsopt at HPC centers that support Singularity without compiling any code.

Singularity Images
^^^^^^^^^^^^^^^^^^

Here we describe how to use simsopt Singularity container on `Stellar cluster located at Princeton University <https://researchcomputing.princeton.edu/systems/stellar>`_. The steps to run simopt Singularity container at other HPC centers shuould be similar to the ones described here. 
format.  `After logging to a Stellar login node
<https://researchcomputing.princeton.edu/systems/stellar#access>`_ check for the singularity
executable:

.. code-block::

   which singularity

Pull simsopt Singularity image file from GitHub by running

.. code-block::

   singularity pull oras://ghcr.io/hiddensymmetries/simsopt:<version_no>

where ``<version_no>`` is the version of your choice, which is
referred to as tag in docker parlance. Once the image is pulled, it
can be found by typing

.. code-block::

   ls simsopt_<version_no>.sif

.. warning::

   The ``master`` branch has the tag ``latest``. This version of image 
   could be stale becaues master branch is always
   changing.  Always re-pull the image if you want to use ``master``
   branch, but keep in mind the results may not be reproducible. For
   reproducible data, users are strongly encouraged to use a container
   with specific version number.

Simsopt Specifics
^^^^^^^^^^^^^^^^^

Simsopt is installed inside a python virtual environment within the
simsopt Singularity container.  The full path for the python executable
installed inside the virtual environment
``/venv/bin/python`` has to be used. Singularity container comes pre-installed with
OpenMPI v4.1.2, which communicates with resource managers such as slurm via PMIx. 


Running the Singularity Container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Serial Jobs
-----------

One can run Singularity on login nodes for small serial jobs. To run a
simsopt python driver script (located in your usual filesystem), you
can type

.. code-block::

   singularity run simsopt_<version_no>.sif /venv/bin/python <script_name>

You can also run the simsopt Singularity container interactively, with

.. code-block::

   singularity run simsopt_<version_no>.sif /venv/bin/python

to enter the python interpreter, or

.. code-block::

   singularity run simsopt_<version_no>.sif /bin/bash

for a shell. 

.. warning::

   Do not abuse the interactive capability by running large scale jobs on login nodes.

Parallel Jobs
-------------

The parallel jobs are demoed with an interactive slurm job using
``salloc`` is shown, but you can use the same logic to submit slurm
batch jobs.

Run salloc to get an interactive session:

.. code-block::
   
   salloc --nodes=1 --ntasks=4 --mem-per-cpu=4G --time=00:20:00

The options to sallaoc are HPC center dependent and the ones shown above are 
specific to Princeton University's Stellar cluster. ``--nodes=1`` option means we want
to run our job on one node. The ``--ntasks=4`` option requests
4 cores on one node,  ``--mem-per-cpu=4G`` request 4 gigabytes of memory per core totaling 16 gigabytes and
``--time=00:20:00`` specifies 20 minutes of allocation time for this job.
After some time, resources are allocated and you can run your jobs. If
you have navigated to a clone of the simsopt repository, you
can run the one of the examples as

.. code-block::
   
   cd <SIMSOPT_ROOT>
   module load openmpi/gcc/4.1.0
   PMIX_MCA_psec=native  mpirun -n 4 singularity run <PATH_TO_SINGULARITY_IMAGE_FILE> /venv/bin/python examples/1_Simple/tracing_fieldline.py 

Pay attention to the PMIx specific environment variable ``PMIX_MCA_psec``.
Specifying ``native`` allows mpirun to use the PMIx installed on the cluster.
Because of this, the same example can be executed by launching the mpirun from
inside the container.

.. code-block::

   cd <SIMSOPT_ROOT>
   module load openmpi/gcc/4.1.0
   PMIX_MCA_psec=native  singularity run <PATH_TO_SINGULARITY_IMAGE_FILE> mpirun -n 4 /venv/bin/python examples/1_Simple/tracing_fieldline.py 

Both approaches take similar time. Alternatively, one can also use ``srun`` command from slurm without loading the openmpi module.
To use srun, we have to specify the PMI standard using the ``mpi`` flag. On Stellar, the following command works.

.. code-block::

   cd <SIMSOPT_ROOT>
   PMIX_MCA_psec=native srun -n 4 --mpi=pmix_v3 singularity run <PATH_TO_SINGULARITY_IMAGE_FILE> /venv/bin/python examples/1_Simple/tracing_fieldline.py 

To know the options that can be supplied for ``mpi`` flag, you can execute

.. code-block::

   srun --mpi=list

which will give a list of PMI standards supported by slurm.
Similarly to run the tests, you can run the below command.

.. code-block::
   
   PMIX_MCA_psec=native srun -n 4 --mpi=pmix_v3 singularity run <PATH_TO_SINGULARITY_IMAGE_FILE> /venv/bin/python -m unittest discover -v -k mpi -s tests

