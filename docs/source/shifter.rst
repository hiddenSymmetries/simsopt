.. _shifter_doc:

Shifter container
================

`Shifter <https://docs.nersc.gov/development/shifter/>`_ is the home-grown container technology deployed at NERSC to circumvent the 
security issues associated with Docker containers. Shifter allows to you use the simsopt Docker
image files hosted on Docker Hub. This page explains on how to take advantage of simsopt Shifter container
so you can run simulation without installing simsopt at NERSC

Shifter Images
^^^^^^^^^^^^^^
Shifter converts Docker images and virtual machines into a common format. 
`After connecting to a NERSC login node <https://docs.nersc.gov/connect/>`_ check for the simsopt shifter images.

.. code-block::

   shifterimg images | grep simsopt 

You should see multiple images similar to ``docker:hiddensymmetries/simsopt:v0.7.0``. If the version you are interested in
is not available, you can pull it by running

.. code-block::

   shifterimg -v pull docker:hiddensymmetries/simsopt:<version_no>

where ``<version_no>`` is the version of your choice, which is referred to as tag in docker parlance. Once the image is pulled,
the corresponding shifter image is made available to all users at NERSC.

.. warning::

   The `master` branch has the tag ``latest``. The image shown by `shifterimg images` may be stale becaues master branch is always changing. 
   Always re-pull the image if you want to use `master` branch, but keep in mind the results may not be reproducible. For reproducible
   data, users are strongly encouraged to use a container with specific version number.

Simsopt Specifics
^^^^^^^^^^^^^^^^^
Simsopt is installed inside a python virtual environment within the simsopt Docker container. On entry, the Docker container automatically activates the python virtual environment. However, Shifter container does not run entrypoint commands unless explicitly told, so the virtual environment is not activated. The full path for the python installed inside the virtual environment `/venv/bin/python` has to be used. 


Running Shifter Container 
^^^^^^^^^^^^^^^^^^^^^^^^^

Login Nodes
-----------
One can run Shifter on login nodes for small serial jobs. To do so, run

.. code-block::

   shifter --image=docker:hiddensymmetries/simsopt:latest --volume=<actual_path_on_cori:container_path> <script_inside_container> [args]

In the above command, for the volume argument two paths need to be supplied, the first one is the full path on cori 
and the second path is the mounted volume inside the container. The two paths are separated by a colon, `:`. One could use 
environment variables such as `$SCRATCH` as arguments to the `volume` flag. Multiple folders can be mounted by supplying 
the `nersc_path1:mounted_path1,nersc_path2:mounted_path2` pairs separted by commas as argument to the `volume` flag. If 
the script is not inside the container, you could bring it inside the container by mounting the folder where the script 
is located using the `volume` flag. Do not forget to use the container pathname. 

With the above approach, the below command could be used to run simsopt Shifter container interactively

.. code-block::

   shifter --image=docker:hiddensymmetries/simsopt:latest --volume=$SCRATCH:/work_dir /bin/bash
   cd /work_dir

The second command `cd /work_dir` takes you to the scratch folder, which is where you want to do more work. Please keep in mind that the bash prompt does not change. So it may appear that the above command does not work. To check if you are inside the simsopt Shifter container, you can run 

.. code-block::

   cat /etc/lsb-release

The output should show `DISTRIB_ID=Ubuntu` along with some other lines.

.. warning::

   Do not abuse the interactive capability by running large scale jobs on login nodes.

Compute Nodes
-------------
The main reason for using Shifter is to run simsopt parallelly with multiple MPI processes on NERSC.
Here interactive slurm job with salloc is shown, but you can use the same logic to submit slurm batch jobs.

Run salloc to get interactive session

.. code-block::
   
   salloc --constraint=haswell -N 1 -p debug --image=hiddensymmetries/simsopt:latest --volume=<cori_path>:/work_dir -t 00:30:00

In the above command, pay attention to ``image`` and ``volume`` options. Both of these shifter related options are passed 
to the slurm commands directly. Here you can not use environment variables such as `$SCRATH`. In their place, full path 
names have to be supplied. The `--constraint=haswell` option means we want to run our job on Haswell nodes on cori. The `-N 1` option specifies that we want one Haswell node, `-p debug` implies debug queue, and `-t 00:30:00` specifies 30 minutes of allocation time for this job.
After some time, resources are allocated and you can run your jobs. Assuming, simsopt was downloaded to `$SCRATCH` folder and `<cori_path>` refers to the output of `echo $SCRATCH` in the above command, you can run the one of the examples in simsopt as
 
.. code-block::
   
   srun -n 4 shifter  /venv/bin/python /work_dir/simsopt/examples/1_Simple/tracing_fieldline.py 

Please remember that one cori Haswell node has 24 cores, so you can use any number upto 24 in place of 4 in the above command.
You can also run the parallel unit tests, by running 
 
.. code-block::
   
   srun -n 4 shifter  /venv/bin/python -m unittest discover -t /work_dir/simsopt/tests -k mpi

