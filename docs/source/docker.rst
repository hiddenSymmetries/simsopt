Docker container
================

A Docker container for simsopt is available, allowing you to use
simsopt without having to compile any code yourself.  The container
includes VMEC and BOOZ_XFORM.

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
