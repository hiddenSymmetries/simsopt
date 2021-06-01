simsopt in a docker container
=====================================

Requirements
^^^^^^^^^^^^
Docker needs to be installed before running the docker container. Docker
software can be obtained from `docker website <https://docs.docker.com/get-docker/>`_.
Check the `docker get started webpage <https://docs.docker.com/get-started/>`_ for installation instructions 
as well as for tutorials to get a feel for docker containers. On linux, you may need to start the docker daemon
before proceeding further.

Install From Docker Hub
^^^^^^^^^^^^^^^
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
------------

Easiest way is to start ipython shell and import the simsopt library. But this approach
is only useful if a few commands need to be executed or to load a python module and execute it.

.. code-block::

    docker run -it --rm hiddensymmetries/simsopt ipython

Bash Shell
----------

In this approach, you write a simsopt based driver script for your optimization problem. One
needs to mount the host directory in the container before running scripts. Use the `-v` flag 
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
    <container ###> jupyter notebook --ip 0.0.0.0 --no-browser # On linux use --allow-root option 

Running the above command gives a link to open the jupyter console on the browser that typically 
starts with **http://127.0.0.1:8888/?token=**. Copy the full link and paste it on any browser in your
computer.


