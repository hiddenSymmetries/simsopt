Installation
============

This page provides general information on installation.  Detailed
installation instructions for some specific systems – such as `Mac <https://github.com/hiddenSymmetries/simsopt/wiki/Mac-Conda-script-installation>`_ or `NERSC Perlmutter <https://github.com/hiddenSymmetries/simsopt/wiki/NERSC-Perlmutter>`_ – can also be found
on `the wiki <https://github.com/hiddenSymmetries/simsopt/wiki>`_.

Requirements
^^^^^^^^^^^^

.. _simsopt: https://github.com/hiddenSymmetries/simsopt

`simsopt`_ is a python package focused on stellarator optimization
and requires python version 3.9 or higher. `simsopt`_ also requires
some mandatory python packages, listed in
:simsopt_file:`requirements.txt`
and in the ``[dependencies]`` section of
:simsopt_file:`pyproject.toml`.
These packages are all installed automatically when you install simsopt using
``pip`` or another python package manager such as ``conda``, as
discussed below.  You can install
these python packages manually using ``pip`` or ``conda``, e.g.
with ``pip install -r requirements.txt``.

Optional Packages
-----------------

Several other optional packages are needed for certain aspects of
simsopt, such as running SPEC or VMEC, visualization/graphics, and building the
documentation.  These requirements can be found in the
``[project.optional-dependencies]`` section of :simsopt_file:`pyproject.toml`.
Also,

- For MPI support:
    * `mpi4py <https://github.com/mpi4py/mpi4py>`_
- For VMEC support:
    * `VMEC2000 <https://github.com/hiddenSymmetries/vmec2000>`_. Note that the
      python extension in this repository is required to run VMEC or
      optimize VMEC configurations, but is not needed for computing
      properties of existing ``wout`` output files.
- For computing Boozer coordinates:
    * `booz_xform <https://hiddensymmetries.github.io/booz_xform/>`_
- For SPEC support:
    * `py_spec <https://github.com/PrincetonUniversity/SPEC/tree/master/Utilities/pythontools>`_
    * `pyoculus <https://github.com/zhisong/pyoculus>`_
    * `h5py <https://github.com/h5py/h5py>`_

For requirements of separate physics modules like VMEC, see the
documentation of the module you wish to use.


Virtual Environments
^^^^^^^^^^^^^^^^^^^^


This is an optional step, but users are strongly encouraged to use a python virtual environment
to install simsopt. There are two popular ways to create a python virtual environment using 
either `venv <https://docs.python.org/3/library/venv.html>`_ module supplied with python or the conda virtual environment.

venv
----

A python virtual envionment can be created with venv using

.. code-block::

    python3 -m venv <path/to/new/virtual/environment>

Activate the newly created virtual environmnet (for bash shell)

.. code-block::
   
    . <path/to/new/virtual/environment>/bin/activate

If you are on a different shell, use the ``activate`` file with an appropriate extension reflecting the shell type.
For more information, please refer to `venv official documentation <https://docs.python.org/3/library/venv.html>`_.

conda
-----
Install either `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or `anaconda <https://www.anaconda.com/>`_.
If you are on a HPC system, anaconda is either available by default or via a module.

A conda python virtual environment can be created by running

.. code-block::

    conda create -n <your_virtual_env_name> python=3.10

For the new virtual environment, python version 3.10 was chosen in the above command, but you are free to choose any version you want. 
The newly created virtual environment can be activated with a simple command

.. code-block::

    conda activate <your_virtual_env_name>

After activating the conda virtual environment, the name of the environment should appear in the shell prompt.

Installation methods
^^^^^^^^^^^^^^^^^^^^

PyPi
----

This works for both venv and conda virtual environments.

.. code-block::

    pip install simsopt

Running the above command will install simsopt and all of its mandatory dependencies. To install
optional dependencies related to SPEC and MPI, run the following command:

.. code-block::

    pip install simsopt[MPI,SPEC]
    
On some systems, you may not have permission to install packages to
the default location. In this case, add the ``--user`` flag to ``pip``
so the package can be installed for your user only::

    pip install --user simsopt
    
conda
-----

A pre-compiled conda package for simsopt is available. This
installation approach works only with conda virtual environments.
First we need to add conda-forge as one of the channels.

.. code-block::

    conda config --add channels conda-forge
    conda config --set channel_priority strict

Then install simsopt by running

.. code-block::

    conda install -c hiddensymmetries simsopt


From source
-----------

This approach works for both venv and conda virtual environments.
First, install ``git`` if not already installed. Then clone the repository using

.. code-block::

    git clone https://github.com/hiddenSymmetries/simsopt.git

Then install the package to your local python environment with

.. code-block::

    cd simsopt
    pip install -e .

The ``-e`` flag makes the installation "editable", meaning that the
installed package is a pointer to your local repository rather than
being a copy of the source files at the time of installation. Hence,
edits to code in your local repository are immediately reflected in
the package you can import.

Again, if you do not have permission to install python packages to the
default location, add the ``--user`` flag to ``pip`` so the package
can be installed for your user only::

    pip install --user -e .
    
.. warning::
    Installation from local source creates a directory called **build**. If you are reinstalling simsopt from source after updating the code by making local changes or by git pull, remove the directory **build** before reinstalling.

If you want to build SIMSOPT locally with the optional dependencies,
you can run

.. code-block::

    pip install --user -e .[MPI,SPEC]

However, if you're using a zsh terminal (example: latest Macbook versions),
you'll need to run instead

.. code-block::

    pip install --user -e ".[MPI,SPEC]"


Docker container
----------------

A docker image with simsopt along with its dependencies, VMEC, SPEC,
and BOOZ_XFORM pre-installed is available from docker hub. This
container allows you to use simsopt without having to compile any code
yourself.  After `installing docker
<https://docs.docker.com/get-docker/>`_, you can run the simsopt
container directly from the docker image uploaded to Docker Hub.

.. code-block::

   docker run -it --rm hiddensymmetries/simsopt python

The above command should load the python shell that comes with the
simsopt docker container. When you run it first time, the image is
downloaded automatically, so be patient. More information about using
simsopt with Docker can be found :doc:`here <containers>`.

Post-Installation
^^^^^^^^^^^^^^^^^

If the installation is successful, ``simsopt`` will be added to your
python environment. You should now be able to import the module from
python::

  >>> import simsopt

