Getting started
===============


Requirements
^^^^^^^^^^^^

``simsopt`` requires python version 3.6+
and some mandatory python packages, listed in
``requirements.txt``.  These packages are all installed automatically
when you install using ``pip``, as discussed below.  If you prefer to
install via ``python setup.py install`` or ``python setup.py
develop``, you will need to install these python packages manually
using ``pip`` or another python package manager such as ``conda``.

Mandatory Packages
------------------
- numpy
- jax
- jaxlib
- scipy
- pandas
- nptyping
- ruamel.yaml
- importlib_metadata if python version is less than 3.8

Optional Packages
-----------------
- For MPI support:
    * mpi4py
    * MPILogger
- For SPEC support:
    * py_spec
    * pyoculus
    * h5py
    * f90nml
- For VMEC support
    * https://github.com/hiddenSymmetries/vmec2000 (For VMEC interface)
    * f90nml
- For quasisymmetry optimization
    * `booz_xform <https://hiddensymmetries.github.io/booz_xform/>`_,

For requirements of separate physics modules like VMEC, see the
documentation of the module you wish to use.


Installation
^^^^^^^^^^^^

From PyPi
---------
Currently, older versions of ``simsopt`` are avaialable at test.pypi.org.
You can install the latest among them using

.. code-block::

    pip install --index-url https://test.pypi.org/simple/ simsopt
    
From Source
-----------
This is the preferred method. First, install ``git`` if not already installed. Then clone the repository using

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

On some systems, you may not have permission to install packages to
the default location. In this case, add the ``--user`` flag to ``pip``
so the package can be installed for your user only::

    pip install --user -e .

From docker image
-----------------
A docker image with simsopt along with all of its dependencies such as
SPEC and VMEC pre-installed is available from docker hub. After 
`installing docker <https://docs.docker.com/get-docker/>`_, you can run
the simsopt container directly from the simsopt docker image uploaded to
Docker Hub.

.. code-block::

   docker run -it --rm medbha/simsopt_ubuntu_focal python

The above command shoud load the python shell that comes with the simsopt
docker container. When you run it first time, the image is downloaded
automatically, so be patient.

Post-Installation
-----------------

If the installation is successful, ``simsopt`` will be added to your
python environment. You should now be able to import the module from
python::

  >>> import simsopt

