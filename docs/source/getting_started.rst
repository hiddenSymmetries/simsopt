Getting started
===============


Requirements
^^^^^^^^^^^^

``simsopt`` is a python package focused on stellarator optimization and requires
python 3.6+.  ``simsopt``
also requires some mandatory python packages, listed in
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
- ruamel.yaml

Optional Packages
-----------------
- mpi4py (for MPI)
- py_spec
- pyoculus
- h5py
- f90nml (Last four packages for interfacing with SPEC code)
- https://github.com/hiddenSymmetries/vmec2000 (For VMEC interface)
- `booz_xform <https://hiddensymmetries.github.io/booz_xform/>`_,
  for optimizing for quasisymmetry

For requirements of separate physics modules like VMEC, see the
documentation of the module you wish to use.


Installation
^^^^^^^^^^^^

From PyPi
---------
Currently, ``simsopt`` is avaialable at test.pypi.org.
You can install the latest version of simsopt using 

.. code-block::

    pip install --index-url https://test.pypi.org/simple/ simsopt
    
From Source
-----------
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

On some systems, you may not have permission to install packages to
the default location. In this case, add the ``--user`` flag to ``pip``
so the package can be installed for your user only::

    pip install --user -e .

Post-Installation
-----------------

If the installation is successful, ``simsopt`` will be added to your
python environment. You should now be able to import the module from
python::

  >>> import simsopt

