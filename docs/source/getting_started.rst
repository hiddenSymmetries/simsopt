Getting started
===============


Requirements
^^^^^^^^^^^^

``simsopt`` requires python 3.x.  For now, ``simsopt`` itself (as
opposed to the separate physics modules like VMEC) is pure python, and
so it does not have any mandatory requirements other than certain
python packages, listed in ``requirements.txt``.  These packages are
all installed automatically when you install using ``pip``, as
discussed below.  If you prefer to install via ``python setup.py
install``, you will need to install these python packages manually
using ``pip`` or another python package manager such as ``conda``.

For requirements of separate physics modules like VMEC, see the
documentation of the module you wish to use.

Installation
^^^^^^^^^^^^

Installation from PyPI is not yet available, but it is planned soon.

First, clone the repository using

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


If the installation is successful, ``simsopt`` will be added to your
python environment. You should now be able to import the module from
python::

  >>> import simsopt

On some systems, you may not have permission to install packages to
the default location. In this case, add the ``--user`` flag to ``pip``
so the package can be installed for your user only::

    pip install --user -e .

