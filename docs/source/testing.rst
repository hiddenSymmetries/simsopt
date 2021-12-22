Testing
^^^^^^^

``simsopt`` includes unit and regression tests, and continuous integration.

Python test suite
*****************

The main test suite is based on the standard ``unittest`` python module.
Source code for the python tests is located in the ``tests`` directory.
These tests will use the installed version of the ``simsopt`` python package,
which may differ from the code in your local repository if you did not
make an editable install (see :doc:`installation`).

To run all of the tests in the test suite on one processor, you can type

.. code-block::

    ./run_tests

from the command line in the repository's home directory. Equivalently,
you can run

.. code-block::

    python -m unittest

from the ``tests`` directory.

For some of the tests involving MPI, it is useful to execute the tests
for various numbers of processes to make sure the tests pass in each
case. For this purpose, it is not necessary to run the entire test
suite, only the tests for which MPI is involved.  For convenience, you
can run the script

.. code-block::

    ./run_tests_mpi

in the repository's home directory. This script runs only the tests
that have ``mpi`` in the name, and the tests are run on 1, 2, and 3
processors.

The tests make use of data files in the ``tests/test_files`` directory.


Longer examples
***************

For convenience, the main test suite is designed to run in no more than a few minutes.
This means that some more complicated integrated and regression tests that require substantial time
are not included. You may wish to run some of these more complicated tests by hand during development.
A number of such examples can be found in the ``examples`` subdirectory.
Also, ``simsopt`` and ``stellopt`` have been benchmarked for several problems in the
`stellopt_scenarios collection <https://github.com/landreman/stellopt_scenarios>`_,
which includes the corresponding ``simsopt`` input files.


Continuous integration
**********************

The serial and MPI tests are automatically run after every commit to
the repository.  This automation is handled by GitHub Actions, and
controlled by the script ``.github/workflows/ci.yml``.
To view the results of the continuous integration runs, you can click on the "Actions"
link from the `GitHub repository page <https://github.com/hiddenSymmetries/simsopt>`_,
or you can directly visit `<https://github.com/hiddenSymmetries/simsopt/actions>`_.
