Testing
^^^^^^^

``simsopt`` includes unit and regression tests, and continuous integration.

Python test suite
*****************

The main test suite is based on the standard ``unittest`` python module.
Source code for the python tests is located in the :simsopt:`tests` directory.
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

The tests make use of data files in the :simsopt:`tests/test_files` directory.

Modular testing
***************

Often you may want to run fewer tests focusing on a module or a submodule or a single class or a method inside a class.
To run all tests within a given folder, let's say ``geo``, run

.. code-block::

    python -m unittest discover -t . -s geo

from the ``tests`` directory. For evaluating tests within a single file, for example ``geo/test_curve.py``, run

.. code-block::

    python -m unittest geo.test_curve

from the ``tests`` directory. By using the ``dot`` operator, you can run a single test suite or a single test case also.

.. code-block::

    python -m unittest geo.test_curve.Testing.test_curve_helical_xyzfourier


Parallel Testing
****************

``unittest`` executes tests in serial, which takes a really long time especially when all the simsopt tests are checked. If you have a beefier machine with multiple cores, you can speed up testing by running ``pytest`` in parallel mode. Install ``pytest`` and ``pytest-xdist`` and then run

.. code-block::

    pytest -n <NUMBER_OF_CORES>

from the ``tests`` directory, where ``<NUMBER_OF_CORES>`` is the number of CPU cores of the machine.


Longer examples
***************

For convenience, the main test suite is designed to run in no more than a few minutes.
This means that some more complicated integrated and regression tests that require substantial time
are not included. You may wish to run some of these more complicated tests by hand during development.
A number of such examples can be found in the :simsopt:`examples` directory.
Also, ``simsopt`` and ``stellopt`` have been benchmarked for several problems in the
`stellopt_scenarios collection <https://github.com/landreman/stellopt_scenarios>`_,
which includes the corresponding ``simsopt`` input files.


Continuous integration
**********************

The serial and MPI tests are automatically run after every commit to
the repository.  This automation is handled by GitHub Actions, and
controlled by the script :simsopt_file:`.github/workflows/ci.yml`.
To view the results of the continuous integration runs, you can click on the "Actions"
link from the `GitHub repository page <https://github.com/hiddenSymmetries/simsopt>`_,
or you can directly visit `<https://github.com/hiddenSymmetries/simsopt/actions>`_.
