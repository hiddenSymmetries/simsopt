# Simsopt testing framework

## Overview

This directory contains integrated/regression tests. Source code for unit tests of each component is stored in the subdirectory for that component.

The layout of the subfolders within **tests** nearly mimics that of the simsopt code in **src/simsopt** folder. The test files (inputs, outputs or any other data files) are all collected into **tests/test_files**.

## Running tests

### With unittest 

To run the tests, you must first install simsopt with `pip install .` from the main simsopt directory.
Then, change to the `tests` directory, and run `python -m unittest` to run all tests.

To run unittests only in one folder, for example, `geo`, run `python -m unittest -t . -s geo`.

To run unittests only in one file for example `geo/test_surface.py", run `python -m unittest geo.test_surface`.

To run only a single suite of unittests such as `QuadpointsTests` in `geo/test_surface.py` run `python -m unittest geo.test_surface.QuadpointsTests`

In this fashion, we can run only a single unit test: `python -m unittest geo.test_surface.QuadpointsTests.test_theta`.

See the [python unittest documentation](https://docs.python.org/3/library/unittest.html) for more options.


### Parallel testing with pytest

Requires installing pytest along with pytest-xdist or pytest-parallel.  Install either of them with pip. With pytest-xdist, run `pytest -n <N> geo` to run all tests in the `geo` folder in parallel. For pytest-parallel, use `pytest --workers <N> geo`. Here <N> is the number of cores you want to use. In place of a specific number for <N>, use `auto` to automatically use all the avaialable cores in the machine.
