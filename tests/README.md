# Simsopt testing framework

## Overview

This directory contains integrated/regression tests. Source code for unit tests of each component is stored in the subdirectory for that component.

The layout of the subfolders within **tests** nearly mimics that of the simsopt code in **src/simsopt** folder. The test files (inputs, outputs or any other data files) are all collected into **tests/test_files**.

## Running tests

To run the tests, you must first install simsopt with `pip install .` from the main simsopt directory.
Then, change to the `tests` directory, and run `python -m unittest` to run all tests.
See the [python unittest documentation](https://docs.python.org/3/library/unittest.html) for more options.
