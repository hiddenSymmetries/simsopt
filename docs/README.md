# Building documentation 

This is a guide to building documentation locally. Online documentation gets
updated whenever master branch is updated.

## Prerequsites

1. Install *doxygen*. On Linux machines, you can use system package managers to install doxygen.
   On Mac use homebrew.

2. Install *sphinx*, *sphinx-rtd-theme*, *sphinxcontrib-napoleon*,
*sphinx-autodoc-napoleon-typehints*, and *breathe*  with pip.

## Build
1. Use sphinx-build

```bash
cd docs
sphinx-build -b html source build
```
The documentation is generated in html format and is in the **docs/build**
folder. Start with index.html file.

2. Use the supplied makefile

```bash
cd docs
make html 
```
The documentation is generated in html format and is in the **docs/build/html**
folder. Start with index.html file.

## Update the documentation

Whenever the code is updated, repopulate the code tree and run either step-1  or step-2.

```bash
cd docs
sphinx-apidoc -f -o source ../src/simsopt 
```

## Refering to code on GitHub

To point to the simsopt repo on GitHub, use the directives ``simsopt``, ``simsoptpy``, ``simsoptpp``, ``examples``, ``tests``.
They linkage is given below:

*. ``simsopt``: https://github.com/hiddenSymmetries/simsopt
*. ``simsoptpy``: https://github.com/hiddenSymmetries/simsopt/tree/master/src/simsopt
*. ``simsoptpp``: https://github.com/hiddenSymmetries/simsopt/tree/master/src/simsoptpp
*. ``examples``: https://github.com/hiddenSymmetries/simsopt/tree/master/examples

So, to point to the ``ci`` folder, use :simsopt:`ci`. Similarly to point to the ``geo`` folder, we use :simsoptpy:`geo`.
For pointing to files, use ``simsoptpy_file``, ``simsoptpp_file``, ``example_file``, ``tests_file``.
