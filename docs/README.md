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
