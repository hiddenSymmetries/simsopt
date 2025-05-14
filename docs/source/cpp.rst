Simsopt C++ backend
*******************

``SIMSOPT`` uses C++ for performance critical functions such as the Biot Savart law, many of the geometric classes, and particle tracing.
This section is aimed at advanced developers of ``SIMSOPT`` to give an overview over the interface between C++ and Python and to help avoid common pitfalls. For most users of ``SIMSOPT`` this is not relevant.

The C++ code can be found in the folder :simsopt:`src/simsoptpp`.


pybind11
^^^^^^^^

To expose the C++ code to python, we use the 
`pybind11 <https://github.com/pybind/pybind11>`_ library.
The interfacing happens in the ``python.cpp`` and ``python_*.cpp`` files.

Trampoline classes:
In many cases we define some parent class in C++ that has virtual functions and then we want to inherit from this class on the python side and overload these functions.
A good example for this is the :mod:`simsoptpp.MagneticField` class. This is the base class for magnetic fields and it takes care of things like caching, or coordinate systems for evaluating magnetic fields. When you create a new magnetic field, you overload functions such as ``B_impl``, in order to compute the magnetic field at given locations. In order for this overload to work on the python side, pybind requires so called `trampoline classes <https://pybind11-jagerman.readthedocs.io/en/latest/advanced/classes.html#overriding-virtual-functions-in-python>`_. In the case of magnetic fields, this can be found in :simsoptpp_file:`pymagneticfield.h`.


Lifetime of objects:
memory management in hybrid Python/C++ codes can be difficult. To guarantee that objects managed by C++ are not deleted even though they are still used on the Python side, we make sure that we hold a reference to them in the Python code. A good example for this is the :mod:`simsopt.field.biotsavart.BiotSavart` class that keeps a reference to the underlying coils. See `this pull request <https://github.com/hiddenSymmetries/simsopt/pull/147>`_ for a discussion of this issue and further references.

xtensor and xtensor-python
^^^^^^^^^^^^^^^^^^^^^^^^^^

We use ``numpy`` array heavily on the python side; in order to use these (without copying them) in the C++ code, we employ the `xtensor <https://github.com/xtensor-stack/xtensor>`_ library and its `xtensor-python <https://github.com/xtensor-stack/xtensor-python>`_ interface to python.

``xarray`` vs ``pyarray``:

One pitfall to be aware of here is the following: there are different types for arrays that are managed from the python side, vs those managed on the C++ side. This means that for code that is purely C++, one should use ``xarray`` from ``xtensor``, but for code that is used from python, one uses ``pyarray`` from ``xtensor-python``. While most of ``simsoptpp`` is only ever used from the python side, it can be useful for profiling purposes (or for future usecases of the library by other C++ codes) to allow for the use of ``xarray``. For this reason, many functions and classes are templated on the data type, so that either is possible.


OpenMP
^^^^^^
Some of the code is parallelized using OpenMP. OpenMP can be turned of by setting
``export OMP_NUM_THREADS=1``
before running a ``SIMSOPT`` script. This is recommended when debugging bugs that are assumed to be in the C++ code. We have found that creating new ``xtensor`` arrays or tensors in OpenMP threads leads to memory issues, so we always create those in the serial part of the code and then simply fill them in parallel.


SIMD
^^^^
For simple computations that are compute bound, we use SIMD (`Single Instruction Multiple Data <https://en.wikipedia.org/wiki/Single_instruction,_multiple_data>`_) instructions to make use of the AVX/AVX2/AVX512 instruction sets on modern CPUs. To simplify the use of these instructions, we use the `xsimd <https://github.com/xtensor-stack/xsimd>`_ library.

CMake
^^^^^

When editing the C++ code, it may be useful to use ``CMake`` and ``make`` directly to only recompile those parts of the code that changed. This can be achieved as follows.

First, install ``SIMSOPT`` in the usual way::

    git clone --recursive git@github.com:hiddenSymmetries/simsopt.git
    cd simsopt
    pip install -e .

Install ``cmake`` and ``pybind11`` using ``pip`` if they are not already installed::

    pip install cmake
    pip install pybind11

Then, recompile the C++ code in a new directory (``cmake_build``)::

    mkdir cmake-build
    cd cmake-build
    cmake .. -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.__path__[0])")/share/cmake/pybind11/
    make -j

If the compilation executed successfully, there should now be a shared object file (ending with ``.so``) for ``simsoptpp`` in the ``cmake-build`` directory. After verifying that this ``.so`` file exists, execute the following command::

    ln -sf $(realpath *.so) $(python -c "import simsoptpp; print(simsoptpp.__file__)")

From then on, you can always just call ``make -j`` inside the ``cmake-build`` directory to recompile the C++ part of the code.
