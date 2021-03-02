import sys
from setuptools import find_packages

try:
    from skbuild import setup
except ImportError:
    print('Please update pip, you need pip 10 or greater,\n'
          ' or you need to install the PEP 518 requirements in pyproject.toml yourself', file=sys.stderr)
    raise

setup(
    name="simsopt",
    version="0.0.3",
    install_requires=[
        "numpy >= v1.19.0",
        "monty >= v4.0.0",
        "mpi4py >= 3.0.3",
        "jax >= 0.2.4",
        "jaxlib >= 0.1.56",
        "scipy >= 1.5.4",
        "py_spec >= 3.0.1",
        "pyoculus >= 0.1.1",
        "h5py >= 3.1.0",
        "f90nml >= 1.2",
        "scikit-build >= 0.11.1"
    ],
    package_dir={'': 'src'},
    packages= find_packages(
        where='src'),
    include_package_data=True,
    package_data = {
        "": ["input.default", "defaults.sp"]
    }
)
