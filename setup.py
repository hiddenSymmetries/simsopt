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
        "scikit-build >= 0.11.1"
    ],
    package_dir={'': 'src'},
    packages= find_packages(
        where='src'),
    package_data = {
        "simsopt.mhd": ["input.default"]
    }
)
