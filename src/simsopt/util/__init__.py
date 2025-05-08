import os

from .mpi import *
from .logger import *
from .famus_helpers import *
from .polarization_project import *
from .permanent_magnet_helper_functions import *

"""Boolean indicating if we are in the GitHub actions CI"""
in_github_actions = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']

__all__ = (
    mpi.__all__
    + logger.__all__
    + famus_helpers.__all__
    + polarization_project.__all__
    + permanent_magnet_helper_functions.__all__
    + ['in_github_actions']
)
