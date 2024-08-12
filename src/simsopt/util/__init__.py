import os

from .mpi import *
from .logger import *

"""Boolean indicating if we are in the GitHub actions CI"""
in_github_actions = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']

__all__ = (
    mpi.__all__ 
    + logger.__all__ 
    + ['in_github_actions']
)
