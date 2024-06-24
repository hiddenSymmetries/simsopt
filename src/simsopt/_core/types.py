# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""
This module contains small utility functions and classes.
"""

from typing import Union, Sequence, Any
from numbers import Integral, Real

# from nptyping import NDArray, Float, Int, Bool
import numpy as np
from numpy.typing import NDArray

# RealArray is a type alias for numpy double array or RealArray
RealArray = Union[Sequence[Real], NDArray[np.double]]
IntArray = Union[Sequence[Integral], NDArray[np.int_]]
StrArray = Sequence[str]
BoolArray = Union[Sequence[bool], NDArray[np.bool]]
Key = Union[Integral, str]

