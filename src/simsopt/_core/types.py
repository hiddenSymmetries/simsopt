# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module contains small utility functions and classes.
"""

from typing import Union, Sequence, Any
from numbers import Integral, Real

from nptyping import NDArray, Float, Int, Bool

# Array = Union[Sequence, NDArray]
RealArray = Union[Sequence[Real], NDArray[Any, Float]]
IntArray = Union[Sequence[Integral], NDArray[Any, Int]]
StrArray = Sequence[str]
BoolArray = Union[Sequence[bool], NDArray[Any, Bool]]
Key = Union[Integral, str]

