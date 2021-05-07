# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module contains small utility functions and classes.
"""

from typing import Union, Sequence
from numbers import Integral, Real

from nptyping import NDArray, Float, Int, Bool

Array = Union[Sequence, NDArray]
RealArray = Union[Sequence[Real], NDArray[Float]]
IntArray = Union[Sequence[Integral], NDArray[Int]]
StrArray = Union[Sequence[str], NDArray[str]]
BoolArray = Union[Sequence[bool], NDArray[Bool]]
Key = Union[Integral, str]

