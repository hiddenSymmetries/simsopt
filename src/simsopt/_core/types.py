# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""
This module contains small utility functions and classes.
"""

from typing import Union, Sequence  # , Any
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

RealArray = Union[Sequence[Real], NDArray[np.double]]
IntArray = Union[Sequence[Integral], NDArray[np.int_]]
StrArray = Sequence[str]
BoolArray = Union[Sequence[bool], NDArray[bool]]
Key = Union[Integral, str]
