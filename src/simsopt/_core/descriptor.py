from abc import ABC, abstractmethod
from typing import Callable
from dataclasses import dataclass

import numbers

__all__ = ['OneofStrings', 'OneofIntegers', 'String', 'Integer', 'Float',
           'PositiveInteger', 'PositiveFloat']


class Validator(ABC):
    """
    Validator is the abstract base class implementing the descriptor protocol.
    It implements the ``__get__`` and ``__set__`` methods as well as the
    ``__set_name__`` method of the descriptor protocol. The implementation
    follows the strategy outlined in the official python documentation.
    """

    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = f"_{name}"

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        self.validate(value)
        setattr(obj, self.private_name, value)

    @abstractmethod
    def validate(self, value):
        """
        All subclasses implement validate method to check for various constraints
        """
        pass


class OneOf(Validator):
    """
    OneOf is used to define descriptors which take only specific values
    as inputs.

    Args:
        args: All the specific values that the descriptor can accept
    """

    def __init__(self, *args):
        self.options = set(args)

    def validate(self, value):
        if value not in self.options:
            raise ValueError(f'{value!r} is not a valid option for {self.public_name!r}.\n'
                             f'Valid options are one of {self.options!r}')
        super().validate(value)


@dataclass
class String(Validator):
    """
    Validates that the input is a string and implements some other checks
    for string object.

    Args:
        minsize: Minimum size of the input string
        maxsize: Maximum size of the input string
        predicate: Callable function that is used to do any other checks
            on the string
    """
    minsize: int = None
    maxsize: int = None
    predicate: Callable[[str], bool] = None

    def validate(self, value):
        if not isinstance(value, str):
            raise TypeError(f'Expected {value!r} to be an str')
        if self.minsize is not None and len(value) < self.minsize:
            raise ValueError(
                f'Expected {value!r} to be no smaller than {self.minsize!r}')
        if self.maxsize is not None and len(value) > self.maxsize:
            raise ValueError(
                f'Expected {value!r} to be no bigger than {self.maxsize!r}')
        if self.predicate is not None and not self.predicate(value):
            raise ValueError(
                f'Expected {self.predicate} to be true for {value!r}')
        super().validate(value)


class _Real(Validator):
    """
    Validates that the input is a real number.
    """

    def validate(self, value):
        if not isinstance(value, numbers.Real):
            raise TypeError(
                f'Expected {value!r} should be of Real type')
        super().validate(value)


class _Integral(Validator):
    """
    Validates that the input is an integer.
    """

    def validate(self, value):
        if not isinstance(value, numbers.Integral):
            raise TypeError(
                f'Expected {value!r} should be of Integer type')
        super().validate(value)


@dataclass
class RangeChecker(Validator):
    """
    Validates that the input number is within specific bounds.

    Args:
        min_value: Minimum acceptable value
        max_value: Maximum acceptable value
    """
    min_value: numbers.Real = None
    max_value: numbers.Real = None

    def validate(self, value):
        if self.min_value is not None and value < self.min_value:
            raise ValueError(
                f'Expected {value!r} to be at least {self.min_value!r}')
        if self.max_value is not None and value > self.max_value:
            raise ValueError(
                f'Expected {value!r} to be no more than {self.max_value!r}')
        super().validate(value)


class PositiveChecker(RangeChecker):
    """
    Validates that the input number is a positive number.
    """

    def __init__(self):
        super().__init__(min_value=0)


class Integer(_Integral, RangeChecker):
    """
    Validates that the input number is an integer. Optionally the user can specify
    the bounds for the number.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def validate(self, value):
        super().validate(value)


class PositiveInteger(Integer, PositiveChecker):
    """
    Validates that the input number is a positive integer number.
    """

    def validate(self, value):
        super().validate(value)


class Float(_Real, RangeChecker):
    """
    Validates that the input number is a real number. Optionally the user can specify
    the bounds for the number
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def validate(self, value):
        super().validate(value)


class PositiveFloat(_Real, PositiveChecker):
    """
    Validates that the input number is a positive real number.
    """

    def validate(self, value):
        super().validate(value)


class OneofStrings(String, OneOf):
    """
    Validates that the input is among the specified strings.

    Args:
        *args: All the acceptable values for the string
    """

    def __init__(self, *args):
        OneOf.__init__(self, *args)

    def validate(self, value):
        super().validate(value)


class OneofIntegers(_Integral, OneOf):
    """
    Validates that the input is among the specified integers.

    Args:
        *args: All the acceptable values for the integer
    """

    def __init__(self, *args):
        OneOf.__init__(self, *args)

    def validate(self, value):
        super().validate(value)
