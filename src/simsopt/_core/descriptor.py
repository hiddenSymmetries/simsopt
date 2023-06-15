from abc import ABC, abstractmethod
from typing import Callable
from dataclasses import dataclass

import numbers

__all__ = ['OneofStrings', 'OneofIntegers', 'String', 'Integer', 'PositiveInteger',
           'Float', 'PositiveFloat']


class Validator(ABC):
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
        pass


class OneOf(Validator):
    def __init__(self, *args):
        self.options = set(args)

    def validate(self, value):
        if value not in self.options:
            raise ValueError(f'{value!r} is not a valid option for {self.public_name!r}.\n'
                             f'Valid options are one of {self.options!r}')
        super().validate(value)


@dataclass
class String(Validator):
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


class Real(Validator):
    def validate(self, value):
        if not isinstance(value, numbers.Real):
            raise TypeError(
                f'Expected {value!r} should be of Real type')
        super().validate(value)


class Integral(Validator):
    def validate(self, value):
        if not isinstance(value, numbers.Integral):
            raise TypeError(
                f'Expected {value!r} should be of Integer type')
        super().validate(value)


@dataclass
class RangeChecker(Validator):
    min_value : numbers.Real = None
    max_value : numbers.Real = None

    def validate(self, value):
        if self.min_value is not None and value < self.min_value:
            raise ValueError(
                f'Expected {value!r} to be at least {self.min_value!r}')
        if self.max_value is not None and value > self.max_value:
            raise ValueError(
                f'Expected {value!r} to be no more than {self.max_value!r}')
        super().validate(value)


class PositiveChecker(RangeChecker):
    def __init__(self):
        super().__init__(min_value = 0)


class Integer(Integral, RangeChecker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def validate(self, value):
        super().validate(value)


class PositiveInteger(Integer, PositiveChecker):
    def validate(self, value):
        super().validate(value)


class Float(Real, RangeChecker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def validate(self, value):
        super().validate(value)


class PositiveFloat(Real, PositiveChecker):
    def validate(self, value):
        super().validate(value)


class OneofStrings(String, OneOf):
    def __init__(self, *args):
        OneOf.__init__(self, *args)
    def validate(self, value):
        super().validate(value)


class OneofIntegers(Integral, OneOf):
    def __init__(self, *args):
        OneOf().__init__(self, *args)

    def validate(self, value):
        super().validate(value)
