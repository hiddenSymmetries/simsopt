from unittest import TestCase
from dataclasses import dataclass
from typing import Any

from simsopt._core.descriptor import OneOf, OneofStrings, OneofIntegers, \
    Float, Integer, String, PositiveInteger, PositiveFloat


@dataclass
class OneOfTstComposite:
    a: Any = OneOf("one", 1, "two")
    b: Any = OneOf("PI", 360)


class OneOfTest(TestCase):
    def test_valid_args(self):
        OneOfTstComposite("one", 360)
        OneOfTstComposite(a=1, b="PI")
        OneOfTstComposite(b=360, a="two")

    def test_invalid_args(self):
        with self.assertRaises(ValueError):
            OneOfTstComposite("three", 180)     # Invalid args
        with self.assertRaises(ValueError):
            OneOfTstComposite(a="PI", b=1)      # Opposite type args
        with self.assertRaises(ValueError):
            OneOfTstComposite(b="one", a="PI")  # Opposite order
        with self.assertRaises(ValueError):
            OneOfTstComposite("1", "PI")        # Wrong option for one arg
        with self.assertRaises(ValueError):
            OneOfTstComposite('two', 180)


@dataclass
class IntComposite:
    i: int = Integer()
    j: int = Integer(10)
    k: int = Integer(10, 100)
    l: int = Integer(min_value=20)
    m: int = Integer(max_value=200)
    n: int = Integer(max_value=200, min_value=20)


class IntegerTests(TestCase):
    def test_valid_args(self):
        IntComposite(10000, 101, 11, 201, -1, 50)
        IntComposite(-10, 11, 99, 10000, 20, 200)

    def test_invalid_args(self):
        with self.assertRaises(TypeError):
            IntComposite(10.5, 10, 10, 20, 200, 20)
        with self.assertRaises(TypeError):
            IntComposite(10, "ten", 10, 20, 200, 20)

        with self.assertRaises(ValueError):
            IntComposite(-10, 9, 10, 20, 200, 20)
        with self.assertRaises(ValueError):
            IntComposite(-10, 11, 101, 20, 200, 20)
        with self.assertRaises(ValueError):
            IntComposite(-10, 11, 100, 19, 200, 20)
        with self.assertRaises(ValueError):
            IntComposite(-10, 11, 100, 20, 201, 20)
        with self.assertRaises(ValueError):
            IntComposite(-10, 11, 100, 20, 200, -10)


@dataclass
class RealComposite:
    a: float = Float()
    b: float = Float(0.5)
    c: float = Float(0.5, 0.76)
    d: float = Float(min_value=1.5)
    e: float = Float(max_value=1.5)
    f: float = Float(max_value=1.5, min_value=0.5)


class FloatTests(TestCase):
    def test_valid_args(self):
        RealComposite(10000, 101, 0.55, 201, -1.5, 1.0)
        RealComposite(1000.1, 101.1, 0.55, 201.1, -1.5, 1)
        RealComposite(-10.5, 0.75, 0.76, 1.5, 1.5, 1.2)

    def test_invalid_args(self):
        with self.assertRaises(TypeError):
            RealComposite([10.5, 10, 10, 20, 200, 20])
        with self.assertRaises(TypeError):
            RealComposite(10, "ten", 10, 20, 200, 20)

        with self.assertRaises(ValueError):
            RealComposite(-0.5, -0.5, 0.6, 20.1, 1.4, 0.75)
        with self.assertRaises(ValueError):
            RealComposite(-0.5, 0.5, -0.5, 20.1, 1.4, 0.75)
        with self.assertRaises(ValueError):
            RealComposite(-0.5, 0.5, 0.6, 0.1, 1.4, 0.75)
        with self.assertRaises(ValueError):
            RealComposite(-0.5, 0.5, 0.6, 20.1, 1.6, 0.75)
        with self.assertRaises(ValueError):
            RealComposite(-0.5, 0.5, 0.6, 20.1, 1.4, 7.5)


@dataclass
class PositiveComposite:
    i: int = PositiveInteger()
    a: int = PositiveFloat()


class PositiveTests(TestCase):
    def test_valid_args(self):
        PositiveComposite(0, 0.0)
        PositiveComposite(5, 0.5)

    def test_invalid_args(self):
        with self.assertRaises(TypeError):
            PositiveComposite(1.0, 0.5)

        with self.assertRaises(ValueError):
            PositiveComposite(-2, 0.4)
        with self.assertRaises(ValueError):
            PositiveComposite(2, -0.4)


@dataclass
class OneOfIntsComposite:
    flag: int = OneofIntegers(0, 1)
    degree: int = OneofIntegers(1, 2, 3)


class OneofIntTest(TestCase):
    def test_valid_args(self):
        OneOfIntsComposite(0, 1)
        OneOfIntsComposite(flag=1, degree=3)
        OneOfIntsComposite(degree=2, flag=0)

    def test_invalid_args(self):
        with self.assertRaises(TypeError):
            OneOfIntsComposite(1, "one")
        with self.assertRaises(TypeError):
            OneOfIntsComposite(degree=2, flag="False")  # Note boolean gets converted to int

        with self.assertRaises(ValueError):
            OneOfIntsComposite(2, 4)  # Invalid args
        with self.assertRaises(ValueError):
            OneOfIntsComposite(flag=2, degree=0)  # Opposite type args
        with self.assertRaises(ValueError):
            OneOfIntsComposite(degree=0, flag=3)  # Opposite order
        with self.assertRaises(ValueError):
            OneOfIntsComposite(2, 3)  # Wrong option for one arg
        with self.assertRaises(ValueError):
            OneOfIntsComposite(0, 4)


@dataclass
class StrComposite:
    a: str = String()
    b: str = String(5)
    c: str = String(maxsize=8)
    d: str = String(minsize=5, maxsize=10)
    e: str = String(predicate=lambda x: x.isupper())
    f: str = String(minsize=3, maxsize=8,
                    predicate=lambda x: x.islower())


class StrTests(TestCase):
    def test_valid_args(self):
        StrComposite("My", "argument", "size", "greater", "THAN", "five")

    def test_invalid_args(self):
        with self.assertRaises(TypeError):
            StrComposite(10.5, "argument", "size", "greater", "THAN", "five")

        with self.assertRaises(ValueError):
            StrComposite("My", "arg", "size", "greater", "THAN", "five")
        with self.assertRaises(ValueError):
            StrComposite("My", "argument", "dimensions", "greater", "THAN", "five")
        with self.assertRaises(ValueError):
            StrComposite("My", "arg", "size", ">", "THAN", "five")
        with self.assertRaises(ValueError):
            StrComposite("My", "arg", "size", "greater", "than", "five")
        with self.assertRaises(ValueError):
            StrComposite("My", "arg", "size", "greater", "than", "FIVE")


@dataclass
class OneOfStringsTstComposite:
    flag: str = OneofStrings("one", "two", "three")
    coords: str = OneofStrings("cartesian", "spherical", "cylindrical")


class OneofStringTest(TestCase):
    def test_valid_args(self):
        OneOfStringsTstComposite("one", "spherical")
        OneOfStringsTstComposite(flag="two", coords="cartesian")
        OneOfStringsTstComposite(coords="cylindrical", flag="three")

    def test_invalid_args(self):
        with self.assertRaises(TypeError):
            OneOfStringsTstComposite(1, "cartesian")
        with self.assertRaises(TypeError):
            OneOfStringsTstComposite(coords="spherical", flag=2)

        with self.assertRaises(ValueError):
            OneOfStringsTstComposite("four", "boozer")  # Invalid args
        with self.assertRaises(ValueError):
            OneOfStringsTstComposite(flag="cylindrical", coords="one")  # Opposite type args
        with self.assertRaises(ValueError):
            OneOfStringsTstComposite(coords="two", flag="spherical")  # Opposite order
        with self.assertRaises(ValueError):
            OneOfStringsTstComposite("1", "cartesian")  # Wrong option for one arg
        with self.assertRaises(ValueError):
            OneOfStringsTstComposite('two', "boozer")
