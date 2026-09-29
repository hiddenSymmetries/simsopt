import unittest
import warnings

try:
    import numpy as np
except ImportError:
    np = None

from simsopt._core.dev import SimsoptRequires, deprecated
from simsopt._core.optimizable import Optimizable


@SimsoptRequires(np is not None, "numpy is not installed.")
class TestClass(Optimizable):
    def __init__(self):
        x = np.array([1.2, 0.9, -0.4])
        fixed = np.full(3, False)
        super().__init__(x0=x, fixed=fixed)

    def J(self):
        return np.exp(self.full_x[0] ** 2 - np.exp(self.full_x[1])
                      + np.sin(self.full_x[2]))

    return_fn_map = {'J': J}


class SimsoptRequiresTest(unittest.TestCase):
    def test_subclass_check(self):
        tf = TestClass()
        self.assertTrue(issubclass(type(tf), Optimizable))

    def test_unmet_condition_raises_on_call(self):
        @SimsoptRequires(False, "dependency missing for test")
        def needs_dep():
            return 42

        with self.assertRaises(RuntimeError) as cm:
            needs_dep()
        self.assertIn("dependency missing for test", str(cm.exception))

    def test_met_condition_calls_through(self):
        @SimsoptRequires(True, "should not raise")
        def works(a, b=2):
            return a + b

        self.assertEqual(works(3), 5)
        self.assertEqual(works(3, b=4), 7)

    def test_wraps_preserves_metadata(self):
        @SimsoptRequires(True, "irrelevant")
        def documented():
            """docstring"""
            return None

        self.assertEqual(documented.__name__, "documented")
        self.assertEqual(documented.__doc__, "docstring")


def _replacement_function():
    """Replacement target used by deprecated() tests."""
    return "new"


class _ReplacementHolder:
    @property
    def replacement_prop(self):
        return "prop"

    @classmethod
    def replacement_classmethod(cls):
        return "cm"


class DeprecatedTest(unittest.TestCase):
    def test_warns_with_future_warning_by_default(self):
        @deprecated()
        def old():
            return "result"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(old(), "result")

        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, FutureWarning)
        self.assertIn("old is deprecated", str(caught[0].message))

    def test_warns_with_deprecation_warning_category(self):
        @deprecated(category=DeprecationWarning)
        def old():
            return "x"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            old()

        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, DeprecationWarning)

    def test_warning_message_includes_replacement_function(self):
        @deprecated(replacement=_replacement_function)
        def old():
            return None

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            old()

        self.assertEqual(len(caught), 1)
        msg = str(caught[0].message)
        self.assertIn("_replacement_function", msg)
        self.assertIn(_replacement_function.__module__, msg)

    def test_warning_message_includes_replacement_property(self):
        @deprecated(replacement=_ReplacementHolder.__dict__["replacement_prop"])
        def old():
            return None

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            old()

        self.assertEqual(len(caught), 1)
        self.assertIn("replacement_prop", str(caught[0].message))

    def test_warning_message_includes_replacement_classmethod(self):
        @deprecated(replacement=_ReplacementHolder.__dict__["replacement_classmethod"])
        def old():
            return None

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            old()

        self.assertEqual(len(caught), 1)
        self.assertIn("replacement_classmethod", str(caught[0].message))

    def test_extra_message_is_appended(self):
        @deprecated(message="see docs")
        def old():
            return None

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            old()

        self.assertEqual(len(caught), 1)
        self.assertIn("see docs", str(caught[0].message))

    def test_decorator_preserves_return_value_and_metadata(self):
        @deprecated()
        def old(a, b=1):
            """legacy"""
            return a * b

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertEqual(old(3, b=4), 12)
        self.assertEqual(old.__name__, "old")
        self.assertEqual(old.__doc__, "legacy")


if __name__ == '__main__':
    unittest.main()
