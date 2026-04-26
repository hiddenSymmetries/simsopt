import os
import py_compile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

EXAMPLES = [
    "examples/2_Intermediate/B_external_normal_jax.py",
    "examples/2_Intermediate/QH_fixed_resolution_jax.py",
    "examples/2_Intermediate/QH_fixed_resolution_boozer_jax.py",
    "examples/2_Intermediate/stage_two_optimization_finite_beta_jax.py",
    "examples/3_Advanced/single_stage_optimization_jax.py",
    "examples/3_Advanced/single_stage_optimization_finite_beta_jax.py",
]


class JaxExamplesTests(unittest.TestCase):
    def test_requested_jax_examples_compile(self):
        for example in EXAMPLES:
            with self.subTest(example=example):
                py_compile.compile(os.path.join(ROOT, example), doraise=True)


if __name__ == "__main__":
    unittest.main()
