import os
import py_compile
import unittest

try:
    import vmec_jax
except ImportError:
    vmec_jax = None

try:
    from simsopt.mhd import VmecJax
except ImportError:
    VmecJax = None


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

    @unittest.skipIf(vmec_jax is None or VmecJax is None, "vmec_jax not found")
    def test_single_stage_exact_specs_match_surface_dofs(self):
        filename = os.path.join(
            ROOT,
            "examples",
            "3_Advanced",
            "inputs",
            "input.nfp4_QH_warm_start",
        )
        vmec = VmecJax(filename, mpi=None, verbose=False, nphi=34, ntheta=34, range_surface="half period")
        surf = vmec.boundary
        surf.fix_all()
        surf.fixed_range(mmin=0, mmax=1, nmin=-1, nmax=1, fixed=False)
        surf.fix("rc(0,0)")

        def label(name):
            local_name = name.split(":")[-1]
            coeff, indices = local_name.split("(")
            m_str, n_str = indices.rstrip(")").split(",")
            return f"{coeff}{int(m_str)}{int(n_str)}"

        cfg, indata = vmec_jax.load_config(filename)
        static = vmec_jax.build_static(cfg)
        boundary = vmec_jax.boundary_from_indata(indata, static.modes)
        specs = vmec_jax.boundary_param_specs(
            boundary,
            static.modes,
            max_mode=1,
            min_coeff=0.0,
            include=("rc", "zs"),
            fix=("rc00",),
        )
        surface_labels = {label(name) for name in surf.dof_names}
        self.assertEqual({spec.name for spec in specs}, surface_labels)


if __name__ == "__main__":
    unittest.main()
