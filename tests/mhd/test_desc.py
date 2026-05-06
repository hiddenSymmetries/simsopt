import unittest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from simsopt.mhd import ProfilePolynomial, ProfileSpline
from simsopt.geo import SurfaceRZFourier
from simsopt.mhd.desc import DescOptimizable, DescEquilibriumProtocol


class TestProfilesToDesc(unittest.TestCase):
    """Test conversion from simsopt to DESC profiles."""

    def test_polynomial_profile_values_match(self):
        """Test ProfilePolynomial conversion preserves values at key points in rho-space."""
        # Create polynomial in rho-space: f(rho) = 1 - rho^2 (symmetric/even powers only)
        prof = ProfilePolynomial([1.0, 0.0, -1.0])

        # Convert to DESC
        prof_desc = DescOptimizable._profile_to_desc(prof)

        # Check values at several rho points
        rho_test = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

        # Original polynomial values in rho-space
        original_values = prof(rho_test)

        # DESC profile values (also in rho-space)
        desc_values = prof_desc(rho_test)

        np.testing.assert_allclose(original_values, desc_values, rtol=1e-10)

    def test_spline_profile_values_match(self):
        """Test ProfileSpline conversion preserves values at key points in rho-space."""
        # Create spline with a few knots in rho-space
        rho_knots = np.linspace(0, 1, 4)
        f_values = 1 - rho_knots
        prof = ProfileSpline(rho_knots, f_values, degree=3)

        # Convert to DESC
        prof_desc = DescOptimizable._profile_to_desc(prof)

        # Check values at rho test points
        rho_test = np.linspace(0, 1, 5)

        original_values = prof(rho_test)
        desc_values = prof_desc(rho_test)

        np.testing.assert_allclose(original_values, desc_values, rtol=1e-6, atol=1e-12)


class TestProfilesFromDesc(unittest.TestCase):
    """Test roundtrip conversion: simsopt -> DESC -> simsopt."""

    def test_polynomial_roundtrip(self):
        """Test polynomial -> DESC -> polynomial roundtrip in rho-space."""
        # Create polynomial in rho-space: f(rho) = 1 - rho^2 + 0.5*rho^4 (even powers only)
        prof_orig = ProfilePolynomial([1.0, 0.0, -1.0, 0.0, 0.5])

        # Convert: simsopt -> DESC
        prof_desc = DescOptimizable._profile_to_desc(prof_orig)

        # Convert back: DESC -> simsopt
        prof_simsopt = DescOptimizable._profile_from_desc(
            prof_desc, n_knots=10, degree=3
        )

        # Should return a ProfilePolynomial, not a spline
        self.assertIsInstance(prof_simsopt, ProfilePolynomial)

        # Check values at test points in rho-space
        rho_test = np.linspace(0, 1, 20)
        original_values = prof_orig(rho_test)
        roundtrip_values = prof_simsopt(rho_test)

        # Exact tolerance since polynomial roundtrips should be exact
        np.testing.assert_allclose(original_values, roundtrip_values, atol=1e-12)

    def test_spline_roundtrip(self):
        """Test spline -> DESC -> spline roundtrip in rho-space."""
        n_pts = 6
        rho_knots = np.linspace(0.0, 1.0, n_pts)
        f_values = 1.0 - rho_knots**2  # Profile f(rho) = 1 - rho^2
        prof_orig = ProfileSpline(rho_knots, f_values, degree=3)

        # Roundtrip conversion
        prof_desc = DescOptimizable._profile_to_desc(prof_orig)
        prof_simsopt = DescOptimizable._profile_from_desc(
            prof_desc, n_knots=n_pts, degree=3
        )

        # Check values in rho-space
        rho_test = np.linspace(0, 1, 20)
        original_values = prof_orig(rho_test)
        roundtrip_values = prof_simsopt(rho_test)

        np.testing.assert_allclose(original_values, roundtrip_values, atol=1e-12)


class TestBoundaryToDesc(unittest.TestCase):
    """Test conversion of plasma boundary surface to DESC format."""

    @classmethod
    def setUpClass(cls):
        """Load test boundary once for all tests in this class."""
        input_file = "ml_fast_ion/vmec_input_files/input.vacuum_template"
        cls.boundary = SurfaceRZFourier.from_vmec_input(input_file)

    def test_nfp_preserved(self):
        """Test that NFP is preserved in boundary conversion."""
        desc_surface = DescOptimizable.surface_to_desc(self.boundary)

        self.assertEqual(desc_surface.NFP, self.boundary.nfp)
        self.assertEqual(desc_surface.NFP, 2)  # input.vacuum_template has NFP=2

    def test_major_radius_preserved(self):
        """Test that the (0,0) Fourier mode (major radius) is approximately preserved."""
        boundary_rz = self.boundary.to_RZFourier()
        desc_surface = DescOptimizable.surface_to_desc(boundary_rz)

        # Get the R(0,0) coefficient, which is the major radius
        R_00, _ = desc_surface.get_coeffs(0, 0)

        # From input.vacuum_template: RBC(0,0) = 1.0
        self.assertAlmostEqual(R_00, 1.0, places=12)

    def test_roundtrip_to_desc_and_back(self):
        """Test surface_to_desc followed by surface_from_desc roundtrip."""
        from desc.equilibrium import Equilibrium as DescEquilibrium

        # Convert surface to DESC format
        desc_surface = DescOptimizable.surface_to_desc(self.boundary)

        # Create a minimal DESC equilibrium with just the surface
        eq = DescEquilibrium(surface=desc_surface)

        # Convert back to simsopt surface
        boundary_from_desc = DescOptimizable.surface_from_desc(eq)
        boundary_roundtrip = boundary_from_desc.copy(
            quadpoints_phi=self.boundary.quadpoints_phi,
            quadpoints_theta=self.boundary.quadpoints_theta,
        )

        # Check key properties are preserved
        self.assertEqual(boundary_roundtrip.nfp, self.boundary.nfp)
        self.assertEqual(boundary_roundtrip.stellsym, self.boundary.stellsym)

        # Check metric/geometry is preserved via gamma
        gamma_flat = self.boundary.gamma().reshape((-1, 3))
        gamma_rt_flat = boundary_roundtrip.gamma().reshape((-1, 3))
        from scipy.spatial.distance import cdist

        distances = cdist(gamma_flat, gamma_rt_flat)
        gamma_err = np.max(np.min(distances, axis=1))
        self.assertAlmostEqual(gamma_err, 0.0, places=12)


class TestBuildDescEquilibrium(unittest.TestCase):
    """Test building a DESC equilibrium from DescOptimizable."""

    def setUp(self):
        """Create a simple test equilibrium."""
        input_file = "ml_fast_ion/vmec_input_files/input.vacuum_template"
        boundary = SurfaceRZFourier.from_vmec_input(input_file)

        # Create simple profiles
        # Symmetric polynomials (even powers only): f(rho) = c0 + c2*rho^2 + c4*rho^4 + ...
        pressure = ProfilePolynomial([1e4, 0.0, -1e4])  # f(rho) = 1e4 * (1 - rho^2)
        iota = ProfilePolynomial([0.4, 0.0, 0.1])  # f(rho) = 0.4 + 0.1 * rho^2

        self.psi = 1.0
        self.desc_opt = DescOptimizable(
            boundary=boundary,
            psi=self.psi,
            pressure_profile=pressure,
            iota_profile=iota,
        )

    def test_returns_protocol_instance(self):
        """Test that the equilibrium is a valid protocol instance."""
        self.assertIsInstance(self.desc_opt.eq, DescEquilibriumProtocol)

    def test_psi_preserved(self):
        """Test that the edge toroidal flux is preserved."""
        self.assertAlmostEqual(self.desc_opt.eq.Psi, self.psi, places=10)

    def test_nfp_preserved_in_equilibrium(self):
        """Test that NFP is preserved in the equilibrium surface."""
        self.assertEqual(self.desc_opt.eq.surface.NFP, 2)


class TestDescOptimizableDOFs(unittest.TestCase):
    """Test degrees of freedom (DOFs) of DescOptimizable."""

    def setUp(self):
        """Create a test DescOptimizable."""
        input_file = "ml_fast_ion/vmec_input_files/input.vacuum_template"
        boundary = SurfaceRZFourier.from_vmec_input(input_file)
        pressure = ProfilePolynomial([1e4, -1e4])
        iota = ProfilePolynomial([0.4, 0.1])

        self.psi = 2.5
        self.desc_opt = DescOptimizable(
            boundary=boundary,
            psi=self.psi,
            pressure_profile=pressure,
            iota_profile=iota,
        )

    def test_get_dofs_returns_psi(self):
        """Test that get returns the PSI value."""
        dofs = self.desc_opt.local_full_x
        self.assertEqual(len(dofs), 1)
        self.assertAlmostEqual(dofs[0], self.psi, places=10)

    def test_set_dofs_updates_psi(self):
        """Test that set_dofs correctly updates the PSI value."""
        new_psi = 5.0
        self.desc_opt.need_to_run_code = False
        self.desc_opt.set("psi", new_psi)

        # Check that psi was updated
        self.assertAlmostEqual(self.desc_opt.get("psi"), new_psi, places=10)
        self.assertAlmostEqual(self.desc_opt.need_to_run_code, True, places=10)


class TestFromFile(unittest.TestCase):
    """Test the from_file classmethod for loading saved equilibria."""

    def setUp(self):
        """Create and save a test equilibrium."""
        input_file = "ml_fast_ion/vmec_input_files/input.padidar_A"
        boundary = SurfaceRZFourier.from_vmec_input(input_file)

        pressure = ProfilePolynomial([1e1, 0.0, -1e1])
        iota = ProfilePolynomial([0.4, 0.0, 0.1])

        self.psi = 1.0
        self.nfp = boundary.nfp

        # Create and build equilibrium
        self.desc_opt_orig = DescOptimizable(
            boundary=boundary,
            psi=self.psi,
            pressure_profile=pressure,
            iota_profile=iota,
        )
        self.eq_orig = self.desc_opt_orig.eq

    def test_from_file_roundtrip(self):
        """Test saving and loading an equilibrium."""
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            tmp_path = tmp.name

        # Save the equilibrium
        self.eq_orig.save(tmp_path)

        # Load it back
        desc_opt_loaded = DescOptimizable.from_file(tmp_path)

        # compare boundaries
        boundary_orig = self.desc_opt_orig.boundary
        boundary_loaded = desc_opt_loaded.boundary
        gamma_err = np.max(np.abs(boundary_loaded.gamma() - boundary_orig.gamma()))
        self.assertAlmostEqual(gamma_err, 0.0, places=13)

        # compare profiles in rho-space
        rho_test = np.linspace(0, 1, 30)
        pressure_orig = self.desc_opt_orig.pressure_profile
        pressure_loaded = desc_opt_loaded.pressure_profile
        pressure_err = np.max(
            np.abs(pressure_loaded(rho_test) - pressure_orig(rho_test))
        )
        self.assertAlmostEqual(pressure_err, 0.0, places=13)
        iota_orig = self.desc_opt_orig.iota_profile
        iota_loaded = desc_opt_loaded.iota_profile
        iota_err = np.max(np.abs(iota_loaded(rho_test) - iota_orig(rho_test)))
        self.assertAlmostEqual(iota_err, 0.0, places=13)

        # Check that key properties match
        self.assertEqual(desc_opt_loaded.boundary.nfp, self.nfp)
        self.assertAlmostEqual(desc_opt_loaded.get("psi"), self.psi, places=10)

        # Check surface parameters
        self.assertEqual(desc_opt_loaded.eq.surface.NFP, self.nfp)

        # Clean up
        Path(tmp_path).unlink(missing_ok=True)


class TestFromEq(unittest.TestCase):
    """Test the from_eq classmethod for creating from DESC Equilibrium objects."""

    def setUp(self):
        """Create a test DESC Equilibrium."""
        from desc.equilibrium import Equilibrium as DescEquilibrium

        input_file = "ml_fast_ion/vmec_input_files/input.vacuum_template"
        boundary = SurfaceRZFourier.from_vmec_input(input_file)

        # Create a DescOptimizable to get a DESC Equilibrium
        pressure = ProfilePolynomial([1e4, -1e4])
        iota = ProfilePolynomial([0.4, 0.1])

        self.desc_opt_orig = DescOptimizable(
            boundary=boundary, psi=1.0, pressure_profile=pressure, iota_profile=iota
        )
        self.eq = self.desc_opt_orig.eq

    def test_from_eq_creates_optimizable(self):
        """Test that from_eq creates a valid DescOptimizable from a DESC Equilibrium."""
        desc_opt = DescOptimizable.from_eq(self.eq)

        # Check that basic properties are preserved
        self.assertIsNotNone(desc_opt.boundary)
        self.assertIsNotNone(desc_opt.eq)
        self.assertEqual(desc_opt.eq, self.eq)  # Should use the same equilibrium
        self.assertAlmostEqual(desc_opt.get("psi"), self.eq.Psi, places=10)

    def test_from_eq_preserves_boundary(self):
        """Test that from_eq preserves boundary properties."""
        desc_opt = DescOptimizable.from_eq(self.eq)

        self.assertEqual(desc_opt.boundary.nfp, self.desc_opt_orig.boundary.nfp)
        self.assertEqual(
            desc_opt.boundary.stellsym, self.desc_opt_orig.boundary.stellsym
        )

    def test_from_eq_preserves_profiles(self):
        """Test that from_eq interpolates profiles correctly."""
        desc_opt = DescOptimizable.from_eq(self.eq)

        s_test = np.linspace(0, 1, 10)
        pressure_orig = self.desc_opt_orig.pressure_profile(s_test)
        pressure_loaded = desc_opt.pressure_profile(s_test)

        np.testing.assert_allclose(pressure_loaded, pressure_orig, rtol=1e-6)


class TestFromInputFile(unittest.TestCase):
    """Test the from_input_file classmethod for loading VMEC/DESC input files."""

    def setUp(self):
        """Create test data and save to a VMEC input file."""
        input_file = "ml_fast_ion/vmec_input_files/input.vacuum_template"
        boundary = SurfaceRZFourier.from_vmec_input(input_file)
        pressure = ProfilePolynomial([1e4, -1e4])
        iota = ProfilePolynomial([0.4, 0.1])

        self.desc_opt_orig = DescOptimizable(
            boundary=boundary, psi=1.0, pressure_profile=pressure, iota_profile=iota
        )
        self.nfp = boundary.nfp

        # Create a temporary VMEC input file
        with tempfile.NamedTemporaryFile(
            suffix="", delete=False, prefix="input."
        ) as tmp:
            self.input_file_path = tmp.name

        self.desc_opt_orig.to_vmec_input(self.input_file_path)

    def tearDown(self):
        """Clean up temporary files."""
        Path(self.input_file_path).unlink(missing_ok=True)

    def test_from_input_file_loads_vmec(self):
        """Test that from_input_file can load a VMEC input file."""
        desc_opt = DescOptimizable.from_input_file(self.input_file_path)

        self.assertIsNotNone(desc_opt.boundary)
        self.assertIsNotNone(desc_opt.eq)

    def test_from_input_file_preserves_nfp(self):
        """Test that from_input_file preserves NFP."""
        desc_opt = DescOptimizable.from_input_file(self.input_file_path)

        self.assertEqual(desc_opt.boundary.nfp, self.nfp)

    def test_from_input_file_creates_optimizable(self):
        """Test that from_input_file creates a valid DescOptimizable."""
        desc_opt = DescOptimizable.from_input_file(self.input_file_path)

        # The eq should be a valid DESC Equilibrium
        self.assertIsNotNone(desc_opt.eq)
        self.assertIsInstance(desc_opt.eq, DescEquilibriumProtocol)

    def test_from_input_file_with_custom_knots(self):
        """Test that from_input_file respects custom knot parameters."""
        n_knots = 15
        degree = 2
        desc_opt = DescOptimizable.from_input_file(
            self.input_file_path, n_knots=n_knots, degree=degree
        )

        self.assertEqual(desc_opt.n_knots, n_knots)
        self.assertEqual(desc_opt.degree, degree)


class TestFromFileMethod(unittest.TestCase):
    """Test the from_file classmethod for loading HDF5/pickle files."""

    def setUp(self):
        """Create test data and save to an HDF5 file."""
        input_file = "ml_fast_ion/vmec_input_files/input.vacuum_template"
        boundary = SurfaceRZFourier.from_vmec_input(input_file)
        pressure = ProfilePolynomial([1e4, -1e4])
        iota = ProfilePolynomial([0.4, 0.1])

        self.psi = 1.0
        self.nfp = boundary.nfp

        self.desc_opt_orig = DescOptimizable(
            boundary=boundary,
            psi=self.psi,
            pressure_profile=pressure,
            iota_profile=iota,
        )

        # Create a temporary HDF5 file
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            self.hdf5_file_path = tmp.name

        self.desc_opt_orig.eq.save(self.hdf5_file_path)

    def tearDown(self):
        """Clean up temporary files."""
        Path(self.hdf5_file_path).unlink(missing_ok=True)

    def test_from_file_loads_hdf5(self):
        """Test that from_file can load an HDF5 file."""
        desc_opt = DescOptimizable.from_file(self.hdf5_file_path)

        self.assertIsNotNone(desc_opt.boundary)
        self.assertIsNotNone(desc_opt.eq)

    def test_from_file_preserves_boundary(self):
        """Test that from_file preserves boundary properties."""
        desc_opt = DescOptimizable.from_file(self.hdf5_file_path)

        # Check NFP and stellsym are preserved
        self.assertEqual(desc_opt.boundary.nfp, self.desc_opt_orig.boundary.nfp)
        self.assertEqual(
            desc_opt.boundary.stellsym, self.desc_opt_orig.boundary.stellsym
        )

    def test_from_file_preserves_psi(self):
        """Test that from_file preserves psi."""
        desc_opt = DescOptimizable.from_file(self.hdf5_file_path)

        self.assertAlmostEqual(desc_opt.get("psi"), self.psi, places=10)

    def test_from_file_with_custom_knots(self):
        """Test that from_file respects custom knot parameters."""
        n_knots = 15
        degree = 2
        desc_opt = DescOptimizable.from_file(
            self.hdf5_file_path, n_knots=n_knots, degree=degree
        )

        self.assertEqual(desc_opt.n_knots, n_knots)
        self.assertEqual(desc_opt.degree, degree)


class TestSurfaceFromDesc(unittest.TestCase):
    """Test the surface_from_desc static method."""

    def test_surface_from_desc_roundtrip(self):
        """Test that surface_from_desc correctly converts DESC surface back to simsopt."""
        from desc.equilibrium import Equilibrium as DescEquilibrium

        input_file = "ml_fast_ion/vmec_input_files/input.padidar_A"
        boundary_orig = SurfaceRZFourier.from_vmec_input(input_file)

        # Convert to DESC and back
        desc_surface = DescOptimizable.surface_to_desc(boundary_orig)
        eq = DescEquilibrium(surface=desc_surface)
        boundary_roundtrip = DescOptimizable.surface_from_desc(eq)
        boundary_from_desc = boundary_roundtrip.copy(
            quadpoints_phi=boundary_orig.quadpoints_phi,
            quadpoints_theta=boundary_orig.quadpoints_theta,
        )

        # Check NFP and stellsym are preserved
        self.assertEqual(boundary_from_desc.nfp, boundary_orig.nfp)
        self.assertEqual(boundary_from_desc.stellsym, boundary_orig.stellsym)

        # Check geometry is preserved via gamma
        # Check metric/geometry is preserved via gamma
        gamma_flat = boundary_orig.gamma().reshape((-1, 3))
        gamma_rt_flat = boundary_from_desc.gamma().reshape((-1, 3))
        from scipy.spatial.distance import cdist

        distances = cdist(gamma_flat, gamma_rt_flat)
        gamma_err = np.max(np.min(distances, axis=1))
        self.assertAlmostEqual(gamma_err, 0.0, places=12)


class TestToWout(unittest.TestCase):
    """Test the to_wout method."""

    def test_to_wout_creates_file(self):
        """Test that to_wout creates a valid VMEC output file."""
        input_file = "ml_fast_ion/vmec_input_files/input.vacuum_template"
        boundary = SurfaceRZFourier.from_vmec_input(input_file)
        pressure = ProfilePolynomial([1e4, -1e4])
        iota = ProfilePolynomial([0.4, 0.1])

        desc_opt = DescOptimizable(
            boundary=boundary, psi=1.0, pressure_profile=pressure, iota_profile=iota
        )

        # Write to temporary file
        with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Call to_wout
            desc_opt.to_wout(tmp_path)

            # Check that file was created
            self.assertTrue(Path(tmp_path).exists())
            # Check file size is reasonable (not empty)
            self.assertGreater(Path(tmp_path).stat().st_size, 0)
        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)


class TestToVmecInput(unittest.TestCase):
    """Test the to_vmec_input method."""

    def test_to_vmec_input_creates_file(self):
        """Test that to_vmec_input creates a VMEC input file."""
        input_file = "ml_fast_ion/vmec_input_files/input.vacuum_template"
        boundary = SurfaceRZFourier.from_vmec_input(input_file)
        pressure = ProfilePolynomial([1e4, -1e4])
        iota = ProfilePolynomial([0.4, 0.1])

        desc_opt = DescOptimizable(
            boundary=boundary, psi=1.0, pressure_profile=pressure, iota_profile=iota
        )

        # Write to temporary file
        with tempfile.NamedTemporaryFile(
            suffix="", delete=False, prefix="input."
        ) as tmp:
            tmp_path = tmp.name

        try:
            # Call to_vmec_input
            desc_opt.to_vmec_input(tmp_path)

            # Check that file was created
            self.assertTrue(Path(tmp_path).exists())
            # Check file size is reasonable (not empty)
            self.assertGreater(Path(tmp_path).stat().st_size, 0)
        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)

    def test_vmec_input_contains_boundary_data(self):
        """Test that to_vmec_input file contains boundary Fourier coefficients."""
        input_file = "ml_fast_ion/vmec_input_files/input.vacuum_template"
        boundary = SurfaceRZFourier.from_vmec_input(input_file)
        pressure = ProfilePolynomial([1e4, -1e4])
        iota = ProfilePolynomial([0.4, 0.1])

        desc_opt = DescOptimizable(
            boundary=boundary, psi=1.0, pressure_profile=pressure, iota_profile=iota
        )

        # Write to temporary file
        with tempfile.NamedTemporaryFile(
            suffix="", delete=False, prefix="input."
        ) as tmp:
            tmp_path = tmp.name

        try:
            # Save via to_vmec_input
            desc_opt.to_vmec_input(tmp_path)

            # Read and verify the file contains boundary data
            with open(tmp_path, "r") as f:
                content = f.read()

            # Check for key VMEC parameters
            self.assertIn("&INDATA", content)
            self.assertIn("RBC", content)
            self.assertIn("ZBS", content)
            self.assertIn("NFP", content)
            self.assertIn("AM", content)  # Pressure coefficients
            self.assertIn("AI", content)  # Iota coefficients
        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)


class TestToVmec(unittest.TestCase):
    """Test the to_vmec method."""

    def _create_mock_vmec(self):
        """Create a mock Vmec object with proper support for nested attributes."""
        from unittest.mock import MagicMock

        mock_vmec = MagicMock()
        mock_vmec.indata.ns_array = [16, 32, 64, 101]
        mock_vmec.indata.niter_array = [1000, 2000, 3000, 10000]
        mock_vmec.indata.ftol_array = [1.0e-16, 1.0e-16, 1.0e-16, 1.0e-13]
        return mock_vmec

    @patch("ml_fast_ion.desc_optimizable.Vmec")
    def test_to_vmec_returns_vmec_object(self, mock_vmec_class):
        """Test that to_vmec returns a Vmec object."""
        mock_vmec = self._create_mock_vmec()
        mock_vmec_class.return_value = mock_vmec

        input_file = "ml_fast_ion/vmec_input_files/input.vacuum_template"
        boundary = SurfaceRZFourier.from_vmec_input(input_file)
        pressure = ProfilePolynomial([1e4, -1e4])
        iota = ProfilePolynomial([0.4, 0.1])

        desc_opt = DescOptimizable(
            boundary=boundary, psi=1.0, pressure_profile=pressure, iota_profile=iota
        )

        vmec = desc_opt.to_vmec()
        self.assertEqual(vmec, mock_vmec)

    @patch("ml_fast_ion.desc_optimizable.Vmec")
    def test_to_vmec_preserves_boundary(self, mock_vmec_class):
        """Test that to_vmec preserves the boundary."""
        mock_vmec = self._create_mock_vmec()
        mock_vmec_class.return_value = mock_vmec

        input_file = "ml_fast_ion/vmec_input_files/input.vacuum_template"
        boundary = SurfaceRZFourier.from_vmec_input(input_file)
        pressure = ProfilePolynomial([1e4, -1e4])
        iota = ProfilePolynomial([0.4, 0.1])

        desc_opt = DescOptimizable(
            boundary=boundary, psi=1.0, pressure_profile=pressure, iota_profile=iota
        )

        vmec = desc_opt.to_vmec()
        self.assertEqual(mock_vmec.boundary, boundary)

    @patch("ml_fast_ion.desc_optimizable.Vmec")
    def test_to_vmec_preserves_psi(self, mock_vmec_class):
        """Test that to_vmec preserves psi (phiedge)."""
        mock_vmec = self._create_mock_vmec()
        mock_vmec_class.return_value = mock_vmec

        input_file = "ml_fast_ion/vmec_input_files/input.vacuum_template"
        boundary = SurfaceRZFourier.from_vmec_input(input_file)
        pressure = ProfilePolynomial([1e4, -1e4])
        iota = ProfilePolynomial([0.4, 0.1])
        psi = 2.5

        desc_opt = DescOptimizable(
            boundary=boundary, psi=psi, pressure_profile=pressure, iota_profile=iota
        )

        vmec = desc_opt.to_vmec()
        self.assertAlmostEqual(mock_vmec.indata.phiedge, psi)

    @patch("ml_fast_ion.desc_optimizable.Vmec")
    def test_to_vmec_preserves_nfp(self, mock_vmec_class):
        """Test that to_vmec preserves nfp."""
        mock_vmec = self._create_mock_vmec()
        mock_vmec_class.return_value = mock_vmec

        input_file = "ml_fast_ion/vmec_input_files/input.vacuum_template"
        boundary = SurfaceRZFourier.from_vmec_input(input_file)
        pressure = ProfilePolynomial([1e4, -1e4])
        iota = ProfilePolynomial([0.4, 0.1])

        desc_opt = DescOptimizable(
            boundary=boundary, psi=1.0, pressure_profile=pressure, iota_profile=iota
        )

        vmec = desc_opt.to_vmec()
        self.assertEqual(mock_vmec.indata.nfp, boundary.nfp)

    @patch("ml_fast_ion.desc_optimizable.Vmec")
    def test_to_vmec_sets_pressure_profile(self, mock_vmec_class):
        """Test that to_vmec sets the pressure profile."""
        mock_vmec = self._create_mock_vmec()
        mock_vmec_class.return_value = mock_vmec

        input_file = "ml_fast_ion/vmec_input_files/input.vacuum_template"
        boundary = SurfaceRZFourier.from_vmec_input(input_file)
        pressure = ProfilePolynomial([1e4, -1e4])
        iota = ProfilePolynomial([0.4, 0.1])

        desc_opt = DescOptimizable(
            boundary=boundary, psi=1.0, pressure_profile=pressure, iota_profile=iota
        )

        vmec = desc_opt.to_vmec()
        self.assertEqual(mock_vmec.pressure_profile, pressure)

    @patch("ml_fast_ion.desc_optimizable.Vmec")
    def test_to_vmec_sets_iota_profile_when_auto(self, mock_vmec_class):
        """Test that to_vmec sets iota profile when only iota is provided."""
        mock_vmec = self._create_mock_vmec()
        mock_vmec_class.return_value = mock_vmec

        input_file = "ml_fast_ion/vmec_input_files/input.vacuum_template"
        boundary = SurfaceRZFourier.from_vmec_input(input_file)
        pressure = ProfilePolynomial([1e4, -1e4])
        iota = ProfilePolynomial([0.4, 0.1])

        desc_opt = DescOptimizable(
            boundary=boundary,
            psi=1.0,
            pressure_profile=pressure,
            iota_profile=iota,
            which_profile="auto",
        )

        vmec = desc_opt.to_vmec()
        self.assertEqual(mock_vmec.iota_profile, iota)
        self.assertEqual(mock_vmec.indata.ncurr, 0)

    @patch("ml_fast_ion.desc_optimizable.Vmec")
    def test_to_vmec_sets_current_profile_when_auto(self, mock_vmec_class):
        """Test that to_vmec sets current profile when current is available."""
        mock_vmec = self._create_mock_vmec()
        mock_vmec_class.return_value = mock_vmec

        input_file = "ml_fast_ion/vmec_input_files/input.vacuum_template"
        boundary = SurfaceRZFourier.from_vmec_input(input_file)
        pressure = ProfilePolynomial([1e4, -1e4])
        current = ProfilePolynomial([1e5, -1e5])

        desc_opt = DescOptimizable(
            boundary=boundary,
            psi=1.0,
            pressure_profile=pressure,
            current_profile=current,
            which_profile="auto",
        )

        vmec = desc_opt.to_vmec()
        self.assertEqual(mock_vmec.current_profile, current)
        self.assertEqual(mock_vmec.indata.ncurr, 1)


class TestToDescProfiles(unittest.TestCase):
    """Test the profiles_for_eq method."""

    def test_to_desc_profiles_with_iota(self):
        """Test profiles_for_eq returns correct tuple with iota profile."""
        input_file = "ml_fast_ion/vmec_input_files/input.vacuum_template"
        boundary = SurfaceRZFourier.from_vmec_input(input_file)
        pressure = ProfilePolynomial([1e4, -1e4])
        iota = ProfilePolynomial([0.4, 0.1])

        desc_opt = DescOptimizable(
            boundary=boundary, psi=1.0, pressure_profile=pressure, iota_profile=iota
        )

        # Get the converted profiles
        pressure_desc, current_desc, iota_desc = desc_opt.profiles_for_eq()

        # Check that pressure and iota are converted, current is None
        self.assertIsNotNone(pressure_desc)
        self.assertIsNone(current_desc)
        self.assertIsNotNone(iota_desc)

        # Check that the converted profiles are DESC profile objects
        from desc.profiles import PowerSeriesProfile, SplineProfile

        self.assertIsInstance(pressure_desc, (PowerSeriesProfile, SplineProfile))
        self.assertIsInstance(iota_desc, (PowerSeriesProfile, SplineProfile))

    def test_to_desc_profiles_with_current(self):
        """Test profiles_for_eq returns correct tuple with current profile."""
        input_file = "ml_fast_ion/vmec_input_files/input.vacuum_template"
        boundary = SurfaceRZFourier.from_vmec_input(input_file)
        pressure = ProfilePolynomial([1e4, -1e4])
        current = ProfilePolynomial([0.0, 0.0, 1e5])  # symmetric: c2*rho^2

        desc_opt = DescOptimizable(
            boundary=boundary,
            psi=1.0,
            pressure_profile=pressure,
            current_profile=current,
        )

        # Get the converted profiles
        pressure_desc, current_desc, iota_desc = desc_opt.profiles_for_eq()

        # Check that pressure and current are converted, iota is None
        self.assertIsNotNone(pressure_desc)
        self.assertIsNotNone(current_desc)
        self.assertIsNone(iota_desc)

        # Check that the converted profiles are DESC profile objects
        from desc.profiles import PowerSeriesProfile, SplineProfile

        self.assertIsInstance(pressure_desc, (PowerSeriesProfile, SplineProfile))
        self.assertIsInstance(current_desc, (PowerSeriesProfile, SplineProfile))


class TestRescale(unittest.TestCase):
    """Test the rescale method."""

    def setUp(self):
        """Create a test DescOptimizable for rescaling."""
        input_file = "ml_fast_ion/vmec_input_files/input.vacuum_template"
        boundary = SurfaceRZFourier.from_vmec_input(input_file)
        pressure = ProfilePolynomial([1e4, -1e4])
        iota = ProfilePolynomial([0.4, 0.1])

        self.desc_opt = DescOptimizable(
            boundary=boundary, psi=1.0, pressure_profile=pressure, iota_profile=iota
        )

    def test_rescale_updates_simsopt_state(self):
        """Test that rescale updates the simsopt surface, psi, and profiles correctly."""
        # Record original values before rescaling
        psi_orig = self.desc_opt.get("psi")
        s_test = np.linspace(0, 1, 10)

        # Rescale
        self.desc_opt.rescale(L=("a", 1.2325), B=("<B>", np.pi))

        # Check that psi is synced between simsopt and DESC
        psi_new = self.desc_opt.get("psi")
        eq_psi = self.desc_opt.eq.Psi
        self.assertAlmostEqual(psi_new, eq_psi, places=10)
        # Verify that rescaling actually changed values
        # (rescaling with L=1.0, B=1.0 should change psi, and likely other properties)
        self.assertNotAlmostEqual(psi_orig, psi_new, places=5)

        # Check that boundary is synced: simsopt boundary should match eq surface
        boundary_from_eq = DescOptimizable.surface_from_desc(self.desc_opt.eq).copy(
            quadpoints_phi=self.desc_opt.boundary.quadpoints_phi,
            quadpoints_theta=self.desc_opt.boundary.quadpoints_theta,
        )
        gamma_new = self.desc_opt.boundary.gamma()
        gamma_from_eq = boundary_from_eq.gamma()
        np.testing.assert_allclose(gamma_new, gamma_from_eq, rtol=1e-10)

        # Check that pressure profile is synced
        pressure_from_eq = DescOptimizable._profile_from_desc(
            self.desc_opt.eq.pressure,
            n_knots=self.desc_opt.n_knots,
            degree=self.desc_opt.degree,
        )(s_test)
        pressure_new = self.desc_opt.pressure_profile(s_test)
        np.testing.assert_allclose(pressure_new, pressure_from_eq, rtol=1e-10)

        # Check that iota profile is synced
        iota_from_eq = DescOptimizable._profile_from_desc(
            self.desc_opt.eq.iota,
            n_knots=self.desc_opt.n_knots,
            degree=self.desc_opt.degree,
        )(s_test)
        iota_new = self.desc_opt.iota_profile(s_test)
        np.testing.assert_allclose(iota_new, iota_from_eq, rtol=1e-10)

    def test_rescale_with_current_profile(self):
        """Test rescale with a current profile instead of iota profile."""
        # Create DescOptimizable with current profile
        input_file = "ml_fast_ion/vmec_input_files/input.vacuum_template"
        boundary = SurfaceRZFourier.from_vmec_input(input_file)
        pressure = ProfilePolynomial([1e4, -1e4])
        current = ProfilePolynomial([0.0, 0.0, 1e5])  # symmetric: c2*rho^2

        desc_opt = DescOptimizable(
            boundary=boundary,
            psi=1.0,
            pressure_profile=pressure,
            current_profile=current,
        )

        # Record original values before rescaling
        psi_orig = desc_opt.get("psi")
        s_test = np.linspace(0, 1, 10)

        # Rescale
        desc_opt.rescale(L=("a", 1.2325), B=("<B>", np.pi))

        # Check that psi is synced between simsopt and DESC
        psi_new = desc_opt.get("psi")
        eq_psi = desc_opt.eq.Psi
        self.assertAlmostEqual(psi_new, eq_psi, places=10)

        # Verify that rescaling actually changed values
        self.assertNotAlmostEqual(psi_orig, psi_new, places=5)

        # Check that boundary is synced
        boundary_from_eq = DescOptimizable.surface_from_desc(desc_opt.eq).copy(
            quadpoints_phi=desc_opt.boundary.quadpoints_phi,
            quadpoints_theta=desc_opt.boundary.quadpoints_theta,
        )
        gamma_new = desc_opt.boundary.gamma()
        gamma_from_eq = boundary_from_eq.gamma()
        np.testing.assert_allclose(gamma_new, gamma_from_eq, rtol=1e-10)

        # Check that pressure profile is synced
        pressure_from_eq = DescOptimizable._profile_from_desc(
            desc_opt.eq.pressure,
            n_knots=desc_opt.n_knots,
            degree=desc_opt.degree,
        )(s_test)
        pressure_new = desc_opt.pressure_profile(s_test)
        np.testing.assert_allclose(pressure_new, pressure_from_eq, rtol=1e-10)

        # Check that current profile is synced
        current_from_eq = DescOptimizable._profile_from_desc(
            desc_opt.eq.current,
            n_knots=desc_opt.n_knots,
            degree=desc_opt.degree,
        )(s_test)
        current_new = desc_opt.current_profile(s_test)
        np.testing.assert_allclose(current_new, current_from_eq, rtol=1e-10)

    def test_rescale_vacuum_equilibrium(self):
        """Test rescale with vacuum (zero pressure and current)."""
        # Create DescOptimizable with vacuum profiles
        input_file = "ml_fast_ion/vmec_input_files/input.vacuum_template"
        boundary = SurfaceRZFourier.from_vmec_input(input_file)
        pressure = ProfilePolynomial([0.0, 0.0])  # vacuum: p = 0
        current = ProfilePolynomial([0.0, 0.0])  # vacuum: I = 0

        desc_opt = DescOptimizable(
            boundary=boundary,
            psi=1.0,
            pressure_profile=pressure,
            current_profile=current,
        )

        # Record original values before rescaling
        psi_orig = desc_opt.get("psi")
        s_test = np.linspace(0, 1, 10)

        # Rescale
        desc_opt.rescale(L=("a", 1.2325), B=("<B>", np.pi))

        # Check that psi is synced between simsopt and DESC
        psi_new = desc_opt.get("psi")
        eq_psi = desc_opt.eq.Psi
        self.assertAlmostEqual(psi_new, eq_psi, places=10)

        # Verify that rescaling actually changed values
        self.assertNotAlmostEqual(psi_orig, psi_new, places=5)

        # Check that boundary is synced
        boundary_from_eq = DescOptimizable.surface_from_desc(desc_opt.eq).copy(
            quadpoints_phi=desc_opt.boundary.quadpoints_phi,
            quadpoints_theta=desc_opt.boundary.quadpoints_theta,
        )
        gamma_new = desc_opt.boundary.gamma()
        gamma_from_eq = boundary_from_eq.gamma()
        np.testing.assert_allclose(gamma_new, gamma_from_eq, rtol=1e-10)

        # Check that pressure profile is synced (should be zero)
        pressure_from_eq = DescOptimizable._profile_from_desc(
            desc_opt.eq.pressure,
            n_knots=desc_opt.n_knots,
            degree=desc_opt.degree,
        )(s_test)
        pressure_new = desc_opt.pressure_profile(s_test)
        np.testing.assert_allclose(pressure_new, pressure_from_eq, rtol=1e-10)
        np.testing.assert_allclose(pressure_new, 0.0, atol=1e-12)

        # Check that current profile is synced (should be zero)
        current_from_eq = DescOptimizable._profile_from_desc(
            desc_opt.eq.current,
            n_knots=desc_opt.n_knots,
            degree=desc_opt.degree,
        )(s_test)
        current_new = desc_opt.current_profile(s_test)
        np.testing.assert_allclose(current_new, current_from_eq, rtol=1e-10)
        np.testing.assert_allclose(current_new, 0.0, atol=1e-12)


class TestComputeMethod(unittest.TestCase):
    """Test the compute method of DescOptimizable."""

    def setUp(self):
        """Create a test DescOptimizable."""
        input_file = "ml_fast_ion/vmec_input_files/input.vacuum_template"
        boundary = SurfaceRZFourier.from_vmec_input(input_file)
        pressure = ProfilePolynomial([1e4, -1e4])
        current = ProfilePolynomial([0.0])

        self.desc_opt = DescOptimizable(
            boundary=boundary,
            psi=1.0,
            pressure_profile=pressure,
            current_profile=current,
        )

    def test_compute_runs_and_returns_dict(self):
        """Test that compute() runs and returns a dict."""
        from desc.grid import LinearGrid

        # Create a simple 1D grid in rho for computing on
        grid = LinearGrid(rho=np.linspace(0, 1, 5))

        # Request computation of some basic quantities
        result = self.desc_opt.compute(grid=grid, names=["R", "Z"])

        # Check that result is a dict
        self.assertIsInstance(result, dict)

        # Check that requested quantities are in the result
        self.assertIn("R", result)
        self.assertIn("Z", result)

    def test_compute_quasisymmetry_metric(self):
        """Test that compute() can calculate quasisymmetry metrics."""
        from desc.grid import LinearGrid

        # Create a simple 1D grid in rho for computing on
        grid = LinearGrid(rho=np.linspace(0, 1, 5))

        # Compute magnetic field strength (key QS metric) and rotational transform
        result = self.desc_opt.compute(grid=grid, names=["B", "iota", "f_C"])

        # Check that result is a dict
        self.assertIsInstance(result, dict)

        # Check that requested quantities were computed
        self.assertIn("f_C", result)
        self.assertIn("iota", result)


class TestDetermineWhichProfile(unittest.TestCase):
    """Test the _determine_which_profile method."""

    def setUp(self):
        """Create test DescOptimizable objects with different profile configurations."""
        input_file = "ml_fast_ion/vmec_input_files/input.vacuum_template"
        self.boundary = SurfaceRZFourier.from_vmec_input(input_file)
        self.pressure = ProfilePolynomial([1e4, -1e4])
        self.iota = ProfilePolynomial([0.4, 0.1])
        self.current = ProfilePolynomial([1e5, 0.0, -1e5])  # symmetric: c0 + c2*rho^2

    def test_auto_with_current_only(self):
        """Test that 'auto' defaults to current when only current is provided."""
        desc_opt = DescOptimizable(
            boundary=self.boundary,
            psi=1.0,
            pressure_profile=self.pressure,
            current_profile=self.current,
            which_profile="auto",
        )
        result = desc_opt._determine_which_profile()
        self.assertTrue(result)

    def test_auto_with_iota_only(self):
        """Test that 'auto' uses iota when only iota is provided."""
        desc_opt = DescOptimizable(
            boundary=self.boundary,
            psi=1.0,
            pressure_profile=self.pressure,
            iota_profile=self.iota,
            which_profile="auto",
        )
        result = desc_opt._determine_which_profile()
        self.assertFalse(result)

    def test_auto_with_both_profiles(self):
        """Test that 'auto' defaults to current when both profiles are provided."""
        desc_opt = DescOptimizable(
            boundary=self.boundary,
            psi=1.0,
            pressure_profile=self.pressure,
            current_profile=self.current,
            iota_profile=self.iota,
            which_profile="auto",
        )
        result = desc_opt._determine_which_profile()
        self.assertTrue(result)

    def test_explicit_current(self):
        """Test that explicitly setting which_profile to 'current' works."""
        desc_opt = DescOptimizable(
            boundary=self.boundary,
            psi=1.0,
            pressure_profile=self.pressure,
            current_profile=self.current,
            iota_profile=self.iota,
            which_profile="current",
        )
        result = desc_opt._determine_which_profile()
        self.assertTrue(result)

    def test_explicit_iota(self):
        """Test that explicitly setting which_profile to 'iota' works."""
        desc_opt = DescOptimizable(
            boundary=self.boundary,
            psi=1.0,
            pressure_profile=self.pressure,
            current_profile=self.current,
            iota_profile=self.iota,
            which_profile="iota",
        )
        result = desc_opt._determine_which_profile()
        self.assertFalse(result)

    def test_invalid_which_profile(self):
        """Test that invalid which_profile value raises ValueError."""
        desc_opt = DescOptimizable(
            boundary=self.boundary,
            psi=1.0,
            pressure_profile=self.pressure,
            current_profile=self.current,
            which_profile="auto",
        )
        # Try to set an invalid value through the property setter
        with self.assertRaises(ValueError):
            desc_opt.which_profile = "invalid_option"

    def test_which_profile_property_setter(self):
        """Test that the which_profile property can be updated."""
        desc_opt = DescOptimizable(
            boundary=self.boundary,
            psi=1.0,
            pressure_profile=self.pressure,
            current_profile=self.current,
            iota_profile=self.iota,
            which_profile="auto",
        )
        # Initially should be auto, determining to current
        self.assertEqual(desc_opt.which_profile, "auto")
        self.assertTrue(desc_opt._determine_which_profile())

        # Update to iota
        desc_opt.which_profile = "iota"
        self.assertEqual(desc_opt.which_profile, "iota")
        self.assertFalse(desc_opt._determine_which_profile())

        # Update back to auto
        desc_opt.which_profile = "auto"
        self.assertEqual(desc_opt.which_profile, "auto")
        self.assertTrue(desc_opt._determine_which_profile())


if __name__ == "__main__":
    unittest.main()
