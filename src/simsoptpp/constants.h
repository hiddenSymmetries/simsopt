#pragma once

// Vacuum permeability in N/A^2, CODATA 2022 (same as scipy.constants.mu_0).
// tests/field/test_biotsavart.py::test_cpp_mu0_matches_scipy checks this against scipy.
constexpr double MU0 = 1.25663706127e-6;
// mu0 / (4 pi), the prefactor of the Biot-Savart law.
constexpr double MU0_OVER_4PI = MU0 / (4 * 3.14159265358979323846);
