#include "dipole_field.h"
#pragma omp declare reduction (+: xt::xarray<double> : omp_out)

// convert to low-level code with for loops and pyarrays
Array dipole_field_B(Array& points, Array& m_points, xt::xarray<double>& m){
    int num_points = points.size();
    xt::xarray<double> B = xt::zeros<double>({points.shape(0), points.shape(1)});
#pragma omp parallel for reduction(+: B)
    for (int i = 0; i < num_points; ++i) {
	auto row_points = xt::row(points, i);
	auto r_dot_m = xt::linalg::tensordot(row_points, m, {1}, {1});
	auto r_mag = xt::norm_l2(row_points - m_points, {1});
	auto r_mag5 = r_dot_m / xt::pow(r_mag, 5);
        auto ones = xt::ones<double>({r_mag.shape(0)});
	auto r_mag3 = ones / xt::pow(r_mag, 3);
	auto first_fac = 3.0 * xt::linalg::tensordot(r_mag5, points, {0}, {0});
	auto second_fac = - xt::linalg::tensordot(r_mag3, m, {0}, {0});
	xt::xarray<double> B_vec = first_fac + second_fac;
	// xt::xarray<double> B_vec = 3.0 * xt::linalg::tensordot(xt::linalg::tensordot(xt::row(points, i), m, {1}, {1}) / xt::pow(xt::norm_l2(xt::row(points, i) - m_points, {1}), 5), points, {0}, {0}) - xt::linalg::tensordot(xt::linalg::tensordot(xt::row(points, i), m, {1}, {1}) / xt::pow(xt::norm_l2(xt::row(points, i) - m_points, {1}), 3), m, {0}, {0})
	B(i, xt::all()) = B_vec(xt::all());
    }
    return B * 1e-7;
}
