#include "dipole_field.h"
#pragma omp declare reduction (merge : Array : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

Array dipole_field_B(Array& points, Array& m_points, vector<double>& m){
    // int num_dipoles = m_points.size();
//#pragma omp parallel for reduction(merge: B)
//    for (int i = 0; i < num_dipoles; ++i) {
//	Array r_dot_m = xt::linalg::tensor_dot(points, m_points[i, :], 1, 1)
//        Array r_mag = xt::norm_l2(points - m_points[i, :], 1)
//        B += 3 * r_dot_m / xt::pow(rmag, 5) * points - m_points / xt::pow(rmag, 3)
//    }
    int num_points = points.size();
    Array B = xt::zeros<double>({points.shape(0), points.shape(1)});
#pragma omp parallel for reduction(merge: B)
    for (int i = 0; i < num_points; ++i) {
	Array r_dot_m = xt::linalg::tensor_dot(points(i, :), m_points, {1}, {1});
        Array r_mag = xt::linalg::norm_l2(points(i, :) - m_points, {1});
        B(i, :) = xt::sum(3 * r_dot_m / xt::pow(r_mag, 5) * points - m_points / xt::pow(r_mag, 3))();
    }
    return B * 1e-7;
}
