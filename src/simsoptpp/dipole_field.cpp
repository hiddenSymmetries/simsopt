#include "dipole_field.h"
#include "simdhelpers.h"
#include "vec3dsimd.h"

// Calculate the B field at a set of evaluation points from N dipoles
// points: where to evaluate the field
// m_points: where the dipoles are located
// m: dipole moments ('orientation')
// everything in xyz coordinates
Array dipole_field_B(Array& points, Array& m_points, Array& m) {
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(m_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("m_points needs to be in row-major storage order");
    if(m.layout() != xt::layout_type::row_major)
          throw std::runtime_error("m needs to be in row-major storage order");

    int num_points = points.shape(0);
    int num_dipoles = m_points.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    Array B = xt::zeros<double>({points.shape(0), points.shape(1)});
   
    // initialize pointers to the beginning of m and the dipole grid
    double* m_points_ptr = &(m_points(0, 0));
    double* m_ptr = &(m(0, 0));
    double fak = 1e-7;  // mu0 divided by 4 * pi factor

    // Loop through the evaluation points by chunks of simd_size
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_points; i += simd_size) {
        auto point_i = Vec3dSimd();
        auto B_i = Vec3dSimd();
        // check that i + k isn't bigger than num_points
        int klimit = std::min(simd_size, num_points - i);
        for(int k = 0; k < klimit; k++){
            for (int d = 0; d < 3; ++d) {
                point_i[d][k] = points(i + k, d);
            }
        }
        for (int j = 0; j < num_dipoles; ++j) {
            Vec3dSimd m_j = Vec3dSimd(m_ptr[3 * j + 0], m_ptr[3 * j + 1], m_ptr[3 * j + 2]);
            Vec3dSimd mp_j = Vec3dSimd(m_points_ptr[3 * j + 0], m_points_ptr[3 * j + 1], m_points_ptr[3 * j + 2]);
            Vec3dSimd r = point_i - mp_j;
            simd_t rmag_2     = normsq(r);
            simd_t rmag_inv   = rsqrt(rmag_2);
            simd_t rmag_inv_3 = rmag_inv * (rmag_inv * rmag_inv);
            simd_t rmag_inv_5 = rmag_inv_3 * (rmag_inv * rmag_inv);
            simd_t rdotm = inner(r, m_j);
            B_i.x += 3.0 * rdotm * r.x * rmag_inv_5 - m_j.x * rmag_inv_3;
            B_i.y += 3.0 * rdotm * r.y * rmag_inv_5 - m_j.y * rmag_inv_3;
            B_i.z += 3.0 * rdotm * r.z * rmag_inv_5 - m_j.z * rmag_inv_3;
        } 
        for(int k = 0; k < klimit; k++){
            B(i + k, 0) = fak * B_i.x[k];
            B(i + k, 1) = fak * B_i.y[k];
            B(i + k, 2) = fak * B_i.z[k];
        }
    }
    return B;
}

Array dipole_field_dB(Array& points, Array& m_points, Array& m) {
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(m_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("m_points needs to be in row-major storage order");
    if(m.layout() != xt::layout_type::row_major)
          throw std::runtime_error("m needs to be in row-major storage order");

    int num_points = points.shape(0);
    int num_dipoles = m_points.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    Array dB = xt::zeros<double>({points.shape(0), points.shape(1), points.shape(1)});
    double* m_points_ptr = &(m_points(0, 0));
    double* m_ptr = &(m(0, 0));
    double fak = 1e-7;
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_points; i += simd_size) {
        auto point_i = Vec3dSimd();
        auto dB_i1   = Vec3dSimd();
        auto dB_i2   = Vec3dSimd();
        int klimit = std::min(simd_size, num_points - i);
        for(int k = 0; k < klimit; k++){
            for (int d = 0; d < 3; ++d) {
                point_i[d][k] = points(i + k, d);
            }
        }
        for (int j = 0; j < num_dipoles; ++j) {
            Vec3dSimd m_j = Vec3dSimd(m_ptr[3 * j], m_ptr[3 * j + 1], m_ptr[3 * j + 2]);
            Vec3dSimd mp_j = Vec3dSimd(m_points_ptr[3 * j], m_points_ptr[3 * j + 1], m_points_ptr[3 * j + 2]);
            Vec3dSimd r = point_i - mp_j;
            simd_t rmag_2     = normsq(r);
            simd_t rmag_inv   = rsqrt(rmag_2);
	    simd_t rmag_inv_2 = rmag_inv * rmag_inv;
            simd_t rmag_inv_3 = rmag_inv * rmag_inv_2;
            simd_t rmag_inv_5 = rmag_inv_3 * rmag_inv_2; 
            simd_t rdotm = inner(r, m_j);
            dB_i1.x += 3.0 * rmag_inv_5 * ((2.0 * m_j.x * r.x + rdotm) - 5.0 * rdotm * r.x * r.x * rmag_inv_2);
            dB_i1.y += 3.0 * rmag_inv_5 * ((m_j.x * r.y + m_j.y * r.x) - 5.0 * rdotm * r.x * r.y * rmag_inv_2);
            dB_i1.z += 3.0 * rmag_inv_5 * ((m_j.x * r.z + m_j.z * r.x) - 5.0 * rdotm * r.x * r.z * rmag_inv_2);
            dB_i2.x += 3.0 * rmag_inv_5 * ((2.0 * m_j.y * r.y + rdotm) - 5.0 * rdotm * r.y * r.y * rmag_inv_2);
            dB_i2.y += 3.0 * rmag_inv_5 * ((m_j.y * r.z + m_j.z * r.y) - 5.0 * rdotm * r.y * r.z * rmag_inv_2);
            dB_i2.z += 3.0 * rmag_inv_5 * ((2.0 * m_j.z * r.z + rdotm) - 5.0 * rdotm * r.z * r.z * rmag_inv_2);
        } 
        for(int k = 0; k < klimit; k++){
            dB(i + k, 0, 0) = fak * dB_i1.x[k];
            dB(i + k, 0, 1) = fak * dB_i1.y[k];
            dB(i + k, 0, 2) = fak * dB_i1.z[k];
            dB(i + k, 1, 1) = fak * dB_i2.x[k];
            dB(i + k, 1, 2) = fak * dB_i2.y[k];
            dB(i + k, 2, 2) = fak * dB_i2.z[k];
	    dB(i + k, 1, 0) = dB(i + k, 0, 1);
	    dB(i + k, 2, 0) = dB(i + k, 0, 2);
	    dB(i + k, 2, 1) = dB(i + k, 1, 2);
	}
    }
    return dB;
}
