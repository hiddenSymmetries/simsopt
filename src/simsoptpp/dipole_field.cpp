#include "dipole_field.h"
#include "simdhelpers.h"
#include "vec3dsimd.h"
#include <cmath>
#include <Eigen/Dense>

#if defined(USE_XSIMD)
// Calculate the B field at a set of evaluation points from N dipoles:
// B = mu0 / (4 * pi) sum_{i=1}^N 3(m_i * r_i)r_i / |r_i|^5 - m_i / |r_i|^3
// points: where to evaluate the field
// m_points: where the dipoles are located
// m: dipole moments (vectors)
// everything in xyz coordinates
// r_i = points - m_points
// m_i = m
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
        // Loops through all the dipoles
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

// A = mu0 / (4 * pi) sum_{i=1}^N m_i x r_i / |r_i|^3
Array dipole_field_A(Array& points, Array& m_points, Array& m) {
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
    Array A = xt::zeros<double>({points.shape(0), points.shape(1)});
   
    // initialize pointers to the beginning of m and the dipole grid
    double* m_points_ptr = &(m_points(0, 0));
    double* m_ptr = &(m(0, 0));
    double fak = 1e-7;  // mu0 divided by 4 * pi factor

    // Loop through the evaluation points by chunks of simd_size
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_points; i += simd_size) {
        auto point_i = Vec3dSimd();
        auto A_i = Vec3dSimd();

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
            Vec3dSimd mcrossr = cross(m_j, r);
            A_i.x += mcrossr.x * rmag_inv_3;
            A_i.y += mcrossr.y * rmag_inv_3;
            A_i.z += mcrossr.z * rmag_inv_3;
        } 
        for(int k = 0; k < klimit; k++){
            A(i + k, 0) = fak * A_i.x[k];
            A(i + k, 1) = fak * A_i.y[k];
            A(i + k, 2) = fak * A_i.z[k];
        }
    }
    return A;
}

// For each dipole i:
// dB_j/dr_k = mu0 / (4 * pi)
// [
//    3(m_k * r_j + m_j * r_k) / |r|^5
//    + 3 (m_l * r_l) * delta_{jk} / |r|^5 
//    - 15 (m_l * r_l) * (r_j * r_k) / |r|^7
// ]
// where here the indices on m, r, and B denote the spatial components.
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

// For each dipole i:
// dA_j/dr_k = mu0 / (4 * pi)
// [
//    eps_jlk * m_l / |r|^3
//    - 3 (m cross r)_j * r_k / |r|^5 
// ]
// where here the indices on m, r, and A denote the spatial components,
// eps_jlk is the Levi-Civita symbol, and the cross product is taken in 3D.
Array dipole_field_dA(Array& points, Array& m_points, Array& m) {
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
    Array dA = xt::zeros<double>({points.shape(0), points.shape(1), points.shape(1)});
    double* m_points_ptr = &(m_points(0, 0));
    double* m_ptr = &(m(0, 0));
    double fak = 1e-7;
    
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_points; i += simd_size) {
        auto point_i = Vec3dSimd();
        auto dA_i1   = Vec3dSimd();
        auto dA_i2   = Vec3dSimd();
        auto dA_i3   = Vec3dSimd();
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
            Vec3dSimd mcrossr = cross(m_j, r);
            dA_i1.x += rmag_inv_3 * (- 3.0 * mcrossr.x * r.x * rmag_inv_2);
            dA_i1.y += rmag_inv_3 * (- m_j.z - 3.0 * mcrossr.x * r.y * rmag_inv_2);
            dA_i1.z += rmag_inv_3 * (m_j.y - 3.0 * mcrossr.x * r.z * rmag_inv_2);
            dA_i2.x += rmag_inv_3 * (m_j.z - 3.0 * mcrossr.y * r.x * rmag_inv_2);
            dA_i2.y += rmag_inv_3 * (- 3.0 * mcrossr.y * r.y * rmag_inv_2);
            dA_i2.z += rmag_inv_3 * (- m_j.x - 3.0 * mcrossr.y * r.z * rmag_inv_2);
            dA_i3.x += rmag_inv_3 * (- m_j.y - 3.0 * mcrossr.z * r.x * rmag_inv_2);
            dA_i3.y += rmag_inv_3 * (m_j.x - 3.0 * mcrossr.z * r.y * rmag_inv_2);
            dA_i3.z += rmag_inv_3 * (- 3.0 * mcrossr.z * r.z * rmag_inv_2);
	    } 
        for(int k = 0; k < klimit; k++){
            dA(i + k, 0, 0) = fak * dA_i1.x[k];
            dA(i + k, 0, 1) = fak * dA_i1.y[k];
            dA(i + k, 0, 2) = fak * dA_i1.z[k];
            dA(i + k, 1, 0) = fak * dA_i2.x[k];
            dA(i + k, 1, 1) = fak * dA_i2.y[k];
            dA(i + k, 1, 2) = fak * dA_i2.z[k];
	    dA(i + k, 2, 0) = fak * dA_i3.x[k];
	    dA(i + k, 2, 1) = fak * dA_i3.y[k]; 
            dA(i + k, 2, 2) = fak * dA_i3.z[k];
	}
    }
    return dA;
}

// Calculate the geometric factor A needed for the permanent magnet optimization
// Bnormal * n = A * m - b, where n is the unit normal to the plasma surface.
// A = [g_1, ..., g_num_dipoles]
// g_i = mu0 / (4 * pi) [3(n_i * r_i)r_i / |r_i|^5 - n_i / |r_i|^3]
// points: where to evaluate the field
// m_points: where the dipoles are located
// unitnormal: unit normal vectors from the plasma surface
// nfp: field-period symmetry of the plasma surface
// stellsym: stellarator symmetry (True/False) of the plasma surface
// b: Bnormal component corresponding to the non-magnet fields (e.g. external coils)
// coordinate_flag: which coordinate system should be considered "grid-aligned"
// R0: Major radius of the device, needed if a simple toroidal coordinate system is desired
// returns the optimization matrix, or inductance, A
Array dipole_field_Bn(Array& points, Array& m_points, Array& unitnormal, int nfp, int stellsym, Array& b, std::string coordinate_flag, double R0)
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(m_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("m_points needs to be in row-major storage order");
    if(unitnormal.layout() != xt::layout_type::row_major)
          throw std::runtime_error("unit normal needs to be in row-major storage order");
    if(b.layout() != xt::layout_type::row_major)
          throw std::runtime_error("b needs to be in row-major storage order");

    int num_points = points.shape(0);
    int num_dipoles = m_points.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    Array A = xt::zeros<double>({num_points, num_dipoles, 3});

    std::string cylindrical_str = "cylindrical";
    std::string toroidal_str = "toroidal";

    // initialize pointer to the beginning of the dipole grid
    double* m_points_ptr = &(m_points(0, 0));
    double fak = 1e-7;  // mu0 divided by 4 * pi factor

    // Loop through the evaluation points by chunks of simd_size
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_points; i += simd_size) {
        auto point_i = Vec3dSimd();
        auto n_i = Vec3dSimd();

        // check that i + k isn't bigger than num_points
        int klimit = std::min(simd_size, num_points - i);
        for(int k = 0; k < klimit; k++){
            for (int d = 0; d < 3; ++d) {
                point_i[d][k] = points(i + k, d);
                n_i[d][k] = unitnormal(i + k, d);
            }
        }
        // Loop through all the dipoles, using all the symmetries
        for (int j = 0; j < num_dipoles; ++j) {
            Vec3dSimd mp_j = Vec3dSimd(m_points_ptr[3 * j + 0], m_points_ptr[3 * j + 1], m_points_ptr[3 * j + 2]);
            for (int stell = 0; stell < (stellsym + 1); ++stell) {
                for(int fp = 0; fp < nfp; ++fp) {
                    simd_t phi0 = (2 * M_PI / ((simd_t) nfp)) * fp;
                    simd_t sphi0 = xsimd::sin(phi0);
                    simd_t cphi0 = xsimd::cos(phi0);
                    auto G_i = Vec3dSimd();

                    // Calculate new dipole location after accounting for the symmetries
                    // reflect the y and z-components and then rotate by phi0
                    simd_t mp_x_new = mp_j.x * cphi0 - mp_j.y * sphi0 * pow(-1, stell);
                    simd_t mp_y_new = mp_j.x * sphi0 + mp_j.y * cphi0 * pow(-1, stell);
                    simd_t mp_z_new = mp_j.z * pow(-1, stell);
                    Vec3dSimd mp_j_new = Vec3dSimd(mp_x_new, mp_y_new, mp_z_new);

                    // Calculate new phi location if switching to cylindrical coordinates
                    simd_t mp_phi_new = xsimd::atan2(mp_j.y, mp_j.x);
                    simd_t mp_theta_new = xsimd::atan2(mp_j.z, sqrt(mp_j.x * mp_j.x + mp_j.y * mp_j.y) - R0);
                    simd_t sphi_new = xsimd::sin(mp_phi_new);
                    simd_t stheta_new = xsimd::sin(mp_theta_new);
                    simd_t cphi_new = xsimd::cos(mp_phi_new);
                    simd_t ctheta_new = xsimd::cos(mp_theta_new);

                    // Compute the unsymmetrized inductance matrix
                    Vec3dSimd r = point_i - mp_j_new;
                    simd_t rmag_2 = normsq(r);
                    simd_t rmag_inv   = rsqrt(rmag_2);
                    simd_t rmag_inv_3 = rmag_inv * (rmag_inv * rmag_inv);
                    simd_t rmag_inv_5 = rmag_inv_3 * (rmag_inv * rmag_inv);
                    simd_t rdotn = inner(r, n_i);
                    G_i.x = 3.0 * rdotn * r.x * rmag_inv_5 - n_i.x * rmag_inv_3;
                    G_i.y = 3.0 * rdotn * r.y * rmag_inv_5 - n_i.y * rmag_inv_3;
                    G_i.z = 3.0 * rdotn * r.z * rmag_inv_5 - n_i.z * rmag_inv_3;
                    for(int k = 0; k < klimit; k++){
                        if (coordinate_flag == cylindrical_str) {
                            double Ax_temp = (G_i.x[k] * cphi0[k] + G_i.y[k] * sphi0[k]) * pow(-1, stell);
                            double Ay_temp = (- G_i.x[k] * sphi0[k] + G_i.y[k] * cphi0[k]);
                            A(i + k, j, 0) += fak * (Ax_temp * cphi_new[k] + Ay_temp * sphi_new[k]);
                            A(i + k, j, 1) += fak * ( - Ax_temp * sphi_new[k] + Ay_temp * cphi_new[k]);
                            A(i + k, j, 2) += fak * G_i.z[k];
                        }
                        else if (coordinate_flag == toroidal_str) {

                            double Ax_temp = (G_i.x[k] * cphi0[k] + G_i.y[k] * sphi0[k]) * pow(-1, stell);
                            double Ay_temp = (- G_i.x[k] * sphi0[k] + G_i.y[k] * cphi0[k]);
                            double Az_temp = G_i.z[k];
                            A(i + k, j, 0) += fak * (Ax_temp * cphi_new[k] * ctheta_new[k] + Ay_temp * sphi_new[k] * ctheta_new[k] + Az_temp * stheta_new[k]);
                            A(i + k, j, 1) += fak * ( - Ax_temp * sphi_new[k] + Ay_temp * cphi_new[k]);
                            A(i + k, j, 2) += fak * (- Ax_temp * cphi_new[k] * stheta_new[k] - Ay_temp * sphi_new[k] * stheta_new[k] + Az_temp * ctheta_new[k]);
                        }
                        else {
                            // rotate by -phi0 and then flip x component
                            // This should be the reverse of what is done to the m vector and the dipole grid
                            // because A * m = A * R^T * R * m and R is an orthogonal matrix both
                            // for a reflection and a rotation.
                            A(i + k, j, 0) += fak * (G_i.x[k] * cphi0[k] + G_i.y[k] * sphi0[k]) * pow(-1, stell);
                            A(i + k, j, 1) += fak * (- G_i.x[k] * sphi0[k] + G_i.y[k] * cphi0[k]);
                            A(i + k, j, 2) += fak * G_i.z[k];
                        }
                    }
                }
            }
        }
    }
    return A;
}

#else
// Calculate the B field at a set of evaluation points from N dipoles:
// B = mu0 / (4 * pi) sum_{i=1}^N 3(m_i * r_i)r_i / |r_i|^5 - m_i / |r_i|^3
// points: where to evaluate the field
// m_points: where the dipoles are located
// m: dipole moments (vectors)
// everything in xyz coordinates
// r_i = points - m_points
// m_i = m
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
    Array B = xt::zeros<double>({points.shape(0), points.shape(1)});

    // initialize pointers to the beginning of m and the dipole grid
    double* m_points_ptr = &(m_points(0, 0));
    double* m_ptr = &(m(0, 0));
    double fak = 1e-7;  // mu0 divided by 4 * pi factor

    // Loop through the evaluation points by chunks of simd_size
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_points; i++) {
        auto point_i = Vec3dStd();
        auto B_i = Vec3dStd();

        for (int d = 0; d < 3; ++d) {
            point_i[d] = points(i, d);
        }
        // Loops through all the dipoles
        for (int j = 0; j < num_dipoles; ++j) {
                auto m_j = Vec3dStd(m_ptr[3 * j + 0], m_ptr[3 * j + 1], m_ptr[3 * j + 2]);
                auto mp_j = Vec3dStd(m_points_ptr[3 * j + 0], m_points_ptr[3 * j + 1], m_points_ptr[3 * j + 2]);
                auto r = point_i - mp_j;
                auto rmag_2     = normsq(r);
                auto rmag_inv   = rsqrt(rmag_2);
                auto rmag_inv_3 = rmag_inv * (rmag_inv * rmag_inv);
                auto rmag_inv_5 = rmag_inv_3 * (rmag_inv * rmag_inv);
                auto rdotm = inner(r, m_j);
                B_i.x += 3.0 * rdotm * r.x * rmag_inv_5 - m_j.x * rmag_inv_3;
                B_i.y += 3.0 * rdotm * r.y * rmag_inv_5 - m_j.y * rmag_inv_3;
                B_i.z += 3.0 * rdotm * r.z * rmag_inv_5 - m_j.z * rmag_inv_3;
            }
            B(i, 0) = fak * B_i.x;
            B(i, 1) = fak * B_i.y;
            B(i, 2) = fak * B_i.z;
    }
    return B;
}

// A = mu0 / (4 * pi) sum_{i=1}^N m_i x r_i / |r_i|^3
Array dipole_field_A(Array& points, Array& m_points, Array& m) {
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(m_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("m_points needs to be in row-major storage order");
    if(m.layout() != xt::layout_type::row_major)
          throw std::runtime_error("m needs to be in row-major storage order");

    int num_points = points.shape(0);
    int num_dipoles = m_points.shape(0);
    Array A = xt::zeros<double>({points.shape(0), points.shape(1)});

    // initialize pointers to the beginning of m and the dipole grid
    double* m_points_ptr = &(m_points(0, 0));
    double* m_ptr = &(m(0, 0));
    double fak = 1e-7;  // mu0 divided by 4 * pi factor

    // Loop through the evaluation points by chunks of simd_size
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_points; i++) {
        auto point_i = Vec3dStd();
        auto A_i = Vec3dStd();

        for (int d = 0; d < 3; ++d) {
            point_i[d] = points(i, d);
        }
        for (int j = 0; j < num_dipoles; ++j) {
            auto m_j = Vec3dStd(m_ptr[3 * j + 0], m_ptr[3 * j + 1], m_ptr[3 * j + 2]);
            auto mp_j = Vec3dStd(m_points_ptr[3 * j + 0], m_points_ptr[3 * j + 1], m_points_ptr[3 * j + 2]);
            auto r = point_i - mp_j;
            auto rmag_2     = normsq(r);
            auto rmag_inv   = rsqrt(rmag_2);
            auto rmag_inv_3 = rmag_inv * (rmag_inv * rmag_inv);
            auto mcrossr = cross(m_j, r);
            A_i.x += mcrossr.x * rmag_inv_3;
            A_i.y += mcrossr.y * rmag_inv_3;
            A_i.z += mcrossr.z * rmag_inv_3;
        }
        A(i, 0) = fak * A_i.x;
        A(i, 1) = fak * A_i.y;
        A(i, 2) = fak * A_i.z;
    }
    return A;
}

// For each dipole i:
// dB_j/dr_k = mu0 / (4 * pi)
// [
//    3(m_k * r_j + m_j * r_k) / |r|^5
//    + 3 (m_l * r_l) * delta_{jk} / |r|^5
//    - 15 (m_l * r_l) * (r_j * r_k) / |r|^7
// ]
// where here the indices on m, r, and B denote the spatial components.
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
    Array dB = xt::zeros<double>({points.shape(0), points.shape(1), points.shape(1)});
    double* m_points_ptr = &(m_points(0, 0));
    double* m_ptr = &(m(0, 0));
    double fak = 1e-7;

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_points; i++) {
        auto point_i = Vec3dStd();
        auto dB_i1   = Vec3dStd();
        auto dB_i2   = Vec3dStd();
        for (int d = 0; d < 3; ++d) {
            point_i[d] = points(i, d);
        }
        for (int j = 0; j < num_dipoles; ++j) {
            auto m_j = Vec3dStd(m_ptr[3 * j], m_ptr[3 * j + 1], m_ptr[3 * j + 2]);
            auto mp_j = Vec3dStd(m_points_ptr[3 * j], m_points_ptr[3 * j + 1], m_points_ptr[3 * j + 2]);
            auto r = point_i - mp_j;
            auto rmag_2     = normsq(r);
            auto rmag_inv   = rsqrt(rmag_2);
	    auto rmag_inv_2 = rmag_inv * rmag_inv;
            auto rmag_inv_3 = rmag_inv * rmag_inv_2;
            auto rmag_inv_5 = rmag_inv_3 * rmag_inv_2;
            auto rdotm = inner(r, m_j);
            dB_i1.x += 3.0 * rmag_inv_5 * ((2.0 * m_j.x * r.x + rdotm) - 5.0 * rdotm * r.x * r.x * rmag_inv_2);
            dB_i1.y += 3.0 * rmag_inv_5 * ((m_j.x * r.y + m_j.y * r.x) - 5.0 * rdotm * r.x * r.y * rmag_inv_2);
            dB_i1.z += 3.0 * rmag_inv_5 * ((m_j.x * r.z + m_j.z * r.x) - 5.0 * rdotm * r.x * r.z * rmag_inv_2);
            dB_i2.x += 3.0 * rmag_inv_5 * ((2.0 * m_j.y * r.y + rdotm) - 5.0 * rdotm * r.y * r.y * rmag_inv_2);
            dB_i2.y += 3.0 * rmag_inv_5 * ((m_j.y * r.z + m_j.z * r.y) - 5.0 * rdotm * r.y * r.z * rmag_inv_2);
            dB_i2.z += 3.0 * rmag_inv_5 * ((2.0 * m_j.z * r.z + rdotm) - 5.0 * rdotm * r.z * r.z * rmag_inv_2);
        }
        dB(i, 0, 0) = fak * dB_i1.x;
        dB(i, 0, 1) = fak * dB_i1.y;
        dB(i, 0, 2) = fak * dB_i1.z;
        dB(i, 1, 1) = fak * dB_i2.x;
        dB(i, 1, 2) = fak * dB_i2.y;
        dB(i, 2, 2) = fak * dB_i2.z;
        dB(i, 1, 0) = dB(i, 0, 1);
        dB(i, 2, 0) = dB(i, 0, 2);
        dB(i, 2, 1) = dB(i, 1, 2);
    }
    return dB;
}

// For each dipole i:
// dA_j/dr_k = mu0 / (4 * pi)
// [
//    eps_jlk * m_l / |r|^3
//    - 3 (m cross r)_j * r_k / |r|^5
// ]
// where here the indices on m, r, and A denote the spatial components,
// eps_jlk is the Levi-Civita symbol, and the cross product is taken in 3D.
Array dipole_field_dA(Array& points, Array& m_points, Array& m) {
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(m_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("m_points needs to be in row-major storage order");
    if(m.layout() != xt::layout_type::row_major)
          throw std::runtime_error("m needs to be in row-major storage order");

    int num_points = points.shape(0);
    int num_dipoles = m_points.shape(0);
    Array dA = xt::zeros<double>({points.shape(0), points.shape(1), points.shape(1)});
    double* m_points_ptr = &(m_points(0, 0));
    double* m_ptr = &(m(0, 0));
    double fak = 1e-7;

    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_points; i++) {
        auto point_i = Vec3dStd();
        auto dA_i1   = Vec3dStd();
        auto dA_i2   = Vec3dStd();
        auto dA_i3   = Vec3dStd();
        for (int d = 0; d < 3; ++d) {
            point_i[d] = points(i, d);
        }
        for (int j = 0; j < num_dipoles; ++j) {
            auto m_j = Vec3dStd(m_ptr[3 * j], m_ptr[3 * j + 1], m_ptr[3 * j + 2]);
            auto mp_j = Vec3dStd(m_points_ptr[3 * j], m_points_ptr[3 * j + 1], m_points_ptr[3 * j + 2]);
            auto r = point_i - mp_j;
            auto rmag_2     = normsq(r);
            auto rmag_inv   = rsqrt(rmag_2);
	    auto rmag_inv_2 = rmag_inv * rmag_inv;
            auto rmag_inv_3 = rmag_inv * rmag_inv_2;
            auto mcrossr = cross(m_j, r);
            dA_i1.x += rmag_inv_3 * (- 3.0 * mcrossr.x * r.x * rmag_inv_2);
            dA_i1.y += rmag_inv_3 * (- m_j.z - 3.0 * mcrossr.x * r.y * rmag_inv_2);
            dA_i1.z += rmag_inv_3 * (m_j.y - 3.0 * mcrossr.x * r.z * rmag_inv_2);
            dA_i2.x += rmag_inv_3 * (m_j.z - 3.0 * mcrossr.y * r.x * rmag_inv_2);
            dA_i2.y += rmag_inv_3 * (- 3.0 * mcrossr.y * r.y * rmag_inv_2);
            dA_i2.z += rmag_inv_3 * (- m_j.x - 3.0 * mcrossr.y * r.z * rmag_inv_2);
            dA_i3.x += rmag_inv_3 * (- m_j.y - 3.0 * mcrossr.z * r.x * rmag_inv_2);
            dA_i3.y += rmag_inv_3 * (m_j.x - 3.0 * mcrossr.z * r.y * rmag_inv_2);
            dA_i3.z += rmag_inv_3 * (- 3.0 * mcrossr.z * r.z * rmag_inv_2);
	    }
        dA(i, 0, 0) = fak * dA_i1.x;
        dA(i, 0, 1) = fak * dA_i1.y;
        dA(i, 0, 2) = fak * dA_i1.z;
        dA(i, 1, 0) = fak * dA_i2.x;
        dA(i, 1, 1) = fak * dA_i2.y;
        dA(i, 1, 2) = fak * dA_i2.z;
        dA(i, 2, 0) = fak * dA_i3.x;
        dA(i, 2, 1) = fak * dA_i3.y;
        dA(i, 2, 2) = fak * dA_i3.z;
    }
    return dA;
}


// Calculate the geometric factor A needed for the permanent magnet optimization
// Bnormal * n = A * m - b, where n is the unit normal to the plasma surface.
// A = [g_1, ..., g_num_dipoles]
// g_i = mu0 / (4 * pi) [3(n_i * r_i)r_i / |r_i|^5 - n_i / |r_i|^3]
// points: where to evaluate the field
// m_points: where the dipoles are located
// unitnormal: unit normal vectors from the plasma surface
// nfp: field-period symmetry of the plasma surface
// stellsym: stellarator symmetry (True/False) of the plasma surface
// b: Bnormal component corresponding to the non-magnet fields (e.g. external coils)
// coordinate_flag: which coordinate system should be considered "grid-aligned"
// R0: Major radius of the device, needed if a simple toroidal coordinate system is desired
// returns the optimization matrix, or inductance, A
Array dipole_field_Bn(Array& points, Array& m_points, Array& unitnormal, int nfp, int stellsym, Array& b, std::string coordinate_flag, double R0) 
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(m_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("m_points needs to be in row-major storage order");
    if(unitnormal.layout() != xt::layout_type::row_major)
          throw std::runtime_error("unit normal needs to be in row-major storage order");
    if(b.layout() != xt::layout_type::row_major)
          throw std::runtime_error("b needs to be in row-major storage order");
    
    int num_points = points.shape(0);
    int num_dipoles = m_points.shape(0);
    Array A = xt::zeros<double>({num_points, num_dipoles, 3});
  
    std::string cylindrical_str = "cylindrical"; 
    std::string toroidal_str = "toroidal"; 
    
    // initialize pointer to the beginning of the dipole grid
    double* m_points_ptr = &(m_points(0, 0));
    double fak = 1e-7;  // mu0 divided by 4 * pi factor

    // Loop through the evaluation points by chunks of simd_size
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_points; i++) {
        auto point_i = Vec3dStd();
        auto n_i = Vec3dStd();
        
        for (int d = 0; d < 3; ++d) {
            point_i[d] = points(i, d);
            n_i[d] = unitnormal(i, d);
        }
	// Loop through all the dipoles, using all the symmetries
        for (int j = 0; j < num_dipoles; ++j) {
            auto mp_j = Vec3dStd(m_points_ptr[3 * j + 0], m_points_ptr[3 * j + 1], m_points_ptr[3 * j + 2]);
            for (int stell = 0; stell < (stellsym + 1); ++stell) {
                for(int fp = 0; fp < nfp; ++fp) {
                    auto phi0 = (2 * M_PI / ((double) nfp)) * fp;
                    auto sphi0 = std::sin(phi0);
                    auto cphi0 = std::cos(phi0);
                    auto G_i = Vec3dStd();

                    // Calculate new dipole location after accounting for the symmetries
                    // reflect the y and z-components and then rotate by phi0
                    auto mp_x_new = mp_j.x * cphi0 - mp_j.y * sphi0 * pow(-1, stell);
                    auto mp_y_new = mp_j.x * sphi0 + mp_j.y * cphi0 * pow(-1, stell);
                    auto mp_z_new = mp_j.z * pow(-1, stell);
                    auto mp_j_new = Vec3dStd(mp_x_new, mp_y_new, mp_z_new);

                    // Calculate new phi location if switching to cylindrical coordinates
                    auto mp_phi_new = std::atan2(mp_j.y, mp_j.x);
                    auto mp_theta_new = std::atan2(mp_j.z, sqrt(mp_j.x * mp_j.x + mp_j.y * mp_j.y) - R0);
                    auto sphi_new = std::sin(mp_phi_new);
                    auto stheta_new = std::sin(mp_theta_new);
                    auto cphi_new = std::cos(mp_phi_new);
                    auto ctheta_new = std::cos(mp_theta_new);

                    // Compute the unsymmetrized inductance matrix
                    auto r = point_i - mp_j_new;
                    auto rmag_2 = normsq(r);
                    auto rmag_inv   = rsqrt(rmag_2);
                    auto rmag_inv_3 = rmag_inv * (rmag_inv * rmag_inv);
                    auto rmag_inv_5 = rmag_inv_3 * (rmag_inv * rmag_inv);
                    auto rdotn = inner(r, n_i);
                    G_i.x = 3.0 * rdotn * r.x * rmag_inv_5 - n_i.x * rmag_inv_3;
                    G_i.y = 3.0 * rdotn * r.y * rmag_inv_5 - n_i.y * rmag_inv_3;
                    G_i.z = 3.0 * rdotn * r.z * rmag_inv_5 - n_i.z * rmag_inv_3;

                    if (coordinate_flag == cylindrical_str) {
                        auto Ax_temp = (G_i.x * cphi0 + G_i.y * sphi0) * pow(-1, stell);
                        auto Ay_temp = (- G_i.x * sphi0 + G_i.y * cphi0);
                        A(i, j, 0) += fak * (Ax_temp * cphi_new + Ay_temp * sphi_new);
                        A(i, j, 1) += fak * (-Ax_temp * sphi_new + Ay_temp * cphi_new);
                        A(i, j, 2) += fak * G_i.z;
                    }
                    else if (coordinate_flag == toroidal_str) {

                        auto Ax_temp = (G_i.x * cphi0 + G_i.y * sphi0) * pow(-1, stell);
                        auto Ay_temp = (- G_i.x * sphi0 + G_i.y * cphi0);
                        auto Az_temp = G_i.z;
                        A(i, j, 0) += fak * (Ax_temp * cphi_new * ctheta_new + Ay_temp * sphi_new * ctheta_new + Az_temp * stheta_new);
                        A(i, j, 1) += fak * (-Ax_temp * sphi_new + Ay_temp * cphi_new);
                        A(i, j, 2) += fak * (-Ax_temp * cphi_new * stheta_new - Ay_temp * sphi_new * stheta_new + Az_temp * ctheta_new);
                    }
                    else {
                        // rotate by -phi0 and then flip x component
                        // This should be the reverse of what is done to the m vector and the dipole grid
                        // because A * m = A * R^T * R * m and R is an orthogonal matrix both
                        // for a reflection and a rotation.
                        A(i, j, 0) += fak * (G_i.x * cphi0 + G_i.y * sphi0) * pow(-1, stell);
                        A(i, j, 1) += fak * (-G_i.x * sphi0 + G_i.y * cphi0);
                        A(i, j, 2) += fak * G_i.z;
                    }
                }
            }
        }
    }
    return A;
}

#endif

// Takes a uniform CARTESIAN grid of dipoles, and loops through
// and creates a final set of points which lie between the
// inner and outer toroidal surfaces defined by extending the plasma
// boundary by its normal vectors * some minimum distance.
Array define_a_uniform_cartesian_grid_between_two_toroidal_surfaces(Array& normal_inner, Array& normal_outer, Array& xyz_uniform, Array& xyz_inner, Array& xyz_outer)
{
    // For each toroidal cross-section:
    // For each dipole location:
    //     1. Find nearest point from dipole to the inner surface
    //     2. Find nearest point from dipole to the outer surface
    //     3. Select nearest point that is closest to the dipole
    //     4. Get normal vector of this inner/outer surface point
    //     5. Draw ray from dipole location in the direction of this normal vector
    //     6. If closest point between inner surface and the ray is the
    //           start of the ray, conclude point is outside the inner surface.
    //     7. If closest point between outer surface and the ray is the
    //           start of the ray, conclude point is outside the outer surface.
    //     8. If Step 4 was True but Step 5 was False, add the point to the final grid.

    if(normal_inner.layout() != xt::layout_type::row_major)
          throw std::runtime_error("normal_inner needs to be in row-major storage order");
    if(normal_outer.layout() != xt::layout_type::row_major)
          throw std::runtime_error("normal_outer needs to be in row-major storage order");
    if(xyz_uniform.layout() != xt::layout_type::row_major)
          throw std::runtime_error("xyz_uniform needs to be in row-major storage order");
    if(xyz_inner.layout() != xt::layout_type::row_major)
          throw std::runtime_error("xyz_inner needs to be in row-major storage order");
    if(xyz_outer.layout() != xt::layout_type::row_major)
          throw std::runtime_error("xyz_outer needs to be in row-major storage order");

    int num_inner = xyz_inner.shape(0);
    int num_outer = xyz_outer.shape(0);
    int ngrid = xyz_uniform.shape(0);
    int num_ray = 2000;
    Array final_grid = xt::zeros<double>({ngrid, 3});

    // Loop through every dipole
#pragma omp parallel for schedule(static)
    for (int i = 0; i < ngrid; i++) {
        double X = xyz_uniform(i, 0);
        double Y = xyz_uniform(i, 1);
        double Z = xyz_uniform(i, 2);

        // find nearest point on inner/outer toroidal surface
        double min_dist_inner = 1e5;
        double min_dist_outer = 1e5;
        int inner_loc = 0;
        int outer_loc = 0;
        for (int k = 0; k < num_inner; k++) {
            double x_inner = xyz_inner(k, 0);
            double y_inner = xyz_inner(k, 1);
            double z_inner = xyz_inner(k, 2);
            double dist_inner = (x_inner - X) * (x_inner - X) + (y_inner - Y) * (y_inner - Y) + (z_inner - Z) * (z_inner - Z);
            if (dist_inner < min_dist_inner) {
                min_dist_inner = dist_inner;
                inner_loc = k;
            }
	    }
        for (int k = 0; k < num_outer; k++) {
            double x_outer = xyz_outer(k, 0);
            double y_outer = xyz_outer(k, 1);
            double z_outer = xyz_outer(k, 2);
            double dist_outer = (x_outer - X) * (x_outer - X) + (y_outer - Y) * (y_outer - Y) + (z_outer - Z) * (z_outer - Z);
            if (dist_outer < min_dist_outer) {
                min_dist_outer = dist_outer;
                outer_loc = k;
	        }
	    }
        double nx = 0.0;
        double ny = 0.0;
        double nz = 0.0;
        if (min_dist_inner < min_dist_outer) {
            nx = normal_inner(inner_loc, 0);
            ny = normal_inner(inner_loc, 1);
            nz = normal_inner(inner_loc, 2);
        }
        else {
            nx = normal_outer(outer_loc, 0);
            ny = normal_outer(outer_loc, 1);
	        nz = normal_outer(outer_loc, 2);
	    }
        // normalize the normal vectors
        double norm_vec = sqrt(nx * nx + ny * ny + nz * nz);
        double ray_x = nx / norm_vec;
        double ray_y = ny / norm_vec;
        double ray_z = nz / norm_vec;

        // Compute all the rays and find the location of minimum ray-surface distance
        double dist_inner_ray = 0.0;
        double dist_outer_ray = 0.0;
        double min_dist_inner_ray = 1e5;
        double min_dist_outer_ray = 1e5;
        int nearest_loc_inner = 0;
        int nearest_loc_outer = 0;
        double ray_equation_x = 0.0;
        double ray_equation_y = 0.0;
        double ray_equation_z = 0.0;
        for (int k = 0; k < num_ray; k++) {
            ray_equation_x = X + ray_x * (4.0 / ((double) num_ray)) * k;
            ray_equation_y = Y + ray_y * (4.0 / ((double) num_ray)) * k;
            ray_equation_z = Z + ray_z * (4.0 / ((double) num_ray)) * k;
            dist_inner_ray = (xyz_inner(inner_loc, 0) - ray_equation_x) * (xyz_inner(inner_loc, 0) - ray_equation_x) + (xyz_inner(inner_loc, 1) - ray_equation_y) * (xyz_inner(inner_loc, 1) - ray_equation_y) + (xyz_inner(inner_loc, 2) - ray_equation_z) * (xyz_inner(inner_loc, 2) - ray_equation_z);
            dist_outer_ray = (xyz_outer(outer_loc, 0) - ray_equation_x) * (xyz_outer(outer_loc, 0) - ray_equation_x) + (xyz_outer(outer_loc, 1) - ray_equation_y) * (xyz_outer(outer_loc, 1) - ray_equation_y) + (xyz_outer(outer_loc, 2) - ray_equation_z) * (xyz_outer(outer_loc, 2) - ray_equation_z);
            if (dist_inner_ray < min_dist_inner_ray) {
                min_dist_inner_ray = dist_inner_ray;
                nearest_loc_inner = k;
            }
            if (dist_outer_ray < min_dist_outer_ray) {
                min_dist_outer_ray = dist_outer_ray;
                nearest_loc_outer = k;
            }
	    }

        // nearest distance from the inner surface to the ray should be just the original point
        if (nearest_loc_inner > 0) continue;

        // nearest distance from the outer surface to the ray should NOT be the original point
        if (nearest_loc_outer > 0) {
            final_grid(i, 0) = X;
            final_grid(i, 1) = Y;
            final_grid(i, 2) = Z;
        }
    }
    return final_grid;
}