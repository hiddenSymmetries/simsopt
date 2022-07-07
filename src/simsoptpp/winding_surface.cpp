#include "winding_surface.h"
#include "simdhelpers.h"
#include "vec3dsimd.h"
#include <Eigen/Dense>

// Calculate the B field at a set of evaluation points on a winding surface
// points: where to evaluate the field
// ws_points: evaluation points on the winding surface
// ws_normal: normal vectors for the evaluation points on the winding surface
// K: surface current vectors for the evaluation points on the winding surface
// everything in xyz coordinates
Array WindingSurfaceB(Array& points, Array& ws_points, Array& ws_normal, Array& K)
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(ws_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("winding surface points needs to be in row-major storage order");
    if(ws_normal.layout() != xt::layout_type::row_major)
          throw std::runtime_error("winding surface normal vector needs to be in row-major storage order");
    if(K.layout() != xt::layout_type::row_major)
          throw std::runtime_error("surface_current needs to be in row-major storage order");

    int num_points = points.shape(0);
    int num_ws_points = ws_points.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    Array B        = xt::zeros<double>({points.shape(0), points.shape(1)});

    // initialize pointer to the beginning of ws_points
    double* ws_points_ptr = &(ws_points(0, 0));
    double* ws_normal_ptr = &(ws_normal(0, 0));
    double* K_ptr = &(K(0, 0));
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
        // Sum contributions from all the winding surface points
        // i.e. do the surface integral over the winding surface
        for (int j = 0; j < num_ws_points; ++j) {
            Vec3dSimd r_j = Vec3dSimd(ws_points_ptr[3 * j + 0], ws_points_ptr[3 * j + 1], ws_points_ptr[3 * j + 2]);
            Vec3dSimd n_j = Vec3dSimd(ws_normal_ptr[3 * j + 0], ws_normal_ptr[3 * j + 1], ws_normal_ptr[3 * j + 2]);
            Vec3dSimd K_j = Vec3dSimd(K_ptr[3 * j + 0], K_ptr[3 * j + 1], K_ptr[3 * j + 2]);
            Vec3dSimd r = point_i - r_j;
            simd_t rmag_2     = normsq(r);
            simd_t rmag_inv   = rsqrt(rmag_2);
            simd_t rmag_inv_3 = rmag_inv * (rmag_inv * rmag_inv);
            simd_t nmag = sqrt(normsq(n_j));
            Vec3dSimd Kcrossr = cross(K_j, r);
            B_i.x += nmag * Kcrossr.x * rmag_inv_3;
            B_i.y += nmag * Kcrossr.y * rmag_inv_3;
            B_i.z += nmag * Kcrossr.z * rmag_inv_3;
        }
        for(int k = 0; k < klimit; k++){
            B(i + k, 0) = fak * B_i.x[k];
            B(i + k, 1) = fak * B_i.y[k];
            B(i + k, 2) = fak * B_i.z[k];
        }
    }
    return B;
}

Array WindingSurfacedB(Array& points, Array& ws_points, Array& ws_normal, Array& K)
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(ws_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("winding surface points needs to be in row-major storage order");
    if(ws_normal.layout() != xt::layout_type::row_major)
          throw std::runtime_error("winding surface normal vector needs to be in row-major storage order");
    if(K.layout() != xt::layout_type::row_major)
          throw std::runtime_error("surface_current needs to be in row-major storage order");

    int num_points = points.shape(0);
    int num_ws_points = ws_points.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    Array dB        = xt::zeros<double>({points.shape(0), points.shape(1), points.shape(1)});

    // initialize pointer to the beginning of ws_points
    double* ws_points_ptr = &(ws_points(0, 0));
    double* ws_normal_ptr = &(ws_normal(0, 0));
    double* K_ptr = &(K(0, 0));
    double fak = 1e-7;  // mu0 divided by 4 * pi factor

    // Loop through the evaluation points by chunks of simd_size
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_points; i += simd_size) {
        auto point_i = Vec3dSimd();
        auto dB_i1 = Vec3dSimd();
        auto dB_i2 = Vec3dSimd();
        auto dB_i3 = Vec3dSimd();

        // check that i + k isn't bigger than num_points
        int klimit = std::min(simd_size, num_points - i);
        for(int k = 0; k < klimit; k++){
            for (int d = 0; d < 3; ++d) {
                point_i[d][k] = points(i + k, d);
            }
        }
        // Sum contributions from all the winding surface points
        // i.e. do the surface integral over the winding surface
        for (int j = 0; j < num_ws_points; ++j) {
            Vec3dSimd r_j = Vec3dSimd(ws_points_ptr[3 * j + 0], ws_points_ptr[3 * j + 1], ws_points_ptr[3 * j + 2]);
            Vec3dSimd n_j = Vec3dSimd(ws_normal_ptr[3 * j + 0], ws_normal_ptr[3 * j + 1], ws_normal_ptr[3 * j + 2]);
            Vec3dSimd K_j = Vec3dSimd(K_ptr[3 * j + 0], K_ptr[3 * j + 1], K_ptr[3 * j + 2]);
            Vec3dSimd r = point_i - r_j;
            simd_t rmag_2     = normsq(r);
            simd_t rmag_inv   = rsqrt(rmag_2);
            simd_t rmag_inv_3 = rmag_inv * (rmag_inv * rmag_inv);
            simd_t rmag_inv_5 = rmag_inv_3 * rmag_inv * rmag_inv;
            simd_t nmag = sqrt(normsq(n_j));
            Vec3dSimd Kcrossr = cross(K_j, r);
            Vec3dSimd ex = Vec3dSimd(1, 0, 0);
            Vec3dSimd ey = Vec3dSimd(0, 1, 0);
            Vec3dSimd ez = Vec3dSimd(0, 1, 0);
            Vec3dSimd Kcrossex = cross(K_j, ex);
            Vec3dSimd Kcrossey = cross(K_j, ey);
            Vec3dSimd Kcrossez = cross(K_j, ez);
            dB_i1.x += nmag * (Kcrossex.x * rmag_inv_3 - 3.0 * Kcrossr.x * rmag_inv_5 * r.x);
            dB_i1.y += nmag * (Kcrossex.y * rmag_inv_3 - 3.0 * Kcrossr.y * rmag_inv_5 * r.x);
            dB_i1.z += nmag * (Kcrossex.z * rmag_inv_3 - 3.0 * Kcrossr.z * rmag_inv_5 * r.x);
            dB_i2.x += nmag * (Kcrossey.x * rmag_inv_3 - 3.0 * Kcrossr.x * rmag_inv_5 * r.y);
            dB_i2.y += nmag * (Kcrossey.y * rmag_inv_3 - 3.0 * Kcrossr.y * rmag_inv_5 * r.y);
            dB_i2.z += nmag * (Kcrossey.z * rmag_inv_3 - 3.0 * Kcrossr.z * rmag_inv_5 * r.y);
            dB_i3.x += nmag * (Kcrossez.x * rmag_inv_3 - 3.0 * Kcrossr.x * rmag_inv_5 * r.z);
            dB_i3.y += nmag * (Kcrossez.y * rmag_inv_3 - 3.0 * Kcrossr.y * rmag_inv_5 * r.z);
            dB_i3.z += nmag * (Kcrossez.z * rmag_inv_3 - 3.0 * Kcrossr.z * rmag_inv_5 * r.z);
        }
        for(int k = 0; k < klimit; k++){
            dB(i + k, 0, 0) = fak * dB_i1.x[k];
            dB(i + k, 0, 1) = fak * dB_i1.y[k];
            dB(i + k, 0, 2) = fak * dB_i1.z[k];
            dB(i + k, 1, 0) = fak * dB_i2.x[k];
            dB(i + k, 1, 1) = fak * dB_i2.y[k];
            dB(i + k, 1, 2) = fak * dB_i2.z[k];
            dB(i + k, 2, 0) = fak * dB_i3.x[k];
            dB(i + k, 2, 1) = fak * dB_i3.y[k];
            dB(i + k, 2, 2) = fak * dB_i3.z[k];
	      }
    }
    return dB;
}

Array WindingSurfaceA(Array& points, Array& ws_points, Array& ws_normal, Array& K)
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(ws_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("winding surface points needs to be in row-major storage order");
    if(ws_normal.layout() != xt::layout_type::row_major)
          throw std::runtime_error("winding surface normal vector needs to be in row-major storage order");
    if(K.layout() != xt::layout_type::row_major)
          throw std::runtime_error("surface_current needs to be in row-major storage order");

    int num_points = points.shape(0);
    int num_ws_points = ws_points.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    Array A        = xt::zeros<double>({points.shape(0), points.shape(1)});

    // initialize pointer to the beginning of ws_points
    double* ws_points_ptr = &(ws_points(0, 0));
    double* ws_normal_ptr = &(ws_normal(0, 0));
    double* K_ptr = &(K(0, 0));
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
        // Sum contributions from all the winding surface points
        // i.e. do the surface integral over the winding surface
        for (int j = 0; j < num_ws_points; ++j) {
            Vec3dSimd r_j = Vec3dSimd(ws_points_ptr[3 * j + 0], ws_points_ptr[3 * j + 1], ws_points_ptr[3 * j + 2]);
            Vec3dSimd n_j = Vec3dSimd(ws_normal_ptr[3 * j + 0], ws_normal_ptr[3 * j + 1], ws_normal_ptr[3 * j + 2]);
            Vec3dSimd K_j = Vec3dSimd(K_ptr[3 * j + 0], K_ptr[3 * j + 1], K_ptr[3 * j + 2]);
            Vec3dSimd r = point_i - r_j;
            simd_t rmag_2     = normsq(r);
            simd_t rmag_inv   = rsqrt(rmag_2);
            simd_t nmag = sqrt(normsq(n_j));
            Vec3dSimd Kcrossr = cross(K_j, r);
            A_i.x += nmag * K_j.x * rmag_inv;
            A_i.y += nmag * K_j.y * rmag_inv;
            A_i.z += nmag * K_j.z * rmag_inv;
        }
        for(int k = 0; k < klimit; k++){
            A(i + k, 0) = fak * A_i.x[k];
            A(i + k, 1) = fak * A_i.y[k];
            A(i + k, 2) = fak * A_i.z[k];
        }
    }
    return A;
}

Array WindingSurfacedA(Array& points, Array& ws_points, Array& ws_normal, Array& K)
{
    // warning: row_major checks below do NOT throw an error correctly on a compute node on Cori
    if(points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("points needs to be in row-major storage order");
    if(ws_points.layout() != xt::layout_type::row_major)
          throw std::runtime_error("winding surface points needs to be in row-major storage order");
    if(ws_normal.layout() != xt::layout_type::row_major)
          throw std::runtime_error("winding surface normal vector needs to be in row-major storage order");
    if(K.layout() != xt::layout_type::row_major)
          throw std::runtime_error("surface_current needs to be in row-major storage order");

    int num_points = points.shape(0);
    int num_ws_points = ws_points.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    Array dA        = xt::zeros<double>({points.shape(0), points.shape(1), points.shape(1)});

    // initialize pointer to the beginning of ws_points
    double* ws_points_ptr = &(ws_points(0, 0));
    double* ws_normal_ptr = &(ws_normal(0, 0));
    double* K_ptr = &(K(0, 0));
    double fak = 1e-7;  // mu0 divided by 4 * pi factor

    // Loop through the evaluation points by chunks of simd_size
    #pragma omp parallel for schedule(static)
    for(int i = 0; i < num_points; i += simd_size) {
        auto point_i = Vec3dSimd();
        auto dA_i1   = Vec3dSimd();
        auto dA_i2   = Vec3dSimd();
        auto dA_i3   = Vec3dSimd();

        // check that i + k isn't bigger than num_points
        int klimit = std::min(simd_size, num_points - i);
        for(int k = 0; k < klimit; k++){
            for (int d = 0; d < 3; ++d) {
                point_i[d][k] = points(i + k, d);
            }
        }
        // Sum contributions from all the winding surface points
        // i.e. do the surface integral over the winding surface
        for (int j = 0; j < num_ws_points; ++j) {
            Vec3dSimd r_j = Vec3dSimd(ws_points_ptr[3 * j + 0], ws_points_ptr[3 * j + 1], ws_points_ptr[3 * j + 2]);
            Vec3dSimd n_j = Vec3dSimd(ws_normal_ptr[3 * j + 0], ws_normal_ptr[3 * j + 1], ws_normal_ptr[3 * j + 2]);
            Vec3dSimd K_j = Vec3dSimd(K_ptr[3 * j + 0], K_ptr[3 * j + 1], K_ptr[3 * j + 2]);
            Vec3dSimd r = point_i - r_j;
            simd_t rmag_2     = normsq(r);
            simd_t rmag_inv   = rsqrt(rmag_2);
            simd_t rmag_inv_3 = rmag_inv * (rmag_inv * rmag_inv);
            simd_t nmag = sqrt(normsq(n_j));
            dA_i1.x += - nmag * K_j.x * r.x * rmag_inv_3;
            dA_i1.y += - nmag * K_j.y * r.x * rmag_inv_3;
            dA_i1.z += - nmag * K_j.z * r.x * rmag_inv_3;
            dA_i2.x += - nmag * K_j.x * r.y * rmag_inv_3;
            dA_i2.y += - nmag * K_j.y * r.y * rmag_inv_3;
            dA_i2.z += - nmag * K_j.z * r.y * rmag_inv_3;
            dA_i3.x += - nmag * K_j.x * r.z * rmag_inv_3;
            dA_i3.y += - nmag * K_j.y * r.z * rmag_inv_3;
            dA_i3.z += - nmag * K_j.z * r.z * rmag_inv_3;
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
