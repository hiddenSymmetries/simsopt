#include "winding_surface.h"
#include <math.h>

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
            simd_t nmag = norm(n_j);
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
    int num_points = points.shape(0);
    int num_ws_points = ws_points.shape(0);
    Array dB       = xt::zeros<double>({points.shape(0), points.shape(1), points.shape(1)});
    // compute BiotSavart dB from surface current K
    return dB;
}

Array WindingSurfaceA(Array& points, Array& ws_points, Array& ws_normal, Array& K)
{
    int num_points = points.shape(0);
    int num_ws_points = ws_points.shape(0);
    Array A       = xt::zeros<double>({points.shape(0), points.shape(1)});
    // compute BiotSavart A from surface current K
    return A;
}

Array WindingSurfacedA(Array& points, Array& ws_points, Array& ws_normal, Array& K)
{
    int num_points = points.shape(0);
    int num_ws_points = ws_points.shape(0);
    Array dA       = xt::zeros<double>({points.shape(0), points.shape(1), points.shape(1)});
    // compute BiotSavart dA from surface current K
    return dA;
}
