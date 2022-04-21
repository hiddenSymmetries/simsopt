#include "dipole_field.h"
#include "simdhelpers.h"
#include "vec3dsimd.h"

// Calculate the B field at a set of evaluation points from N dipoles
// points: where to evaluate the field
// m_points: where the dipoles are located
// m: dipole moments ('orientation')
// everything in xyz coordinates
Array dipole_field_B(Array& points, Array& m_points, Array& m) {
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
    double* m_points_ptr = &(m_points(0, 0));
    double* m_ptr = &(m(0, 0));
    double fak = 1e-7;
#pragma omp parallel for schedule(static)
    for(int i = 0; i < num_points; i += simd_size) {
        auto point_i = Vec3dSimd();
        auto B_i   = Vec3dSimd();
        // check that i+k isn't bigger than num_points
        int klimit = std::min(simd_size, num_points-i);
        for(int k=0; k<klimit; k++){
            for (int d = 0; d < 3; ++d) {
                point_i[d][k] = points(i+k, d);
            }
        }
        for (int j = 0; j < num_dipoles; ++j) {
            Vec3dSimd m_j = Vec3dSimd(m_ptr[3*j+0], m_ptr[3*j+1], m_ptr[3*j+2]);
            Vec3dSimd mp_j = Vec3dSimd(m_points_ptr[3*j+0], m_points_ptr[3*j+1], m_points_ptr[3*j+2]);
            Vec3dSimd r = point_i - mp_j;
            simd_t rmag_2     = normsq(r);
            simd_t rmag_inv   = rsqrt(rmag_2);
            simd_t rmag_inv_3 = rmag_inv*(rmag_inv*rmag_inv);
            simd_t rmag_inv_5 = rmag_inv_3*(rmag_inv*rmag_inv);
            simd_t rdotm = inner(r, m_j);
            B_i.x += 3.0 * rdotm * r.x * rmag_inv_5 - m_j.x * rmag_inv_3;
            B_i.y += 3.0 * rdotm * r.y * rmag_inv_5 - m_j.y * rmag_inv_3;
            B_i.z += 3.0 * rdotm * r.z * rmag_inv_5 - m_j.z * rmag_inv_3;
        } 
        for(int k=0; k<klimit; k++){
            B(i+k, 0) = fak * B_i.x[k];
            B(i+k, 1) = fak * B_i.y[k];
            B(i+k, 2) = fak * B_i.z[k];
        }
    }
    return B;
}

// Calculate the gradient of the B field at a set of evaluation points from N dipoles
Array dipole_field_dB(Array& points, Array& m_points, Array& m) {
    int num_points = points.shape(0);
    int num_dipoles = m_points.shape(0);
    Array dB = xt::zeros<double>({points.shape(0), points.shape(1), points.shape(1)});
    double x, y, z, mx, my, mz, mpx, mpy, mpz, rx, ry, rz, rmag, rdotm, r5;
    //#pragma omp parallel for private(x, y, z, mx, my, mz, mpx, mpy, mpz, rx, ry, rz, rmag, rdotm, r5)
    for (int i = 0; i < num_points; ++i) {
        x = points(i, 0);
        y = points(i, 1);
        z = points(i, 2);
        for (int j = 0; j < num_dipoles; ++j) {
            mpx = m_points(j, 0);
            mpy = m_points(j, 1);
            mpz = m_points(j, 2);
            mx = m(j, 0);
            my = m(j, 1);
            mz = m(j, 2);
            rx = x - mpx;
            ry = y - mpy;
            rz = z - mpz;
            rmag = sqrt(rx * rx + ry * ry + rz * rz);
            rdotm = rx * mx + ry * my + rz * mz;
            r5 = 3.0 / pow(rmag, 5);
            dB(i, 0, 0) += r5 * ((2 * mx * rx + rdotm) - 5 * rdotm / pow(rmag, 2) * rx * rx); 
            dB(i, 0, 1) += r5 * ((mx * ry + my * rx) - 5 * rdotm / pow(rmag, 2) * rx * ry);
            dB(i, 0, 2) += r5 * ((mx * rz + mz * rx) - 5 * rdotm / pow(rmag, 2) * rx * rz);
            dB(i, 1, 1) += r5 * ((2 * my * ry + rdotm) - 5 * rdotm / pow(rmag, 2) * ry * ry); 
            dB(i, 1, 2) += r5 * ((my * rz + mz * ry) - 5 * rdotm / pow(rmag, 2) * ry * rz);
            dB(i, 2, 2) += r5 * ((2 * mz * rz + rdotm) - 5 * rdotm / pow(rmag, 2) * rz * rz); 
        }
        dB(i, 2, 1) = dB(i, 1, 2);
        dB(i, 2, 0) = dB(i, 0, 2);
        dB(i, 1, 0) = dB(i, 0, 1);
    }
    return dB * 1e-7;
}
