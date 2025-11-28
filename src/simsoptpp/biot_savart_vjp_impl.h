#pragma once

#include "simdhelpers.h"
#include "vec3dsimd.h"
#include <stdexcept>
#include <vector>
#include "xtensor/xlayout.hpp"

// When compiled with C++17, then we use `if constexpr` to check for
// derivatives that need to be computed.  These are actually evaluated at
// compile time, e.g. the compiler creates two different functions, one that
// only computes the vjp for B, and one that computes the vjp for B and ∇B.
//
#if __cplusplus >= 201703L
#  define MYIF(c) if constexpr(c)
#else
#  define MYIF(c) if(c)
#endif

// ============================================================================
//  BIOT-SAVART VJP (B and ∇B adjoint, plus optional Hessian / grad-grad-B)
// ============================================================================

#if defined(USE_XSIMD)

// ------------------------- XSIMD / Vec3dSimd branch -------------------------

template<class T, int derivs>
void biot_savart_vjp_kernel(
    AlignedPaddedVec &pointsx,
    AlignedPaddedVec &pointsy,
    AlignedPaddedVec &pointsz,
    T &gamma,
    T &dgamma_by_dphi,
    T &v,
    T &res_gamma,
    T &res_dgamma_by_dphi,
    T &vgrad,
    T &res_grad_gamma,
    T &res_grad_dgamma_by_dphi)
{
    if (gamma.layout() != xt::layout_type::row_major)
        throw std::runtime_error("gamma needs to be in row-major storage order");
    if (dgamma_by_dphi.layout() != xt::layout_type::row_major)
        throw std::runtime_error("dgamma_by_dphi needs to be in row-major storage order");
    if (res_gamma.layout() != xt::layout_type::row_major)
        throw std::runtime_error("res_gamma needs to be in row-major storage order");
    if (res_dgamma_by_dphi.layout() != xt::layout_type::row_major)
        throw std::runtime_error("res_dgamma_by_dphi needs to be in row-major storage order");
    if (res_grad_gamma.layout() != xt::layout_type::row_major)
        throw std::runtime_error("res_grad_gamma needs to be in row-major storage order");
    if (res_grad_dgamma_by_dphi.layout() != xt::layout_type::row_major)
        throw std::runtime_error("res_grad_dgamma_by_dphi needs to be in row-major storage order");

    const int num_points      = static_cast<int>(pointsx.size());
    const int num_quad_points = static_cast<int>(gamma.shape(0));
    constexpr int simd_size   = xsimd::simd_type<double>::size;

    double *gamma_j_ptr                 = &(gamma(0, 0));
    double *dgamma_j_by_dphi_ptr        = &(dgamma_by_dphi(0, 0));
    double *res_dgamma_by_dphi_ptr      = &(res_dgamma_by_dphi(0, 0));
    double *res_gamma_ptr               = &(res_gamma(0, 0));
    double *res_grad_dgamma_by_dphi_ptr = &(res_grad_dgamma_by_dphi(0, 0));
    double *res_grad_gamma_ptr          = &(res_grad_gamma(0, 0));

    // SIMD bulk
    for (int i = 0; i < num_points - num_points % simd_size; i += simd_size)
    {
        Vec3dSimd point_i(&(pointsx[i]), &(pointsy[i]), &(pointsz[i]));

        Vec3dSimd v_i;
        std::vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, XSIMD_DEFAULT_ALIGNMENT>> vgrad_i{
            Vec3dSimd(), Vec3dSimd(), Vec3dSimd()
        };

#pragma unroll
        for (int k = 0; k < simd_size; ++k)
        {
            for (int d = 0; d < 3; ++d)
            {
                v_i[d][k] = v(i + k, d);
                MYIF(derivs > 0)
                {
#pragma unroll
                    for (int dd = 0; dd < 3; ++dd)
                    {
                        vgrad_i[dd][d][k] = vgrad(i + k, dd, d);
                    }
                }
            }
        }

        for (int j = 0; j < num_quad_points; ++j)
        {
            Vec3d dgamma_j_by_dphi_vec{
                dgamma_j_by_dphi_ptr[3 * j + 0],
                dgamma_j_by_dphi_ptr[3 * j + 1],
                dgamma_j_by_dphi_ptr[3 * j + 2]
            };

            Vec3dSimd diff = point_i - Vec3dSimd(
                gamma_j_ptr[3 * j + 0],
                gamma_j_ptr[3 * j + 1],
                gamma_j_ptr[3 * j + 2]);

            simd_t norm_diff_2          = normsq(diff);
            simd_t norm_diff_inv        = rsqrt(norm_diff_2);
            simd_t norm_diff_2_inv      = norm_diff_inv * norm_diff_inv;
            simd_t norm_diff_3_inv      = norm_diff_2_inv * norm_diff_inv;
            simd_t norm_diff_5_inv      = norm_diff_3_inv * norm_diff_2_inv;
            simd_t norm_diff_5_inv_times_3 = simd_t(3.0) * norm_diff_5_inv;

            // ----- vjp wrt dgamma_by_dphi (B) -----
            Vec3dSimd res_dgamma_by_dphi_add = cross(diff, v_i) * norm_diff_3_inv;
            res_dgamma_by_dphi_ptr[3 * j + 0] += xsimd::hadd(res_dgamma_by_dphi_add.x);
            res_dgamma_by_dphi_ptr[3 * j + 1] += xsimd::hadd(res_dgamma_by_dphi_add.y);
            res_dgamma_by_dphi_ptr[3 * j + 2] += xsimd::hadd(res_dgamma_by_dphi_add.z);

            // ----- vjp wrt gamma (∇B) -----
            Vec3dSimd cross_dgamma_j_by_dphi_diff = cross(dgamma_j_by_dphi_vec, diff);
            Vec3dSimd res_gamma_add = cross(dgamma_j_by_dphi_vec, v_i) * norm_diff_3_inv;
            res_gamma_add += diff * inner(cross_dgamma_j_by_dphi_diff, v_i) * norm_diff_5_inv_times_3;

            res_gamma_ptr[3 * j + 0] += xsimd::hadd(res_gamma_add.x);
            res_gamma_ptr[3 * j + 1] += xsimd::hadd(res_gamma_add.y);
            res_gamma_ptr[3 * j + 2] += xsimd::hadd(res_gamma_add.z);

            // ----- ∇B derivatives + optional Hessian -----
            MYIF(derivs > 0)
            {
                simd_t norm_diff_7_inv = norm_diff_5_inv * norm_diff_2_inv;

                Vec3dSimd res_grad_dgamma_by_dphi_add; // starts at zero
                Vec3dSimd res_grad_gamma_add;          // starts at zero

#pragma unroll
                for (int k = 0; k < 3; ++k)
                {
                    Vec3dSimd eksimd;
                    eksimd[k] += simd_t(1.0);

                    Vec3d ek = Vec3d::Zero();
                    ek[k]    = 1.0;

                    res_grad_dgamma_by_dphi_add += cross(k, vgrad_i[k]) * norm_diff_3_inv;
                    res_grad_dgamma_by_dphi_add -= cross(diff, vgrad_i[k]) * (diff[k] * norm_diff_5_inv_times_3);

                    res_grad_gamma_add += diff * (inner(cross(dgamma_j_by_dphi_vec, ek), vgrad_i[k]) * norm_diff_5_inv_times_3);
                    res_grad_gamma_add += eksimd * (inner(cross_dgamma_j_by_dphi_diff, vgrad_i[k]) * norm_diff_5_inv_times_3);
                    res_grad_gamma_add += cross(vgrad_i[k], dgamma_j_by_dphi_vec) * (norm_diff_5_inv_times_3 * diff[k]);
                    res_grad_gamma_add -= diff * (simd_t(15.0) * diff[k] *
                                                  inner(cross_dgamma_j_by_dphi_diff, vgrad_i[k]) * norm_diff_7_inv);

                    // MYIF(derivs > 1) { ... }  // place for second-derivative-of-∇B if needed
                }

                res_grad_dgamma_by_dphi_ptr[3 * j + 0] += xsimd::hadd(res_grad_dgamma_by_dphi_add.x);
                res_grad_dgamma_by_dphi_ptr[3 * j + 1] += xsimd::hadd(res_grad_dgamma_by_dphi_add.y);
                res_grad_dgamma_by_dphi_ptr[3 * j + 2] += xsimd::hadd(res_grad_dgamma_by_dphi_add.z);

                res_grad_gamma_ptr[3 * j + 0] += xsimd::hadd(res_grad_gamma_add.x);
                res_grad_gamma_ptr[3 * j + 1] += xsimd::hadd(res_grad_gamma_add.y);
                res_grad_gamma_ptr[3 * j + 2] += xsimd::hadd(res_grad_gamma_add.z);
            }
        }
    }

    // Scalar tail for remaining points
    for (int i = num_points - num_points % simd_size; i < num_points; ++i)
    {
        Vec3d point_i{pointsx[i], pointsy[i], pointsz[i]};

        Vec3d v_i = Vec3d::Zero();
        std::vector<Vec3d> vgrad_i{
            Vec3d::Zero(), Vec3d::Zero(), Vec3d::Zero()
        };

#pragma unroll
        for (int d = 0; d < 3; ++d)
        {
            v_i[d] = v(i, d);
            MYIF(derivs > 0)
            {
                for (int dd = 0; dd < 3; ++dd)
                {
                    vgrad_i[dd][d] = vgrad(i, dd, d);
                }
            }
        }

        for (int j = 0; j < num_quad_points; ++j)
        {
            Vec3d diff{
                point_i[0] - gamma_j_ptr[3 * j + 0],
                point_i[1] - gamma_j_ptr[3 * j + 1],
                point_i[2] - gamma_j_ptr[3 * j + 2]
            };
            Vec3d dgamma_j_by_dphi_vec{
                dgamma_j_by_dphi_ptr[3 * j + 0],
                dgamma_j_by_dphi_ptr[3 * j + 1],
                dgamma_j_by_dphi_ptr[3 * j + 2]
            };

            double norm_diff    = norm(diff);
            double norm_diff_inv = 1.0 / norm_diff;
            double norm_diff_2_inv = norm_diff_inv * norm_diff_inv;
            double norm_diff_3_inv = norm_diff_2_inv * norm_diff_inv;
            double norm_diff_5_inv = norm_diff_3_inv * norm_diff_2_inv;
            double norm_diff_5_inv_times_3 = 3.0 * norm_diff_5_inv;

            // vjp wrt dgamma_by_dphi
            Vec3d res_dgamma_by_dphi_add = cross(diff, v_i) * norm_diff_3_inv;
            res_dgamma_by_dphi(j, 0) += res_dgamma_by_dphi_add[0];
            res_dgamma_by_dphi(j, 1) += res_dgamma_by_dphi_add[1];
            res_dgamma_by_dphi(j, 2) += res_dgamma_by_dphi_add[2];

            // vjp wrt gamma
            Vec3d cross_dgamma_j_by_dphi_diff = cross(dgamma_j_by_dphi_vec, diff);
            Vec3d res_gamma_add = cross(dgamma_j_by_dphi_vec, v_i) * norm_diff_3_inv;
            res_gamma_add += diff * inner(cross_dgamma_j_by_dphi_diff, v_i) * norm_diff_5_inv_times_3;

            res_gamma(j, 0) += res_gamma_add[0];
            res_gamma(j, 1) += res_gamma_add[1];
            res_gamma(j, 2) += res_gamma_add[2];

            MYIF(derivs > 0)
            {
                double norm_diff_7_inv = norm_diff_5_inv * norm_diff_2_inv;
                Vec3d res_grad_dgamma_by_dphi_add = Vec3d::Zero();
                Vec3d res_grad_gamma_add          = Vec3d::Zero();

#pragma unroll
                for (int k = 0; k < 3; ++k)
                {
                    Vec3d ek = Vec3d::Zero();
                    ek[k]    = 1.0;

                    res_grad_dgamma_by_dphi_add += cross(k, vgrad_i[k]) * norm_diff_3_inv;
                    res_grad_dgamma_by_dphi_add -= cross(diff, vgrad_i[k]) * (diff[k] * norm_diff_5_inv_times_3);

                    res_grad_gamma_add += diff * (inner(cross(dgamma_j_by_dphi_vec, ek), vgrad_i[k]) * norm_diff_5_inv_times_3);
                    res_grad_gamma_add += ek * (inner(cross_dgamma_j_by_dphi_diff, vgrad_i[k]) * norm_diff_5_inv_times_3);
                    res_grad_gamma_add += cross(vgrad_i[k], dgamma_j_by_dphi_vec) * (norm_diff_5_inv_times_3 * diff[k]);
                    res_grad_gamma_add -= diff * (15.0 * diff[k] *
                                                  inner(cross_dgamma_j_by_dphi_diff, vgrad_i[k]) * norm_diff_7_inv);

                    // MYIF(derivs > 1) { ... } // place for 2nd order if desired
                }

                res_grad_dgamma_by_dphi(j, 0) += res_grad_dgamma_by_dphi_add[0];
                res_grad_dgamma_by_dphi(j, 1) += res_grad_dgamma_by_dphi_add[1];
                res_grad_dgamma_by_dphi(j, 2) += res_grad_dgamma_by_dphi_add[2];

                res_grad_gamma(j, 0) += res_grad_gamma_add[0];
                res_grad_gamma(j, 1) += res_grad_gamma_add[1];
                res_grad_gamma(j, 2) += res_grad_gamma_add[2];
            }
        }
    }
}

#else  // !USE_XSIMD

// ------------------------- Scalar / Vec3dStd branch -------------------------

template<class T, int derivs>
void biot_savart_vjp_kernel(
    AlignedPaddedVec &pointsx,
    AlignedPaddedVec &pointsy,
    AlignedPaddedVec &pointsz,
    T &gamma,
    T &dgamma_by_dphi,
    T &v,
    T &res_gamma,
    T &res_dgamma_by_dphi,
    T &vgrad,
    T &res_grad_gamma,
    T &res_grad_dgamma_by_dphi)
{
    if (gamma.layout() != xt::layout_type::row_major)
        throw std::runtime_error("gamma needs to be in row-major storage order");
    if (dgamma_by_dphi.layout() != xt::layout_type::row_major)
        throw std::runtime_error("dgamma_by_dphi needs to be in row-major storage order");
    if (res_gamma.layout() != xt::layout_type::row_major)
        throw std::runtime_error("res_gamma needs to be in row-major storage order");
    if (res_dgamma_by_dphi.layout() != xt::layout_type::row_major)
        throw std::runtime_error("res_dgamma_by_dphi needs to be in row-major storage order");
    if (res_grad_gamma.layout() != xt::layout_type::row_major)
        throw std::runtime_error("res_grad_gamma needs to be in row-major storage order");
    if (res_grad_dgamma_by_dphi.layout() != xt::layout_type::row_major)
        throw std::runtime_error("res_grad_dgamma_by_dphi needs to be in row-major storage order");

    const int num_points      = static_cast<int>(pointsx.size());
    const int num_quad_points = static_cast<int>(gamma.shape(0));

    double *gamma_j_ptr                 = &(gamma(0, 0));
    double *dgamma_j_by_dphi_ptr        = &(dgamma_by_dphi(0, 0));
    double *res_dgamma_by_dphi_ptr      = &(res_dgamma_by_dphi(0, 0));
    double *res_gamma_ptr               = &(res_gamma(0, 0));
    double *res_grad_dgamma_by_dphi_ptr = &(res_grad_dgamma_by_dphi(0, 0));
    double *res_grad_gamma_ptr          = &(res_grad_gamma(0, 0));

    for (int i = 0; i < num_points; ++i)
    {
        Vec3dStd point_i(&(pointsx[i]), &(pointsy[i]), &(pointsz[i]));
        Vec3dStd v_i;

        std::vector<Vec3dStd> vgrad_i{
            Vec3dStd(), Vec3dStd(), Vec3dStd()
        };

#pragma unroll
        for (int d = 0; d < 3; ++d)
        {
            v_i[d] = v(i, d);
            MYIF(derivs > 0)
            {
                for (int dd = 0; dd < 3; ++dd)
                {
                    vgrad_i[dd][d] = vgrad(i, dd, d);
                }
            }
        }

        for (int j = 0; j < num_quad_points; ++j)
        {
            Vec3d dgamma_j_by_dphi_vec{
                dgamma_j_by_dphi_ptr[3 * j + 0],
                dgamma_j_by_dphi_ptr[3 * j + 1],
                dgamma_j_by_dphi_ptr[3 * j + 2]
            };

            Vec3dStd diff(
                gamma_j_ptr[3 * j + 0],
                gamma_j_ptr[3 * j + 1],
                gamma_j_ptr[3 * j + 2]);
            diff = point_i - diff;

            double r2                   = normsq(diff);
            double norm_diff_inv        = rsqrt(r2);
            double norm_diff_2_inv      = norm_diff_inv * norm_diff_inv;
            double norm_diff_3_inv      = norm_diff_2_inv * norm_diff_inv;
            double norm_diff_5_inv      = norm_diff_3_inv * norm_diff_2_inv;
            double norm_diff_5_inv_times_3 = 3.0 * norm_diff_5_inv;

            // vjp wrt dgamma_by_dphi
            Vec3dStd res_dgamma_by_dphi_add = cross(diff, v_i) * norm_diff_3_inv;
            res_dgamma_by_dphi_ptr[3 * j + 0] += res_dgamma_by_dphi_add.x;
            res_dgamma_by_dphi_ptr[3 * j + 1] += res_dgamma_by_dphi_add.y;
            res_dgamma_by_dphi_ptr[3 * j + 2] += res_dgamma_by_dphi_add.z;

            // vjp wrt gamma
            Vec3dStd cross_dgamma_j_by_dphi_diff = cross(dgamma_j_by_dphi_vec, diff);
            Vec3dStd res_gamma_add = cross(dgamma_j_by_dphi_vec, v_i) * norm_diff_3_inv;
            res_gamma_add += diff * inner(cross_dgamma_j_by_dphi_diff, v_i) * norm_diff_5_inv_times_3;

            res_gamma_ptr[3 * j + 0] += res_gamma_add.x;
            res_gamma_ptr[3 * j + 1] += res_gamma_add.y;
            res_gamma_ptr[3 * j + 2] += res_gamma_add.z;

            MYIF(derivs > 0)
            {
                double norm_diff_7_inv = norm_diff_5_inv * norm_diff_2_inv;
                Vec3dStd res_grad_dgamma_by_dphi_add;
                Vec3dStd res_grad_gamma_add;

#pragma unroll
                for (int k = 0; k < 3; ++k)
                {
                    Vec3dStd eksimd;
                    eksimd[k] += 1.0;

                    Vec3d ek = Vec3d::Zero();
                    ek[k]    = 1.0;

                    res_grad_dgamma_by_dphi_add += cross(k, vgrad_i[k]) * norm_diff_3_inv;
                    res_grad_dgamma_by_dphi_add -= cross(diff, vgrad_i[k]) * (diff[k] * norm_diff_5_inv_times_3);

                    res_grad_gamma_add += diff * (inner(cross(dgamma_j_by_dphi_vec, ek), vgrad_i[k]) * norm_diff_5_inv_times_3);
                    res_grad_gamma_add += eksimd * (inner(cross_dgamma_j_by_dphi_diff, vgrad_i[k]) * norm_diff_5_inv_times_3);
                    res_grad_gamma_add += cross(vgrad_i[k], dgamma_j_by_dphi_vec) * (norm_diff_5_inv_times_3 * diff[k]);
                    res_grad_gamma_add -= diff * (15.0 * diff[k] *
                                                  inner(cross_dgamma_j_by_dphi_diff, vgrad_i[k]) * norm_diff_7_inv);

                    // MYIF(derivs > 1) { ... }  // extension for 2nd order if you ever want it
                }

                res_grad_dgamma_by_dphi_ptr[3 * j + 0] += res_grad_dgamma_by_dphi_add.x;
                res_grad_dgamma_by_dphi_ptr[3 * j + 1] += res_grad_dgamma_by_dphi_add.y;
                res_grad_dgamma_by_dphi_ptr[3 * j + 2] += res_grad_dgamma_by_dphi_add.z;

                res_grad_gamma_ptr[3 * j + 0] += res_grad_gamma_add.x;
                res_grad_gamma_ptr[3 * j + 1] += res_grad_gamma_add.y;
                res_grad_gamma_ptr[3 * j + 2] += res_grad_gamma_add.z;
            }
        }
    }
}

#endif  // USE_XSIMD

// ============================================================================
//  VECTOR POTENTIAL VJP (A and its derivatives)
// ============================================================================

#if defined(USE_XSIMD)

// ---------------------- XSIMD / Vec3dSimd branch ---------------------------

template<class T, int derivs>
void biot_savart_vector_potential_vjp_kernel(
    AlignedPaddedVec &pointsx,
    AlignedPaddedVec &pointsy,
    AlignedPaddedVec &pointsz,
    T &gamma,
    T &dgamma_by_dphi,
    T &v,
    T &res_gamma,
    T &res_dgamma_by_dphi,
    T &vgrad,
    T &res_grad_gamma,
    T &res_grad_dgamma_by_dphi)
{
    if (gamma.layout() != xt::layout_type::row_major)
        throw std::runtime_error("gamma needs to be in row-major storage order");
    if (dgamma_by_dphi.layout() != xt::layout_type::row_major)
        throw std::runtime_error("dgamma_by_dphi needs to be in row-major storage order");
    if (res_gamma.layout() != xt::layout_type::row_major)
        throw std::runtime_error("res_gamma needs to be in row-major storage order");
    if (res_dgamma_by_dphi.layout() != xt::layout_type::row_major)
        throw std::runtime_error("res_dgamma_by_dphi needs to be in row-major storage order");
    if (res_grad_gamma.layout() != xt::layout_type::row_major)
        throw std::runtime_error("res_grad_gamma needs to be in row-major storage order");
    if (res_grad_dgamma_by_dphi.layout() != xt::layout_type::row_major)
        throw std::runtime_error("res_grad_dgamma_by_dphi needs to be in row-major storage order");

    const int num_points      = static_cast<int>(pointsx.size());
    const int num_quad_points = static_cast<int>(gamma.shape(0));
    constexpr int simd_size   = xsimd::simd_type<double>::size;

    double *gamma_j_ptr                 = &(gamma(0, 0));
    double *dgamma_j_by_dphi_ptr        = &(dgamma_by_dphi(0, 0));
    double *res_dgamma_by_dphi_ptr      = &(res_dgamma_by_dphi(0, 0));
    double *res_gamma_ptr               = &(res_gamma(0, 0));
    double *res_grad_dgamma_by_dphi_ptr = &(res_grad_dgamma_by_dphi(0, 0));
    double *res_grad_gamma_ptr          = &(res_grad_gamma(0, 0));

    // SIMD bulk
    for (int i = 0; i < num_points - num_points % simd_size; i += simd_size)
    {
        Vec3dSimd point_i(&(pointsx[i]), &(pointsy[i]), &(pointsz[i]));

        Vec3dSimd v_i;
        std::vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, XSIMD_DEFAULT_ALIGNMENT>> vgrad_i{
            Vec3dSimd(), Vec3dSimd(), Vec3dSimd()
        };

#pragma unroll
        for (int k = 0; k < simd_size; ++k)
        {
            for (int d = 0; d < 3; ++d)
            {
                v_i[d][k] = v(i + k, d);
                MYIF(derivs > 0)
                {
#pragma unroll
                    for (int dd = 0; dd < 3; ++dd)
                    {
                        vgrad_i[dd][d][k] = vgrad(i + k, dd, d);
                    }
                }
            }
        }

        for (int j = 0; j < num_quad_points; ++j)
        {
            Vec3d dgamma_j_by_dphi_vec{
                dgamma_j_by_dphi_ptr[3 * j + 0],
                dgamma_j_by_dphi_ptr[3 * j + 1],
                dgamma_j_by_dphi_ptr[3 * j + 2]
            };

            Vec3dSimd diff = point_i - Vec3dSimd(
                gamma_j_ptr[3 * j + 0],
                gamma_j_ptr[3 * j + 1],
                gamma_j_ptr[3 * j + 2]);

            simd_t norm_diff_2    = normsq(diff);
            simd_t norm_diff_inv  = rsqrt(norm_diff_2);
            simd_t norm_diff_inv_3 = norm_diff_inv * norm_diff_inv * norm_diff_inv;

            Vec3dSimd res_dgamma_by_dphi_add = v_i * norm_diff_inv;
            res_dgamma_by_dphi_ptr[3 * j + 0] += xsimd::hadd(res_dgamma_by_dphi_add.x);
            res_dgamma_by_dphi_ptr[3 * j + 1] += xsimd::hadd(res_dgamma_by_dphi_add.y);
            res_dgamma_by_dphi_ptr[3 * j + 2] += xsimd::hadd(res_dgamma_by_dphi_add.z);

            simd_t vi_dot_dgamma_dphi_j = inner(v_i, dgamma_j_by_dphi_vec);
            Vec3dSimd res_gamma_add = diff * (vi_dot_dgamma_dphi_j * norm_diff_inv_3);

            res_gamma_ptr[3 * j + 0] += xsimd::hadd(res_gamma_add.x);
            res_gamma_ptr[3 * j + 1] += xsimd::hadd(res_gamma_add.y);
            res_gamma_ptr[3 * j + 2] += xsimd::hadd(res_gamma_add.z);

            MYIF(derivs > 0)
            {
                simd_t norm_diff_inv_5 = norm_diff_inv_3 * norm_diff_inv * norm_diff_inv;

                Vec3dSimd res_grad_dgamma_by_dphi_add;
                Vec3dSimd res_grad_gamma_add;

#pragma unroll
                for (int k = 0; k < 3; ++k)
                {
                    res_grad_dgamma_by_dphi_add -= vgrad_i[k] * norm_diff_inv_3 * diff[k];
                    res_grad_gamma_add -= diff * inner(vgrad_i[k], dgamma_j_by_dphi_vec)
                                                * (simd_t(3.0) * diff[k]) * norm_diff_inv_5;

                    // MYIF(derivs > 1) { ... }  // slot for true second-derivative wrt gamma if needed
                }

                res_grad_gamma_add.x += inner(vgrad_i[0], dgamma_j_by_dphi_vec) * norm_diff_inv_3;
                res_grad_gamma_add.y += inner(vgrad_i[1], dgamma_j_by_dphi_vec) * norm_diff_inv_3;
                res_grad_gamma_add.z += inner(vgrad_i[2], dgamma_j_by_dphi_vec) * norm_diff_inv_3;

                res_grad_dgamma_by_dphi_ptr[3 * j + 0] += xsimd::hadd(res_grad_dgamma_by_dphi_add.x);
                res_grad_dgamma_by_dphi_ptr[3 * j + 1] += xsimd::hadd(res_grad_dgamma_by_dphi_add.y);
                res_grad_dgamma_by_dphi_ptr[3 * j + 2] += xsimd::hadd(res_grad_dgamma_by_dphi_add.z);

                res_grad_gamma_ptr[3 * j + 0] += xsimd::hadd(res_grad_gamma_add.x);
                res_grad_gamma_ptr[3 * j + 1] += xsimd::hadd(res_grad_gamma_add.y);
                res_grad_gamma_ptr[3 * j + 2] += xsimd::hadd(res_grad_gamma_add.z);
            }
        }
    }

    // scalar tail
    for (int i = num_points - num_points % simd_size; i < num_points; ++i)
    {
        Vec3d point_i{pointsx[i], pointsy[i], pointsz[i]};

        Vec3d v_i   = Vec3d::Zero();
        std::vector<Vec3d> vgrad_i{
            Vec3d::Zero(), Vec3d::Zero(), Vec3d::Zero()
        };

#pragma unroll
        for (int d = 0; d < 3; ++d)
        {
            v_i[d] = v(i, d);
            MYIF(derivs > 0)
            {
                for (int dd = 0; dd < 3; ++dd)
                {
                    vgrad_i[dd][d] = vgrad(i, dd, d);
                }
            }
        }

        for (int j = 0; j < num_quad_points; ++j)
        {
            Vec3d diff{
                point_i[0] - gamma_j_ptr[3 * j + 0],
                point_i[1] - gamma_j_ptr[3 * j + 1],
                point_i[2] - gamma_j_ptr[3 * j + 2]
            };
            Vec3d dgamma_j_by_dphi_vec{
                dgamma_j_by_dphi_ptr[3 * j + 0],
                dgamma_j_by_dphi_ptr[3 * j + 1],
                dgamma_j_by_dphi_ptr[3 * j + 2]
            };

            double norm_diff       = norm(diff);
            double norm_diff_inv   = 1.0 / norm_diff;
            double norm_diff_inv_3 = norm_diff_inv * norm_diff_inv * norm_diff_inv;

            Vec3d res_dgamma_by_dphi_add = v_i * norm_diff_inv;
            res_dgamma_by_dphi(j, 0) += res_dgamma_by_dphi_add[0];
            res_dgamma_by_dphi(j, 1) += res_dgamma_by_dphi_add[1];
            res_dgamma_by_dphi(j, 2) += res_dgamma_by_dphi_add[2];

            double vi_dot_dgamma_dphi_j = inner(v_i, dgamma_j_by_dphi_vec);
            Vec3d res_gamma_add         = diff * (vi_dot_dgamma_dphi_j * norm_diff_inv_3);

            res_gamma(j, 0) += res_gamma_add[0];
            res_gamma(j, 1) += res_gamma_add[1];
            res_gamma(j, 2) += res_gamma_add[2];

            MYIF(derivs > 0)
            {
                double norm_diff_inv_5 = norm_diff_inv_3 * norm_diff_inv * norm_diff_inv;
                Vec3d res_grad_dgamma_by_dphi_add{0., 0., 0.};
                Vec3d res_grad_gamma_add{0., 0., 0.};

#pragma unroll
                for (int k = 0; k < 3; ++k)
                {
                    res_grad_dgamma_by_dphi_add -= vgrad_i[k] * norm_diff_inv_3 * diff[k];
                    res_grad_gamma_add -= diff * inner(vgrad_i[k], dgamma_j_by_dphi_vec)
                                                * (3.0 * diff[k]) * norm_diff_inv_5;

                    // MYIF(derivs > 1) { ... }  // slot for 2nd order if desired
                }

                res_grad_gamma_add += Vec3d{
                    inner(vgrad_i[0], dgamma_j_by_dphi_vec),
                    inner(vgrad_i[1], dgamma_j_by_dphi_vec),
                    inner(vgrad_i[2], dgamma_j_by_dphi_vec)
                } * norm_diff_inv_3;

                res_grad_dgamma_by_dphi(j, 0) += res_grad_dgamma_by_dphi_add[0];
                res_grad_dgamma_by_dphi(j, 1) += res_grad_dgamma_by_dphi_add[1];
                res_grad_dgamma_by_dphi(j, 2) += res_grad_dgamma_by_dphi_add[2];

                res_grad_gamma(j, 0) += res_grad_gamma_add[0];
                res_grad_gamma(j, 1) += res_grad_gamma_add[1];
                res_grad_gamma(j, 2) += res_grad_gamma_add[2];
            }
        }
    }
}

#else  // !USE_XSIMD

// ------------------ Scalar / Vec3dStd branch -------------------------------

template<class T, int derivs>
void biot_savart_vector_potential_vjp_kernel(
    AlignedPaddedVec &pointsx,
    AlignedPaddedVec &pointsy,
    AlignedPaddedVec &pointsz,
    T &gamma,
    T &dgamma_by_dphi,
    T &v,
    T &res_gamma,
    T &res_dgamma_by_dphi,
    T &vgrad,
    T &res_grad_gamma,
    T &res_grad_dgamma_by_dphi)
{
    if (gamma.layout() != xt::layout_type::row_major)
        throw std::runtime_error("gamma needs to be in row-major storage order");
    if (dgamma_by_dphi.layout() != xt::layout_type::row_major)
        throw std::runtime_error("dgamma_by_dphi needs to be in row-major storage order");
    if (res_gamma.layout() != xt::layout_type::row_major)
        throw std::runtime_error("res_gamma needs to be in row-major storage order");
    if (res_dgamma_by_dphi.layout() != xt::layout_type::row_major)
        throw std::runtime_error("res_dgamma_by_dphi needs to be in row-major storage order");
    if (res_grad_gamma.layout() != xt::layout_type::row_major)
        throw std::runtime_error("res_grad_gamma needs to be in row-major storage order");
    if (res_grad_dgamma_by_dphi.layout() != xt::layout_type::row_major)
        throw std::runtime_error("res_grad_dgamma_by_dphi needs to be in row-major storage order");

    const int num_points      = static_cast<int>(pointsx.size());
    const int num_quad_points = static_cast<int>(gamma.shape(0));

    double *gamma_j_ptr                 = &(gamma(0, 0));
    double *dgamma_j_by_dphi_ptr        = &(dgamma_by_dphi(0, 0));
    double *res_dgamma_by_dphi_ptr      = &(res_dgamma_by_dphi(0, 0));
    double *res_gamma_ptr               = &(res_gamma(0, 0));
    double *res_grad_dgamma_by_dphi_ptr = &(res_grad_dgamma_by_dphi(0, 0));
    double *res_grad_gamma_ptr          = &(res_grad_gamma(0, 0));

    for (int i = 0; i < num_points; ++i)
    {
        Vec3dStd point_i(&(pointsx[i]), &(pointsy[i]), &(pointsz[i]));
        Vec3dStd v_i;

        std::vector<Vec3dStd> vgrad_i{
            Vec3dStd(), Vec3dStd(), Vec3dStd()
        };

#pragma unroll
        for (int d = 0; d < 3; ++d)
        {
            v_i[d] = v(i, d);
            MYIF(derivs > 0)
            {
                for (int dd = 0; dd < 3; ++dd)
                {
                    vgrad_i[dd][d] = vgrad(i, dd, d);
                }
            }
        }

        for (int j = 0; j < num_quad_points; ++j)
        {
            Vec3d dgamma_j_by_dphi_vec{
                dgamma_j_by_dphi_ptr[3 * j + 0],
                dgamma_j_by_dphi_ptr[3 * j + 1],
                dgamma_j_by_dphi_ptr[3 * j + 2]
            };

            Vec3dStd diff(
                gamma_j_ptr[3 * j + 0],
                gamma_j_ptr[3 * j + 1],
                gamma_j_ptr[3 * j + 2]);
            diff = point_i - diff;

            double r2               = normsq(diff);
            double norm_diff_inv    = rsqrt(r2);
            double norm_diff_inv_3  = norm_diff_inv * norm_diff_inv * norm_diff_inv;

            Vec3dStd res_dgamma_by_dphi_add = v_i * norm_diff_inv;
            res_dgamma_by_dphi_ptr[3 * j + 0] += res_dgamma_by_dphi_add.x;
            res_dgamma_by_dphi_ptr[3 * j + 1] += res_dgamma_by_dphi_add.y;
            res_dgamma_by_dphi_ptr[3 * j + 2] += res_dgamma_by_dphi_add.z;

            double vi_dot_dgamma_dphi_j = inner(v_i, dgamma_j_by_dphi_vec);
            Vec3dStd res_gamma_add      = diff * (vi_dot_dgamma_dphi_j * norm_diff_inv_3);

            res_gamma_ptr[3 * j + 0] += res_gamma_add.x;
            res_gamma_ptr[3 * j + 1] += res_gamma_add.y;
            res_gamma_ptr[3 * j + 2] += res_gamma_add.z;

            MYIF(derivs > 0)
            {
                double norm_diff_inv_5 = norm_diff_inv_3 * norm_diff_inv * norm_diff_inv;

                Vec3dStd res_grad_dgamma_by_dphi_add;
                Vec3dStd res_grad_gamma_add;

#pragma unroll
                for (int k = 0; k < 3; ++k)
                {
                    res_grad_dgamma_by_dphi_add -= vgrad_i[k] * norm_diff_inv_3 * diff[k];
                    res_grad_gamma_add -= diff * inner(vgrad_i[k], dgamma_j_by_dphi_vec)
                                                * (3.0 * diff[k]) * norm_diff_inv_5;

                    // MYIF(derivs > 1) { ... } // place for higher-order contributions if desired
                }

                // add diagonal parts
                res_grad_gamma_add.x += inner(vgrad_i[0], dgamma_j_by_dphi_vec) * norm_diff_inv_3;
                res_grad_gamma_add.y += inner(vgrad_i[1], dgamma_j_by_dphi_vec) * norm_diff_inv_3;
                res_grad_gamma_add.z += inner(vgrad_i[2], dgamma_j_by_dphi_vec) * norm_diff_inv_3;

                res_grad_dgamma_by_dphi_ptr[3 * j + 0] += res_grad_dgamma_by_dphi_add.x;
                res_grad_dgamma_by_dphi_ptr[3 * j + 1] += res_grad_dgamma_by_dphi_add.y;
                res_grad_dgamma_by_dphi_ptr[3 * j + 2] += res_grad_dgamma_by_dphi_add.z;

                res_grad_gamma_ptr[3 * j + 0] += res_grad_gamma_add.x;
                res_grad_gamma_ptr[3 * j + 1] += res_grad_gamma_add.y;
                res_grad_gamma_ptr[3 * j + 2] += res_grad_gamma_add.z;
            }
        }
    }
}

#endif  // USE_XSIMD
