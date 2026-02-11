#pragma once

#include "simdhelpers.h"
#include "vec3dsimd.h"
#include <stdexcept>
#include <vector>
#include <cmath>
#include "xtensor/xlayout.hpp"

// When compiled with C++17, then we use `if constexpr` to check for
// derivatives that need to be computed. These are evaluated at compile time.
//
#if __cplusplus >= 201703L
#  define MYIF(c) if constexpr(c)
#else
#  define MYIF(c) if(c)
#endif

// -----------------------------------------------------------------------------
// Small helpers for Hessian formulas
// -----------------------------------------------------------------------------
inline double kronecker_delta(int i, int j)
{
    return (i == j) ? 1.0 : 0.0;
}

inline double levi_civita(int i, int j, int k)
{
    // ε_{ijk} with indices in {0,1,2}
    if ((i == 0 && j == 1 && k == 2) ||
        (i == 1 && j == 2 && k == 0) ||
        (i == 2 && j == 0 && k == 1))
        return 1.0;
    if ((i == 0 && j == 2 && k == 1) ||
        (i == 2 && j == 1 && k == 0) ||
        (i == 1 && j == 0 && k == 2))
        return -1.0;
    return 0.0;
}

// ============================================================================
//  BIOT-SAVART VJP (B, ∇B, and ∇∇B adjoint)
// ============================================================================
//
// T is assumed xtensor-like:
//   gamma, dgamma_by_dphi:        (num_quad_points, 3)
//   v:                            (num_points, 3)         (adjoint of B)
//   vgrad: (if derivs > 0)        (num_points, 3, 3)      (adjoint of ∇B)
//   vhess: (if derivs > 1)        (num_points, 3, 3, 3)   (adjoint of ∇∇B)
//
// Coil-space adjoints:
//   res_gamma, res_dgamma_by_dphi           for B
//   res_grad_gamma, res_grad_dgamma_by_dphi for ∇B
//   res_hess_gamma, res_hess_dgamma_by_dphi for ∇∇B
// ----------------------------------------------------------------------------

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
    T &res_grad_dgamma_by_dphi,
    T &vhess,
    T &res_hess_gamma,
    T &res_hess_dgamma_by_dphi
)
{
    if (gamma.layout() != xt::layout_type::row_major)
        throw std::runtime_error("gamma needs to be in row-major storage order");
    if (dgamma_by_dphi.layout() != xt::layout_type::row_major)
        throw std::runtime_error("dgamma_by_dphi needs to be in row-major storage order");
    if (res_gamma.layout() != xt::layout_type::row_major)
        throw std::runtime_error("res_gamma needs to be in row-major storage order");
    if (res_dgamma_by_dphi.layout() != xt::layout_type::row_major)
        throw std::runtime_error("res_dgamma_by_dphi needs to be in row-major storage order");
    MYIF(derivs > 0) {
        if (res_grad_gamma.layout() != xt::layout_type::row_major)
            throw std::runtime_error("res_grad_gamma needs to be in row-major storage order");
        if (res_grad_dgamma_by_dphi.layout() != xt::layout_type::row_major)
            throw std::runtime_error("res_grad_dgamma_by_dphi needs to be in row-major storage order");
    }

    MYIF(derivs > 1) {
        if (res_hess_gamma.layout() != xt::layout_type::row_major)
            throw std::runtime_error("res_hess_gamma needs to be in row-major storage order");
        if (res_hess_dgamma_by_dphi.layout() != xt::layout_type::row_major)
            throw std::runtime_error("res_hess_dgamma_by_dphi needs to be in row-major storage order");
    }
    const int num_points      = static_cast<int>(pointsx.size());
    const int num_quad_points = static_cast<int>(gamma.shape(0));
    constexpr int simd_size   = xsimd::simd_type<double>::size;

    double *gamma_j_ptr                 = &(gamma(0, 0));
    double *dgamma_j_by_dphi_ptr        = &(dgamma_by_dphi(0, 0));
    double *res_dgamma_by_dphi_ptr      = &(res_dgamma_by_dphi(0, 0));
    double *res_gamma_ptr               = &(res_gamma(0, 0));
    double *res_grad_dgamma_by_dphi_ptr = &(res_grad_dgamma_by_dphi(0, 0));
    double *res_grad_gamma_ptr          = &(res_grad_gamma(0, 0));
    double *res_hess_dgamma_by_dphi_ptr = &(res_hess_dgamma_by_dphi(0, 0));
    double *res_hess_gamma_ptr          = &(res_hess_gamma(0, 0));

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

            // ----- vjp wrt gamma (B) -----
            Vec3dSimd cross_dgamma_j_by_dphi_diff = cross(dgamma_j_by_dphi_vec, diff);
            Vec3dSimd res_gamma_add = cross(dgamma_j_by_dphi_vec, v_i) * norm_diff_3_inv;
            res_gamma_add += diff * inner(cross_dgamma_j_by_dphi_diff, v_i) * norm_diff_5_inv_times_3;

            res_gamma_ptr[3 * j + 0] += xsimd::hadd(res_gamma_add.x);
            res_gamma_ptr[3 * j + 1] += xsimd::hadd(res_gamma_add.y);
            res_gamma_ptr[3 * j + 2] += xsimd::hadd(res_gamma_add.z);

            // ----- ∇B derivatives -----
            MYIF(derivs > 0)
            {
                simd_t norm_diff_7_inv = norm_diff_5_inv * norm_diff_2_inv;

                Vec3dSimd res_grad_dgamma_by_dphi_add; // zero-init
                Vec3dSimd res_grad_gamma_add;          // zero-init

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
                }

                res_grad_dgamma_by_dphi_ptr[3 * j + 0] += xsimd::hadd(res_grad_dgamma_by_dphi_add.x);
                res_grad_dgamma_by_dphi_ptr[3 * j + 1] += xsimd::hadd(res_grad_dgamma_by_dphi_add.y);
                res_grad_dgamma_by_dphi_ptr[3 * j + 2] += xsimd::hadd(res_grad_dgamma_by_dphi_add.z);

                res_grad_gamma_ptr[3 * j + 0] += xsimd::hadd(res_grad_gamma_add.x);
                res_grad_gamma_ptr[3 * j + 1] += xsimd::hadd(res_grad_gamma_add.y);
                res_grad_gamma_ptr[3 * j + 2] += xsimd::hadd(res_grad_gamma_add.z);
            }

            // ----- ∇∇B (Hessian) adjoints -----
            MYIF(derivs > 1)
            {
                // per-lane scalar loop for Hessian (still inside SIMD bulk)
                for (int b = 0; b < 3; ++b)        // B component
                {
                    for (int jg = 0; jg < 3; ++jg) // first derivative index
                    {
                        for (int kg = 0; kg < 3; ++kg) // second derivative index
                        {
                            double accum_a[3] = {0.0, 0.0, 0.0};
                            double accum_g[3] = {0.0, 0.0, 0.0};

                            for (int lane = 0; lane < simd_size; ++lane)
                            {
                                int ip = i + lane;
                                if (ip >= num_points) break;

                                double v_h = vhess(ip, b, jg, kg);
                                if (v_h == 0.0)
                                    continue;

                                double r[3] = {
                                    diff.x[lane],
                                    diff.y[lane],
                                    diff.z[lane]
                                };

                                double r2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
                                if (r2 == 0.0)
                                    continue;

                                double inv_r  = 1.0 / std::sqrt(r2);
                                double inv_r2 = inv_r * inv_r;
                                double inv_r3 = inv_r2 * inv_r;
                                double inv_r5 = inv_r3 * inv_r2;
                                double inv_r7 = inv_r5 * inv_r2;
                                double inv_r9 = inv_r7 * inv_r2;

                                // G_{n,jg,kg} = ∂²F_n / ∂x_j ∂x_k
                                double G[3];
                                for (int n = 0; n < 3; ++n)
                                {
                                    double term1 =
                                        -3.0 * ( kronecker_delta(n, jg) * r[kg]
                                               + kronecker_delta(n, kg) * r[jg]
                                               + kronecker_delta(jg, kg) * r[n] ) * inv_r5;
                                    double term2 =
                                        15.0 * r[n] * r[jg] * r[kg] * inv_r7;
                                    G[n] = term1 + term2;
                                }

                                // dL/da_p from Hessian: Σ_n ε_{b p n} G[n]
                                for (int p = 0; p < 3; ++p)
                                {
                                    double sum = 0.0;
                                    for (int n = 0; n < 3; ++n)
                                    {
                                        double eps = levi_civita(b, p, n);
                                        if (eps != 0.0)
                                            sum += eps * G[n];
                                    }
                                    accum_a[p] += v_h * sum;
                                }

                                // dL/dgamma_l from Hessian:
                                // ∂HessB_b(jg,kg)/∂gamma_l = - Σ_mn ε_{b m n} a_m H_{n,jg,kg,l}
                                for (int l = 0; l < 3; ++l)
                                {
                                    double sum_g = 0.0;
                                    for (int n = 0; n < 3; ++n)
                                    {
                                        double delta_nj = kronecker_delta(n, jg);
                                        double delta_nk = kronecker_delta(n, kg);
                                        double delta_nl = kronecker_delta(n, l);
                                        double delta_jk = kronecker_delta(jg, kg);
                                        double delta_jl = kronecker_delta(jg, l);
                                        double delta_kl = kronecker_delta(kg, l);

                                        double termA =
                                            -3.0 * ( delta_nj * delta_kl
                                                   + delta_nk * delta_jl
                                                   + delta_nl * delta_jk ) * inv_r5;

                                        double termB =
                                            15.0 * (
                                                delta_nj * r[kg] * r[l]
                                              + delta_nk * r[jg] * r[l]
                                              + r[n] * delta_jk * r[l]
                                              + delta_nl * r[jg] * r[kg]
                                              + r[n] * delta_jl * r[kg]
                                              + r[n] * r[jg] * delta_kl
                                            ) * inv_r7;

                                        double termC =
                                            -105.0 * r[n] * r[jg] * r[kg] * r[l] * inv_r9;

                                        double H_njkl = termA + termB + termC;

                                        for (int m = 0; m < 3; ++m)
                                        {
                                            double eps = levi_civita(b, m, n);
                                            if (eps != 0.0)
                                            {
                                                double a_m = dgamma_j_by_dphi_vec[m];
                                                sum_g += eps * a_m * H_njkl;
                                            }
                                        }
                                    }
                                    accum_g[l] -= v_h * sum_g;
                                }
                            } // lane

                            for (int p = 0; p < 3; ++p)
                                res_hess_dgamma_by_dphi_ptr[3 * j + p] += accum_a[p];

                            for (int l = 0; l < 3; ++l)
                                res_hess_gamma_ptr[3 * j + l] += accum_g[l];

                        } // kg
                    }     // jg
                }         // b
            }             // derivs > 1
        }                 // j
    }                     // SIMD bulk

    // Scalar tail for remaining points (still in SIMD branch for consistency)
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

            double r2              = diff.squaredNorm();
            double norm_diff_inv   = 1.0 / std::sqrt(r2);
            double norm_diff_2_inv = norm_diff_inv * norm_diff_inv;
            double norm_diff_3_inv = norm_diff_2_inv * norm_diff_inv;
            double norm_diff_5_inv = norm_diff_3_inv * norm_diff_2_inv;
            double norm_diff_5_inv_times_3 = 3.0 * norm_diff_5_inv;

            // vjp wrt dgamma_by_dphi (B)
            Vec3d res_dgamma_by_dphi_add = cross(diff, v_i) * norm_diff_3_inv;
            res_dgamma_by_dphi_ptr[3 * j + 0] += res_dgamma_by_dphi_add[0];
            res_dgamma_by_dphi_ptr[3 * j + 1] += res_dgamma_by_dphi_add[1];
            res_dgamma_by_dphi_ptr[3 * j + 2] += res_dgamma_by_dphi_add[2];

            // vjp wrt gamma (B)
            Vec3d cross_dgamma_j_by_dphi_diff = cross(dgamma_j_by_dphi_vec, diff);
            Vec3d res_gamma_add = cross(dgamma_j_by_dphi_vec, v_i) * norm_diff_3_inv;
            res_gamma_add += diff * inner(cross_dgamma_j_by_dphi_diff, v_i) * norm_diff_5_inv_times_3;

            res_gamma_ptr[3 * j + 0] += res_gamma_add[0];
            res_gamma_ptr[3 * j + 1] += res_gamma_add[1];
            res_gamma_ptr[3 * j + 2] += res_gamma_add[2];

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
                }

                res_grad_dgamma_by_dphi_ptr[3 * j + 0] += res_grad_dgamma_by_dphi_add[0];
                res_grad_dgamma_by_dphi_ptr[3 * j + 1] += res_grad_dgamma_by_dphi_add[1];
                res_grad_dgamma_by_dphi_ptr[3 * j + 2] += res_grad_dgamma_by_dphi_add[2];

                res_grad_gamma_ptr[3 * j + 0] += res_grad_gamma_add[0];
                res_grad_gamma_ptr[3 * j + 1] += res_grad_gamma_add[1];
                res_grad_gamma_ptr[3 * j + 2] += res_grad_gamma_add[2];
            }

            MYIF(derivs > 1)
            {
                double inv_r   = norm_diff_inv;
                double inv_r2  = norm_diff_2_inv;
                double inv_r3  = norm_diff_3_inv;
                double inv_r5  = norm_diff_5_inv;
                double norm_diff_7_inv = norm_diff_5_inv * norm_diff_2_inv;
                double inv_r7  = norm_diff_7_inv;
                double inv_r9  = inv_r7 * inv_r2;

                double r[3] = {diff[0], diff[1], diff[2]};

                for (int b = 0; b < 3; ++b)
                {
                    for (int jg = 0; jg < 3; ++jg)
                    {
                        for (int kg = 0; kg < 3; ++kg)
                        {
                            double v_h = vhess(i, b, jg, kg);
                            if (v_h == 0.0)
                                continue;

                            double G[3];
                            for (int n = 0; n < 3; ++n)
                            {
                                double term1 =
                                    -3.0 * ( kronecker_delta(n, jg) * r[kg]
                                           + kronecker_delta(n, kg) * r[jg]
                                           + kronecker_delta(jg, kg) * r[n] ) * inv_r5;
                                double term2 =
                                    15.0 * r[n] * r[jg] * r[kg] * inv_r7;
                                G[n] = term1 + term2;
                            }

                            for (int p = 0; p < 3; ++p)
                            {
                                double sum = 0.0;
                                for (int n = 0; n < 3; ++n)
                                {
                                    double eps = levi_civita(b, p, n);
                                    if (eps != 0.0)
                                        sum += eps * G[n];
                                }
                                res_hess_dgamma_by_dphi_ptr[3 * j + p] += v_h * sum;
                            }

                            for (int l = 0; l < 3; ++l)
                            {
                                double sum_g = 0.0;
                                for (int n = 0; n < 3; ++n)
                                {
                                    double delta_nj = kronecker_delta(n, jg);
                                    double delta_nk = kronecker_delta(n, kg);
                                    double delta_nl = kronecker_delta(n, l);
                                    double delta_jk = kronecker_delta(jg, kg);
                                    double delta_jl = kronecker_delta(jg, l);
                                    double delta_kl = kronecker_delta(kg, l);

                                    double termA =
                                        -3.0 * ( delta_nj * delta_kl
                                               + delta_nk * delta_jl
                                               + delta_nl * delta_jk ) * inv_r5;

                                    double termB =
                                        15.0 * (
                                            delta_nj * r[kg] * r[l]
                                          + delta_nk * r[jg] * r[l]
                                          + r[n] * delta_jk * r[l]
                                          + delta_nl * r[jg] * r[kg]
                                          + r[n] * delta_jl * r[kg]
                                          + r[n] * r[jg] * delta_kl
                                        ) * inv_r7;

                                    double termC =
                                        -105.0 * r[n] * r[jg] * r[kg] * r[l] * inv_r9;

                                    double H_njkl = termA + termB + termC;

                                    for (int m = 0; m < 3; ++m)
                                    {
                                        double eps = levi_civita(b, m, n);
                                        if (eps != 0.0)
                                        {
                                            double a_m = dgamma_j_by_dphi_vec[m];
                                            sum_g += eps * a_m * H_njkl;
                                        }
                                    }
                                }
                                res_hess_gamma_ptr[3 * j + l] -= v_h * sum_g;
                            }
                        }
                    }
                }
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
    T &res_grad_dgamma_by_dphi,
    T &vhess,
    T &res_hess_gamma,
    T &res_hess_dgamma_by_dphi
)
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
    if (res_hess_gamma.layout() != xt::layout_type::row_major)
        throw std::runtime_error("res_hess_gamma needs to be in row-major storage order");
    if (res_hess_dgamma_by_dphi.layout() != xt::layout_type::row_major)
        throw std::runtime_error("res_hess_dgamma_by_dphi needs to be in row-major storage order");

    const int num_points      = static_cast<int>(pointsx.size());
    const int num_quad_points = static_cast<int>(gamma.shape(0));

    double *gamma_j_ptr                 = &(gamma(0, 0));
    double *dgamma_j_by_dphi_ptr        = &(dgamma_by_dphi(0, 0));
    double *res_dgamma_by_dphi_ptr      = &(res_dgamma_by_dphi(0, 0));
    double *res_gamma_ptr               = &(res_gamma(0, 0));
    double *res_grad_dgamma_by_dphi_ptr = &(res_grad_dgamma_by_dphi(0, 0));
    double *res_grad_gamma_ptr          = &(res_grad_gamma(0, 0));
    double *res_hess_dgamma_by_dphi_ptr = &(res_hess_dgamma_by_dphi(0, 0));
    double *res_hess_gamma_ptr          = &(res_hess_gamma(0, 0));

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

            double r2              = diff.norm() * diff.norm(); // or diff.squaredNorm() if available
            double norm_diff       = std::sqrt(r2);
            double norm_diff_inv   = 1.0 / norm_diff;
            double norm_diff_2_inv = norm_diff_inv * norm_diff_inv;
            double norm_diff_3_inv = norm_diff_2_inv * norm_diff_inv;
            double norm_diff_5_inv = norm_diff_3_inv * norm_diff_2_inv;
            double norm_diff_5_inv_times_3 = 3.0 * norm_diff_5_inv;

            // vjp wrt dgamma_by_dphi (B)
            Vec3dStd res_dgamma_by_dphi_add = cross(diff, v_i) * norm_diff_3_inv;
            res_dgamma_by_dphi_ptr[3 * j + 0] += res_dgamma_by_dphi_add.x;
            res_dgamma_by_dphi_ptr[3 * j + 1] += res_dgamma_by_dphi_add.y;
            res_dgamma_by_dphi_ptr[3 * j + 2] += res_dgamma_by_dphi_add.z;

            // vjp wrt gamma (B)
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
                }

                res_grad_dgamma_by_dphi_ptr[3 * j + 0] += res_grad_dgamma_by_dphi_add.x;
                res_grad_dgamma_by_dphi_ptr[3 * j + 1] += res_grad_dgamma_by_dphi_add.y;
                res_grad_dgamma_by_dphi_ptr[3 * j + 2] += res_grad_dgamma_by_dphi_add.z;

                res_grad_gamma_ptr[3 * j + 0] += res_grad_gamma_add.x;
                res_grad_gamma_ptr[3 * j + 1] += res_grad_gamma_add.y;
                res_grad_gamma_ptr[3 * j + 2] += res_grad_gamma_add.z;
            }

            MYIF(derivs > 1)
            {
                double inv_r   = norm_diff_inv;
                double inv_r2  = norm_diff_2_inv;
                double inv_r3  = norm_diff_3_inv;
                double inv_r5  = norm_diff_5_inv;
                double norm_diff_7_inv = norm_diff_5_inv * norm_diff_2_inv;
                double inv_r7  = norm_diff_7_inv;
                double inv_r9  = inv_r7 * inv_r2;

                double r[3] = {diff.x, diff.y, diff.z};

                for (int b = 0; b < 3; ++b)
                {
                    for (int jg = 0; jg < 3; ++jg)
                    {
                        for (int kg = 0; kg < 3; ++kg)
                        {
                            double v_h = vhess(i, b, jg, kg);
                            if (v_h == 0.0)
                                continue;

                            // G_{n,jg,kg} (2nd derivative kernel)
                            double G[3];
                            for (int n = 0; n < 3; ++n)
                            {
                                double term1 =
                                    -3.0 * ( kronecker_delta(n, jg) * r[kg]
                                           + kronecker_delta(n, kg) * r[jg]
                                           + kronecker_delta(jg, kg) * r[n] ) * inv_r5;
                                double term2 =
                                    15.0 * r[n] * r[jg] * r[kg] * inv_r7;
                                G[n] = term1 + term2;
                            }

                            // dL/da_p from Hessian
                            for (int p = 0; p < 3; ++p)
                            {
                                double sum = 0.0;
                                for (int n = 0; n < 3; ++n)
                                {
                                    double eps = levi_civita(b, p, n);
                                    if (eps != 0.0)
                                        sum += eps * G[n];
                                }
                                res_hess_dgamma_by_dphi_ptr[3 * j + p] += v_h * sum;
                            }

                            // dL/dgamma_l from Hessian
                            for (int l = 0; l < 3; ++l)
                            {
                                double sum_g = 0.0;
                                for (int n = 0; n < 3; ++n)
                                {
                                    double delta_nj = kronecker_delta(n, jg);
                                    double delta_nk = kronecker_delta(n, kg);
                                    double delta_nl = kronecker_delta(n, l);
                                    double delta_jk = kronecker_delta(jg, kg);
                                    double delta_jl = kronecker_delta(jg, l);
                                    double delta_kl = kronecker_delta(kg, l);

                                    double termA =
                                        -3.0 * ( delta_nj * delta_kl
                                               + delta_nk * delta_jl
                                               + delta_nl * delta_jk ) * inv_r5;

                                    double termB =
                                        15.0 * (
                                            delta_nj * r[kg] * r[l]
                                          + delta_nk * r[jg] * r[l]
                                          + r[n] * delta_jk * r[l]
                                          + delta_nl * r[jg] * r[kg]
                                          + r[n] * delta_jl * r[kg]
                                          + r[n] * r[jg] * delta_kl
                                        ) * inv_r7;

                                    double termC =
                                        -105.0 * r[n] * r[jg] * r[kg] * r[l] * inv_r9;

                                    double H_njkl = termA + termB + termC;

                                    for (int m = 0; m < 3; ++m)
                                    {
                                        double eps = levi_civita(b, m, n);
                                        if (eps != 0.0)
                                        {
                                            double a_m = dgamma_j_by_dphi_vec[m];
                                            sum_g += eps * a_m * H_njkl;
                                        }
                                    }
                                }
                                res_hess_gamma_ptr[3 * j + l] -= v_h * sum_g;
                            }
                        }
                    }
                }
            }
        }
    }
}

#endif  // USE_XSIMD

// ============================================================================
//  VECTOR POTENTIAL VJP STUB
// ============================================================================
//
// This is provided so that biot_savart_vjp_py.cpp can link. It *throws* at
// runtime if actually called. Replace with a proper implementation if/when
// you need VJP of the vector potential itself.
// ----------------------------------------------------------------------------

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
    T &res_grad_dgamma_by_dphi,
    T &vhess,
    T &res_hess_gamma,
    T &res_hess_dgamma_by_dphi
)
{
    (void)pointsx; (void)pointsy; (void)pointsz;
    (void)gamma; (void)dgamma_by_dphi;
    (void)v; (void)res_gamma; (void)res_dgamma_by_dphi;
    (void)vgrad; (void)res_grad_gamma; (void)res_grad_dgamma_by_dphi;
    (void)vhess; (void)res_hess_gamma; (void)res_hess_dgamma_by_dphi;

    throw std::runtime_error(
        "biot_savart_vector_potential_vjp_kernel is not implemented in this build "
        "(only B, grad B, and grad grad B VJPs are implemented)."
    );
}
