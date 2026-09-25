#include "simdhelpers.h"
#include "vec3dsimd.h"
#include <stdexcept>
#include "xtensor/core/xlayout.hpp"

// When compiled with C++17, then we use `if constexpr` to check for
// derivatives that need to be computed.  These are actually evaluated at
// compile time, e.g. the compiler creates two different functions, one that
// only computes the vjp for B, and one that computes the vjp for B and \nabla B.
//
#if __cplusplus >= 201703L
#define MYIF(c) if constexpr(c)
#else
#define MYIF(c) if(c)
#endif

#if defined(USE_XSIMD)

template<class T, int derivs>
void biot_savart_vjp_kernel(AlignedPaddedVec& pointsx, AlignedPaddedVec& pointsy, AlignedPaddedVec& pointsz,
            T& gamma, T& dgamma_by_dphi, T& v, T& res_gamma, T& res_dgamma_by_dphi, T& vgrad, T& res_grad_gamma,
            T& res_grad_dgamma_by_dphi) {
    if(gamma.layout() != xt::layout_type::row_major)
          throw std::runtime_error("gamma needs to be in row-major storage order");
    if(dgamma_by_dphi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("dgamma_by_dphi needs to be in row-major storage order");
    if(res_gamma.layout() != xt::layout_type::row_major)
          throw std::runtime_error("res_gamma needs to be in row-major storage order");
    if(res_dgamma_by_dphi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("res_dgamma_by_dphi needs to be in row-major storage order");
    if(res_grad_gamma.layout() != xt::layout_type::row_major)
          throw std::runtime_error("res_grad_gamma needs to be in row-major storage order");
    if(res_grad_dgamma_by_dphi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("res_grad_dgamma_by_dphi needs to be in row-major storage order");
    int num_points         = pointsx.size();
    int num_quad_points    = gamma.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    double* gamma_j_ptr = &(gamma(0, 0));
    double* dgamma_j_by_dphi_ptr = &(dgamma_by_dphi(0, 0));
    double* res_dgamma_by_dphi_ptr = &(res_dgamma_by_dphi(0, 0));
    double* res_gamma_ptr = &(res_gamma(0, 0));
    double* res_grad_dgamma_by_dphi_ptr = &(res_grad_dgamma_by_dphi(0, 0));
    double* res_grad_gamma_ptr = &(res_grad_gamma(0, 0));
    for(int i = 0; i < num_points-num_points%simd_size; i += simd_size) {
        Vec3dSimd point_i = Vec3dSimd(&(pointsx[i]), &(pointsy[i]), &(pointsz[i]));
        auto v_i   = Vec3dSimd();
        auto vgrad_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, xs::default_arch::alignment()>>{
                Vec3dSimd(), Vec3dSimd(), Vec3dSimd()
            };
        // v/vgrad are interleaved (point, component[, component]) arrays and batches
        // have no per-lane write via operator[] anymore; gather the lanes into small
        // contiguous buffers first, then load each buffer into a batch in one shot.
        alignas(xs::default_arch::alignment()) double v_x[simd_size], v_y[simd_size], v_z[simd_size];
        alignas(xs::default_arch::alignment()) double vgrad_x[3][simd_size], vgrad_y[3][simd_size], vgrad_z[3][simd_size];
#pragma unroll
        for(int k=0; k<simd_size; k++){
            v_x[k] = v(i+k, 0);
            v_y[k] = v(i+k, 1);
            v_z[k] = v(i+k, 2);
            MYIF(derivs>0) {
#pragma unroll
                for (int dd = 0; dd < 3; ++dd) {
                    vgrad_x[dd][k] = vgrad(i+k, dd, 0);
                    vgrad_y[dd][k] = vgrad(i+k, dd, 1);
                    vgrad_z[dd][k] = vgrad(i+k, dd, 2);
                }
            }
        }
        v_i = Vec3dSimd(xs::load_aligned(v_x), xs::load_aligned(v_y), xs::load_aligned(v_z));
        MYIF(derivs>0) {
#pragma unroll
            for (int dd = 0; dd < 3; ++dd) {
                vgrad_i[dd] = Vec3dSimd(xs::load_aligned(vgrad_x[dd]), xs::load_aligned(vgrad_y[dd]), xs::load_aligned(vgrad_z[dd]));
            }
        }

        for (int j = 0; j < num_quad_points; ++j) {
            auto dgamma_j_by_dphi = Vec3d{ dgamma_j_by_dphi_ptr[3*j+0], dgamma_j_by_dphi_ptr[3*j+1], dgamma_j_by_dphi_ptr[3*j+2] };
            auto diff = point_i - Vec3dSimd(gamma_j_ptr[3*j+0], gamma_j_ptr[3*j+1], gamma_j_ptr[3*j+2]);
            auto norm_diff_2 = normsq(diff);
            auto norm_diff_inv = rsqrt(norm_diff_2);
            auto norm_diff_2_inv = norm_diff_inv*norm_diff_inv;
            auto norm_diff_3_inv = norm_diff_2_inv*norm_diff_inv;
            auto norm_diff_5_inv = norm_diff_3_inv*norm_diff_2_inv;
            auto norm_diff_5_inv_times_3 = 3.*norm_diff_5_inv;

            auto res_dgamma_by_dphi_add = cross(diff, v_i) * norm_diff_3_inv;
            res_dgamma_by_dphi_ptr[3*j+0] += xsimd::reduce_add(res_dgamma_by_dphi_add.x);
            res_dgamma_by_dphi_ptr[3*j+1] += xsimd::reduce_add(res_dgamma_by_dphi_add.y);
            res_dgamma_by_dphi_ptr[3*j+2] += xsimd::reduce_add(res_dgamma_by_dphi_add.z);

            auto cross_dgamma_j_by_dphi_diff = cross(dgamma_j_by_dphi, diff);
            auto res_gamma_add = cross(dgamma_j_by_dphi, v_i) * norm_diff_3_inv;
            res_gamma_add += diff * inner(cross_dgamma_j_by_dphi_diff, v_i) * (norm_diff_5_inv_times_3);
            res_gamma_ptr[3*j+0] += xsimd::reduce_add(res_gamma_add.x);
            res_gamma_ptr[3*j+1] += xsimd::reduce_add(res_gamma_add.y);
            res_gamma_ptr[3*j+2] += xsimd::reduce_add(res_gamma_add.z);

            MYIF(derivs>0) {
                auto norm_diff_7_inv = norm_diff_5_inv*norm_diff_2_inv;
                auto res_grad_dgamma_by_dphi_add = Vec3dSimd();
                auto res_grad_gamma_add = Vec3dSimd();

#pragma unroll
                for(int k=0; k<3; k++){
                    auto eksimd = Vec3dSimd();
                    eksimd[k] += 1.;
                    Vec3d ek = Vec3d::Zero();
                    ek[k] = 1.;
                    res_grad_dgamma_by_dphi_add += cross(k, vgrad_i[k]) * norm_diff_3_inv;
                    res_grad_dgamma_by_dphi_add -= cross(diff, vgrad_i[k]) * (diff[k] * norm_diff_5_inv_times_3);

                    res_grad_gamma_add += diff * (inner(cross(dgamma_j_by_dphi, ek), vgrad_i[k]) * norm_diff_5_inv_times_3);
                    res_grad_gamma_add += eksimd * (inner(cross_dgamma_j_by_dphi_diff, vgrad_i[k]) * norm_diff_5_inv_times_3);
                    res_grad_gamma_add += cross(vgrad_i[k], dgamma_j_by_dphi) * (norm_diff_5_inv_times_3 * diff[k]);
                    res_grad_gamma_add -= diff * (15. * diff[k] * inner(cross_dgamma_j_by_dphi_diff, vgrad_i[k]) * norm_diff_7_inv);
                }
                res_grad_dgamma_by_dphi_ptr[3*j+0] += xsimd::reduce_add(res_grad_dgamma_by_dphi_add.x);
                res_grad_dgamma_by_dphi_ptr[3*j+1] += xsimd::reduce_add(res_grad_dgamma_by_dphi_add.y);
                res_grad_dgamma_by_dphi_ptr[3*j+2] += xsimd::reduce_add(res_grad_dgamma_by_dphi_add.z);
                res_grad_gamma_ptr[3*j+0] += xsimd::reduce_add(res_grad_gamma_add.x);
                res_grad_gamma_ptr[3*j+1] += xsimd::reduce_add(res_grad_gamma_add.y);
                res_grad_gamma_ptr[3*j+2] += xsimd::reduce_add(res_grad_gamma_add.z);
            }
        }
    }
    for (int i = num_points - num_points % simd_size; i < num_points; ++i) {
        auto point_i = Vec3d{pointsx[i], pointsy[i], pointsz[i]};
        Vec3d v_i   = Vec3d::Zero();
        auto vgrad_i = vector<Vec3d>{
            Vec3d::Zero(), Vec3d::Zero(), Vec3d::Zero()
            };
#pragma unroll
        for (int d = 0; d < 3; ++d) {
            v_i[d] = v(i, d);
            MYIF(derivs>0) {
                for (int dd = 0; dd < 3; ++dd) {
                    vgrad_i[dd][d] = vgrad(i, dd, d);
                }
            }
        }
        for (int j = 0; j < num_quad_points; ++j) {
            Vec3d diff = point_i - Vec3d{gamma_j_ptr[3*j+0], gamma_j_ptr[3*j+1], gamma_j_ptr[3*j+2]};
            Vec3d dgamma_j_by_dphi = Vec3d{dgamma_j_by_dphi_ptr[3*j+0], dgamma_j_by_dphi_ptr[3*j+1], dgamma_j_by_dphi_ptr[3*j+2]};
            double norm_diff = norm(diff);
            double norm_diff_inv = 1/norm_diff;
            double norm_diff_2_inv = norm_diff_inv*norm_diff_inv;
            double norm_diff_3_inv = norm_diff_2_inv*norm_diff_inv;
            double norm_diff_5_inv = norm_diff_3_inv*norm_diff_2_inv;
            double norm_diff_5_inv_times_3 = 3.*norm_diff_5_inv;

            Vec3d res_dgamma_by_dphi_add = cross(diff, v_i) * norm_diff_3_inv;
            res_dgamma_by_dphi(j, 0) += res_dgamma_by_dphi_add.coeff(0);
            res_dgamma_by_dphi(j, 1) += res_dgamma_by_dphi_add.coeff(1);
            res_dgamma_by_dphi(j, 2) += res_dgamma_by_dphi_add.coeff(2);

            Vec3d cross_dgamma_j_by_dphi_diff = cross(dgamma_j_by_dphi, diff);
            Vec3d res_gamma_add = cross(dgamma_j_by_dphi, v_i) * norm_diff_3_inv;
            res_gamma_add += diff * inner(cross_dgamma_j_by_dphi_diff, v_i) * (norm_diff_5_inv_times_3);
            res_gamma(j, 0) += res_gamma_add.coeff(0);
            res_gamma(j, 1) += res_gamma_add.coeff(1);
            res_gamma(j, 2) += res_gamma_add.coeff(2);

            MYIF(derivs>0) {
                double norm_diff_7_inv = norm_diff_5_inv*norm_diff_2_inv;
                Vec3d res_grad_dgamma_by_dphi_add = Vec3d::Zero();
                Vec3d res_grad_gamma_add = Vec3d::Zero();

#pragma unroll
                for(int k=0; k<3; k++){
                    Vec3d ek = Vec3d::Zero();
                    ek[k] = 1.;
                    res_grad_dgamma_by_dphi_add += cross(k, vgrad_i[k]) * norm_diff_3_inv;
                    res_grad_dgamma_by_dphi_add -= cross(diff, vgrad_i[k]) * (diff[k] * norm_diff_5_inv_times_3);

                    res_grad_gamma_add += diff * (inner(cross(dgamma_j_by_dphi, k), vgrad_i[k]) * norm_diff_5_inv_times_3);
                    res_grad_gamma_add += ek * (inner(cross_dgamma_j_by_dphi_diff, vgrad_i[k]) * norm_diff_5_inv_times_3);
                    res_grad_gamma_add += cross(vgrad_i[k], dgamma_j_by_dphi) * (norm_diff_5_inv_times_3 * diff[k]);
                    res_grad_gamma_add -= diff * (15. * diff[k] * inner(cross_dgamma_j_by_dphi_diff, vgrad_i[k]) * norm_diff_7_inv);
                }
                res_grad_dgamma_by_dphi(j, 0) += res_grad_dgamma_by_dphi_add.coeff(0);
                res_grad_dgamma_by_dphi(j, 1) += res_grad_dgamma_by_dphi_add.coeff(1);
                res_grad_dgamma_by_dphi(j, 2) += res_grad_dgamma_by_dphi_add.coeff(2);
                res_grad_gamma(j, 0) += res_grad_gamma_add.coeff(0);
                res_grad_gamma(j, 1) += res_grad_gamma_add.coeff(1);
                res_grad_gamma(j, 2) += res_grad_gamma_add.coeff(2);
            }
        }
    }
}

#else

template<class T, int derivs>
void biot_savart_vjp_kernel(AlignedPaddedVec& pointsx, AlignedPaddedVec& pointsy, AlignedPaddedVec& pointsz,
            T& gamma, T& dgamma_by_dphi, T& v, T& res_gamma, T& res_dgamma_by_dphi, T& vgrad, T& res_grad_gamma,
            T& res_grad_dgamma_by_dphi) {
    if(gamma.layout() != xt::layout_type::row_major)
          throw std::runtime_error("gamma needs to be in row-major storage order");
    if(dgamma_by_dphi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("dgamma_by_dphi needs to be in row-major storage order");
    if(res_gamma.layout() != xt::layout_type::row_major)
          throw std::runtime_error("res_gamma needs to be in row-major storage order");
    if(res_dgamma_by_dphi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("res_dgamma_by_dphi needs to be in row-major storage order");
    if(res_grad_gamma.layout() != xt::layout_type::row_major)
          throw std::runtime_error("res_grad_gamma needs to be in row-major storage order");
    if(res_grad_dgamma_by_dphi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("res_grad_dgamma_by_dphi needs to be in row-major storage order");
    int num_points         = pointsx.size();
    int num_quad_points    = gamma.shape(0);
    double* gamma_j_ptr = &(gamma(0, 0));
    double* dgamma_j_by_dphi_ptr = &(dgamma_by_dphi(0, 0));
    double* res_dgamma_by_dphi_ptr = &(res_dgamma_by_dphi(0, 0));
    double* res_gamma_ptr = &(res_gamma(0, 0));
    double* res_grad_dgamma_by_dphi_ptr = &(res_grad_dgamma_by_dphi(0, 0));
    double* res_grad_gamma_ptr = &(res_grad_gamma(0, 0));
    for(int i = 0; i < num_points; i++) {
        Vec3dStd point_i = Vec3dStd(&(pointsx[i]), &(pointsy[i]), &(pointsz[i]));
        auto v_i   = Vec3dStd();
        auto vgrad_i = vector<Vec3dStd>{
                Vec3dStd(), Vec3dStd(), Vec3dStd()
            };
#pragma unroll
            for (int d = 0; d < 3; ++d) {
                v_i[d] = v(i, d);
                MYIF(derivs>0) {
#pragma unroll
                    for (int dd = 0; dd < 3; ++dd) {
                        vgrad_i[dd][d] = vgrad(i, dd, d);
                    }
                }
            }

        for (int j = 0; j < num_quad_points; ++j) {
            auto dgamma_j_by_dphi = Vec3d{ dgamma_j_by_dphi_ptr[3*j+0], dgamma_j_by_dphi_ptr[3*j+1], dgamma_j_by_dphi_ptr[3*j+2] };
            auto diff = point_i - Vec3dStd(gamma_j_ptr[3*j+0], gamma_j_ptr[3*j+1], gamma_j_ptr[3*j+2]);
            auto norm_diff_2 = normsq(diff);
            auto norm_diff_inv = rsqrt(norm_diff_2);
            auto norm_diff_2_inv = norm_diff_inv*norm_diff_inv;
            auto norm_diff_3_inv = norm_diff_2_inv*norm_diff_inv;
            auto norm_diff_5_inv = norm_diff_3_inv*norm_diff_2_inv;
            auto norm_diff_5_inv_times_3 = 3.*norm_diff_5_inv;

            auto res_dgamma_by_dphi_add = cross(diff, v_i) * norm_diff_3_inv;
            res_dgamma_by_dphi_ptr[3*j+0] += res_dgamma_by_dphi_add.x;
            res_dgamma_by_dphi_ptr[3*j+1] += res_dgamma_by_dphi_add.y;
            res_dgamma_by_dphi_ptr[3*j+2] += res_dgamma_by_dphi_add.z;

            auto cross_dgamma_j_by_dphi_diff = cross(dgamma_j_by_dphi, diff);
            auto res_gamma_add = cross(dgamma_j_by_dphi, v_i) * norm_diff_3_inv;
            res_gamma_add += diff * inner(cross_dgamma_j_by_dphi_diff, v_i) * (norm_diff_5_inv_times_3);
            res_gamma_ptr[3*j+0] += res_gamma_add.x;
            res_gamma_ptr[3*j+1] += res_gamma_add.y;
            res_gamma_ptr[3*j+2] += res_gamma_add.z;

            MYIF(derivs>0) {
                auto norm_diff_7_inv = norm_diff_5_inv*norm_diff_2_inv;
                auto res_grad_dgamma_by_dphi_add = Vec3dStd();
                auto res_grad_gamma_add = Vec3dStd();

#pragma unroll
                for(int k=0; k<3; k++){
                    auto eksimd = Vec3dStd();
                    eksimd[k] += 1.;
                    Vec3d ek = Vec3d::Zero();
                    ek[k] = 1.;
                    res_grad_dgamma_by_dphi_add += cross(k, vgrad_i[k]) * norm_diff_3_inv;
                    res_grad_dgamma_by_dphi_add -= cross(diff, vgrad_i[k]) * (diff[k] * norm_diff_5_inv_times_3);

                    res_grad_gamma_add += diff * (inner(cross(dgamma_j_by_dphi, ek), vgrad_i[k]) * norm_diff_5_inv_times_3);
                    res_grad_gamma_add += eksimd * (inner(cross_dgamma_j_by_dphi_diff, vgrad_i[k]) * norm_diff_5_inv_times_3);
                    res_grad_gamma_add += cross(vgrad_i[k], dgamma_j_by_dphi) * (norm_diff_5_inv_times_3 * diff[k]);
                    res_grad_gamma_add -= diff * (15. * diff[k] * inner(cross_dgamma_j_by_dphi_diff, vgrad_i[k]) * norm_diff_7_inv);
                }
                res_grad_dgamma_by_dphi_ptr[3*j+0] += res_grad_dgamma_by_dphi_add.x;
                res_grad_dgamma_by_dphi_ptr[3*j+1] += res_grad_dgamma_by_dphi_add.y;
                res_grad_dgamma_by_dphi_ptr[3*j+2] += res_grad_dgamma_by_dphi_add.z;
                res_grad_gamma_ptr[3*j+0] += res_grad_gamma_add.x;
                res_grad_gamma_ptr[3*j+1] += res_grad_gamma_add.y;
                res_grad_gamma_ptr[3*j+2] += res_grad_gamma_add.z;
            }
        }
    }
}

#endif

#if defined(USE_XSIMD)

template<class T, int derivs>
void biot_savart_vector_potential_vjp_kernel(
            AlignedPaddedVec& pointsx, AlignedPaddedVec& pointsy, AlignedPaddedVec& pointsz,
            T& gamma, T& dgamma_by_dphi, T& v, T& res_gamma, T& res_dgamma_by_dphi, T& vgrad, T& res_grad_gamma, T& res_grad_dgamma_by_dphi) {

    if(gamma.layout() != xt::layout_type::row_major)
          throw std::runtime_error("gamma needs to be in row-major storage order");
    if(dgamma_by_dphi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("dgamma_by_dphi needs to be in row-major storage order");
    if(res_gamma.layout() != xt::layout_type::row_major)
          throw std::runtime_error("res_gamma needs to be in row-major storage order");
    if(res_dgamma_by_dphi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("res_dgamma_by_dphi needs to be in row-major storage order");
    if(res_grad_gamma.layout() != xt::layout_type::row_major)
          throw std::runtime_error("res_grad_gamma needs to be in row-major storage order");
    if(res_grad_dgamma_by_dphi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("res_grad_dgamma_by_dphi needs to be in row-major storage order");
    int num_points         = pointsx.size();
    int num_quad_points    = gamma.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    double* gamma_j_ptr = &(gamma(0, 0));
    double* dgamma_j_by_dphi_ptr = &(dgamma_by_dphi(0, 0));
    double* res_dgamma_by_dphi_ptr = &(res_dgamma_by_dphi(0, 0));
    double* res_gamma_ptr = &(res_gamma(0, 0));
    double* res_grad_dgamma_by_dphi_ptr = &(res_grad_dgamma_by_dphi(0, 0));
    double* res_grad_gamma_ptr = &(res_grad_gamma(0, 0));
    for(int i = 0; i < num_points-num_points%simd_size; i += simd_size) {
        Vec3dSimd point_i = Vec3dSimd(&(pointsx[i]), &(pointsy[i]), &(pointsz[i]));
        auto v_i   = Vec3dSimd();
        auto vgrad_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, xs::default_arch::alignment()>>{
                Vec3dSimd(), Vec3dSimd(), Vec3dSimd()
            };
        // v/vgrad are interleaved (point, component[, component]) arrays and batches
        // have no per-lane write via operator[] anymore; gather the lanes into small
        // contiguous buffers first, then load each buffer into a batch in one shot.
        alignas(xs::default_arch::alignment()) double v_x[simd_size], v_y[simd_size], v_z[simd_size];
        alignas(xs::default_arch::alignment()) double vgrad_x[3][simd_size], vgrad_y[3][simd_size], vgrad_z[3][simd_size];
#pragma unroll
        for(int k=0; k<simd_size; k++){
            v_x[k] = v(i+k, 0);
            v_y[k] = v(i+k, 1);
            v_z[k] = v(i+k, 2);
            MYIF(derivs>0) {
#pragma unroll
                for (int dd = 0; dd < 3; ++dd) {
                    vgrad_x[dd][k] = vgrad(i+k, dd, 0);
                    vgrad_y[dd][k] = vgrad(i+k, dd, 1);
                    vgrad_z[dd][k] = vgrad(i+k, dd, 2);
                }
            }
        }
        v_i = Vec3dSimd(xs::load_aligned(v_x), xs::load_aligned(v_y), xs::load_aligned(v_z));
        MYIF(derivs>0) {
#pragma unroll
            for (int dd = 0; dd < 3; ++dd) {
                vgrad_i[dd] = Vec3dSimd(xs::load_aligned(vgrad_x[dd]), xs::load_aligned(vgrad_y[dd]), xs::load_aligned(vgrad_z[dd]));
            }
        }

        for (int j = 0; j < num_quad_points; ++j) {
            auto dgamma_j_by_dphi = Vec3d{ dgamma_j_by_dphi_ptr[3*j+0], dgamma_j_by_dphi_ptr[3*j+1], dgamma_j_by_dphi_ptr[3*j+2] };
            auto diff = point_i - Vec3dSimd(gamma_j_ptr[3*j+0], gamma_j_ptr[3*j+1], gamma_j_ptr[3*j+2]);
            auto norm_diff_2 = normsq(diff);
            auto norm_diff_inv = rsqrt(norm_diff_2);
            auto norm_diff_inv_3 = norm_diff_inv * norm_diff_inv * norm_diff_inv;

            auto res_dgamma_by_dphi_add = v_i * norm_diff_inv;
            res_dgamma_by_dphi_ptr[3*j+0] += xsimd::reduce_add(res_dgamma_by_dphi_add.x);
            res_dgamma_by_dphi_ptr[3*j+1] += xsimd::reduce_add(res_dgamma_by_dphi_add.y);
            res_dgamma_by_dphi_ptr[3*j+2] += xsimd::reduce_add(res_dgamma_by_dphi_add.z);

            auto vi_dot_dgamma_dphi_j = inner(v_i, dgamma_j_by_dphi); 
            auto res_gamma_add = diff * (vi_dot_dgamma_dphi_j * norm_diff_inv_3);
            res_gamma_ptr[3*j+0] += xsimd::reduce_add(res_gamma_add.x);
            res_gamma_ptr[3*j+1] += xsimd::reduce_add(res_gamma_add.y);
            res_gamma_ptr[3*j+2] += xsimd::reduce_add(res_gamma_add.z);

            MYIF(derivs>0) {
                auto norm_diff_inv_5 = norm_diff_inv_3 * norm_diff_inv * norm_diff_inv;
                auto res_grad_dgamma_by_dphi_add = Vec3dSimd();
                auto res_grad_gamma_add = Vec3dSimd();
#pragma unroll
                for(int k=0; k<3; k++){
                    res_grad_dgamma_by_dphi_add -= vgrad_i[k] * norm_diff_inv_3 * diff[k]  ;
                    res_grad_gamma_add -= diff * inner(vgrad_i[k], dgamma_j_by_dphi) * (3 * diff[k]) * norm_diff_inv_5;
                }

                res_grad_gamma_add.x += inner(vgrad_i[0], dgamma_j_by_dphi) * norm_diff_inv_3;
                res_grad_gamma_add.y += inner(vgrad_i[1], dgamma_j_by_dphi) * norm_diff_inv_3;
                res_grad_gamma_add.z += inner(vgrad_i[2], dgamma_j_by_dphi) * norm_diff_inv_3;
                res_grad_dgamma_by_dphi_ptr[3*j+0] += xsimd::reduce_add(res_grad_dgamma_by_dphi_add.x);
                res_grad_dgamma_by_dphi_ptr[3*j+1] += xsimd::reduce_add(res_grad_dgamma_by_dphi_add.y);
                res_grad_dgamma_by_dphi_ptr[3*j+2] += xsimd::reduce_add(res_grad_dgamma_by_dphi_add.z);
                res_grad_gamma_ptr[3*j+0] += xsimd::reduce_add(res_grad_gamma_add.x);
                res_grad_gamma_ptr[3*j+1] += xsimd::reduce_add(res_grad_gamma_add.y);
                res_grad_gamma_ptr[3*j+2] += xsimd::reduce_add(res_grad_gamma_add.z);
            }
        }
    }
    for (int i = num_points - num_points % simd_size; i < num_points; ++i) {
        auto point_i = Vec3d{pointsx[i], pointsy[i], pointsz[i]};
        
        Vec3d v_i   = Vec3d::Zero();
        auto vgrad_i = vector<Vec3d>{
            Vec3d::Zero(), Vec3d::Zero(), Vec3d::Zero()
            };
#pragma unroll
        for (int d = 0; d < 3; ++d) {
            v_i[d] = v(i, d);
            MYIF(derivs>0) {
                for (int dd = 0; dd < 3; ++dd) {
                    vgrad_i[dd][d] = vgrad(i, dd, d);
                }
            }
        }
        for (int j = 0; j < num_quad_points; ++j) {
            Vec3d diff = point_i - Vec3d{gamma_j_ptr[3*j+0], gamma_j_ptr[3*j+1], gamma_j_ptr[3*j+2]};
            Vec3d dgamma_j_by_dphi = Vec3d{dgamma_j_by_dphi_ptr[3*j+0], dgamma_j_by_dphi_ptr[3*j+1], dgamma_j_by_dphi_ptr[3*j+2]};
            auto norm_diff = norm(diff);
            auto norm_diff_inv = 1./norm_diff;
            auto norm_diff_inv_3 = norm_diff_inv * norm_diff_inv * norm_diff_inv;

            auto res_dgamma_by_dphi_add = v_i * norm_diff_inv;
            res_dgamma_by_dphi(j, 0) += res_dgamma_by_dphi_add.coeff(0);
            res_dgamma_by_dphi(j, 1) += res_dgamma_by_dphi_add.coeff(1);
            res_dgamma_by_dphi(j, 2) += res_dgamma_by_dphi_add.coeff(2);
            
            auto vi_dot_dgamma_dphi_j = inner(v_i, dgamma_j_by_dphi); 
            auto res_gamma_add = diff * (vi_dot_dgamma_dphi_j * norm_diff_inv_3);
            res_gamma(j, 0) += res_gamma_add.coeff(0);
            res_gamma(j, 1) += res_gamma_add.coeff(1);
            res_gamma(j, 2) += res_gamma_add.coeff(2);
             
            MYIF(derivs>0) {
                auto norm_diff_inv_5 = norm_diff_inv_3 * norm_diff_inv * norm_diff_inv;
                auto res_grad_dgamma_by_dphi_add = Vec3d{0.,0.,0.};
                auto res_grad_gamma_add = Vec3d{0.,0.,0.};
#pragma unroll
                for(int k=0; k<3; k++){
                    res_grad_dgamma_by_dphi_add -= vgrad_i[k] * norm_diff_inv_3 * diff[k]  ;
                    res_grad_gamma_add -= diff * inner(vgrad_i[k], dgamma_j_by_dphi) * (3 * diff[k]) * norm_diff_inv_5;
                }
                res_grad_gamma_add += Vec3d{inner(vgrad_i[0], dgamma_j_by_dphi), inner(vgrad_i[1], dgamma_j_by_dphi) , inner(vgrad_i[2], dgamma_j_by_dphi)}  * norm_diff_inv_3;

                res_grad_dgamma_by_dphi(j, 0) += res_grad_dgamma_by_dphi_add.coeff(0);
                res_grad_dgamma_by_dphi(j, 1) += res_grad_dgamma_by_dphi_add.coeff(1);
                res_grad_dgamma_by_dphi(j, 2) += res_grad_dgamma_by_dphi_add.coeff(2);
                res_grad_gamma(j, 0) += res_grad_gamma_add.coeff(0);
                res_grad_gamma(j, 1) += res_grad_gamma_add.coeff(1);
                res_grad_gamma(j, 2) += res_grad_gamma_add.coeff(2);
            }
        }
    }
}

#else

template<class T, int derivs>
void biot_savart_vector_potential_vjp_kernel(
            AlignedPaddedVec& pointsx, AlignedPaddedVec& pointsy, AlignedPaddedVec& pointsz,
            T& gamma, T& dgamma_by_dphi, T& v, T& res_gamma, T& res_dgamma_by_dphi, T& vgrad, T& res_grad_gamma, T& res_grad_dgamma_by_dphi) {

    if(gamma.layout() != xt::layout_type::row_major)
          throw std::runtime_error("gamma needs to be in row-major storage order");
    if(dgamma_by_dphi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("dgamma_by_dphi needs to be in row-major storage order");
    if(res_gamma.layout() != xt::layout_type::row_major)
          throw std::runtime_error("res_gamma needs to be in row-major storage order");
    if(res_dgamma_by_dphi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("res_dgamma_by_dphi needs to be in row-major storage order");
    if(res_grad_gamma.layout() != xt::layout_type::row_major)
          throw std::runtime_error("res_grad_gamma needs to be in row-major storage order");
    if(res_grad_dgamma_by_dphi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("res_grad_dgamma_by_dphi needs to be in row-major storage order");
    int num_points         = pointsx.size();
    int num_quad_points    = gamma.shape(0);
    double* gamma_j_ptr = &(gamma(0, 0));
    double* dgamma_j_by_dphi_ptr = &(dgamma_by_dphi(0, 0));
    double* res_dgamma_by_dphi_ptr = &(res_dgamma_by_dphi(0, 0));
    double* res_gamma_ptr = &(res_gamma(0, 0));
    double* res_grad_dgamma_by_dphi_ptr = &(res_grad_dgamma_by_dphi(0, 0));
    double* res_grad_gamma_ptr = &(res_grad_gamma(0, 0));
    for(int i = 0; i < num_points; i++) {
        Vec3dStd point_i = Vec3dStd(&(pointsx[i]), &(pointsy[i]), &(pointsz[i]));
        auto v_i   = Vec3dStd();
        auto vgrad_i = vector<Vec3dStd>{
                Vec3dStd(), Vec3dStd(), Vec3dStd()
            };
#pragma unroll
            for (int d = 0; d < 3; ++d) {
                v_i[d] = v(i, d);
                MYIF(derivs>0) {
#pragma unroll
                    for (int dd = 0; dd < 3; ++dd) {
                        vgrad_i[dd][d] = vgrad(i, dd, d);
                    }
                }
            }


        for (int j = 0; j < num_quad_points; ++j) {
            auto dgamma_j_by_dphi = Vec3d{ dgamma_j_by_dphi_ptr[3*j+0], dgamma_j_by_dphi_ptr[3*j+1], dgamma_j_by_dphi_ptr[3*j+2] };
            auto diff = point_i - Vec3dStd(gamma_j_ptr[3*j+0], gamma_j_ptr[3*j+1], gamma_j_ptr[3*j+2]);
            auto norm_diff_2 = normsq(diff);
            auto norm_diff_inv = rsqrt(norm_diff_2);
            auto norm_diff_inv_3 = norm_diff_inv * norm_diff_inv * norm_diff_inv;

            auto res_dgamma_by_dphi_add = v_i * norm_diff_inv;
            res_dgamma_by_dphi_ptr[3*j+0] += res_dgamma_by_dphi_add.x;
            res_dgamma_by_dphi_ptr[3*j+1] += res_dgamma_by_dphi_add.y;
            res_dgamma_by_dphi_ptr[3*j+2] += res_dgamma_by_dphi_add.z;

            auto vi_dot_dgamma_dphi_j = inner(v_i, dgamma_j_by_dphi);
            auto res_gamma_add = diff * (vi_dot_dgamma_dphi_j * norm_diff_inv_3);
            res_gamma_ptr[3*j+0] += res_gamma_add.x;
            res_gamma_ptr[3*j+1] += res_gamma_add.y;
            res_gamma_ptr[3*j+2] += res_gamma_add.z;

            MYIF(derivs>0) {
                auto norm_diff_inv_5 = norm_diff_inv_3 * norm_diff_inv * norm_diff_inv;
                auto res_grad_dgamma_by_dphi_add = Vec3dStd();
                auto res_grad_gamma_add = Vec3dStd();
#pragma unroll
                for(int k=0; k<3; k++){
                    res_grad_dgamma_by_dphi_add -= vgrad_i[k] * norm_diff_inv_3 * diff[k]  ;
                    res_grad_gamma_add -= diff * inner(vgrad_i[k], dgamma_j_by_dphi) * (3 * diff[k]) * norm_diff_inv_5;
                }

                res_grad_gamma_add.x += inner(vgrad_i[0], dgamma_j_by_dphi) * norm_diff_inv_3;
                res_grad_gamma_add.y += inner(vgrad_i[1], dgamma_j_by_dphi) * norm_diff_inv_3;
                res_grad_gamma_add.z += inner(vgrad_i[2], dgamma_j_by_dphi) * norm_diff_inv_3;

                res_grad_dgamma_by_dphi_ptr[3*j+0] += res_grad_dgamma_by_dphi_add.x;
                res_grad_dgamma_by_dphi_ptr[3*j+1] += res_grad_dgamma_by_dphi_add.y;
                res_grad_dgamma_by_dphi_ptr[3*j+2] += res_grad_dgamma_by_dphi_add.z;
                res_grad_gamma_ptr[3*j+0] += res_grad_gamma_add.x;
                res_grad_gamma_ptr[3*j+1] += res_grad_gamma_add.y;
                res_grad_gamma_ptr[3*j+2] += res_grad_gamma_add.z;
            }
        }
    }
}

#endif

// VJP kernel for \nabla\nabla B (the Hessian of the field with respect to the
// evaluation point). Given a seed `vgradgrad` with the same shape as
// d2B_by_dXdX (num_points, 3, 3, 3), this accumulates the vector Jacobian
// product with respect to the geometric quantities gamma and dgamma_by_dphi.
//
// For a single evaluation point x, single quadrature point (with gamma, and
// tangent G = dgamma_by_dphi), let d = x - gamma, r = |d|, and P = G x d. The
// (unscaled) forward second derivative reads
//
//   H_{k1,k2} = -3 r^-5 [ d_{k1} (G x e_{k2}) + d_{k2} (G x e_{k1}) + delta_{k1,k2} P ]
//               + 15 r^-7 d_{k1} d_{k2} P.
//
// With seed w = vgradgrad(i, k1, k2, :), we compute the contributions
//   d(w.H)/dG    -> res_dgamma_by_dphi   (H is linear in G)
//   d(w.H)/dgamma = -d(w.H)/dd -> res_gamma.
// The overall current/quadrature factor is applied by the caller.
#if defined(USE_XSIMD)

template<class T>
void biot_savart_gradgradB_vjp_kernel(AlignedPaddedVec& pointsx, AlignedPaddedVec& pointsy, AlignedPaddedVec& pointsz,
            T& gamma, T& dgamma_by_dphi, T& vgradgrad, T& res_gamma, T& res_dgamma_by_dphi) {
    if(gamma.layout() != xt::layout_type::row_major)
          throw std::runtime_error("gamma needs to be in row-major storage order");
    if(dgamma_by_dphi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("dgamma_by_dphi needs to be in row-major storage order");
    if(res_gamma.layout() != xt::layout_type::row_major)
          throw std::runtime_error("res_gamma needs to be in row-major storage order");
    if(res_dgamma_by_dphi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("res_dgamma_by_dphi needs to be in row-major storage order");
    int num_points         = pointsx.size();
    int num_quad_points    = gamma.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    double* gamma_j_ptr = &(gamma(0, 0));
    double* dgamma_j_by_dphi_ptr = &(dgamma_by_dphi(0, 0));
    double* res_dgamma_by_dphi_ptr = &(res_dgamma_by_dphi(0, 0));
    double* res_gamma_ptr = &(res_gamma(0, 0));
    for(int i = 0; i < num_points-num_points%simd_size; i += simd_size) {
        Vec3dSimd point_i = Vec3dSimd(&(pointsx[i]), &(pointsy[i]), &(pointsz[i]));
        // w_i[3*k1+k2] holds the seed vgradgrad(i, k1, k2, :), packed over the simd lanes
        auto w_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, XSIMD_DEFAULT_ALIGNMENT>>{
                Vec3dSimd(), Vec3dSimd(), Vec3dSimd(),
                Vec3dSimd(), Vec3dSimd(), Vec3dSimd(),
                Vec3dSimd(), Vec3dSimd(), Vec3dSimd()
            };
#pragma unroll
        for(int k=0; k<simd_size; k++){
            for (int k1 = 0; k1 < 3; ++k1) {
                for (int k2 = 0; k2 < 3; ++k2) {
                    for (int c = 0; c < 3; ++c) {
                        w_i[3*k1+k2][c][k] = vgradgrad(i+k, k1, k2, c);
                    }
                }
            }
        }

        for (int j = 0; j < num_quad_points; ++j) {
            auto G = Vec3d{ dgamma_j_by_dphi_ptr[3*j+0], dgamma_j_by_dphi_ptr[3*j+1], dgamma_j_by_dphi_ptr[3*j+2] };
            auto diff = point_i - Vec3dSimd(gamma_j_ptr[3*j+0], gamma_j_ptr[3*j+1], gamma_j_ptr[3*j+2]);
            auto norm_diff_2 = normsq(diff);
            auto norm_diff_inv = rsqrt(norm_diff_2);
            auto norm_diff_2_inv = norm_diff_inv*norm_diff_inv;
            auto norm_diff_5_inv = norm_diff_2_inv*norm_diff_2_inv*norm_diff_inv;
            auto norm_diff_7_inv = norm_diff_5_inv*norm_diff_2_inv;
            auto norm_diff_9_inv = norm_diff_7_inv*norm_diff_2_inv;
            auto p3_r5   = 3.*norm_diff_5_inv;
            auto m3_r5   = (-3.)*norm_diff_5_inv;
            auto p15_r7  = 15.*norm_diff_7_inv;
            auto p105_r9 = 105.*norm_diff_9_inv;

            // quantities that only depend on the quadrature point and the point index
            auto G_cross_diff = cross(G, diff);           // P = G x d
            Vec3d G_cross_e[3] = { cross(G, 0), cross(G, 1), cross(G, 2) };

            auto res_gamma_add = Vec3dSimd();
            auto res_dgamma_by_dphi_add = Vec3dSimd();
#pragma unroll
            for(int k1=0; k1<3; k1++){
                auto dk1 = diff[k1];
#pragma unroll
                for(int k2=0; k2<3; k2++){
                    auto& w = w_i[3*k1+k2];
                    auto dk2 = diff[k2];
                    double delta = (k1 == k2) ? 1. : 0.;

                    auto alpha1 = inner(G_cross_e[k1], w);    // (G x e_{k1}) . w
                    auto alpha2 = inner(G_cross_e[k2], w);    // (G x e_{k2}) . w
                    auto beta   = inner(G_cross_diff, w);     // (G x d)     . w
                    auto w_cross_G    = cross(w, G);          // w x G
                    auto diff_cross_w = cross(diff, w);       // d x w
                    auto k1_cross_w   = cross(k1, w);         // e_{k1} x w
                    auto k2_cross_w   = cross(k2, w);         // e_{k2} x w

                    // d(w.H)/dG
                    res_dgamma_by_dphi_add += k2_cross_w * (m3_r5 * dk1);
                    res_dgamma_by_dphi_add += k1_cross_w * (m3_r5 * dk2);
                    res_dgamma_by_dphi_add += diff_cross_w * (m3_r5 * delta + p15_r7 * dk1 * dk2);

                    // d(w.H)/dgamma = -d(w.H)/dd
                    auto scalar_d = (-p15_r7) * (alpha2*dk1 + alpha1*dk2 + delta*beta) + p105_r9 * dk1 * dk2 * beta;
                    auto coeff_ek1 = p3_r5*alpha2 - p15_r7*beta*dk2;
                    auto coeff_ek2 = p3_r5*alpha1 - p15_r7*beta*dk1;
                    auto coeff_wG  = p3_r5*delta  - p15_r7*dk1*dk2;
                    res_gamma_add += diff * scalar_d;
                    res_gamma_add += w_cross_G * coeff_wG;
                    res_gamma_add[k1] += coeff_ek1;
                    res_gamma_add[k2] += coeff_ek2;
                }
            }
            res_dgamma_by_dphi_ptr[3*j+0] += xsimd::hadd(res_dgamma_by_dphi_add.x);
            res_dgamma_by_dphi_ptr[3*j+1] += xsimd::hadd(res_dgamma_by_dphi_add.y);
            res_dgamma_by_dphi_ptr[3*j+2] += xsimd::hadd(res_dgamma_by_dphi_add.z);
            res_gamma_ptr[3*j+0] += xsimd::hadd(res_gamma_add.x);
            res_gamma_ptr[3*j+1] += xsimd::hadd(res_gamma_add.y);
            res_gamma_ptr[3*j+2] += xsimd::hadd(res_gamma_add.z);
        }
    }
    for (int i = num_points - num_points % simd_size; i < num_points; ++i) {
        auto point_i = Vec3d{pointsx[i], pointsy[i], pointsz[i]};
        Vec3d w_i[9];
        for (int k1 = 0; k1 < 3; ++k1)
            for (int k2 = 0; k2 < 3; ++k2)
                w_i[3*k1+k2] = Vec3d{vgradgrad(i, k1, k2, 0), vgradgrad(i, k1, k2, 1), vgradgrad(i, k1, k2, 2)};
        for (int j = 0; j < num_quad_points; ++j) {
            Vec3d G = Vec3d{dgamma_j_by_dphi_ptr[3*j+0], dgamma_j_by_dphi_ptr[3*j+1], dgamma_j_by_dphi_ptr[3*j+2]};
            Vec3d diff = point_i - Vec3d{gamma_j_ptr[3*j+0], gamma_j_ptr[3*j+1], gamma_j_ptr[3*j+2]};
            double norm_diff = norm(diff);
            double norm_diff_inv = 1/norm_diff;
            double norm_diff_2_inv = norm_diff_inv*norm_diff_inv;
            double norm_diff_5_inv = norm_diff_2_inv*norm_diff_2_inv*norm_diff_inv;
            double norm_diff_7_inv = norm_diff_5_inv*norm_diff_2_inv;
            double norm_diff_9_inv = norm_diff_7_inv*norm_diff_2_inv;
            double p3_r5   = 3.*norm_diff_5_inv;
            double m3_r5   = (-3.)*norm_diff_5_inv;
            double p15_r7  = 15.*norm_diff_7_inv;
            double p105_r9 = 105.*norm_diff_9_inv;

            Vec3d G_cross_diff = cross(G, diff);
            Vec3d G_cross_e[3] = { cross(G, 0), cross(G, 1), cross(G, 2) };

            Vec3d res_gamma_add = Vec3d::Zero();
            Vec3d res_dgamma_by_dphi_add = Vec3d::Zero();
#pragma unroll
            for(int k1=0; k1<3; k1++){
                double dk1 = diff[k1];
#pragma unroll
                for(int k2=0; k2<3; k2++){
                    Vec3d w = w_i[3*k1+k2];
                    double dk2 = diff[k2];
                    double delta = (k1 == k2) ? 1. : 0.;

                    double alpha1 = inner(G_cross_e[k1], w);
                    double alpha2 = inner(G_cross_e[k2], w);
                    double beta   = inner(G_cross_diff, w);
                    Vec3d w_cross_G    = cross(w, G);
                    Vec3d diff_cross_w = cross(diff, w);
                    Vec3d k1_cross_w   = cross(k1, w);
                    Vec3d k2_cross_w   = cross(k2, w);

                    res_dgamma_by_dphi_add += k2_cross_w * (m3_r5 * dk1);
                    res_dgamma_by_dphi_add += k1_cross_w * (m3_r5 * dk2);
                    res_dgamma_by_dphi_add += diff_cross_w * (m3_r5 * delta + p15_r7 * dk1 * dk2);

                    double scalar_d = (-p15_r7) * (alpha2*dk1 + alpha1*dk2 + delta*beta) + p105_r9 * dk1 * dk2 * beta;
                    double coeff_ek1 = p3_r5*alpha2 - p15_r7*beta*dk2;
                    double coeff_ek2 = p3_r5*alpha1 - p15_r7*beta*dk1;
                    double coeff_wG  = p3_r5*delta  - p15_r7*dk1*dk2;
                    res_gamma_add += diff * scalar_d;
                    res_gamma_add += w_cross_G * coeff_wG;
                    res_gamma_add[k1] += coeff_ek1;
                    res_gamma_add[k2] += coeff_ek2;
                }
            }
            res_dgamma_by_dphi(j, 0) += res_dgamma_by_dphi_add.coeff(0);
            res_dgamma_by_dphi(j, 1) += res_dgamma_by_dphi_add.coeff(1);
            res_dgamma_by_dphi(j, 2) += res_dgamma_by_dphi_add.coeff(2);
            res_gamma(j, 0) += res_gamma_add.coeff(0);
            res_gamma(j, 1) += res_gamma_add.coeff(1);
            res_gamma(j, 2) += res_gamma_add.coeff(2);
        }
    }
}

#else

template<class T>
void biot_savart_gradgradB_vjp_kernel(AlignedPaddedVec& pointsx, AlignedPaddedVec& pointsy, AlignedPaddedVec& pointsz,
            T& gamma, T& dgamma_by_dphi, T& vgradgrad, T& res_gamma, T& res_dgamma_by_dphi) {
    if(gamma.layout() != xt::layout_type::row_major)
          throw std::runtime_error("gamma needs to be in row-major storage order");
    if(dgamma_by_dphi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("dgamma_by_dphi needs to be in row-major storage order");
    if(res_gamma.layout() != xt::layout_type::row_major)
          throw std::runtime_error("res_gamma needs to be in row-major storage order");
    if(res_dgamma_by_dphi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("res_dgamma_by_dphi needs to be in row-major storage order");
    int num_points         = pointsx.size();
    int num_quad_points    = gamma.shape(0);
    double* gamma_j_ptr = &(gamma(0, 0));
    double* dgamma_j_by_dphi_ptr = &(dgamma_by_dphi(0, 0));
    double* res_dgamma_by_dphi_ptr = &(res_dgamma_by_dphi(0, 0));
    double* res_gamma_ptr = &(res_gamma(0, 0));
    for(int i = 0; i < num_points; i++) {
        Vec3dStd point_i = Vec3dStd(&(pointsx[i]), &(pointsy[i]), &(pointsz[i]));
        Vec3dStd w_i[9];
        for (int k1 = 0; k1 < 3; ++k1)
            for (int k2 = 0; k2 < 3; ++k2)
                w_i[3*k1+k2] = Vec3dStd(vgradgrad(i, k1, k2, 0), vgradgrad(i, k1, k2, 1), vgradgrad(i, k1, k2, 2));

        for (int j = 0; j < num_quad_points; ++j) {
            auto G = Vec3d{ dgamma_j_by_dphi_ptr[3*j+0], dgamma_j_by_dphi_ptr[3*j+1], dgamma_j_by_dphi_ptr[3*j+2] };
            auto diff = point_i - Vec3dStd(gamma_j_ptr[3*j+0], gamma_j_ptr[3*j+1], gamma_j_ptr[3*j+2]);
            auto norm_diff_2 = normsq(diff);
            auto norm_diff_inv = rsqrt(norm_diff_2);
            auto norm_diff_2_inv = norm_diff_inv*norm_diff_inv;
            auto norm_diff_5_inv = norm_diff_2_inv*norm_diff_2_inv*norm_diff_inv;
            auto norm_diff_7_inv = norm_diff_5_inv*norm_diff_2_inv;
            auto norm_diff_9_inv = norm_diff_7_inv*norm_diff_2_inv;
            auto p3_r5   = 3.*norm_diff_5_inv;
            auto m3_r5   = (-3.)*norm_diff_5_inv;
            auto p15_r7  = 15.*norm_diff_7_inv;
            auto p105_r9 = 105.*norm_diff_9_inv;

            auto G_cross_diff = cross(G, diff);
            Vec3d G_cross_e[3] = { cross(G, 0), cross(G, 1), cross(G, 2) };

            auto res_gamma_add = Vec3dStd();
            auto res_dgamma_by_dphi_add = Vec3dStd();
#pragma unroll
            for(int k1=0; k1<3; k1++){
                auto dk1 = diff[k1];
#pragma unroll
                for(int k2=0; k2<3; k2++){
                    auto& w = w_i[3*k1+k2];
                    auto dk2 = diff[k2];
                    double delta = (k1 == k2) ? 1. : 0.;

                    auto alpha1 = inner(G_cross_e[k1], w);
                    auto alpha2 = inner(G_cross_e[k2], w);
                    auto beta   = inner(G_cross_diff, w);
                    auto w_cross_G    = cross(w, G);
                    auto diff_cross_w = cross(diff, w);
                    auto k1_cross_w   = cross(k1, w);
                    auto k2_cross_w   = cross(k2, w);

                    res_dgamma_by_dphi_add += k2_cross_w * (m3_r5 * dk1);
                    res_dgamma_by_dphi_add += k1_cross_w * (m3_r5 * dk2);
                    res_dgamma_by_dphi_add += diff_cross_w * (m3_r5 * delta + p15_r7 * dk1 * dk2);

                    auto scalar_d = (-p15_r7) * (alpha2*dk1 + alpha1*dk2 + delta*beta) + p105_r9 * dk1 * dk2 * beta;
                    auto coeff_ek1 = p3_r5*alpha2 - p15_r7*beta*dk2;
                    auto coeff_ek2 = p3_r5*alpha1 - p15_r7*beta*dk1;
                    auto coeff_wG  = p3_r5*delta  - p15_r7*dk1*dk2;
                    res_gamma_add += diff * scalar_d;
                    res_gamma_add += w_cross_G * coeff_wG;
                    res_gamma_add[k1] += coeff_ek1;
                    res_gamma_add[k2] += coeff_ek2;
                }
            }
            res_dgamma_by_dphi_ptr[3*j+0] += res_dgamma_by_dphi_add.x;
            res_dgamma_by_dphi_ptr[3*j+1] += res_dgamma_by_dphi_add.y;
            res_dgamma_by_dphi_ptr[3*j+2] += res_dgamma_by_dphi_add.z;
            res_gamma_ptr[3*j+0] += res_gamma_add.x;
            res_gamma_ptr[3*j+1] += res_gamma_add.y;
            res_gamma_ptr[3*j+2] += res_gamma_add.z;
        }
    }
}

#endif
