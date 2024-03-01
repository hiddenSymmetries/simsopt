#include "simdhelpers.h"
#include "vec3dsimd.h"
#include <stdexcept>
#include "xtensor/xlayout.hpp"

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

#include <iostream>
template<class T, int derivs>
void biot_savart_vjp_kernel(AlignedPaddedVec& pointsx, AlignedPaddedVec& pointsy, AlignedPaddedVec& pointsz, T& gamma, T& dgamma_by_dphi, T& v, T& res_gamma, T& res_dgamma_by_dphi, T& vgrad, T& res_grad_gamma, T& res_grad_dgamma_by_dphi) {
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
        auto vgrad_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, XSIMD_DEFAULT_ALIGNMENT>>{
                Vec3dSimd(), Vec3dSimd(), Vec3dSimd()
            };
#pragma unroll
        for(int k=0; k<simd_size; k++){
            for (int d = 0; d < 3; ++d) {
                v_i[d][k] = v(i+k, d);
                MYIF(derivs>0) {
#pragma unroll
                    for (int dd = 0; dd < 3; ++dd) {
                        vgrad_i[dd][d][k] = vgrad(i+k, dd, d);
                    }
                }
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
            res_dgamma_by_dphi_ptr[3*j+0] += xsimd::hadd(res_dgamma_by_dphi_add.x);
            res_dgamma_by_dphi_ptr[3*j+1] += xsimd::hadd(res_dgamma_by_dphi_add.y);
            res_dgamma_by_dphi_ptr[3*j+2] += xsimd::hadd(res_dgamma_by_dphi_add.z);

            auto cross_dgamma_j_by_dphi_diff = cross(dgamma_j_by_dphi, diff);
            auto res_gamma_add = cross(dgamma_j_by_dphi, v_i) * norm_diff_3_inv;
            res_gamma_add += diff * inner(cross_dgamma_j_by_dphi_diff, v_i) * (norm_diff_5_inv_times_3);
            res_gamma_ptr[3*j+0] += xsimd::hadd(res_gamma_add.x);
            res_gamma_ptr[3*j+1] += xsimd::hadd(res_gamma_add.y);
            res_gamma_ptr[3*j+2] += xsimd::hadd(res_gamma_add.z);

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
                res_grad_dgamma_by_dphi_ptr[3*j+0] += xsimd::hadd(res_grad_dgamma_by_dphi_add.x);
                res_grad_dgamma_by_dphi_ptr[3*j+1] += xsimd::hadd(res_grad_dgamma_by_dphi_add.y);
                res_grad_dgamma_by_dphi_ptr[3*j+2] += xsimd::hadd(res_grad_dgamma_by_dphi_add.z);
                res_grad_gamma_ptr[3*j+0] += xsimd::hadd(res_grad_gamma_add.x);
                res_grad_gamma_ptr[3*j+1] += xsimd::hadd(res_grad_gamma_add.y);
                res_grad_gamma_ptr[3*j+2] += xsimd::hadd(res_grad_gamma_add.z);
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


template<class T, int derivs>
void biot_savart_vector_potential_vjp_kernel(AlignedPaddedVec& pointsx, AlignedPaddedVec& pointsy, AlignedPaddedVec& pointsz, T& gamma, T& dgamma_by_dphi, T& v, T& res_gamma, T& res_dgamma_by_dphi, T& vgrad, T& res_grad_gamma, T& res_grad_dgamma_by_dphi) {

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
        auto vgrad_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, XSIMD_DEFAULT_ALIGNMENT>>{
                Vec3dSimd(), Vec3dSimd(), Vec3dSimd()
            };
#pragma unroll
        for(int k=0; k<simd_size; k++){
            for (int d = 0; d < 3; ++d) {
                v_i[d][k] = v(i+k, d);
                MYIF(derivs>0) {
#pragma unroll
                    for (int dd = 0; dd < 3; ++dd) {
                        vgrad_i[dd][d][k] = vgrad(i+k, dd, d);
                    }
                }
            }
        }
        

        for (int j = 0; j < num_quad_points; ++j) {
            auto dgamma_j_by_dphi = Vec3d{ dgamma_j_by_dphi_ptr[3*j+0], dgamma_j_by_dphi_ptr[3*j+1], dgamma_j_by_dphi_ptr[3*j+2] };
            auto diff = point_i - Vec3dSimd(gamma_j_ptr[3*j+0], gamma_j_ptr[3*j+1], gamma_j_ptr[3*j+2]);
            auto norm_diff_2 = normsq(diff);
            auto norm_diff_inv = rsqrt(norm_diff_2);
            auto norm_diff_inv_3 = norm_diff_inv * norm_diff_inv * norm_diff_inv;

            auto res_dgamma_by_dphi_add = v_i * norm_diff_inv;
            res_dgamma_by_dphi_ptr[3*j+0] += xsimd::hadd(res_dgamma_by_dphi_add.x);
            res_dgamma_by_dphi_ptr[3*j+1] += xsimd::hadd(res_dgamma_by_dphi_add.y);
            res_dgamma_by_dphi_ptr[3*j+2] += xsimd::hadd(res_dgamma_by_dphi_add.z);
            
            auto vi_dot_dgamma_dphi_j = inner(v_i, dgamma_j_by_dphi); 
            auto res_gamma_add = diff * (vi_dot_dgamma_dphi_j * norm_diff_inv_3);
            res_gamma_ptr[3*j+0] += xsimd::hadd(res_gamma_add.x);
            res_gamma_ptr[3*j+1] += xsimd::hadd(res_gamma_add.y);
            res_gamma_ptr[3*j+2] += xsimd::hadd(res_gamma_add.z);

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
                
                res_grad_dgamma_by_dphi_ptr[3*j+0] += xsimd::hadd(res_grad_dgamma_by_dphi_add.x);
                res_grad_dgamma_by_dphi_ptr[3*j+1] += xsimd::hadd(res_grad_dgamma_by_dphi_add.y);
                res_grad_dgamma_by_dphi_ptr[3*j+2] += xsimd::hadd(res_grad_dgamma_by_dphi_add.z);
                res_grad_gamma_ptr[3*j+0] += xsimd::hadd(res_grad_gamma_add.x);
                res_grad_gamma_ptr[3*j+1] += xsimd::hadd(res_grad_gamma_add.y);
                res_grad_gamma_ptr[3*j+2] += xsimd::hadd(res_grad_gamma_add.z);
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
