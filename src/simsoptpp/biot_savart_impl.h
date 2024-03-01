#include "simdhelpers.h"
#include "vec3dsimd.h"
#include <stdexcept>
#include "xtensor/xlayout.hpp"


// When compiled with C++17, then we use `if constexpr` to check for
// derivatives that need to be computed.  These are actually evaluated at
// compile time, e.g. the compiler creates three different functions, one that
// only computes B, one that computes B and \nabla B, and one that computes B,
// \nabla B, and \nabla\nabla B.

#if __cplusplus >= 201703L
#define MYIF(c) if constexpr(c)
#else
#define MYIF(c) if(c)
#endif


template<class T, int derivs>
void biot_savart_kernel(AlignedPaddedVec& pointsx, AlignedPaddedVec& pointsy, AlignedPaddedVec& pointsz, T& gamma, T& dgamma_by_dphi, T& B, T& dB_by_dX, T& d2B_by_dXdX) {
    if(gamma.layout() != xt::layout_type::row_major)
          throw std::runtime_error("gamma needs to be in row-major storage order");
    if(dgamma_by_dphi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("dgamma_by_dphi needs to be in row-major storage order");
    int num_points         = pointsx.size();
    int num_quad_points    = gamma.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    auto dB_dX_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, XSIMD_DEFAULT_ALIGNMENT>>();
    MYIF(derivs > 0) {
        dB_dX_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, XSIMD_DEFAULT_ALIGNMENT>>{
            Vec3dSimd(), Vec3dSimd(), Vec3dSimd()
        };
    }
    auto d2B_dXdX_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, XSIMD_DEFAULT_ALIGNMENT>>();
    MYIF(derivs > 1) {
        d2B_dXdX_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, XSIMD_DEFAULT_ALIGNMENT>>{
            Vec3dSimd(), Vec3dSimd(), Vec3dSimd(), 
            Vec3dSimd(), Vec3dSimd(), Vec3dSimd(), 
            Vec3dSimd(), Vec3dSimd(), Vec3dSimd() 
        };
    }
    double fak = (1e-7/num_quad_points);
    double* gamma_j_ptr = &(gamma(0, 0));
    double* dgamma_j_by_dphi_ptr = &(dgamma_by_dphi(0, 0));
    // out vectors pointsx, pointsy, and pointsz are added and aligned, so we
    // don't have to worry about going out of bounds here
    for(int i = 0; i < num_points; i += simd_size) {
        auto point_i = Vec3dSimd(&(pointsx[i]), &(pointsy[i]), &(pointsz[i]));
        auto B_i   = Vec3dSimd();
        MYIF(derivs > 0) {
            dB_dX_i[0] *= 0.;
            dB_dX_i[1] *= 0.;
            dB_dX_i[2] *= 0.;
        }
        MYIF(derivs > 1) {
            d2B_dXdX_i[0] *= 0.; d2B_dXdX_i[1] *= 0.; d2B_dXdX_i[2] *= 0.;
            d2B_dXdX_i[3] *= 0.; d2B_dXdX_i[4] *= 0.; d2B_dXdX_i[5] *= 0.;
            d2B_dXdX_i[6] *= 0.; d2B_dXdX_i[7] *= 0.; d2B_dXdX_i[8] *= 0.;
        }
        for (int j = 0; j < num_quad_points; ++j) {
            auto diff = point_i - Vec3dSimd(gamma_j_ptr[3*j+0], gamma_j_ptr[3*j+1], gamma_j_ptr[3*j+2]);
            auto norm_diff_2     = normsq(diff);
            auto norm_diff_inv   = rsqrt(norm_diff_2);
            auto norm_diff_3_inv = norm_diff_inv*norm_diff_inv*norm_diff_inv;

            auto dgamma_by_dphi_j_simd = Vec3dSimd(dgamma_j_by_dphi_ptr[3*j+0], dgamma_j_by_dphi_ptr[3*j+1], dgamma_j_by_dphi_ptr[3*j+2]);
            auto dgamma_by_dphi_j_cross_diff = cross(dgamma_by_dphi_j_simd, diff);
            B_i.x = xsimd::fma(dgamma_by_dphi_j_cross_diff.x, norm_diff_3_inv, B_i.x);
            B_i.y = xsimd::fma(dgamma_by_dphi_j_cross_diff.y, norm_diff_3_inv, B_i.y);
            B_i.z = xsimd::fma(dgamma_by_dphi_j_cross_diff.z, norm_diff_3_inv, B_i.z);

            MYIF(derivs > 0) {
                auto norm_diff_4_inv = norm_diff_3_inv*norm_diff_inv;
                auto three_dgamma_by_dphi_cross_diff_by_norm_diff = dgamma_by_dphi_j_cross_diff*(3.*norm_diff_inv);
                auto norm_diff = norm_diff_2*norm_diff_inv;
                auto dgamma_by_dphi_j_simd_norm_diff = dgamma_by_dphi_j_simd * norm_diff;
#pragma unroll
                for(int k=0; k<3; k++) {
                    auto numerator1 = cross(dgamma_by_dphi_j_simd_norm_diff, k);
                    //auto numerator2 = three_dgamma_by_dphi_cross_diff_by_norm_diff * diff[k];
                    //auto temp = numerator2-numerator1;
                    //dB_dX_i[k].x = xsimd::fma(temp.x, norm_diff_4_inv, dB_dX_i[k].x);
                    //dB_dX_i[k].y = xsimd::fma(temp.y, norm_diff_4_inv, dB_dX_i[k].y);
                    //dB_dX_i[k].z = xsimd::fma(temp.z, norm_diff_4_inv, dB_dX_i[k].z);

                    auto tempx = xsimd::fnma(three_dgamma_by_dphi_cross_diff_by_norm_diff.x, diff[k], numerator1.x);
                    auto tempy = xsimd::fnma(three_dgamma_by_dphi_cross_diff_by_norm_diff.y, diff[k], numerator1.y);
                    auto tempz = xsimd::fnma(three_dgamma_by_dphi_cross_diff_by_norm_diff.z, diff[k], numerator1.z);
                    dB_dX_i[k].x = xsimd::fma(tempx, norm_diff_4_inv, dB_dX_i[k].x);
                    dB_dX_i[k].y = xsimd::fma(tempy, norm_diff_4_inv, dB_dX_i[k].y);
                    dB_dX_i[k].z = xsimd::fma(tempz, norm_diff_4_inv, dB_dX_i[k].z);
                }
                MYIF(derivs > 1) {
                    auto norm_diff_5_inv = norm_diff_4_inv*norm_diff_inv;;
                    auto norm_diff_7_inv = norm_diff_4_inv*norm_diff_3_inv;
                    auto term124fak = (-3.)*norm_diff_5_inv;
                    auto norm_diff_7_inv_15 = norm_diff_7_inv*15.;
#pragma unroll
                    for(int k1=0; k1<3; k1++) {
#pragma unroll
                        for(int k2=0; k2<=k1; k2++) {
                            auto term12 = cross(dgamma_by_dphi_j_simd, k2)*diff[k1];
                            auto dgamma_by_dphi_j_simd_cross_k1 = cross(dgamma_by_dphi_j_simd, k1);
                            term12.x = xsimd::fma(dgamma_by_dphi_j_simd_cross_k1.x, diff[k2], term12.x);
                            term12.y = xsimd::fma(dgamma_by_dphi_j_simd_cross_k1.y, diff[k2], term12.y);
                            term12.z = xsimd::fma(dgamma_by_dphi_j_simd_cross_k1.z, diff[k2], term12.z);
                            //term12 += cross(dgamma_by_dphi_j_simd, k1)*diff[k2];


                            d2B_dXdX_i[3*k1 + k2].x = xsimd::fma(term124fak, term12.x, d2B_dXdX_i[3*k1 + k2].x);
                            d2B_dXdX_i[3*k1 + k2].y = xsimd::fma(term124fak, term12.y, d2B_dXdX_i[3*k1 + k2].y);
                            d2B_dXdX_i[3*k1 + k2].z = xsimd::fma(term124fak, term12.z, d2B_dXdX_i[3*k1 + k2].z);

                            auto term3fak = diff[k1] * diff[k2] * norm_diff_7_inv_15;
                            if(k1 == k2) {
                                term3fak += term124fak;
                            }
                            d2B_dXdX_i[3*k1 + k2].x = xsimd::fma(term3fak, dgamma_by_dphi_j_cross_diff.x, d2B_dXdX_i[3*k1 + k2].x);
                            d2B_dXdX_i[3*k1 + k2].y = xsimd::fma(term3fak, dgamma_by_dphi_j_cross_diff.y, d2B_dXdX_i[3*k1 + k2].y);
                            d2B_dXdX_i[3*k1 + k2].z = xsimd::fma(term3fak, dgamma_by_dphi_j_cross_diff.z, d2B_dXdX_i[3*k1 + k2].z);
                        }
                    }
                }
            }
        }
        // in the last iteration of the loop over i, we might overshoot. e.g.
        // consider num_points=11.  then in the first two iterations we deal
        // with 0-3, 4-7, in the final iteration we only want 8-10, but have
        // actually computed for i from 8-11 since we always work on full simd
        // vectors. so we have to ignore those results. Disgarding the unneeded
        // entries is actually faster than falling back to scalar operations
        // (which would require treat i = 8, 9, 10 all individually).
        int jlimit = std::min(simd_size, num_points-i);
        for(int j=0; j<jlimit; j++){
            B(i+j, 0) = fak * B_i.x[j];
            B(i+j, 1) = fak * B_i.y[j];
            B(i+j, 2) = fak * B_i.z[j];
            MYIF(derivs > 0) {
                for(int k=0; k<3; k++) {
                    dB_by_dX(i+j, k, 0) = fak*dB_dX_i[k].x[j];
                    dB_by_dX(i+j, k, 1) = fak*dB_dX_i[k].y[j];
                    dB_by_dX(i+j, k, 2) = fak*dB_dX_i[k].z[j];
                }
            }
            MYIF(derivs > 1) {
                for(int k1=0; k1<3; k1++) {
                    for(int k2=0; k2<=k1; k2++) {
                        d2B_by_dXdX(i+j, k1, k2, 0) = fak*d2B_dXdX_i[3*k1 + k2].x[j];
                        d2B_by_dXdX(i+j, k1, k2, 1) = fak*d2B_dXdX_i[3*k1 + k2].y[j];
                        d2B_by_dXdX(i+j, k1, k2, 2) = fak*d2B_dXdX_i[3*k1 + k2].z[j];
                        if(k2 < k1){
                            d2B_by_dXdX(i+j, k2, k1, 0) = fak*d2B_dXdX_i[3*k1 + k2].x[j];
                            d2B_by_dXdX(i+j, k2, k1, 1) = fak*d2B_dXdX_i[3*k1 + k2].y[j];
                            d2B_by_dXdX(i+j, k2, k1, 2) = fak*d2B_dXdX_i[3*k1 + k2].z[j];
                        }
                    }
                }
            }
        }
    }
}



template<class T, int derivs>
void biot_savart_kernel_A(AlignedPaddedVec& pointsx, AlignedPaddedVec& pointsy, AlignedPaddedVec& pointsz, T& gamma, T& dgamma_by_dphi, T& A, T& dA_by_dX, T& d2A_by_dXdX) {
    if(gamma.layout() != xt::layout_type::row_major)
          throw std::runtime_error("gamma needs to be in row-major storage order");
    if(dgamma_by_dphi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("dgamma_by_dphi needs to be in row-major storage order");
    int num_points         = pointsx.size();
    int num_quad_points    = gamma.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    auto dA_dX_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, XSIMD_DEFAULT_ALIGNMENT>>();
    MYIF(derivs > 0) {
        dA_dX_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, XSIMD_DEFAULT_ALIGNMENT>>{
            Vec3dSimd(), Vec3dSimd(), Vec3dSimd()
        };
    }
    auto d2A_dXdX_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, XSIMD_DEFAULT_ALIGNMENT>>();
    MYIF(derivs > 1) {
        d2A_dXdX_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, XSIMD_DEFAULT_ALIGNMENT>>{
            Vec3dSimd(), Vec3dSimd(), Vec3dSimd(), 
            Vec3dSimd(), Vec3dSimd(), Vec3dSimd(), 
            Vec3dSimd(), Vec3dSimd(), Vec3dSimd() 
        };
    }
    double fak = (1e-7/num_quad_points);
    double* gamma_j_ptr = &(gamma(0, 0));
    double* dgamma_j_by_dphi_ptr = &(dgamma_by_dphi(0, 0));
    // out vectors pointsx, pointsy, and pointsz are added and aligned, so we
    // don't have to worry about going out of bounds here
    for(int i = 0; i < num_points; i += simd_size) {
        auto point_i = Vec3dSimd(&(pointsx[i]), &(pointsy[i]), &(pointsz[i]));
        auto A_i   = Vec3dSimd();
        MYIF(derivs > 0) {
            dA_dX_i[0] *= 0.;
            dA_dX_i[1] *= 0.;
            dA_dX_i[2] *= 0.;
        }
        MYIF(derivs > 1) {
            d2A_dXdX_i[0] *= 0.; d2A_dXdX_i[1] *= 0.; d2A_dXdX_i[2] *= 0.;
            d2A_dXdX_i[3] *= 0.; d2A_dXdX_i[4] *= 0.; d2A_dXdX_i[5] *= 0.;
            d2A_dXdX_i[6] *= 0.; d2A_dXdX_i[7] *= 0.; d2A_dXdX_i[8] *= 0.;
        }
        for (int j = 0; j < num_quad_points; ++j) {
            auto diff = point_i - Vec3dSimd(gamma_j_ptr[3*j+0], gamma_j_ptr[3*j+1], gamma_j_ptr[3*j+2]);
            auto norm_diff_2     = normsq(diff);
            auto norm_diff_inv   = rsqrt(norm_diff_2);
            auto norm_diff_3_inv = norm_diff_inv*norm_diff_inv*norm_diff_inv;

            auto dgamma_by_dphi_j_simd = Vec3dSimd(dgamma_j_by_dphi_ptr[3*j+0], dgamma_j_by_dphi_ptr[3*j+1], dgamma_j_by_dphi_ptr[3*j+2]);
            A_i.x = xsimd::fma(dgamma_by_dphi_j_simd.x , norm_diff_inv, A_i.x) ;
            A_i.y = xsimd::fma(dgamma_by_dphi_j_simd.y , norm_diff_inv, A_i.y) ;
            A_i.z = xsimd::fma(dgamma_by_dphi_j_simd.z , norm_diff_inv, A_i.z) ;

            MYIF(derivs > 0) {
#pragma unroll
                for(int k=0; k<3; k++) {
                    auto diffk_norm_diff_3_inv = norm_diff_3_inv * diff[k];
                    dA_dX_i[k].x = xsimd::fnma(dgamma_by_dphi_j_simd.x, diffk_norm_diff_3_inv, dA_dX_i[k].x);
                    dA_dX_i[k].y = xsimd::fnma(dgamma_by_dphi_j_simd.y, diffk_norm_diff_3_inv, dA_dX_i[k].y);
                    dA_dX_i[k].z = xsimd::fnma(dgamma_by_dphi_j_simd.z, diffk_norm_diff_3_inv, dA_dX_i[k].z);
                }
                MYIF(derivs > 1) {
                    auto term124fak = dgamma_by_dphi_j_simd;
                    auto fak5 = 3.*norm_diff_3_inv*norm_diff_inv*norm_diff_inv;
                    term124fak.x *= fak5;
                    term124fak.y *= fak5;
                    term124fak.z *= fak5;
#pragma unroll
                    for(int k1=0; k1<3; k1++) {
#pragma unroll
                        for(int k2=0; k2<=k1; k2++) {
                            auto term12 = diff[k1]*diff[k2];
                            d2A_dXdX_i[3*k1 + k2].x = xsimd::fma(term124fak.x, term12, d2A_dXdX_i[3*k1 + k2].x);
                            d2A_dXdX_i[3*k1 + k2].y = xsimd::fma(term124fak.y, term12, d2A_dXdX_i[3*k1 + k2].y);
                            d2A_dXdX_i[3*k1 + k2].z = xsimd::fma(term124fak.z, term12, d2A_dXdX_i[3*k1 + k2].z);

                            if(k1 == k2) {
                                d2A_dXdX_i[3*k1 + k2].x = xsimd::fnma(norm_diff_3_inv, dgamma_by_dphi_j_simd.x, d2A_dXdX_i[3*k1 + k2].x);
                                d2A_dXdX_i[3*k1 + k2].y = xsimd::fnma(norm_diff_3_inv, dgamma_by_dphi_j_simd.y, d2A_dXdX_i[3*k1 + k2].y);
                                d2A_dXdX_i[3*k1 + k2].z = xsimd::fnma(norm_diff_3_inv, dgamma_by_dphi_j_simd.z, d2A_dXdX_i[3*k1 + k2].z);
                            }
                        }
                    }
                }
            }
        }
        // in the last iteration of the loop over i, we might overshoot. e.g.
        // consider num_points=11.  then in the first two iterations we deal
        // with 0-3, 4-7, in the final iteration we only want 8-10, but have
        // actually computed for i from 8-11 since we always work on full simd
        // vectors. so we have to ignore those results. Disgarding the unneeded
        // entries is actually faster than falling back to scalar operations
        // (which would require treat i = 8, 9, 10 all individually).
        int jlimit = std::min(simd_size, num_points-i);
        for(int j=0; j<jlimit; j++){
            A(i+j, 0) = fak * A_i.x[j];
            A(i+j, 1) = fak * A_i.y[j];
            A(i+j, 2) = fak * A_i.z[j];
            MYIF(derivs > 0) {
                for(int k=0; k<3; k++) {
                    dA_by_dX(i+j, k, 0) = fak*dA_dX_i[k].x[j];
                    dA_by_dX(i+j, k, 1) = fak*dA_dX_i[k].y[j];
                    dA_by_dX(i+j, k, 2) = fak*dA_dX_i[k].z[j];
                }
            }
            MYIF(derivs > 1) {
                for(int k1=0; k1<3; k1++) {
                    for(int k2=0; k2<=k1; k2++) {
                        d2A_by_dXdX(i+j, k1, k2, 0) = fak*d2A_dXdX_i[3*k1 + k2].x[j];
                        d2A_by_dXdX(i+j, k1, k2, 1) = fak*d2A_dXdX_i[3*k1 + k2].y[j];
                        d2A_by_dXdX(i+j, k1, k2, 2) = fak*d2A_dXdX_i[3*k1 + k2].z[j];
                        if(k2 < k1){
                            d2A_by_dXdX(i+j, k2, k1, 0) = fak*d2A_dXdX_i[3*k1 + k2].x[j];
                            d2A_by_dXdX(i+j, k2, k1, 1) = fak*d2A_dXdX_i[3*k1 + k2].y[j];
                            d2A_by_dXdX(i+j, k2, k1, 2) = fak*d2A_dXdX_i[3*k1 + k2].z[j];
                        }
                    }
                }
            }
        }
    }
}
