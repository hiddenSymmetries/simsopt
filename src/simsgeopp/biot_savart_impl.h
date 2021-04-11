#include "simdhelpers.h"
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
void biot_savart_kernel(vector_type& pointsx, vector_type& pointsy, vector_type& pointsz, T& gamma, T& dgamma_by_dphi, T& B, T& dB_by_dX, T& d2B_by_dXdX) {
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
    double* gamma_j_ptr = &(gamma(0, 0));
    double* dgamma_j_by_dphi_ptr = &(dgamma_by_dphi(0, 0));
    for(int i = 0; i < num_points-num_points%simd_size; i += simd_size) {
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
                auto norm_diff = 1./norm_diff_inv;
                auto dgamma_by_dphi_j_simd_norm_diff = dgamma_by_dphi_j_simd * norm_diff;
#pragma unroll
                for(int k=0; k<3; k++) {
                    auto numerator1 = cross(dgamma_by_dphi_j_simd_norm_diff, k);
                    auto numerator2 = three_dgamma_by_dphi_cross_diff_by_norm_diff * diff[k];
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

        for(int j=0; j<simd_size; j++){
            B(i+j, 0) = B_i.x[j];
            B(i+j, 1) = B_i.y[j];
            B(i+j, 2) = B_i.z[j];
            MYIF(derivs > 0) {
                for(int k=0; k<3; k++) {
                    dB_by_dX(i+j, k, 0) = dB_dX_i[k].x[j];
                    dB_by_dX(i+j, k, 1) = dB_dX_i[k].y[j];
                    dB_by_dX(i+j, k, 2) = dB_dX_i[k].z[j];
                }
            }
            MYIF(derivs > 1) {
                for(int k1=0; k1<3; k1++) {
                    for(int k2=0; k2<=k1; k2++) {
                        d2B_by_dXdX(i+j, k1, k2, 0) = d2B_dXdX_i[3*k1 + k2].x[j];
                        d2B_by_dXdX(i+j, k1, k2, 1) = d2B_dXdX_i[3*k1 + k2].y[j];
                        d2B_by_dXdX(i+j, k1, k2, 2) = d2B_dXdX_i[3*k1 + k2].z[j];
                        if(k2 < k1){
                            d2B_by_dXdX(i+j, k2, k1, 0) = d2B_dXdX_i[3*k1 + k2].x[j];
                            d2B_by_dXdX(i+j, k2, k1, 1) = d2B_dXdX_i[3*k1 + k2].y[j];
                            d2B_by_dXdX(i+j, k2, k1, 2) = d2B_dXdX_i[3*k1 + k2].z[j];
                        }
                    }
                }
            }
        }
    }
    for (int i = num_points - num_points % simd_size; i < num_points; ++i) {
        auto point_i = Vec3d{pointsx[i], pointsy[i], pointsz[i]};
        B(i, 0) = 0;
        B(i, 1) = 0;
        B(i, 2) = 0;
        for (int j = 0; j < num_quad_points; ++j) {
            Vec3d diff = point_i - Vec3d{gamma_j_ptr[3*j+0], gamma_j_ptr[3*j+1], gamma_j_ptr[3*j+2]};
            Vec3d dgamma_by_dphi_j = Vec3d{dgamma_j_by_dphi_ptr[3*j+0], dgamma_j_by_dphi_ptr[3*j+1], dgamma_j_by_dphi_ptr[3*j+2]};
            double norm_diff_2 = diff.coeff(0)*diff.coeff(0) + diff.coeff(1)*diff.coeff(1) + diff.coeff(2)*diff.coeff(2);
            double norm_diff = sqrt(norm_diff_2);
            double norm_diff_inv = 1./norm_diff;
            double norm_diff_2_inv = norm_diff_inv*norm_diff_inv;
            double norm_diff_3_inv = norm_diff_2_inv*norm_diff_inv;
            Vec3d dgamma_by_dphi_j_cross_diff = cross(dgamma_by_dphi_j, diff);
            Vec3d B_i = dgamma_by_dphi_j_cross_diff*norm_diff_3_inv;

            B(i, 0) += B_i.coeff(0);
            B(i, 1) += B_i.coeff(1);
            B(i, 2) += B_i.coeff(2);
            MYIF(derivs > 0) {
                double norm_diff_4_inv = norm_diff_2_inv*norm_diff_2_inv;
                Vec3d three_dgamma_by_dphi_cross_diff_by_norm_diff = dgamma_by_dphi_j_cross_diff * 3 * norm_diff_inv;
                Vec3d dgamma_by_dphi_j_norm_diff = dgamma_by_dphi_j*norm_diff;
#pragma unroll
                for(int k=0; k<3; k++) {
                    Vec3d numerator1 = cross(dgamma_by_dphi_j_norm_diff, k);
                    Vec3d numerator2 = three_dgamma_by_dphi_cross_diff_by_norm_diff * diff[k];
                    Vec3d temp = (numerator1-numerator2) * norm_diff_4_inv;
                    dB_by_dX(i, k, 0) += temp.coeff(0);
                    dB_by_dX(i, k, 1) += temp.coeff(1);
                    dB_by_dX(i, k, 2) += temp.coeff(2);
                }
                MYIF(derivs > 1) {
                    double norm_diff_5_inv = norm_diff_4_inv*norm_diff_inv;
                    double norm_diff_7_inv = norm_diff_5_inv*norm_diff_2_inv;
#pragma unroll
                    for(int k1=0; k1<3; k1++) {
#pragma unroll
                        for(int k2=0; k2<3; k2++) {
                            Vec3d term1 = -3 * (diff[k1]*norm_diff_5_inv) * cross(dgamma_by_dphi_j, k2);
                            Vec3d term2 = -3 * (diff[k2]*norm_diff_5_inv) * cross(dgamma_by_dphi_j, k1);
                            Vec3d term3 = 15 * (diff[k1] * diff[k2] * norm_diff_7_inv) * dgamma_by_dphi_j_cross_diff;
                            Vec3d term4 = Vec3d{0., 0., 0.};
                            if(k1 == k2) {
                                term4 = -3 * norm_diff_5_inv * dgamma_by_dphi_j_cross_diff;
                            }
                            Vec3d temp = (term1 + term2 + term3 + term4);
                            d2B_by_dXdX(i, k1, k2, 0) += temp.coeff(0);
                            d2B_by_dXdX(i, k1, k2, 1) += temp.coeff(1);
                            d2B_by_dXdX(i, k1, k2, 2) += temp.coeff(2);
                        }
                    }
                }
            }
        }
    }
    double fak = (1e-7/num_quad_points);
    B *= fak;
    MYIF(derivs > 0)
        dB_by_dX *= fak;
    MYIF(derivs > 1)
        d2B_by_dXdX *= fak;
}
