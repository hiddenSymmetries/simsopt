#include "simdhelpers.h"
#include "vec3dsimd.h"
#include <stdexcept>
#include "xtensor/core/xlayout.hpp"

using namespace std;
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

#if defined(USE_XSIMD)

template<class T, int derivs>
void biot_savart_kernel(AlignedPaddedVec& pointsx, AlignedPaddedVec& pointsy, AlignedPaddedVec& pointsz,
            T& gamma, T& dgamma_by_dphi, T& B, T& dB_by_dX, T& d2B_by_dXdX, T& d3B_by_dXdXdX) {
    if(gamma.layout() != xt::layout_type::row_major)
          throw std::runtime_error("gamma needs to be in row-major storage order");
    if(dgamma_by_dphi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("dgamma_by_dphi needs to be in row-major storage order");
    int num_points         = pointsx.size();
    int num_quad_points    = gamma.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    auto dB_dX_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, xs::default_arch::alignment()>>();
    MYIF(derivs > 0) {
        dB_dX_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, xs::default_arch::alignment()>>{
            Vec3dSimd(), Vec3dSimd(), Vec3dSimd()
        };
    }
    auto d2B_dXdX_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, xs::default_arch::alignment()>>();
    MYIF(derivs > 1) {
        d2B_dXdX_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, xs::default_arch::alignment()>>{
            Vec3dSimd(), Vec3dSimd(), Vec3dSimd(),
            Vec3dSimd(), Vec3dSimd(), Vec3dSimd(),
            Vec3dSimd(), Vec3dSimd(), Vec3dSimd()
        };
    }
    // d3B_dXdXdX_i holds the 27 entries of the third derivative tensor, indexed
    // as 9*k1 + 3*k2 + k3. Only the entries with k1>=k2>=k3 are accumulated (the
    // tensor is fully symmetric in its three derivative indices); the remaining
    // permutations are filled in during the write-out below.
    auto d3B_dXdXdX_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, XSIMD_DEFAULT_ALIGNMENT>>();
    MYIF(derivs > 2) {
        d3B_dXdXdX_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, XSIMD_DEFAULT_ALIGNMENT>>(27, Vec3dSimd());
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
        MYIF(derivs > 2) {
            for(int q=0; q<27; q++)
                d3B_dXdXdX_i[q] *= 0.;
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
                    MYIF(derivs > 2) {
                        auto norm_diff_9_inv = norm_diff_7_inv*norm_diff_inv*norm_diff_inv;
                        auto fak7 = norm_diff_7_inv*15.;
                        auto fak5 = norm_diff_5_inv*(-3.);
                        auto fak9 = norm_diff_9_inv*(-105.);
                        // c_k = dgamma_by_dphi x e_k, the derivative of dgamma_by_dphi x diff w.r.t. x_k
                        Vec3dSimd ck[3] = {
                            cross(dgamma_by_dphi_j_simd, 0),
                            cross(dgamma_by_dphi_j_simd, 1),
                            cross(dgamma_by_dphi_j_simd, 2)
                        };
#pragma unroll
                        for(int k1=0; k1<3; k1++) {
#pragma unroll
                            for(int k2=0; k2<=k1; k2++) {
#pragma unroll
                                for(int k3=0; k3<=k2; k3++) {
                                    auto coef_ck1 = fak7*diff[k2]*diff[k3];
                                    auto coef_ck2 = fak7*diff[k1]*diff[k3];
                                    auto coef_ck3 = fak7*diff[k1]*diff[k2];
                                    if(k2==k3) coef_ck1 = coef_ck1 + fak5;
                                    if(k1==k3) coef_ck2 = coef_ck2 + fak5;
                                    if(k1==k2) coef_ck3 = coef_ck3 + fak5;
                                    auto coef_c = fak9*diff[k1]*diff[k2]*diff[k3];
                                    if(k1==k2) coef_c = coef_c + fak7*diff[k3];
                                    if(k1==k3) coef_c = coef_c + fak7*diff[k2];
                                    if(k2==k3) coef_c = coef_c + fak7*diff[k1];
                                    d3B_dXdXdX_i[9*k1 + 3*k2 + k3] += ck[k1]*coef_ck1 + ck[k2]*coef_ck2 + ck[k3]*coef_ck3 + dgamma_by_dphi_j_cross_diff*coef_c;
                                }
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
        // B/dB_by_dX/d2B_by_dXdX are interleaved (point, component) arrays, so a batch can't
        // be stored directly into them; store each Vec3dSimd component to a small contiguous
        // buffer once, then index that buffer per lane below.
        alignas(xs::default_arch::alignment()) double B_i_x[simd_size], B_i_y[simd_size], B_i_z[simd_size];
        B_i.store_aligned(B_i_x, B_i_y, B_i_z);

        alignas(xs::default_arch::alignment()) double dB_dX_i_x[3][simd_size], dB_dX_i_y[3][simd_size], dB_dX_i_z[3][simd_size];
        MYIF(derivs > 0) {
            for(int k=0; k<3; k++)
                dB_dX_i[k].store_aligned(dB_dX_i_x[k], dB_dX_i_y[k], dB_dX_i_z[k]);
        }

        alignas(xs::default_arch::alignment()) double d2B_dXdX_i_x[9][simd_size], d2B_dXdX_i_y[9][simd_size], d2B_dXdX_i_z[9][simd_size];
        MYIF(derivs > 1) {
            for(int idx=0; idx<9; idx++)
                d2B_dXdX_i[idx].store_aligned(d2B_dXdX_i_x[idx], d2B_dXdX_i_y[idx], d2B_dXdX_i_z[idx]);
        }

        int jlimit = std::min(simd_size, num_points-i);
        for(int j=0; j<jlimit; j++){
            B(i+j, 0) = fak * B_i_x[j];
            B(i+j, 1) = fak * B_i_y[j];
            B(i+j, 2) = fak * B_i_z[j];
            MYIF(derivs > 0) {
                for(int k=0; k<3; k++) {
                    dB_by_dX(i+j, k, 0) = fak*dB_dX_i_x[k][j];
                    dB_by_dX(i+j, k, 1) = fak*dB_dX_i_y[k][j];
                    dB_by_dX(i+j, k, 2) = fak*dB_dX_i_z[k][j];
                }
            }
            MYIF(derivs > 1) {
                for(int k1=0; k1<3; k1++) {
                    for(int k2=0; k2<=k1; k2++) {
                        int idx = 3*k1 + k2;
                        d2B_by_dXdX(i+j, k1, k2, 0) = fak*d2B_dXdX_i_x[idx][j];
                        d2B_by_dXdX(i+j, k1, k2, 1) = fak*d2B_dXdX_i_y[idx][j];
                        d2B_by_dXdX(i+j, k1, k2, 2) = fak*d2B_dXdX_i_z[idx][j];
                        if(k2 < k1){
                            d2B_by_dXdX(i+j, k2, k1, 0) = fak*d2B_dXdX_i_x[idx][j];
                            d2B_by_dXdX(i+j, k2, k1, 1) = fak*d2B_dXdX_i_y[idx][j];
                            d2B_by_dXdX(i+j, k2, k1, 2) = fak*d2B_dXdX_i_z[idx][j];
                        }
                    }
                }
            }
            MYIF(derivs > 2) {
                for(int k1=0; k1<3; k1++) {
                    for(int k2=0; k2<=k1; k2++) {
                        for(int k3=0; k3<=k2; k3++) {
                            double vx = fak*d3B_dXdXdX_i[9*k1 + 3*k2 + k3].x[j];
                            double vy = fak*d3B_dXdXdX_i[9*k1 + 3*k2 + k3].y[j];
                            double vz = fak*d3B_dXdXdX_i[9*k1 + 3*k2 + k3].z[j];
                            // the third derivative is fully symmetric, so scatter the
                            // value computed for the sorted triple (k1>=k2>=k3) to all
                            // permutations (duplicate writes for repeated indices are
                            // harmless as the value is identical).
                            int perms[6][3] = {{k1,k2,k3},{k1,k3,k2},{k2,k1,k3},{k2,k3,k1},{k3,k1,k2},{k3,k2,k1}};
                            for(int m=0; m<6; m++) {
                                d3B_by_dXdXdX(i+j, perms[m][0], perms[m][1], perms[m][2], 0) = vx;
                                d3B_by_dXdXdX(i+j, perms[m][0], perms[m][1], perms[m][2], 1) = vy;
                                d3B_by_dXdXdX(i+j, perms[m][0], perms[m][1], perms[m][2], 2) = vz;
                            }
                        }
                    }
                }
            }
        }
    }
}

#else

template<class T, int derivs>
void biot_savart_kernel(AlignedPaddedVec& pointsx, AlignedPaddedVec& pointsy, AlignedPaddedVec& pointsz,
            T& gamma, T& dgamma_by_dphi, T& B, T& dB_by_dX, T& d2B_by_dXdX, T& d3B_by_dXdXdX) {
    if(gamma.layout() != xt::layout_type::row_major)
          throw std::runtime_error("gamma needs to be in row-major storage order");
    if(dgamma_by_dphi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("dgamma_by_dphi needs to be in row-major storage order");
    int num_points         = pointsx.size();
    int num_quad_points    = gamma.shape(0);
    auto dB_dX_i = vector<Vec3dStd>();
    MYIF(derivs > 0) {
        dB_dX_i = vector<Vec3dStd>{
            Vec3dStd(), Vec3dStd(), Vec3dStd()
        };
    }
    auto d2B_dXdX_i = vector<Vec3dStd>();
    MYIF(derivs > 1) {
        d2B_dXdX_i = vector<Vec3dStd>{
            Vec3dStd(), Vec3dStd(), Vec3dStd(),
            Vec3dStd(), Vec3dStd(), Vec3dStd(),
            Vec3dStd(), Vec3dStd(), Vec3dStd()
        };
    }
    // d3B_dXdXdX_i holds the 27 entries of the third derivative tensor, indexed
    // as 9*k1 + 3*k2 + k3. Only the entries with k1>=k2>=k3 are accumulated (the
    // tensor is fully symmetric in its three derivative indices); the remaining
    // permutations are filled in during the write-out below.
    auto d3B_dXdXdX_i = vector<Vec3dStd>();
    MYIF(derivs > 2) {
        d3B_dXdXdX_i = vector<Vec3dStd>(27, Vec3dStd());
    }
    double fak = (1e-7/num_quad_points);
    double* gamma_j_ptr = &(gamma(0, 0));
    double* dgamma_j_by_dphi_ptr = &(dgamma_by_dphi(0, 0));
    // out vectors pointsx, pointsy, and pointsz are added and aligned, so we
    // don't have to worry about going out of bounds here
    for(int i = 0; i < num_points; i++) {
        auto point_i = Vec3dStd(&(pointsx[i]), &(pointsy[i]), &(pointsz[i]));
        auto B_i   = Vec3dStd();
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
        MYIF(derivs > 2) {
            for(int q=0; q<27; q++)
                d3B_dXdXdX_i[q] *= 0.;
        }
       // #pragma omp simd aligned(pointsx, pointsy, pointsz: 64)
        for (int j = 0; j < num_quad_points; ++j) {
            auto diff = point_i - Vec3dStd(gamma_j_ptr[3*j+0], gamma_j_ptr[3*j+1], gamma_j_ptr[3*j+2]);
            auto norm_diff_2     = normsq(diff);
            auto norm_diff_inv   = rsqrt(norm_diff_2);
            auto norm_diff_3_inv = norm_diff_inv*norm_diff_inv*norm_diff_inv;

            auto dgamma_by_dphi_j_simd = Vec3dStd(dgamma_j_by_dphi_ptr[3*j+0], dgamma_j_by_dphi_ptr[3*j+1], dgamma_j_by_dphi_ptr[3*j+2]);
            auto dgamma_by_dphi_j_cross_diff = cross(dgamma_by_dphi_j_simd, diff);

            B_i += (dgamma_by_dphi_j_cross_diff * norm_diff_3_inv);

            MYIF(derivs > 0) {
                auto norm_diff_4_inv = norm_diff_3_inv*norm_diff_inv;
                auto three_dgamma_by_dphi_cross_diff_by_norm_diff = dgamma_by_dphi_j_cross_diff*(3.*norm_diff_inv);
                auto norm_diff = norm_diff_2*norm_diff_inv;
                auto dgamma_by_dphi_j_simd_norm_diff = dgamma_by_dphi_j_simd * norm_diff;
#pragma unroll
                for(int k=0; k<3; k++) {
                    auto numerator1 = cross(dgamma_by_dphi_j_simd_norm_diff, k);

                    auto temp = numerator1 - three_dgamma_by_dphi_cross_diff_by_norm_diff * diff[k];
                    dB_dX_i[k] += temp * norm_diff_4_inv;

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
                            term12 += dgamma_by_dphi_j_simd_cross_k1 * diff[k2];
                            d2B_dXdX_i[3*k1 + k2] += term124fak * term12;

                            auto term3fak = diff[k1] * diff[k2] * norm_diff_7_inv_15;
                            if(k1 == k2) {
                                term3fak += term124fak;
                            }
                            d2B_dXdX_i[3*k1 + k2] += term3fak * dgamma_by_dphi_j_cross_diff;
                        }
                    }
                    MYIF(derivs > 2) {
                        auto norm_diff_9_inv = norm_diff_7_inv*norm_diff_inv*norm_diff_inv;
                        auto fak7 = norm_diff_7_inv*15.;
                        auto fak5 = norm_diff_5_inv*(-3.);
                        auto fak9 = norm_diff_9_inv*(-105.);
                        // c_k = dgamma_by_dphi x e_k, the derivative of dgamma_by_dphi x diff w.r.t. x_k
                        Vec3dStd ck[3] = {
                            cross(dgamma_by_dphi_j_simd, 0),
                            cross(dgamma_by_dphi_j_simd, 1),
                            cross(dgamma_by_dphi_j_simd, 2)
                        };
#pragma unroll
                        for(int k1=0; k1<3; k1++) {
#pragma unroll
                            for(int k2=0; k2<=k1; k2++) {
#pragma unroll
                                for(int k3=0; k3<=k2; k3++) {
                                    auto coef_ck1 = fak7*diff[k2]*diff[k3];
                                    auto coef_ck2 = fak7*diff[k1]*diff[k3];
                                    auto coef_ck3 = fak7*diff[k1]*diff[k2];
                                    if(k2==k3) coef_ck1 += fak5;
                                    if(k1==k3) coef_ck2 += fak5;
                                    if(k1==k2) coef_ck3 += fak5;
                                    auto coef_c = fak9*diff[k1]*diff[k2]*diff[k3];
                                    if(k1==k2) coef_c += fak7*diff[k3];
                                    if(k1==k3) coef_c += fak7*diff[k2];
                                    if(k2==k3) coef_c += fak7*diff[k1];
                                    d3B_dXdXdX_i[9*k1 + 3*k2 + k3] += ck[k1]*coef_ck1 + ck[k2]*coef_ck2 + ck[k3]*coef_ck3 + dgamma_by_dphi_j_cross_diff*coef_c;
                                }
                            }
                        }
                    }
                }
            }
        }

        B(i, 0) = fak * B_i.x;
        B(i, 1) = fak * B_i.y;
        B(i, 2) = fak * B_i.z;
        MYIF(derivs > 0) {
            for(int k=0; k<3; k++) {
                dB_by_dX(i, k, 0) = fak*dB_dX_i[k].x;
                dB_by_dX(i, k, 1) = fak*dB_dX_i[k].y;
                dB_by_dX(i, k, 2) = fak*dB_dX_i[k].z;
            }
        }
        MYIF(derivs > 1) {
            for(int k1=0; k1<3; k1++) {
                for(int k2=0; k2<=k1; k2++) {
                    d2B_by_dXdX(i, k1, k2, 0) = fak*d2B_dXdX_i[3*k1 + k2].x;
                    d2B_by_dXdX(i, k1, k2, 1) = fak*d2B_dXdX_i[3*k1 + k2].y;
                    d2B_by_dXdX(i, k1, k2, 2) = fak*d2B_dXdX_i[3*k1 + k2].z;
                    if(k2 < k1){
                        d2B_by_dXdX(i, k2, k1, 0) = fak*d2B_dXdX_i[3*k1 + k2].x;
                        d2B_by_dXdX(i, k2, k1, 1) = fak*d2B_dXdX_i[3*k1 + k2].y;
                        d2B_by_dXdX(i, k2, k1, 2) = fak*d2B_dXdX_i[3*k1 + k2].z;
                    }
                }
            }
        }
        MYIF(derivs > 2) {
            for(int k1=0; k1<3; k1++) {
                for(int k2=0; k2<=k1; k2++) {
                    for(int k3=0; k3<=k2; k3++) {
                        double vx = fak*d3B_dXdXdX_i[9*k1 + 3*k2 + k3].x;
                        double vy = fak*d3B_dXdXdX_i[9*k1 + 3*k2 + k3].y;
                        double vz = fak*d3B_dXdXdX_i[9*k1 + 3*k2 + k3].z;
                        // the third derivative is fully symmetric, so scatter the
                        // value computed for the sorted triple (k1>=k2>=k3) to all
                        // permutations (duplicate writes for repeated indices are
                        // harmless as the value is identical).
                        int perms[6][3] = {{k1,k2,k3},{k1,k3,k2},{k2,k1,k3},{k2,k3,k1},{k3,k1,k2},{k3,k2,k1}};
                        for(int m=0; m<6; m++) {
                            d3B_by_dXdXdX(i, perms[m][0], perms[m][1], perms[m][2], 0) = vx;
                            d3B_by_dXdXdX(i, perms[m][0], perms[m][1], perms[m][2], 1) = vy;
                            d3B_by_dXdXdX(i, perms[m][0], perms[m][1], perms[m][2], 2) = vz;
                        }
                    }
                }
            }
        }
    }
}
#endif

#if defined(USE_XSIMD)

template<class T, int derivs>
void biot_savart_kernel_A(AlignedPaddedVec& pointsx, AlignedPaddedVec& pointsy, AlignedPaddedVec& pointsz,
            T& gamma, T& dgamma_by_dphi, T& A, T& dA_by_dX, T& d2A_by_dXdX) {
    if(gamma.layout() != xt::layout_type::row_major)
          throw std::runtime_error("gamma needs to be in row-major storage order");
    if(dgamma_by_dphi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("dgamma_by_dphi needs to be in row-major storage order");
    int num_points         = pointsx.size();
    int num_quad_points    = gamma.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    auto dA_dX_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, xs::default_arch::alignment()>>();
    MYIF(derivs > 0) {
        dA_dX_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, xs::default_arch::alignment()>>{
            Vec3dSimd(), Vec3dSimd(), Vec3dSimd()
        };
    }
    auto d2A_dXdX_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, xs::default_arch::alignment()>>();
    MYIF(derivs > 1) {
        d2A_dXdX_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, xs::default_arch::alignment()>>{
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

        // A/dA_by_dX/d2A_by_dXdX are interleaved (point, component) arrays, so a batch can't
        // be stored directly into them; store each Vec3dSimd component to a small contiguous
        // buffer once, then index that buffer per lane below.
        alignas(xs::default_arch::alignment()) double A_i_x[simd_size], A_i_y[simd_size], A_i_z[simd_size];
        A_i.store_aligned(A_i_x, A_i_y, A_i_z);

        alignas(xs::default_arch::alignment()) double dA_dX_i_x[3][simd_size], dA_dX_i_y[3][simd_size], dA_dX_i_z[3][simd_size];
        MYIF(derivs > 0) {
            for(int k=0; k<3; k++)
                dA_dX_i[k].store_aligned(dA_dX_i_x[k], dA_dX_i_y[k], dA_dX_i_z[k]);
        }

        alignas(xs::default_arch::alignment()) double d2A_dXdX_i_x[9][simd_size], d2A_dXdX_i_y[9][simd_size], d2A_dXdX_i_z[9][simd_size];
        MYIF(derivs > 1) {
            for(int idx=0; idx<9; idx++)
                d2A_dXdX_i[idx].store_aligned(d2A_dXdX_i_x[idx], d2A_dXdX_i_y[idx], d2A_dXdX_i_z[idx]);
        }

        int jlimit = std::min(simd_size, num_points-i);
        for(int j=0; j<jlimit; j++){
            A(i+j, 0) = fak * A_i_x[j];
            A(i+j, 1) = fak * A_i_y[j];
            A(i+j, 2) = fak * A_i_z[j];
            MYIF(derivs > 0) {
                for(int k=0; k<3; k++) {
                    dA_by_dX(i+j, k, 0) = fak*dA_dX_i_x[k][j];
                    dA_by_dX(i+j, k, 1) = fak*dA_dX_i_y[k][j];
                    dA_by_dX(i+j, k, 2) = fak*dA_dX_i_z[k][j];
                }
            }
            MYIF(derivs > 1) {
                for(int k1=0; k1<3; k1++) {
                    for(int k2=0; k2<=k1; k2++) {
                        int idx = 3*k1 + k2;
                        d2A_by_dXdX(i+j, k1, k2, 0) = fak*d2A_dXdX_i_x[idx][j];
                        d2A_by_dXdX(i+j, k1, k2, 1) = fak*d2A_dXdX_i_y[idx][j];
                        d2A_by_dXdX(i+j, k1, k2, 2) = fak*d2A_dXdX_i_z[idx][j];
                        if(k2 < k1){
                            d2A_by_dXdX(i+j, k2, k1, 0) = fak*d2A_dXdX_i_x[idx][j];
                            d2A_by_dXdX(i+j, k2, k1, 1) = fak*d2A_dXdX_i_y[idx][j];
                            d2A_by_dXdX(i+j, k2, k1, 2) = fak*d2A_dXdX_i_z[idx][j];
                        }
                    }
                }
            }
        }
    }
}

#else

template<class T, int derivs>
void biot_savart_kernel_A(AlignedPaddedVec& pointsx, AlignedPaddedVec& pointsy, AlignedPaddedVec& pointsz,
            T& gamma, T& dgamma_by_dphi, T& A, T& dA_by_dX, T& d2A_by_dXdX) {
    if(gamma.layout() != xt::layout_type::row_major)
          throw std::runtime_error("gamma needs to be in row-major storage order");
    if(dgamma_by_dphi.layout() != xt::layout_type::row_major)
          throw std::runtime_error("dgamma_by_dphi needs to be in row-major storage order");
    int num_points         = pointsx.size();
    int num_quad_points    = gamma.shape(0);
    auto dA_dX_i = vector<Vec3dStd>();
    MYIF(derivs > 0) {
        dA_dX_i = vector<Vec3dStd>{
            Vec3dStd(), Vec3dStd(), Vec3dStd()
        };
    }
    auto d2A_dXdX_i = vector<Vec3dStd>();
    MYIF(derivs > 1) {
        d2A_dXdX_i = vector<Vec3dStd>{
            Vec3dStd(), Vec3dStd(), Vec3dStd(),
            Vec3dStd(), Vec3dStd(), Vec3dStd(),
            Vec3dStd(), Vec3dStd(), Vec3dStd()
        };
    }
    double fak = (1e-7/num_quad_points);
    double* gamma_j_ptr = &(gamma(0, 0));
    double* dgamma_j_by_dphi_ptr = &(dgamma_by_dphi(0, 0));
    // out vectors pointsx, pointsy, and pointsz are added and aligned, so we
    // don't have to worry about going out of bounds here
    for(int i = 0; i < num_points; i++) {
        auto point_i = Vec3dStd(&(pointsx[i]), &(pointsy[i]), &(pointsz[i]));
        auto A_i   = Vec3dStd();
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
            auto diff = point_i - Vec3dStd(gamma_j_ptr[3*j+0], gamma_j_ptr[3*j+1], gamma_j_ptr[3*j+2]);
            auto norm_diff_2     = normsq(diff);
            auto norm_diff_inv   = rsqrt(norm_diff_2);
            auto norm_diff_3_inv = norm_diff_inv*norm_diff_inv*norm_diff_inv;

            auto dgamma_by_dphi_j_simd = Vec3dStd(dgamma_j_by_dphi_ptr[3*j+0], dgamma_j_by_dphi_ptr[3*j+1], dgamma_j_by_dphi_ptr[3*j+2]);
            A_i += dgamma_by_dphi_j_simd * norm_diff_inv;

            MYIF(derivs > 0) {
#pragma unroll
                for(int k=0; k<3; k++) {
                    auto diffk_norm_diff_3_inv = norm_diff_3_inv * diff[k];
                    dA_dX_i[k] -= dgamma_by_dphi_j_simd * diffk_norm_diff_3_inv;
                }
                MYIF(derivs > 1) {
                    auto term124fak = dgamma_by_dphi_j_simd;
                    auto fak5 = 3.*norm_diff_3_inv*norm_diff_inv*norm_diff_inv;
                    term124fak *= fak5;
#pragma unroll
                    for(int k1=0; k1<3; k1++) {
#pragma unroll
                        for(int k2=0; k2<=k1; k2++) {
                            auto term12 = diff[k1]*diff[k2];
                            d2A_dXdX_i[3*k1 + k2] += term124fak * term12;

                            if(k1 == k2) {
                                d2A_dXdX_i[3*k1 + k2] -= norm_diff_3_inv * dgamma_by_dphi_j_simd;
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
        A(i, 0) = fak * A_i.x;
        A(i, 1) = fak * A_i.y;
        A(i, 2) = fak * A_i.z;
        MYIF(derivs > 0) {
            for(int k=0; k<3; k++) {
                dA_by_dX(i, k, 0) = fak*dA_dX_i[k].x;
                dA_by_dX(i, k, 1) = fak*dA_dX_i[k].y;
                dA_by_dX(i, k, 2) = fak*dA_dX_i[k].z;
            }
        }
        MYIF(derivs > 1) {
            for(int k1=0; k1<3; k1++) {
                for(int k2=0; k2<=k1; k2++) {
                    d2A_by_dXdX(i, k1, k2, 0) = fak*d2A_dXdX_i[3*k1 + k2].x;
                    d2A_by_dXdX(i, k1, k2, 1) = fak*d2A_dXdX_i[3*k1 + k2].y;
                    d2A_by_dXdX(i, k1, k2, 2) = fak*d2A_dXdX_i[3*k1 + k2].z;
                    if(k2 < k1){
                        d2A_by_dXdX(i, k2, k1, 0) = fak*d2A_dXdX_i[3*k1 + k2].x;
                        d2A_by_dXdX(i, k2, k1, 1) = fak*d2A_dXdX_i[3*k1 + k2].y;
                        d2A_by_dXdX(i, k2, k1, 2) = fak*d2A_dXdX_i[3*k1 + k2].z;
                    }
                }
            }
        }
    }
}

#endif
