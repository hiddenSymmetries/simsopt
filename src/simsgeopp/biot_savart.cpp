#include "biot_savart.h"

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


inline simd_t rsqrt_approx_intrin(const simd_t& r2){
    return simd_t(_mm256_cvtps_pd(_mm_rsqrt_ps(_mm256_cvtpd_ps(r2()))));
}


inline void rsqrt_newton_intrin(simd_t& rinv, const simd_t& r2){
  // Newton iteration: rinv = 0.5 rinv_approx ( 3 - r2 rinv_approx^2 )
  rinv = rinv*(1.5-r2*rinv*rinv);
}

inline simd_t rsqrt_intrin3(simd_t r2){
  simd_t rinv=rsqrt_approx_intrin(r2);
  r2 *= 0.5;
  rsqrt_newton_intrin(rinv, r2);
  rsqrt_newton_intrin(rinv, r2);
  //rsqrt_newton_intrin(rinv, r2);
  return rinv;
}


template<class T, int derivs>
void biot_savart_kernel(vector_type& pointsx, vector_type& pointsy, vector_type& pointsz, T& gamma, T& dgamma_by_dphi, T& B, T& dB_by_dX, T& d2B_by_dXdX) {
    int num_points         = pointsx.size();
    int num_quad_points    = gamma.shape(0);
    constexpr int simd_size = xsimd::simd_type<double>::size;
    auto dB_dX_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, XSIMD_DEFAULT_ALIGNMENT>>();
    auto d2B_dXdX_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, XSIMD_DEFAULT_ALIGNMENT>>();
    for(int i = 0; i < num_points-num_points%simd_size; i += simd_size) {
        auto point_i = Vec3dSimd(&(pointsx[i]), &(pointsy[i]), &(pointsz[i]));
        auto B_i   = Vec3dSimd();
        MYIF(derivs > 0) {
            dB_dX_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, XSIMD_DEFAULT_ALIGNMENT>>{
                Vec3dSimd(), Vec3dSimd(), Vec3dSimd()
            };
            MYIF(derivs > 1) {
                d2B_dXdX_i = vector<Vec3dSimd, xs::aligned_allocator<Vec3dSimd, XSIMD_DEFAULT_ALIGNMENT>>{
                    Vec3dSimd(), Vec3dSimd(), Vec3dSimd(), 
                        Vec3dSimd(), Vec3dSimd(), Vec3dSimd(), 
                        Vec3dSimd(), Vec3dSimd(), Vec3dSimd() 
                };
            }
        }
        for (int j = 0; j < num_quad_points; ++j) {
            auto diff = Vec3dSimd(point_i.x - gamma(j, 0), point_i.y - gamma(j, 1), point_i.z - gamma(j, 2));
            auto dgamma_by_dphi_j_simd = Vec3dSimd(dgamma_by_dphi(j, 0), dgamma_by_dphi(j, 1), dgamma_by_dphi(j, 2));

            auto norm_diff_2   = normsq(diff);
            //auto norm_diff_inv = norm_diff_2;
            //auto norm_diff_inv = rsqrt_intrin3(norm_diff_2);
            auto norm_diff_inv = 1/sqrt(norm_diff_2);

            //auto norm_diff       = sqrt(norm_diff_2);

            auto norm_diff_3_inv = norm_diff_inv*norm_diff_inv*norm_diff_inv;
            auto dgamma_by_dphi_j_cross_diff = cross(dgamma_by_dphi_j_simd, diff);
            B_i.x = xsimd::fma(dgamma_by_dphi_j_cross_diff.x, norm_diff_3_inv, B_i.x);
            B_i.y = xsimd::fma(dgamma_by_dphi_j_cross_diff.y, norm_diff_3_inv, B_i.y);
            B_i.z = xsimd::fma(dgamma_by_dphi_j_cross_diff.z, norm_diff_3_inv, B_i.z);

            //MYIF(derivs > 0) {
            //    auto norm_diff_4_inv = 1/(norm_diff_2*norm_diff_2);
            //    auto three_dgamma_by_dphi_cross_diff_by_norm_diff = dgamma_by_dphi_j_cross_diff * (3/norm_diff);
            //    auto dgamma_by_dphi_j_simd_norm_diff = dgamma_by_dphi_j_simd * norm_diff;
            //    for(int k=0; k<3; k++) {
            //        auto numerator1 = cross(dgamma_by_dphi_j_simd_norm_diff, k);
            //        auto numerator2 = three_dgamma_by_dphi_cross_diff_by_norm_diff * diff[k];
            //        auto temp = (numerator1-numerator2);
            //        dB_dX_i[k].x = xsimd::fma(temp.x, norm_diff_4_inv, dB_dX_i[k].x);
            //        dB_dX_i[k].y = xsimd::fma(temp.y, norm_diff_4_inv, dB_dX_i[k].y);
            //        dB_dX_i[k].z = xsimd::fma(temp.z, norm_diff_4_inv, dB_dX_i[k].z);
            //    }
            //    MYIF(derivs > 1) {
            //        auto norm_diff_5_inv = norm_diff_4_inv/norm_diff;
            //        auto norm_diff_7_inv = norm_diff_5_inv/norm_diff_2;
            //        for(int k1=0; k1<3; k1++) {
            //            for(int k2=0; k2<=k1; k2++) {
            //                auto term12 = cross(dgamma_by_dphi_j_simd, k2)*diff[k1];
            //                term12 += cross(dgamma_by_dphi_j_simd, k1)*diff[k2];

            //                auto term124fak = (-3.)*norm_diff_5_inv;

            //                d2B_dXdX_i[3*k1 + k2].x = xsimd::fma(term124fak, term12.x, d2B_dXdX_i[3*k1 + k2].x);
            //                d2B_dXdX_i[3*k1 + k2].y = xsimd::fma(term124fak, term12.y, d2B_dXdX_i[3*k1 + k2].y);
            //                d2B_dXdX_i[3*k1 + k2].z = xsimd::fma(term124fak, term12.z, d2B_dXdX_i[3*k1 + k2].z);

            //                auto term3fak = (15. * (diff[k1] * diff[k2] * norm_diff_7_inv));
            //                if(k1 == k2) {
            //                    term3fak += term124fak;
            //                }
            //                d2B_dXdX_i[3*k1 + k2].x = xsimd::fma(term3fak, dgamma_by_dphi_j_cross_diff.x, d2B_dXdX_i[3*k1 + k2].x);
            //                d2B_dXdX_i[3*k1 + k2].y = xsimd::fma(term3fak, dgamma_by_dphi_j_cross_diff.y, d2B_dXdX_i[3*k1 + k2].y);
            //                d2B_dXdX_i[3*k1 + k2].z = xsimd::fma(term3fak, dgamma_by_dphi_j_cross_diff.z, d2B_dXdX_i[3*k1 + k2].z);
            //            }
            //        }
            //    }
            //}
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
        auto point = Vec3d{pointsx[i], pointsy[i], pointsz[i]};
        B(i, 0) = 0;
        B(i, 1) = 0;
        B(i, 2) = 0;
        for (int j = 0; j < num_quad_points; ++j) {
            Vec3d gamma_j = Vec3d { gamma(j, 0), gamma(j, 1), gamma(j, 2)};
            Vec3d dgamma_by_dphi_j = Vec3d { dgamma_by_dphi(j, 0), dgamma_by_dphi(j, 1), dgamma_by_dphi(j, 2)};
            Vec3d diff = point - gamma_j;
            double norm_diff = norm(diff);
            Vec3d dgamma_by_dphi_j_cross_diff = cross(dgamma_by_dphi_j, diff);
            Vec3d B_i = dgamma_by_dphi_j_cross_diff / (norm_diff * norm_diff * norm_diff);

            B(i, 0) += B_i[0];
            B(i, 1) += B_i[1];
            B(i, 2) += B_i[2];
            MYIF(derivs > 0) {
                double norm_diff_4_inv = 1/(norm_diff*norm_diff*norm_diff*norm_diff);
                Vec3d three_dgamma_by_dphi_cross_diff_by_norm_diff = dgamma_by_dphi_j_cross_diff * 3 / norm_diff;
                for(int k=0; k<3; k++) {
                    Vec3d ek = Vec3d{0., 0., 0.};
                    ek[k] = 1.0;
                    Vec3d numerator1 = cross(dgamma_by_dphi_j, ek) * norm_diff;
                    Vec3d numerator2 = three_dgamma_by_dphi_cross_diff_by_norm_diff * diff[k];
                    Vec3d temp = (numerator1-numerator2) * norm_diff_4_inv;
                    dB_by_dX(i, k, 0) += temp[0];
                    dB_by_dX(i, k, 1) += temp[1];
                    dB_by_dX(i, k, 2) += temp[2];
                }
                MYIF(derivs > 1) {
                    double norm_diff_5_inv = norm_diff_4_inv/norm_diff;
                    double norm_diff_7_inv = norm_diff_5_inv/(norm_diff*norm_diff);
                    for(int k1=0; k1<3; k1++) {
                        for(int k2=0; k2<3; k2++) {
                            Vec3d ek1 = Vec3d{0., 0., 0.};
                            ek1[k1] = 1.0;
                            Vec3d ek2 = Vec3d{0., 0., 0.};
                            ek2[k2] = 1.0;

                            Vec3d term1 = -3 * (diff[k1]*norm_diff_5_inv) * cross(dgamma_by_dphi_j, ek2);
                            Vec3d term2 = -3 * (diff[k2]*norm_diff_5_inv) * cross(dgamma_by_dphi_j, ek1);
                            Vec3d term3 = 15 * (diff[k1] * diff[k2] * norm_diff_7_inv) * dgamma_by_dphi_j_cross_diff;
                            Vec3d term4 = Vec3d{0., 0., 0.};
                            if(k1 == k2) {
                                term4 = -3 * norm_diff_5_inv * dgamma_by_dphi_j_cross_diff;
                            }
                            Vec3d temp = (term1 + term2 + term3 + term4);
                            d2B_by_dXdX(i, k1, k2, 0) += temp[0];
                            d2B_by_dXdX(i, k1, k2, 1) += temp[1];
                            d2B_by_dXdX(i, k1, k2, 2) += temp[2];
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

template void biot_savart_kernel<xt::xarray<double>, 0>(vector_type&, vector_type&, vector_type&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&);
template void biot_savart_kernel<xt::xarray<double>, 1>(vector_type&, vector_type&, vector_type&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&);
template void biot_savart_kernel<xt::xarray<double>, 2>(vector_type&, vector_type&, vector_type&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&, xt::xarray<double>&);


void biot_savart(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis, vector<Array>& B, vector<Array>& dB_by_dX, vector<Array>& d2B_by_dXdX) {
    auto pointsx = vector_type(points.shape(0), 0);
    auto pointsy = vector_type(points.shape(0), 0);
    auto pointsz = vector_type(points.shape(0), 0);
    int num_points = points.shape(0);
    for (int i = 0; i < num_points; ++i) {
        pointsx[i] = points(i, 0);
        pointsy[i] = points(i, 1);
        pointsz[i] = points(i, 2);
    }
    int num_coils  = gammas.size();

    Array dummyjac = xt::zeros<double>({1, 1, 1});
    Array dummyhess = xt::zeros<double>({1, 1, 1, 1});

    int nderivs = 0;
    if(dB_by_dX.size() == num_coils) {
        nderivs = 1;
        if(d2B_by_dXdX.size() == num_coils) {
            nderivs = 2;
        }
    }

#pragma omp parallel for
    for(int i=0; i<num_coils; i++) {
        if(nderivs == 2)
            biot_savart_kernel<Array, 2>(pointsx, pointsy, pointsz, gammas[i], dgamma_by_dphis[i], B[i], dB_by_dX[i], d2B_by_dXdX[i]);
        else {
            if(nderivs == 1) 
                biot_savart_kernel<Array, 1>(pointsx, pointsy, pointsz, gammas[i], dgamma_by_dphis[i], B[i], dB_by_dX[i], dummyhess);
            else
                biot_savart_kernel<Array, 0>(pointsx, pointsy, pointsz, gammas[i], dgamma_by_dphis[i], B[i], dummyjac, dummyhess);
        }
    }
}

Array biot_savart_B(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis, vector<double>& currents){
    auto dB_by_dXs = vector<Array>();
    auto d2B_by_dXdXs = vector<Array>();
    int num_coils = currents.size();
    auto Bs = vector<Array>(num_coils, Array());
    for (int i = 0; i < num_coils; ++i) {
        Bs[i] = xt::zeros<double>({points.shape(0), points.shape(1)});
    }
    biot_savart(points, gammas, dgamma_by_dphis, Bs, dB_by_dXs, d2B_by_dXdXs);
    Array B = xt::zeros<double>({points.shape(0), points.shape(1)});
    for (int i = 0; i < num_coils; ++i) {
        B += currents[i] * Bs[i];
    }
    return B;
}
