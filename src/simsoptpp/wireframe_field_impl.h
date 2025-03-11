#include "simdhelpers.h"
#include "vec3dsimd.h"
#include <stdexcept>
#include "xtensor/xlayout.hpp"

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
void wireframe_field_kernel(AlignedPaddedVec& pointsx, AlignedPaddedVec& pointsy, AlignedPaddedVec& pointsz,
            std::vector<double>& node0, std::vector<double>& node1, 
            T& B, T& dB_by_dX, T& d2B_by_dXdX) {

    int num_points = pointsx.size();
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

    double fak = 1e-7;
    auto node0_vec = Vec3dSimd(node0[0], node0[1], node0[2]);
    auto node1_vec = Vec3dSimd(node1[0], node1[1], node1[2]);

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

        auto diff0 = point_i - node0_vec;
        auto diff1 = point_i - node1_vec;
        auto norm_diff0_sq = normsq(diff0);
        auto norm_diff1_sq = normsq(diff1);
        auto norm_diff0 = sqrt(norm_diff0_sq);
        auto norm_diff1 = sqrt(norm_diff1_sq);
        auto diff0_diff1 = norm_diff0*norm_diff1;
        auto denom = diff0_diff1 * (diff0_diff1 + inner(diff0, diff1));
        auto factor = (norm_diff0 + norm_diff1) / denom;
        auto diff0_cross_diff1 = cross(diff0, diff1);

        B_i.x = xsimd::fma(diff0_cross_diff1.x, factor, B_i.x);
        B_i.y = xsimd::fma(diff0_cross_diff1.y, factor, B_i.y);
        B_i.z = xsimd::fma(diff0_cross_diff1.z, factor, B_i.z);

        MYIF(derivs > 0) {
            auto p0 = diff0 * norm_diff1;
            auto p1 = diff1 * norm_diff0;
            auto factorsq = factor * factor;
            auto grad_factor = (p0 + p1) * (-factorsq)
                               - (p0*(1.0/norm_diff0_sq) 
                                  + p1*(1.0/norm_diff1_sq))*(1.0/denom);

            dB_dX_i[0].x = grad_factor.x * diff0_cross_diff1.x;
            dB_dX_i[0].y = grad_factor.y * diff0_cross_diff1.x
                           + factor * ( diff1.z - diff0.z);
            dB_dX_i[0].z = grad_factor.z * diff0_cross_diff1.x
                           + factor * ( diff0.y - diff1.y);
            dB_dX_i[1].x = grad_factor.x * diff0_cross_diff1.y
                           + factor * (-diff1.z + diff0.z);
            dB_dX_i[1].y = grad_factor.y * diff0_cross_diff1.y;
            dB_dX_i[1].z = grad_factor.z * diff0_cross_diff1.y
                           + factor * (-diff0.x + diff1.x);
            dB_dX_i[2].x = grad_factor.x * diff0_cross_diff1.z
                           + factor * ( diff1.y - diff0.y);
            dB_dX_i[2].y = grad_factor.y * diff0_cross_diff1.z
                           + factor * ( diff0.x - diff1.x);
            dB_dX_i[2].z = grad_factor.z * diff0_cross_diff1.z;
            
            MYIF(derivs > 1) {

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

#else

template<class T, int derivs>
void wireframe_field_kernel(AlignedPaddedVec& pointsx, AlignedPaddedVec& pointsy, AlignedPaddedVec& pointsz,
            std::vector<double>& node0, std::vector<double>& node1, 
            T& B, T& dB_by_dX, T& d2B_by_dXdX) {

    int num_points = pointsx.size();
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
    double fak = 1e-7;
    auto node0_vec = Vec3dStd(node0[0], node0[1], node0[2]);
    auto node1_vec = Vec3dStd(node1[0], node1[1], node1[2]);

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

        auto diff0 = point_i - node0_vec;
        auto diff1 = point_i - node1_vec;
        auto norm_diff0_sq = normsq(diff0);
        auto norm_diff1_sq = normsq(diff1);
        auto norm_diff0 = sqrt(norm_diff0_sq);
        auto norm_diff1 = sqrt(norm_diff1_sq);
        auto diff0_diff1 = norm_diff0*norm_diff1;
        auto denom = diff0_diff1 * (diff0_diff1 + inner(diff0, diff1));
        auto factor = (norm_diff0 + norm_diff1) / denom;
        auto diff0_cross_diff1 = cross(diff0, diff1);

        B_i += (factor * diff0_cross_diff1);

        MYIF(derivs > 0) {

            auto p0 = diff0 * norm_diff1;
            auto p1 = diff1 * norm_diff0;
            auto factorsq = factor * factor;
            auto grad_factor = (p0 + p1) * (-factorsq)
                               - (p0*(1.0/norm_diff0_sq) 
                                  + p1*(1.0/norm_diff1_sq))*(1.0/denom);

            dB_dX_i[0].x = grad_factor.x * diff0_cross_diff1.x;
            dB_dX_i[0].y = grad_factor.y * diff0_cross_diff1.x
                           + factor * ( diff1.z - diff0.z);
            dB_dX_i[0].z = grad_factor.z * diff0_cross_diff1.x
                           + factor * ( diff0.y - diff1.y);
            dB_dX_i[1].x = grad_factor.x * diff0_cross_diff1.y
                           + factor * (-diff1.z + diff0.z);
            dB_dX_i[1].y = grad_factor.y * diff0_cross_diff1.y;
            dB_dX_i[1].z = grad_factor.z * diff0_cross_diff1.y
                           + factor * (-diff0.x + diff1.x);
            dB_dX_i[2].x = grad_factor.x * diff0_cross_diff1.z
                           + factor * ( diff1.y - diff0.y);
            dB_dX_i[2].y = grad_factor.y * diff0_cross_diff1.z
                           + factor * ( diff0.x - diff1.x);
            dB_dX_i[2].z = grad_factor.z * diff0_cross_diff1.z;
            
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
    }
}
#endif

