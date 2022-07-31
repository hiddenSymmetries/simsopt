#include "currentpotentialfourier.h"
#include "simdhelpers.h"

#include <vector>
using std::vector;

#include <string>
using std::string;

#include <map>
using std::map;
#include <stdexcept>
using std::logic_error;

#include "xtensor/xarray.hpp"
#include "cachedarray.h"
#include <Eigen/Dense>

#define ANGLE_RECOMPUTE 5

template<class Array>
void CurrentPotentialFourier<Array>::Phi_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    int numquadpoints_phi = quadpoints_phi.size();
    int numquadpoints_theta = quadpoints_theta.size();
    constexpr int simd_size = xsimd::simd_type<double>::size;
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for(int k2 = 0; k2 < numquadpoints_theta; k2 += simd_size) {
            simd_t theta;
            for (int l = 0; l < simd_size; ++l) {
                if(k2 + l >= numquadpoints_theta)
                    break;
                theta[l] = 2*M_PI * quadpoints_theta[k2+l];
            }
            simd_t Phi(0.);
            double sin_nfpphi = sin(-nfp*phi);
            double cos_nfpphi = cos(-nfp*phi);
            for (int m = 0; m <= mpol; ++m) {
                simd_t sinterm, costerm;
                for (int i = 0; i < 2*ntor+1; ++i) {
                    int n  = i - ntor;
                    // recompute the angle from scratch every so often, to
                    // avoid accumulating floating point error
                    if(i % ANGLE_RECOMPUTE == 0)
                        xsimd::sincos(m*theta-n*nfp*phi, sinterm, costerm);
                    Phi += phis(m, i) * sinterm;
                    if(!stellsym) {
                        Phi += phic(m, i) * costerm;
                    }
                    if(i % ANGLE_RECOMPUTE != ANGLE_RECOMPUTE - 1){
                        simd_t sinterm_old = sinterm;
                        simd_t costerm_old = costerm;
                        sinterm = cos_nfpphi * sinterm_old + costerm_old * sin_nfpphi;
                        costerm = costerm_old * cos_nfpphi - sinterm_old * sin_nfpphi;
                    }
                }
            }
            for (int l = 0; l < simd_size; ++l) {
                if(k2 + l >= numquadpoints_theta)
                    break;
                data(k1, k2+l) = Phi[l];
            }
        }
    }
}

template<class Array>
void CurrentPotentialFourier<Array>::Phidash1_impl(Array& data) {
    constexpr int simd_size = xsimd::simd_type<double>::size;
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for(int k2 = 0; k2 < numquadpoints_theta; k2 += simd_size) {
            simd_t theta;
            for (int l = 0; l < simd_size; ++l) {
                if(k2 + l >= numquadpoints_theta)
                    break;
                theta[l] = 2*M_PI * quadpoints_theta[k2+l];
            }
            simd_t Phidash1(0.);
            double sin_nfpphi = sin(-nfp*phi);
            double cos_nfpphi = cos(-nfp*phi);
            for (int m = 0; m <= mpol; ++m) {
                simd_t sinterm, costerm;
                for (int i = 0; i < 2*ntor+1; ++i) {
                    int n  = i - ntor;
                    // recompute the angle from scratch every so often, to
                    // avoid accumulating floating point error
                    if(i % ANGLE_RECOMPUTE == 0)
                        xsimd::sincos(m*theta-n*nfp*phi, sinterm, costerm);
                    Phidash1 += (-n*nfp) * phis(m, i) * costerm;
                    if(!stellsym) {
                        Phidash1 += - (-n*nfp) * phic(m, i) * sinterm;
                    }
                    if(i % ANGLE_RECOMPUTE != ANGLE_RECOMPUTE - 1){
                        simd_t sinterm_old = sinterm;
                        simd_t costerm_old = costerm;
                        sinterm = cos_nfpphi * sinterm_old + costerm_old * sin_nfpphi;
                        costerm = costerm_old * cos_nfpphi - sinterm_old * sin_nfpphi;
                    }
                }
            }
            for (int l = 0; l < simd_size; ++l) {
                if(k2 + l >= numquadpoints_theta)
                    break;
                data(k1, k2+l) = Phidash1[l]*2*M_PI;
            }
        }
    }
}

template<class Array>
void CurrentPotentialFourier<Array>::Phidash2_impl(Array& data) {
    constexpr int simd_size = xsimd::simd_type<double>::size;
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for(int k2 = 0; k2 < numquadpoints_theta; k2 += simd_size) {
            simd_t theta;
            for (int l = 0; l < simd_size; ++l) {
                if(k2 + l >= numquadpoints_theta)
                    break;
                theta[l] = 2*M_PI * quadpoints_theta[k2+l];
            }
            simd_t Phidash2(0.);
            double sin_nfpphi = sin(-nfp*phi);
            double cos_nfpphi = cos(-nfp*phi);
            for (int m = 0; m <= mpol; ++m) {
                simd_t sinterm, costerm;
                for (int i = 0; i < 2*ntor+1; ++i) {
                    int n  = i - ntor;
                    // recompute the angle from scratch every so often, to
                    // avoid accumulating floating point error
                    if(i % ANGLE_RECOMPUTE == 0)
                        xsimd::sincos(m*theta-n*nfp*phi, sinterm, costerm);
                    Phidash2 += m * phis(m, i) * costerm;
                    if(!stellsym) {
                        Phidash2 += - m * phic(m, i) * sinterm;
                    }
                    if(i % ANGLE_RECOMPUTE != ANGLE_RECOMPUTE - 1){
                        simd_t sinterm_old = sinterm;
                        simd_t costerm_old = costerm;
                        sinterm = cos_nfpphi * sinterm_old + costerm_old * sin_nfpphi;
                        costerm = costerm_old * cos_nfpphi - sinterm_old * sin_nfpphi;
                    }
                }
            }
            for (int l = 0; l < simd_size; ++l) {
                if(k2 + l >= numquadpoints_theta)
                    break;
                data(k1, k2+l) = Phidash2[l] * 2*M_PI;
            }
        }
    }
}


#include "xtensor-python/pyarray.hpp"
typedef xt::pyarray<double> Array;
template class CurrentPotentialFourier<Array>;
