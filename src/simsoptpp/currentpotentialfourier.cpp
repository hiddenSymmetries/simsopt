#include "currentpotentialfourier.h"
#include "simdhelpers.h"

#define ANGLE_RECOMPUTE 5

#if defined(USE_XSIMD)

// template<template<class Array> class Surface, class Array>
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
                    if (! (m == 0 && n <= 0)) {
		        Phi += phis(m, i) * sinterm;
                        if(!stellsym) {
                            Phi += phic(m, i) * costerm;
                        }
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

#else

template<class Array>
void CurrentPotentialFourier<Array>::Phi_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    int numquadpoints_phi = quadpoints_phi.size();
    int numquadpoints_theta = quadpoints_theta.size();
    constexpr int simd_size = 1;
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for(int k2 = 0; k2 < numquadpoints_theta; k2 += simd_size) {
            double theta = 2*M_PI * quadpoints_theta[k2];
            double Phi = 0.;
            double sin_nfpphi = sin(-nfp*phi);
            double cos_nfpphi = cos(-nfp*phi);
            for (int m = 0; m <= mpol; ++m) {
                double sinterm, costerm;
                for (int i = 0; i < 2*ntor+1; ++i) {
                    int n  = i - ntor;
                    if(i % ANGLE_RECOMPUTE == 0) {
                        sinterm = sin(m*theta - n*nfp*phi);
                        costerm = cos(m*theta - n*nfp*phi);
                    }
                    if (! (m == 0 && n <= 0)) {
                        Phi += phis(m, i) * sinterm;
                        if(!stellsym) {
                            Phi += phic(m, i) * costerm;
                        }
                    }
                    if(i % ANGLE_RECOMPUTE != ANGLE_RECOMPUTE - 1){
                        double sinterm_old = sinterm;
                        double costerm_old = costerm;
                        sinterm = cos_nfpphi * sinterm_old + costerm_old * sin_nfpphi;
                        costerm = costerm_old * cos_nfpphi - sinterm_old * sin_nfpphi;
                    }
                }
            }
            data(k1, k2) = Phi;
        }
    }
}

#endif

#if defined(USE_XSIMD)
// template<template<class Array> class T, class Array>
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
                    if (! (m == 0 && n <= 0)) {
                        Phidash1 += (-n*nfp) * phis(m, i) * costerm;
                        if(!stellsym) {
                            Phidash1 += - (-n*nfp) * phic(m, i) * sinterm;
                        }
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

#else

template<class Array>
void CurrentPotentialFourier<Array>::Phidash1_impl(Array& data) {
    constexpr int simd_size = 1;
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for(int k2 = 0; k2 < numquadpoints_theta; k2 += simd_size) {
            double theta = 2*M_PI * quadpoints_theta[k2];
            double Phidash1 = 0.;
            double sin_nfpphi = sin(-nfp*phi);
            double cos_nfpphi = cos(-nfp*phi);
            for (int m = 0; m <= mpol; ++m) {
                double sinterm, costerm;
                for (int i = 0; i < 2*ntor+1; ++i) {
                    int n  = i - ntor;
                    if(i % ANGLE_RECOMPUTE == 0) {
                        sinterm = sin(m*theta - n*nfp*phi);
                        costerm = cos(m*theta - n*nfp*phi);
                    }
                    if (! (m == 0 && n <= 0)) {
                        Phidash1 += (-n*nfp) * phis(m, i) * costerm;
                        if(!stellsym) {
                            Phidash1 += - (-n*nfp) * phic(m, i) * sinterm;
                        }
                    }
                    if(i % ANGLE_RECOMPUTE != ANGLE_RECOMPUTE - 1){
                        double sinterm_old = sinterm;
                        double costerm_old = costerm;
                        sinterm = cos_nfpphi * sinterm_old + costerm_old * sin_nfpphi;
                        costerm = costerm_old * cos_nfpphi - sinterm_old * sin_nfpphi;
                    }
                }
            }
            data(k1, k2) = Phidash1 * 2*M_PI;
        }
    }
}

#endif

#if defined(USE_XSIMD)
// template<template<class Array> class Surface, class Array>
// template<template<class Array> class T>
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
                    if (! (m == 0 && n <= 0)) {
                        Phidash2 += m * phis(m, i) * costerm;
                        if(!stellsym) {
                            Phidash2 += - m * phic(m, i) * sinterm;
                        }
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

#else

template<class Array>
void CurrentPotentialFourier<Array>::Phidash2_impl(Array& data) {
    constexpr int simd_size = 1;
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for(int k2 = 0; k2 < numquadpoints_theta; k2 += simd_size) {
            double theta = 2*M_PI * quadpoints_theta[k2];
            double Phidash2 = 0.;
            double sin_nfpphi = sin(-nfp*phi);
            double cos_nfpphi = cos(-nfp*phi);
            for (int m = 0; m <= mpol; ++m) {
                double sinterm, costerm;
                for (int i = 0; i < 2*ntor+1; ++i) {
                    int n  = i - ntor;
                    if(i % ANGLE_RECOMPUTE == 0) {
                        sinterm = sin(m*theta - n*nfp*phi);
                        costerm = cos(m*theta - n*nfp*phi);
                    }
                    if (! (m == 0 && n <= 0)) {
                        Phidash2 += m * phis(m, i) * costerm;
                        if(!stellsym) {
                            Phidash2 += - m * phic(m, i) * sinterm;
                        }
                    }
                    if(i % ANGLE_RECOMPUTE != ANGLE_RECOMPUTE - 1){
                        double sinterm_old = sinterm;
                        double costerm_old = costerm;
                        sinterm = cos_nfpphi * sinterm_old + costerm_old * sin_nfpphi;
                        costerm = costerm_old * cos_nfpphi - sinterm_old * sin_nfpphi;
                    }
                }
            }
            data(k1, k2) = Phidash2 * 2*M_PI;
        }
    }
}

#endif

template<class Array>
void CurrentPotentialFourier<Array>::dPhidash2_by_dcoeff_impl(Array& data) {
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            int counter = 0;
            for (int m = 0; m <= mpol; ++m) {
                for (int n = -ntor; n <= ntor; ++n) {
                    if(m==0 && n<=0) continue;
                    data(k1, k2, counter) = 2*M_PI*m * cos(m*theta-n*nfp*phi);
                    counter++;
                }
            }
            if (!stellsym) {
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<=0) continue;
                        data(k1, k2, counter) = 2*M_PI*(-m) * sin(m*theta-n*nfp*phi);
                        counter++;
                    }
                }
            }
        }
    }
}

template<class Array>
void CurrentPotentialFourier<Array>::dPhidash1_by_dcoeff_impl(Array& data) {
#pragma omp parallel for
    for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
        double phi  = 2*M_PI*quadpoints_phi[k1];
        for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
            double theta  = 2*M_PI*quadpoints_theta[k2];
            int counter = 0;
            for (int m = 0; m <= mpol; ++m) {
                for (int n = -ntor; n <= ntor; ++n) {
                    if(m==0 && n<=0) continue;
                    data(k1, k2, counter) = 2*M_PI*(-n*nfp) * cos(m*theta-n*nfp*phi);
                    counter++;
                }
            }
            if (!stellsym) {
                for (int m = 0; m <= mpol; ++m) {
                    for (int n = -ntor; n <= ntor; ++n) {
                        if(m==0 && n<=0) continue;
                        data(k1, k2, counter) = 2*M_PI*(n*nfp) * sin(m*theta-n*nfp*phi);
                        counter++;
                    }
                }
            }
        }
    }
}

#include "xtensor-python/pyarray.hpp"
typedef xt::pyarray<double> Array;
template class CurrentPotentialFourier<Array>;
// template<template<class Array> class T>
// template<template<class Array> class Surface, class Array>
