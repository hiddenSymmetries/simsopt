#include "surface_newquadpts.h"
#include "simdhelpers.h"

// Optimization notes:
// We use two "tricks" in this part of the code to speed up some of the functions.
// 1) We use SIMD instructions to parallelise across the angle theta.
// 2) This parametrization requires the evaluation of
//          sin(m*theta-n*nfp*phi) and cos(m*theta-n*nfp*phi)
//    for many values of n and m. Since trigonometric functions are expensive,
//    we want to avoid lots of calls to sin and cos. Instead, we use the rules
//          sin(a + b) = cos(b) sin(a) + cos(a) sin(b)
//          cos(a + b) = cos(a) cos(b) - sin(a) sin(b)
//    to write
//          sin(m*theta-(n+1)*nfp*phi) = cos(-nfp*phi) * sin(m*theta-n*nfp*phi) + cos(m*theta-n*nfp*phi) * sin(-nfp*phi)
//          cos(m*theta-(n+1)*nfp*phi) = cos(m*theta-n*nfp*phi) * cos(-nfp*phi) + sin(m*theta-n*nfp*phi) * sin(-nfp*phi)
//    In our code we loop over n. So we start with n=-ntor, and then we always
//    just increase the angle by -nfp*phi.

#define ANGLE_RECOMPUTE 5

template<class Array>
void SurfaceNewQuadPts<Array>::gamma_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.gamma_impl(data, quadpoints_phi, quadpoints_theta);
}


template<class Array>
void SurfaceNewQuadPts<Array>::gamma_lin(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.gamma_lin(data, quadpoints_phi, quadpoints_theta);
}


template<class Array>
void SurfaceNewQuadPts<Array>::gammadash1_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.gammadash1_impl(data, quadpoints_phi, quadpoints_theta);
}

template<class Array>
void SurfaceNewQuadPts<Array>::gammadash1dash1_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.gammadash1dash1_impl(data, quadpoints_phi, quadpoints_theta);
}

template<class Array>
void SurfaceNewQuadPts<Array>::gammadash1dash2_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.gammadash1dash2_impl(data, quadpoints_phi, quadpoints_theta);
}

template<class Array>
void SurfaceNewQuadPts<Array>::gammadash2dash2_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.gammadash2dash2_impl(data, quadpoints_phi, quadpoints_theta);
}

template<class Array>
void SurfaceNewQuadPts<Array>::gammadash2_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.gammadash2_impl(data, quadpoints_phi, quadpoints_theta);
}

template<class Array>
Array SurfaceNewQuadPts<Array>::dgamma_by_dcoeff_vjp(Array& v) {
    Array res = xt::zeros<double>({num_dofs()});
    constexpr int simd_size = xsimd::simd_type<double>::size;
    auto resptr = &(res(0));
#pragma omp parallel
    {
        double* resptr_private = new double[num_dofs()];
        for (int i = 0; i < num_dofs(); ++i) {
            resptr_private[i] = 0.;
        }
#pragma omp for
        for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
            double phi  = 2*M_PI*quadpoints_phi[k1];
            double sinphi = sin(phi);
            double cosphi = cos(phi);

            for(int i = 0; i < numquadpoints_theta; i += simd_size) {
                simd_t theta(0.);
                simd_t v0(0.);
                simd_t v1(0.);
                simd_t v2(0.);
                for (int l = 0; l < simd_size; ++l) {
                    if(i + l >= numquadpoints_theta)
                        break;
                    v0[l] = v(k1, i+l, 0);
                    v1[l] = v(k1, i+l, 1);
                    v2[l] = v(k1, i+l, 2);
                    theta[l] = 2*M_PI * quadpoints_theta[i+l];
                }
                int counter = 0;
                int shift0 = -ntor;
                int shift1 = !stellsym ? shift0 + (mpol+1) * (2*ntor+1) - ntor - 1 : shift0;
                int shift2 = !stellsym ? shift1 + (mpol+1) * (2*ntor+1) - ntor : shift1;
                int shift3 = shift2 + (mpol+1)*(2*ntor+1) - ntor - 1;
                double sin_nfpphi = sin(-nfp*phi);
                double cos_nfpphi = cos(-nfp*phi);
                for (int m = 0; m <= mpol; ++m) {
                    simd_t sinterm, costerm;
                    for (int n = -ntor; n <= ntor; ++n) {
                        int i = n + ntor;
                         // recompute the angle from scratch every so often, to
                         // avoid accumulating floating point error
                        if(i % ANGLE_RECOMPUTE == 0)
                            xsimd::sincos(m*theta-n*nfp*phi, sinterm, costerm);
                        if(!(m==0 && n<0)){
                            resptr_private[counter+shift0] += cosphi * xsimd::hadd(costerm * v0);
                            resptr_private[counter+shift0] += sinphi * xsimd::hadd(costerm * v1);
                        }
                        if(!(stellsym) && !(m==0 && n<=0)){
                            resptr_private[counter+shift1] += cosphi * xsimd::hadd(sinterm * v0);
                            resptr_private[counter+shift1] += sinphi * xsimd::hadd(sinterm * v1);
                        }
                        if(!(stellsym) && !(m==0 && n<0)){
                            resptr_private[counter+shift2] += xsimd::hadd(costerm * v2);
                        }
                        if(!(m==0 && n<=0)){
                            resptr_private[counter+shift3] += xsimd::hadd(sinterm * v2);
                        }
                        counter++;
                        if(i % ANGLE_RECOMPUTE != ANGLE_RECOMPUTE - 1){
                            simd_t sinterm_old = sinterm;
                            simd_t costerm_old = costerm;
                            sinterm = cos_nfpphi * sinterm_old + costerm_old * sin_nfpphi;
                            costerm = costerm_old * cos_nfpphi - sinterm_old * sin_nfpphi;
                        }
                    }
                }
            }
        }
#pragma omp critical
        {
            for(int i=0; i<num_dofs(); ++i) {
                resptr[i] += resptr_private[i];
            }
        }
    }
    return res;
}

template<class Array>
void SurfaceNewQuadPts<Array>::dgamma_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.dgamma_by_dcoeff_impl(data, quadpoints_phi, quadpoints_theta);
}

template<class Array>
Array SurfaceNewQuadPts<Array>::dgammadash1_by_dcoeff_vjp(Array& v) {
    Array res = xt::zeros<double>({num_dofs()});
    constexpr int simd_size = xsimd::simd_type<double>::size;
    auto resptr = &(res(0));
#pragma omp parallel
    {
        double* resptr_private = new double[num_dofs()];
        for (int i = 0; i < num_dofs(); ++i) {
            resptr_private[i] = 0.;
        }
#pragma omp for
        for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
            double phi  = 2*M_PI*quadpoints_phi[k1];
            double sinphi = sin(phi);
            double cosphi = cos(phi);

            for(int i = 0; i < numquadpoints_theta; i += simd_size) {
                simd_t theta(0.);
                simd_t v0(0.);
                simd_t v1(0.);
                simd_t v2(0.);
                for (int l = 0; l < simd_size; ++l) {
                    if(i + l >= numquadpoints_theta)
                        break;
                    v0[l] = v(k1, i+l, 0);
                    v1[l] = v(k1, i+l, 1);
                    v2[l] = v(k1, i+l, 2);
                    theta[l] = 2*M_PI * quadpoints_theta[i+l];
                }
                int counter = 0;
                int shift0 = -ntor;
                int shift1 = !stellsym ? shift0 + (mpol+1) * (2*ntor+1) - ntor - 1 : shift0;
                int shift2 = !stellsym ? shift1 + (mpol+1) * (2*ntor+1) - ntor : shift1;
                int shift3 = shift2 + (mpol+1)*(2*ntor+1) - ntor - 1;

                double sin_nfpphi = sin(-nfp*phi);
                double cos_nfpphi = cos(-nfp*phi);
                for (int m = 0; m <= mpol; ++m) {
                    simd_t sinterm, costerm;
                    for (int n = -ntor; n <= ntor; ++n) {
                        int i = n + ntor;
                        // recompute the angle from scratch every so often, to
                        // avoid accumulating floating point error
                        if(i % ANGLE_RECOMPUTE == 0)
                            xsimd::sincos(m*theta-n*nfp*phi, sinterm, costerm);
                        if(!(m==0 && n<0)){
                            resptr_private[counter+shift0] += xsimd::hadd((sinterm * ((n*nfp) * cosphi) - costerm * sinphi) * v0);
                            resptr_private[counter+shift0] += xsimd::hadd((sinterm * ((n*nfp) * sinphi) + costerm * cosphi) * v1);
                        }
                        if(!(stellsym) && !(m==0 && n<=0)){
                            resptr_private[counter+shift1] += xsimd::hadd((costerm * ((-n*nfp)*cosphi) - sinterm * sinphi) * v0);
                            resptr_private[counter+shift1] += xsimd::hadd((costerm * ((-n*nfp)*sinphi) + sinterm * cosphi) * v1);
                        }
                        if(!(stellsym) && !(m==0 && n<0)){
                            resptr_private[counter+shift2] += xsimd::hadd((n*nfp)*sinterm * v2);
                        }
                        if(!(m==0 && n<=0)){
                            resptr_private[counter+shift3] += xsimd::hadd((-n*nfp)*costerm * v2);
                        }
                        counter++;
                        if(i % ANGLE_RECOMPUTE != ANGLE_RECOMPUTE - 1){
                            simd_t sinterm_old = sinterm;
                            simd_t costerm_old = costerm;
                            sinterm = cos_nfpphi * sinterm_old + costerm_old * sin_nfpphi;
                            costerm = costerm_old * cos_nfpphi - sinterm_old * sin_nfpphi;
                        }
                    }
                }
            }
        }
#pragma omp critical
        {
            for(int i=0; i<num_dofs(); ++i) {
                resptr[i] += resptr_private[i];
            }
        }
    }
    res *= 2*M_PI;
    return res;
}

template<class Array>
void SurfaceNewQuadPts<Array>::dgammadash1_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.dgammadash1_by_dcoeff_impl(data, quadpoints_phi, quadpoints_theta);
}

template<class Array>
void SurfaceNewQuadPts<Array>::dgammadash1dash2_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.dgammadash1dash2_by_dcoeff_impl(data, quadpoints_phi, quadpoints_theta);
}

template<class Array>
void SurfaceNewQuadPts<Array>::dgammadash1dash1_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.dgammadash1dash1_by_dcoeff_impl(data, quadpoints_phi, quadpoints_theta);
}

template<class Array>
Array SurfaceNewQuadPts<Array>::dgammadash2_by_dcoeff_vjp(Array& v) {
    Array res = xt::zeros<double>({num_dofs()});
    constexpr int simd_size = xsimd::simd_type<double>::size;
    auto resptr = &(res(0));
#pragma omp parallel
    {
        double* resptr_private = new double[num_dofs()];
        for (int i = 0; i < num_dofs(); ++i) {
            resptr_private[i] = 0.;
        }
#pragma omp for
        for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
            double phi  = 2*M_PI*quadpoints_phi[k1];
            double sinphi = sin(phi);
            double cosphi = cos(phi);

            for(int i = 0; i < numquadpoints_theta; i += simd_size) {
                simd_t theta(0.);
                simd_t v0(0.);
                simd_t v1(0.);
                simd_t v2(0.);
                for (int l = 0; l < simd_size; ++l) {
                    if(i + l >= numquadpoints_theta)
                        break;
                    v0[l] = v(k1, i+l, 0);
                    v1[l] = v(k1, i+l, 1);
                    v2[l] = v(k1, i+l, 2);
                    theta[l] = 2*M_PI * quadpoints_theta[i+l];
                }
                int counter = 0;
                int shift0 = -ntor;
                int shift1 = !stellsym ? shift0 + (mpol+1) * (2*ntor+1) - ntor - 1 : shift0;
                int shift2 = !stellsym ? shift1 + (mpol+1) * (2*ntor+1) - ntor : shift1;
                int shift3 = shift2 + (mpol+1)*(2*ntor+1) - ntor - 1;

                double sin_nfpphi = sin(-nfp*phi);
                double cos_nfpphi = cos(-nfp*phi);
                for (int m = 0; m <= mpol; ++m) {
                    simd_t sinterm, costerm;
                    for (int n = -ntor; n <= ntor; ++n) {
                        int i = n + ntor;
                        // recompute the angle from scratch every so often, to
                        // avoid accumulating floating point error
                        if(i % ANGLE_RECOMPUTE == 0)
                            xsimd::sincos(m*theta-n*nfp*phi, sinterm, costerm);
                        if(!(m==0 && n<0)){
                            resptr_private[counter+shift0] -= (cosphi * m) * xsimd::hadd(sinterm * v0);
                            resptr_private[counter+shift0] -= (sinphi * m) * xsimd::hadd(sinterm * v1);
                        }
                        if(!(stellsym) && !(m==0 && n<=0)){
                            resptr_private[counter+shift1] += (cosphi * m) * xsimd::hadd(costerm * v0);
                            resptr_private[counter+shift1] += (sinphi * m) * xsimd::hadd(costerm * v1);
                        }
                        if(!(stellsym) && !(m==0 && n<0)){
                            resptr_private[counter+shift2] -= m * xsimd::hadd(sinterm * v2);
                        }
                        if(!(m==0 && n<=0)){
                            resptr_private[counter+shift3] += m * xsimd::hadd(costerm * v2);
                        }
                        counter++;
                        if(i % ANGLE_RECOMPUTE != ANGLE_RECOMPUTE - 1){
                            simd_t sinterm_old = sinterm;
                            simd_t costerm_old = costerm;
                            sinterm = cos_nfpphi * sinterm_old + costerm_old * sin_nfpphi;
                            costerm = costerm_old * cos_nfpphi - sinterm_old * sin_nfpphi;
                        }
                    }
                }
            }
        }
#pragma omp critical
        {
            for(int i=0; i<num_dofs(); ++i) {
                resptr[i] += resptr_private[i];
            }
        }
    }
    res *= 2*M_PI;
    return res;
}

template<class Array>
void SurfaceNewQuadPts<Array>::dgammadash2_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.dgammadash2_by_dcoeff_impl(data, quadpoints_phi, quadpoints_theta);
}

template<class Array>
void SurfaceNewQuadPts<Array>::dgammadash2dash2_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.dgammadash2dash2_by_dcoeff_impl(data, quadpoints_phi, quadpoints_theta);
}

#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;
template class SurfaceNewQuadPts<Array>;
