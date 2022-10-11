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
void SurfaceNewQuadPoints<Array>::gamma_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.gamma_impl(data, quadpoints_phi, quadpoints_theta);
}


template<class Array>
void SurfaceNewQuadPoints<Array>::gamma_lin(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.gamma_lin(data, quadpoints_phi, quadpoints_theta);
}


template<class Array>
void SurfaceNewQuadPoints<Array>::gammadash1_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.gammadash1_impl(data, quadpoints_phi, quadpoints_theta);
}

template<class Array>
void SurfaceNewQuadPoints<Array>::gammadash1dash1_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.gammadash1dash1_impl(data, quadpoints_phi, quadpoints_theta);
}

template<class Array>
void SurfaceNewQuadPoints<Array>::gammadash1dash2_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.gammadash1dash2_impl(data, quadpoints_phi, quadpoints_theta);
}

template<class Array>
void SurfaceNewQuadPoints<Array>::gammadash2dash2_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.gammadash2dash2_impl(data, quadpoints_phi, quadpoints_theta);
}

template<class Array>
void SurfaceNewQuadPoints<Array>::gammadash2_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.gammadash2_impl(data, quadpoints_phi, quadpoints_theta);
}

template<class Array>
void SurfaceNewQuadPoints<Array>::dgamma_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.dgamma_by_dcoeff_impl(data, quadpoints_phi, quadpoints_theta);
}

template<class Array>
void SurfaceNewQuadPoints<Array>::dgammadash1_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.dgammadash1_by_dcoeff_impl(data, quadpoints_phi, quadpoints_theta);
}

template<class Array>
void SurfaceNewQuadPoints<Array>::dgammadash1dash2_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.dgammadash1dash2_by_dcoeff_impl(data, quadpoints_phi, quadpoints_theta);
}

template<class Array>
void SurfaceNewQuadPoints<Array>::dgammadash1dash1_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.dgammadash1dash1_by_dcoeff_impl(data, quadpoints_phi, quadpoints_theta);
}

template<class Array>
void SurfaceNewQuadPoints<Array>::dgammadash2_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.dgammadash2_by_dcoeff_impl(data, quadpoints_phi, quadpoints_theta);
}

template<class Array>
void SurfaceNewQuadPoints<Array>::dgammadash2dash2_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
    parent_surface.dgammadash2dash2_by_dcoeff_impl(data, quadpoints_phi, quadpoints_theta);
}

#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;
template class SurfaceNewQuadPoints<Array>;
