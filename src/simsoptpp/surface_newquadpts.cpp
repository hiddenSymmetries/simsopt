#include "surface_newquadpts.h"


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
