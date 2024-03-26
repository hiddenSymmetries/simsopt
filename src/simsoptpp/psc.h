#pragma once

// #define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1
// #include <tr1/cmath>
// #define BOOST_CONFIG_SUPPRESS_OUTDATED_MESSAGE
// #include <boost/lambda/lambda.hpp>
#include <boost/math/special_functions/ellint_1.hpp>  
#include <boost/math/special_functions/ellint_2.hpp>  
#include <cmath>
#include <tuple>  // c++ tuples
#include <string> // for string class
#include <iostream>
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;

Array L_matrix(Array& points, Array& alphas, Array& deltas, Array& phi, double R);

Array L_deriv(Array& points, Array& alphas, Array& deltas, Array& phi, double R);
// 
// Array TF_fluxes(Array& points, Array& alphas, Array& deltas, Array& rho, Array& phi, Array& I, Array& normal, double R);
Array flux_xyz(Array& points, Array& alphas, Array& deltas, Array& rho, Array& phi, Array& normal);

Array flux_integration(Array& B, Array& rho, Array& normal);

Array A_matrix(Array& points, Array& plasma_points, Array& alphas, Array& deltas, Array& plasma_normal, double R);

Array B_PSC(Array& points, Array& plasma_points, Array& alphas, Array& deltas, Array& psc_currents, double R);
// 
// Array A_matrix(Array& plasma_surface_normals, Array& B, Array& alphas, Array& deltas);