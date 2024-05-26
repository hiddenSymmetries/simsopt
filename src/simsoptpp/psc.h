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
#include "simdhelpers.h"
#include "vec3dsimd.h"
typedef xt::pyarray<double> Array;

double Ellint1AGM(double k);

double Ellint2AGM(double k);

simd_t Ellint1AGM_simd(simd_t k);

simd_t Ellint2AGM_simd(simd_t k);

Array L_matrix(Array& points, Array& alphas, Array& deltas, Array& int_points, Array& int_weights);

Array L_deriv(Array& points, Array& alphas, Array& deltas, Array& int_points, Array& int_weights);

Array L_deriv_simd(Array& points, Array& alphas, Array& deltas, Array& int_points, Array& int_weights);

Array flux_xyz(Array& points, Array& alphas, Array& deltas, Array& rho, Array& phi);

Array flux_integration(Array& B, Array& rho, Array& normal);

Array A_matrix(Array& points, Array& plasma_points, Array& alphas, Array& deltas, Array& plasma_normal, double R);
Array A_matrix_simd(Array& points, Array& plasma_points, Array& alphas, Array& deltas, Array& plasma_normal, double R);

Array B_PSC(Array& points, Array& plasma_points, Array& alphas, Array& deltas, Array& psc_currents, double R);

Array dA_dkappa(Array& points, Array& plasma_points, Array& alphas, Array& deltas, Array& plasma_normal, Array& int_points, Array& int_weights, double R);

Array dA_dkappa_simd(Array& points, Array& plasma_points, Array& alphas, Array& deltas, Array& plasma_normal, Array& int_points, Array& int_weights, double R);

Array A_matrix_direct(Array& points, Array& plasma_points, Array& alphas, Array& deltas, Array& plasma_normal, Array& phi, double R);

Array dpsi_dkappa(Array& I_TF, Array& dl_TF, Array& gamma_TF, Array& PSC_points, Array& alphas, Array& deltas, Array& coil_normals, Array& rho, Array& phi, double R);

Array psi_check(Array& I_TF, Array& dl_TF, Array& gamma_TF, Array& PSC_points, Array& alphas, Array& deltas, Array& coil_normals, Array& rho, Array& phi, double R);

Array B_TF(Array& I_TF, Array& dl_TF, Array& gamma_TF, Array& PSC_points);

Array coil_forces(Array& points, Array& B, Array& alphas, Array& deltas, Array& int_points, Array& int_weights);