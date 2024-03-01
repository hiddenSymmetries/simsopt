#pragma once

#include <cmath>
#include <tuple>  // c++ tuples
#include <string> // for string class
#include <iostream>
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;

Array L_matrix(Array& points, Array& alphas, Array& deltas, Array& phi, double R);
// 
Array TF_fluxes(Array& points, Array& alphas, Array& deltas, Array& rho, Array& phi, Array& B_TF, Array& normal);
// 
// Array A_matrix(Array& plasma_surface_normals, Array& B, Array& alphas, Array& deltas);