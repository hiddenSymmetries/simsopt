#pragma once

#include <cmath>
#include <tuple>  // c++ tuples
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;
typedef xt::pyarray<int> Array_INT;
#include <functional>
#include <vector>
#include <algorithm>  // std::min_element function
using std::vector;

Array current_voxels_geo_factors(Array& points, Array& integration_points, Array& plasma_normal, Array& Phi);
std::tuple<Array, Array_INT> current_voxels_flux_jumps(Array& coil_points, Array& Phi, double dx, double dy, double dz);
Array_INT connections(Array& coil_points, double dx, double dy, double dz);
