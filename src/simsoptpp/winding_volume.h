#pragma once

#include <cmath>
#include <tuple>  // c++ tuples
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;
#include <functional>
#include <vector>
#include <algorithm>  // std::min_element function
using std::vector;

Array winding_volume_geo_factors(Array& points, Array& coil_points, Array& integration_points, Array& plasma_normal, Array& Phi);
std::tuple<Array, Array> winding_volume_flux_jumps(Array& coil_points, Array& Phi, double dx, double dy, double dz);
Array connections(Array& dipole_grid_xyz, int Nadjacent, int dx, int dy, int dz);
