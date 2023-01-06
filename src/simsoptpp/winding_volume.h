#pragma once

#include <cmath>
// #include <tuple>  // c++ tuples
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;

Array winding_volume_geo_factors(Array& points, Array& coil_points, Array& integration_points, Array& plasma_normal, Array& Phi);
Array winding_volume_flux_jumps(Array& coil_points, Array& integration_points, Array& Phi, double dx, double dy, double dz);
Array connections(Array& dipole_grid_xyz, int Nadjacent, int dx, int dy, int dz);
