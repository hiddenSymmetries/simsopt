#pragma once

#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;

Array current_voxels_field_B(Array& points, Array& integration_points, Array& J);
Array current_voxels_field_Bext(Array& points, Array& integration_points, Array& Phi, Array& plasma_unitnormal);
Array current_voxels_field_B_SIMD(Array& points, Array& integration_points, Array& J); 
Array current_voxels_field_Bext_SIMD(Array& points, Array& integration_points, Array& Phi, Array& plasma_unitnormal); 
