#pragma once

#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;

Array winding_volume_field_B(Array& points, Array& integration_points, Array& J);
Array winding_volume_field_Bext(Array& points, Array& integration_points, Array& Phi);
