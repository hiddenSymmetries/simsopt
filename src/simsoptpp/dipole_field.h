#pragma once

#include <tuple>  // c++ tuples
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;

std::tuple<Array, Array, Array> dipole_field_Bn(Array& points, Array& m_points, Array& unitnormal, int nfp, int stellsym, Array& phi, Array& b);
Array dipole_field_B(Array& points, Array& m_points, Array& m);
Array dipole_field_dB(Array& points, Array& m_points, Array& m);
