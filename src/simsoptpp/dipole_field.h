#pragma once

#include <cmath>
#include <tuple>  // c++ tuples
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;

Array dipole_field_B(Array& points, Array& m_points, Array& m);

Array dipole_field_A(Array& points, Array& m_points, Array& m);

Array dipole_field_dB(Array& points, Array& m_points, Array& m);

Array dipole_field_dA(Array& points, Array& m_points, Array& m);

std::tuple<Array, Array, Array> dipole_field_Bn(Array& points, Array& m_points, Array& unitnormal, int nfp, int stellsym, Array& phi, Array& b, bool cylindrical);

std::tuple<Array, Array> make_final_surface(Array& phi, Array& normal_inner, Array& normal_outer, Array& dipole_grid_rz, Array& r_inner, Array& r_outer, Array& z_inner, Array& z_outer);
