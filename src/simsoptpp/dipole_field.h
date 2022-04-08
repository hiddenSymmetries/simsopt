#pragma once

#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
#include "xtensor-blas/xlinalg.hpp"       // Numpy functions 
typedef xt::pyarray<double> Array;
using std::vector;
using std::string;

Array dipole_field_B(Array& points, Array& m_points, vector<double>& m);
