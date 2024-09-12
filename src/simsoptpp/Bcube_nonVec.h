#pragma once

#include <cmath>
#include <iostream>
#include <cmath>
#include <omp.h>
#include <tuple>
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;

double heaviside(double x1, double x2);

Array Pd(double phi, double theta);

Array iterate_over_corners(int i, int j, int k, Array& x, Array& y, Array& z);

Array Hd_i_prime(double rx_loc, double ry_loc, double rz_loc, double dimx, double dimy, double dimz);

Array B_direct(Array& points, Array& magPos, Array& M, Array& dims, Array& phiThetas);

Array Bn_direct(Array& point, Array& magPos, Array& M, Array& norms, Array& dims, Array& phiThetas);

std::tuple<double, double, double> gd_i(double rx_loc, double ry_loc, double rz_loc, double nx_loc, \
    double ny_loc, double nz_loc, double dimx, double dimy, double dimz);

Array Acube(Array& points, Array& magPos, Array& norms, Array& dims, Array& phithetas);

Array Bn_fromMat(Array& points, Array& magPos, Array& M, Array& norms, Array& dims, Array& phiThetas);