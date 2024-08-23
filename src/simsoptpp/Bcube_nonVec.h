
#include <cmath>
#include <iostream>
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;

double heaviside(double x1, double x2);

Array Pd(double phi, double theta);

Array iterate_over_corners(Array corner, Array x, Array y, Array z);

Array Hd_i_prime(Array r, Array dims);

Array B_direct(Array points, Array magPos, Array M, Array dims, Array phiThetas);

Array Bn_direct(Array point, Array magPos, Array M, Array norms, Array dims, Array phiThetas);

Array gd_i(Array r_loc, Array n_i_loc, Array dims);

Array Acube(Array points, Array magPos, Array norms, Array dims, Array phithetas);

Array Bn_fromMat(Array points, Array magPos, Array M, Array norms, Array dims, Array phiThetas);