#include <cmath>
#include <iostream>
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;

double heaviside(double x1, double x2);
Array Pd(double phi, double theta);
Array iterate_over_corners(Array corner, double x, double y, double z);