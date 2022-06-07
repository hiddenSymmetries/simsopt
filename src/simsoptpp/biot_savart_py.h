#pragma once

#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;
using std::vector;

void biot_savart(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis, vector<Array>& B, vector<Array>& dB_by_dX, vector<Array>& d2B_by_dXdX);
Array biot_savart_B(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis, vector<double>& currents);
