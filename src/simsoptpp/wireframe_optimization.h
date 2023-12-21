#pragma once

#include <tuple>
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;
typedef xt::pyarray<int> IntArray;

std::tuple<Array,IntArray,Array,Array> GSCO(
    Array& A_obj, Array& b_obj, double current, double max_current, 
    Array& loops, int n_iter, Array& x_init, int nHistory);

void record_history(int hist_ind, int iter, Array& x, double R2, 
                    Array& x_history, Array& R2_history);
