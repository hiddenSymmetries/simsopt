#pragma once

#include <tuple>
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;
typedef xt::pyarray<int> IntArray;

std::tuple<Array,IntArray,Array,Array,Array,Array> GSCO(
    Array& A_obj, Array& b_obj, double current, double max_current, 
    IntArray& cells, IntArray& free_cells, double lambda_P, int nIter, 
    Array& x_init, IntArray& cell_count_init, int nHistory);

void record_history(int hist_ind, int iter, Array& x, double f_B, double f_S, 
                    double f, Array& x_history, Array& f_B_history, 
                    Array& f_S_history, Array& f_history);

double compute_chi2_P(Array& x, double tol);

