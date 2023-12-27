#pragma once

#include <tuple>
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;
typedef xt::pyarray<int> IntArray;

std::tuple<Array,IntArray,Array,Array,Array,Array> GSCO(
    Array& A_obj, Array& b_obj, double current, double max_current, 
    IntArray& cells, IntArray& free_cells, IntArray& cell_neighbors, 
    double lambda_P, int nIter, Array& x_init, IntArray& cell_count_init, 
    int nHistory);

std::tuple<Array,IntArray,Array,Array> GSCO_orig(
    Array& A_obj, Array& b_obj, double current, double max_current, 
    Array& loops, int n_iter, Array& x_init, int nHistory);

void record_history(int hist_ind, int iter, Array& x, double Chi2_B, 
                    double Chi2_P, double Chi2, Array& x_history, 
                    Array& Chi2_B_history, Array& Chi2_P_history, 
                    Array& Chi2_history);

void record_history_orig(int hist_ind, int iter, Array& x, double R2, 
                         Array& x_history, Array& R2_history);

double compute_chi2_P(Array& x, double tol);

