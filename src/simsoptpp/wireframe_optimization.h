#pragma once

#include <tuple>
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
#include "xtensor/xview.hpp"
typedef xt::pyarray<double> Array;
typedef xt::pyarray<int> IntArray;

std::tuple<Array,IntArray,IntArray,Array,Array,Array,Array> GSCO(
    bool no_crossing, bool no_new_coils, bool match_current,
    Array& A_obj, Array& b_obj, double default_current, double max_current, 
    int max_loop_count, IntArray& loops, IntArray& free_loops, 
    IntArray& segments, IntArray& connections, double lambda_P, int nIter, 
    Array& x_init, IntArray& loop_count_init, int nHistory);


void record_history(int hist_ind, int iter, Array& x, double f_B, double f_S, 
                    double f, IntArray& iter_history, Array& x_history, 
                    Array& f_B_history, Array& f_S_history, Array& f_history);

double compute_chi2_P(Array& x, double tol);

void check_eligibility(int nLoops, double default_current, double max_current, 
                       int max_loop_count, bool no_crossing, bool no_new_coils, 
                       bool match_current, double tol, int* loops_rep, 
                       int* freeLoops, int* loop_count, int* segments, 
                       int* connections, double* x, double* current);

