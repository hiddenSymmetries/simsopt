#pragma once

#include <tuple>
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
#include "xtensor/xview.hpp"
typedef xt::pyarray<double> Array;
typedef xt::pyarray<int> IntArray;

std::tuple<Array,IntArray,IntArray,Array,IntArray,Array,Array,Array> GSCO(
    bool no_crossing, bool no_new_coils, bool match_current,
    Array& A_obj, Array& b_obj, double default_current, double max_current, 
    int max_loop_count, IntArray& loops, IntArray& free_loops, 
    IntArray& segments, IntArray& connections, double lambda_P, int max_iter, 
    Array& x_init, IntArray& loop_count_init, int print_interval);


void record_iter(int iter, double curr, int loop_ind, double f_B, double f_S,  
                 double f, int* iter_hist_ptr, double* curr_hist_ptr,
                 int* loop_hist_ptr, double* f_B_hist_ptr, double* f_S_hist_ptr,
                 double* f_hist_ptr);

void print_iter(int iter, double f_B, double f_S, double f);

double compute_f_S(Array& x, double tol);

void check_eligibility(int nLoops, double default_current, double max_current, 
                       int max_loop_count, bool no_crossing, bool no_new_coils, 
                       bool match_current, double tol, int* loops_rep, 
                       int* freeLoops, int* loop_count, int* segments, 
                       int* connections, double* x, double* current);

