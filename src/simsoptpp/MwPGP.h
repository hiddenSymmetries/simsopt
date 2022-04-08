#pragma once

#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;
using std::vector;
using std::string;

Array& phi_MwPGP(const Array& x, const Array& g, const vector<double>& m_maxima);
Array& beta_tilde(const Array& x, const Array& g, const double alpha, const vector<double>& m_maxima);
Array& g_reduced_gradient(Array& x, Array& g, const double alpha, const vector<double> m_maxima);
Array& g_reduced_projected_gradient(Array& x, Array& g, const double alpha, const vector<double> m_maxima);
double find_max_alphaf(const Array& x, const Array& p);
void MwPGP_algorithm(Array& ATA, vector<double>& ATb, vector<double>& m_proxy, vector<double> m0, double nu, double delta, double epsilon, double reg_l0, double reg_l1, double reg_l2, double reg_l2_shifted, int max_iter, bool verbose);
