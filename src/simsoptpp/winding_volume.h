#pragma once

#include <cmath>
#include <tuple>  // c++ tuples
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;
typedef xt::pyarray<int> Array_INT;
#include <functional>
#include <vector>
#include <algorithm>  // std::min_element function
#include <Eigen/Dense>
#include <Eigen/Sparse>
using std::vector;

Array winding_volume_geo_factors(Array& points, Array& integration_points, Array& plasma_normal, Array& Phi);
std::tuple<Array, Array_INT> winding_volume_flux_jumps(Array& coil_points, Array& Phi, double dx, double dy, double dz);
Array make_winding_volume_grid(Array& normal_inner, Array& normal_outer, Array& dipole_grid_xyz, Array& xyz_inner, Array& xyz_outer);
Array_INT connections(Array& coil_points, double dx, double dy, double dz);
Array acc_prox_grad_descent(Eigen::SparseMatrix<double, Eigen::RowMajor> eigen_P, Array& B, Array& I, Array& bB, Array& bI, Array& alpha_initial, double lam, double initial_step, int max_iter);