#pragma once

#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include <vector>
#include <cmath>
#include <tuple>

typedef xt::pyarray<double> PyArray;
typedef xt::pytensor<double, 3, xt::layout_type::row_major> Array3D;
using std::vector;

/**
 * @brief Builds the A_tildeF tensor for a single dipole-dipole interaction
 * 
 * Constructs the rank-3 tensor A_F[j, i, l] such that:
 *     F_j = sum_{i,l} m2_i * A_F[j, i, l] * m1_l
 * matches the classical dipole-dipole force formula.
 * 
 * @param r Displacement vector from dipole 1 to dipole 2
 * @return Array3D The A_tildeF tensor of shape (3, 3, 3)
 */
Array3D build_A_tildeF_tensor(const PyArray& r);

/**
 * @brief Builds the full A_F interaction tensor for N dipoles
 * 
 * Builds a (3N, 3N, 3) tensor for N dipoles that encodes all dipole-dipole
 * force interactions. The tensor A_F[i, j, k] represents the force component k
 * on dipole i due to dipole j.
 * 
 * @param positions Array of shape (3N) containing dipole positions
 * @return Array3D The A_F tensor of shape (3N, 3N, 3)
 */
Array3D build_A_F_tensor(const PyArray& positions);

/**
 * @brief Computes forces on all dipoles using the A_F tensor
 * 
 * This function includes optimization to skip calculations for magnets with zero moments,
 * which can significantly improve performance when many magnets have zero moments.
 * 
 * @param moments Array of shape (3N) containing dipole moments
 * @param A_F The interaction tensor from build_A_F_tensor
 * @return PyArray Array of shape (3N) containing forces on each dipole
 */
PyArray dipole_forces_from_A_F(const PyArray& moments, const Array3D& A_F);

/**
 * @brief Computes the squared 2-norm of an array
 * 
 * @param array Input array
 * @return double Sum of squares of array elements
 */
double two_norm_squared(const PyArray& array);

/**
 * @brief Diagnostic test function for performance and correctness
 * 
 * @param N Number of dipoles
 * @param moments Optional dipole moments (if empty, random values generated)
 * @param positions Optional dipole positions (if empty, random values generated)
 * @return double Squared 2-norm of the net forces
 */
double diagnostic_test(int N, const PyArray& moments = PyArray(), const PyArray& positions = PyArray());

/**
 * @brief Calculate forces when activating a new magnet and return the 2-norm squared of forces
 * 
 * This function takes the current magnet array with moments and forces, and iterates them to account for changes
 * when a new magnet is activated. It identifies active magnets, calculates force interactions between the new
 * magnet j and all existing active magnets, updates a clone of the forces array, and returns the 2-norm squared.
 * The function only returns the modified forces norm if the j magnet is actually used (has non-zero moments).
 * 
 * @param moments Reference to dipole moments array (3N elements)
 * @param forces Reference to forces array (3N elements) - not modified, a clone is used internally
 * @param j_index Index of the dipole being "activated"
 * @param dipole_grid_xyz Array of dipole positions (3N elements)
 * @param sign Orientation of new magnet (default: 1 for positive)
 * @return std::tuple<PyArray, double> The modified forces array and its two-norm squared
 */
std::tuple<PyArray, double> Abbv_Force_Calc(const PyArray& moments, const PyArray& forces, int j_index, const PyArray& dipole_grid_xyz, int sign = 1);

/**
 * @brief Generate random dipole moments and positions
 * 
 * @param N Number of dipoles
 * @param moments Output array for dipole moments
 * @param positions Output array for dipole positions
 */
void random_dipoles_and_positions(int N, PyArray& moments, PyArray& positions); 