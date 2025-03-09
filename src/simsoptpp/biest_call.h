#include "BIEST/biest.hpp" // BIEST
// #include "xtensor-python/pyarray.hpp"     // Numpy bindings

#include <numeric>                    // Standard library import for std::accumulate
#include <functional>                 // Allow the user to choose kernel while reducing duplicate codes
#define FORCE_IMPORT_ARRAY            // numpy C api loading
#include <xtensor/xarray.hpp>
#include <xtensor-python/pyarray.hpp> // Numpy bindings
// Only used in sum_of_sines
#include <xtensor/xmath.hpp> // xtensor import for the C++ universal functions

typedef xt::pyarray<double> Array;
typedef sctl::Vector<biest::Surface<Real>> Surface;
constexpr int DIM = 3;      // dimensions of coordinate space
constexpr int KER_DIM0 = 1; // input degrees-of-freedom of kernel
constexpr int KER_DIM1 = 1; // output degrees-of-freedom of kernel

double sum_of_sines(Array &m);
static void plot_in_vtk(Array &gamma, int digits, int nfp);
static void integrate_multi(
    Array &gamma,
    Array &func_in,
    Array &result,
    bool single,
    int digits,
    int nfp);
void test_double(double a, double b, Array &gamma, Array &f_arr, Array &result);
void test_single(double a, double b, Array &gamma, Array &f_arr, Array &result);

