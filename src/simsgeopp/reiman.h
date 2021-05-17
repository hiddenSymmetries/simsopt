#include "xtensor-python/pyarray.hpp"
typedef xt::pyarray<double> Array;
Array ReimanB(double& iota0, double& iota1, Array& k_theta, Array& epsilon, int& m0_symmetry, Array& points);
Array ReimandB(double& iota0, double& iota1, Array& k_theta, Array& epsilon, int& m0_symmetry, Array& points);
