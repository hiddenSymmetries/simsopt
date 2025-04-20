#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <cstddef> // for size_t
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;

Array net_force_matrix(Array& magnetMoments, Array& magnetPositions);