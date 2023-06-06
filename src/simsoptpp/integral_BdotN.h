#include "xtensor-python/pyarray.hpp"
typedef xt::pyarray<double> PyArray;

double integral_BdotN(PyArray& Bcoil, PyArray& Btarget, PyArray& n, std::string definition);
