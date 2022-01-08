#include "boozermagneticfield.h"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;

template<template<class, std::size_t, xt::layout_type> class T>
xt::pyarray<double> bounce_integral(shared_ptr<BoozerMagneticField<T>> field, double s, double theta0, int nzeta, double lam, int nfp, int nmax, int ntransitmax, bool jpar, bool psidot, bool alphadot, bool ihat, bool khat, bool dkhatdalpha, bool tau, double step_size, int digits, double tol);

template<template<class, std::size_t, xt::layout_type> class T>
double vprime(shared_ptr<BoozerMagneticField<T>> field, double s, double theta0, int nzeta, int nfp, int nmax, double step_size, int digits);
