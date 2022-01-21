#include "boozermagneticfield.h"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;

template<template<class, std::size_t, xt::layout_type> class T>
xt::pyarray<double> bounce_integral(std::vector<double> bouncel, std::vector<double> bouncer, shared_ptr<BoozerMagneticField<T>> field, double s, double theta0, double lam, int nfp, int ntransitmax, bool jpar, bool psidot, bool alphadot, bool ihat, bool khat, bool dkhatdalpha, bool tau, double step_size, double tol, double dt_max, bool adjust);

template<template<class, std::size_t, xt::layout_type> class T>
double vprime(shared_ptr<BoozerMagneticField<T>> field, double s, double theta0, int nzeta, int nfp, int nmax, double step_size, int digits);

template<template<class, std::size_t, xt::layout_type> class T>
std::vector<double> find_bounce_points(shared_ptr<BoozerMagneticField<T>> field, double s, double theta0, double zeta0, int nzeta, double lam, int nfp, int nmax, int digits, int option, double derivative_tol, double argmin_tol, double root_tol);
