#include "boozermagneticfield.h"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;

template<template<class, std::size_t, xt::layout_type> class T>
xt::pyarray<double> bounce_integral(std::vector<double> bouncel, std::vector<double> bouncer, shared_ptr<BoozerMagneticField<T>> field, double s, double theta0, double lam, int nfp, bool jpar, bool psidot, bool alphadot, bool ihat, bool khat, bool dkhatdalpha, bool tau, double step_size=1e-3, double tol=1e-8, bool adjust=false);

template<template<class, std::size_t, xt::layout_type> class T>
std::vector<double> find_bounce_points(shared_ptr<BoozerMagneticField<T>> field, double s, double theta0, double lam, int nfp, int option, int nmax, int nzeta=1000, int digits=16, double derivative_tol=1e-3, double argmin_tol=1e-3, double root_tol=1e-8);

template<template<class, std::size_t, xt::layout_type> class T>
double vprime(shared_ptr<BoozerMagneticField<T>> field, double s,
    double theta0, int nfp, int nmax, double step_size=1e-3);
