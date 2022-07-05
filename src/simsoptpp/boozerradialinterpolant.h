#include "xtensor-python/pyarray.hpp"
typedef xt::pyarray<double> Array;

Array fourier_transform_odd(Array& K, Array& xm, Array& xn, Array& thetas, Array& zetas);
Array fourier_transform_even(Array& K, Array& xm, Array& xn, Array& thetas, Array& zetas);
void inverse_fourier_transform_odd(Array& K, Array& kmns, Array& xm, Array& xn, Array& thetas, Array& zetas);
void inverse_fourier_transform_even(Array& K, Array& kmns, Array& xm, Array& xn, Array& thetas, Array& zetas);
Array compute_kmns(Array& rmnc, Array& drmncds, Array& zmns, Array& dzmnsds, Array& numns, Array& dnumnsds, Array& bmnc, Array& iota, Array& G, Array& I, Array& xm, Array& xn, Array& thetas, Array& zetas);
Array compute_kmnc_kmns(Array& rmnc, Array& drmncds, Array& zmns, Array& dzmnsds,
    Array& numns, Array& dnumnsds, Array& bmnc,
    Array& rmns, Array& drmnsds, Array& zmnc, Array& dzmncds,
    Array& numnc, Array& dnumncds, Array& bmns,
    Array& iota, Array& G, Array& I, Array& xm, Array& xn, Array& thetas, Array& zetas);
