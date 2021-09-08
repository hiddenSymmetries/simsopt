#include "curvexyzfourier.h"

template<class Array>
void CurveXYZFourier<Array>::gamma_impl(Array& data, Array& quadpoints) {
    int numquadpoints = quadpoints.size();
    data *= 0;
    for (int k = 0; k < numquadpoints; ++k) {
        for (int i = 0; i < 3; ++i) {
            data(k, i) += dofs[i][0];
            for (int j = 1; j < order+1; ++j) {
                data(k, i) += dofs[i][2*j-1]*sin(2*M_PI*j*quadpoints[k]);
                data(k, i) += dofs[i][2*j]*cos(2*M_PI*j*quadpoints[k]);
            }
        }
    }
}

template<class Array>
void CurveXYZFourier<Array>::gammadash_impl(Array& data) {
    data *= 0;
    for (int k = 0; k < numquadpoints; ++k) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 1; j < order+1; ++j) {
                data(k, i) += +dofs[i][2*j-1]*2*M_PI*j*cos(2*M_PI*j*quadpoints[k]);
                data(k, i) += -dofs[i][2*j]*2*M_PI*j*sin(2*M_PI*j*quadpoints[k]);
            }
        }
    }
}

template<class Array>
void CurveXYZFourier<Array>::gammadashdash_impl(Array& data) {
    data *= 0;
    for (int k = 0; k < numquadpoints; ++k) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 1; j < order+1; ++j) {
                data(k, i) += -dofs[i][2*j-1] * (2*M_PI*j)*(2*M_PI*j)*sin(2*M_PI*j*quadpoints[k]);
                data(k, i) += -dofs[i][2*j]   * (2*M_PI*j)*(2*M_PI*j)*cos(2*M_PI*j*quadpoints[k]);
            }
        }
    }
}

template<class Array>
void CurveXYZFourier<Array>::gammadashdashdash_impl(Array& data) {
    data *= 0;
    for (int k = 0; k < numquadpoints; ++k) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 1; j < order+1; ++j) {
                data(k, i) += -dofs[i][2*j-1] * (2*M_PI*j)*(2*M_PI*j)*(2*M_PI*j)*cos(2*M_PI*j*quadpoints[k]);
                data(k, i) += +dofs[i][2*j]   * (2*M_PI*j)*(2*M_PI*j)*(2*M_PI*j)*sin(2*M_PI*j*quadpoints[k]);
            }
        }
    }
}

template<class Array>
void CurveXYZFourier<Array>::dgamma_by_dcoeff_impl(Array& data) {
    for (int k = 0; k < numquadpoints; ++k) {
        for (int i = 0; i < 3; ++i) {
            data(k, i, i*(2*order+1)) = 1.;
            for (int j = 1; j < order+1; ++j) {
                data(k, i, i*(2*order+1) + 2*j-1) = sin(2*M_PI*j*quadpoints[k]);
                data(k, i, i*(2*order+1) + 2*j  ) = cos(2*M_PI*j*quadpoints[k]);
            }
        }
    }
}

template<class Array>
void CurveXYZFourier<Array>::dgammadash_by_dcoeff_impl(Array& data) {
    for (int k = 0; k < numquadpoints; ++k) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 1; j < order+1; ++j) {
                data(k, i, i*(2*order+1) + 2*j-1) = +2*M_PI*j*cos(2*M_PI*j*quadpoints[k]);
                data(k, i, i*(2*order+1) + 2*j  ) = -2*M_PI*j*sin(2*M_PI*j*quadpoints[k]);
            }
        }
    }
}

template<class Array>
void CurveXYZFourier<Array>::dgammadashdash_by_dcoeff_impl(Array& data) {
    for (int k = 0; k < numquadpoints; ++k) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 1; j < order+1; ++j) {
                data(k, i, i*(2*order+1) + 2*j-1) = -(2*M_PI*j)*(2*M_PI*j)*sin(2*M_PI*j*quadpoints[k]);
                data(k, i, i*(2*order+1) + 2*j  ) = -(2*M_PI*j)*(2*M_PI*j)*cos(2*M_PI*j*quadpoints[k]);
            }
        }
    }
}

template<class Array>
void CurveXYZFourier<Array>::dgammadashdashdash_by_dcoeff_impl(Array& data) {
    for (int k = 0; k < numquadpoints; ++k) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 1; j < order+1; ++j) {
                data(k, i, i*(2*order+1) + 2*j-1) = -(2*M_PI*j)*(2*M_PI*j)*(2*M_PI*j)*cos(2*M_PI*j*quadpoints[k]);
                data(k, i, i*(2*order+1) + 2*j  ) = +(2*M_PI*j)*(2*M_PI*j)*(2*M_PI*j)*sin(2*M_PI*j*quadpoints[k]);
            }
        }
    }
}
#include "xtensor/xlayout.hpp"

//template<class Array>
//Array CurveXYZFourier<Array>:: dgamma_by_dcoeff_vjp_impl(Array& v) {
//    if(v.layout() != xt::layout_type::row_major)
//          throw std::runtime_error("v needs to be in row-major storage order");
//    Array res = xt::zeros<double>({num_dofs()});
//    double* resptr = res.data();
//    double* vptr = v.data();
//#pragma omp parallel for
//    for (int k = 0; k < numquadpoints; ++k) {
//        for (int i = 0; i < 3; ++i) {
//            resptr[i*(2*order+1)] += vptr[3*k+i];
//            for (int j = 1; j < order+1; ++j) {
//                resptr[i*(2*order+1) + 2*j-1] += sin(2*M_PI*j*quadpoints[k]) * vptr[3*k+i];
//                resptr[i*(2*order+1) + 2*j] += cos(2*M_PI*j*quadpoints[k]) * vptr[3*k+i];
//            }
//        }
//    }
//    return res;
//}

//template<class Array>
//Array CurveXYZFourier<Array>:: dgammadash_by_dcoeff_vjp_impl(Array& v) {
//    if(v.layout() != xt::layout_type::row_major)
//          throw std::runtime_error("v needs to be in row-major storage order");
//    Array res = xt::zeros<double>({num_dofs()});
//    double* resptr = res.data();
//    double* vptr = v.data();
//#pragma omp parallel for
//    for (int k = 0; k < numquadpoints; ++k) {
//        for (int i = 0; i < 3; ++i) {
//            for (int j = 1; j < order+1; ++j) {
//                resptr[i*(2*order+1) + 2*j-1] += +2*M_PI*j*cos(2*M_PI*j*quadpoints[k]) * vptr[3*k+i];
//                resptr[i*(2*order+1) + 2*j] += -2*M_PI*j*sin(2*M_PI*j*quadpoints[k]) * vptr[3*k+i];
//            }
//        }
//    }
//    return res;
//}

#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;
template class CurveXYZFourier<Array>;
