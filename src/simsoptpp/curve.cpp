#include "curve.h"

template<class Array>
void Curve<Array>::least_squares_fit(Array& target_values) {
    if(target_values.shape(0) != numquadpoints)
        throw std::runtime_error("Wrong first dimension for target_values. Should match numquadpoints.");
    if(target_values.shape(1) != 3)
        throw std::runtime_error("Wrong third dimension for target_values. Should be 3.");

    if(!qr){
        auto dg_dc = this->dgamma_by_dcoeff();
        Eigen::MatrixXd A = Eigen::MatrixXd(numquadpoints*3, num_dofs());
        int counter = 0;
        for (int i = 0; i < numquadpoints; ++i) {
            for (int d = 0; d < 3; ++d) {
                for (int c = 0; c  < num_dofs(); ++c ) {
                    A(counter, c) = dg_dc(i, d, c);
                }
                counter++;
            }
        }
        qr = std::make_unique<Eigen::FullPivHouseholderQR<Eigen::MatrixXd>>(A.fullPivHouseholderQr());
    }
    Eigen::VectorXd b = Eigen::VectorXd(numquadpoints*3);
    int counter = 0;
    for (int i = 0; i < numquadpoints; ++i) {
        for (int d = 0; d < 3; ++d) {
            b(counter++) = target_values(i, d);
        }
    }
    Eigen::VectorXd x = qr->solve(b);
    vector<double> dofs(x.data(), x.data() + x.size());
    this->set_dofs(dofs);
}

template<class Array>
void Curve<Array>::incremental_arclength_impl(Array& data) { 
    auto dg = this->gammadash();
    for (int i = 0; i < numquadpoints; ++i) {
        data(i) = std::sqrt(dg(i, 0)*dg(i, 0)+dg(i, 1)*dg(i, 1)+dg(i, 2)*dg(i, 2));
    }
};

template<class Array>
void Curve<Array>::dincremental_arclength_by_dcoeff_impl(Array& data) {
    auto dg = this->gammadash();
    auto dgdc = this->dgammadash_by_dcoeff();
    auto l = this->incremental_arclength();
    for (int i = 0; i < numquadpoints; ++i) {
        for (int c = 0; c < num_dofs(); ++c) {
            data(i, c) = (1./l(i)) * (dg(i, 0) * dgdc(i, 0, c) + dg(i, 1) * dgdc(i, 1, c) + dg(i, 2) * dgdc(i, 2, c));
        }
    }
};


#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;
template class Curve<Array>;
