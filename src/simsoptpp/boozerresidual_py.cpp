#include "boozerresidual_impl.h"
#include "boozerresidual_py.h"

double boozer_residual(double G, double iota, Array& xphi, Array& xtheta, Array& B, bool weight_inv_modB){
    double res = 0.;
    Array dummy  = xt::zeros<double>({1});
    boozer_residual_impl<Array, 0>(G, iota, B, dummy, dummy, xphi, xtheta, dummy, dummy, dummy, res, dummy, dummy, 0, weight_inv_modB);
    return res;
}

std::tuple<double, Array> boozer_residual_ds(double G, double iota, Array& B, Array& dB_dx, Array& xphi, Array& xtheta, Array& dx_ds, Array& dxphi_ds, Array& dxtheta_ds, bool weight_inv_modB){
    size_t ndofs = dx_ds.shape(3);
    
    double res = 0.;
    Array dres  = xt::zeros<double>({ndofs+2});
    Array dummy  = xt::zeros<double>({1});
    boozer_residual_impl<Array, 1>(G, iota, B, dB_dx, dummy, xphi, xtheta, dx_ds, dxphi_ds, dxtheta_ds, res, dres, dummy, ndofs, weight_inv_modB);
    auto tup = std::make_tuple(res, dres);
    return tup;
}

std::tuple<double, Array, Array> boozer_residual_ds2(double G, double iota, Array& B, Array& dB_dx, Array& d2B_dx2, Array& xphi, Array& xtheta, Array& dx_ds, Array& dxphi_ds, Array& dxtheta_ds, bool weight_inv_modB){
    size_t ndofs = dx_ds.shape(3);

    double res = 0.;
    Array dres  = xt::zeros<double>({ndofs+2});
    Array d2res = xt::zeros<double>({ndofs+2, ndofs+2});
    boozer_residual_impl<Array, 2>(G, iota, B, dB_dx, d2B_dx2, xphi, xtheta, dx_ds, dxphi_ds, dxtheta_ds, res, dres, d2res, ndofs, weight_inv_modB);
    auto tup = std::make_tuple(res, dres, d2res);
    return tup;
}

