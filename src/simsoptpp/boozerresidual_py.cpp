#include "boozerresidual_impl.h"
#include "boozerresidual_py.h"

std::complex<double> boozer_residual(std::complex<double> G, std::complex<double> iota, Array& xphi, Array& xtheta, Array& B, bool weight_inv_modB){
    std::complex<double> res = 0.;
    Array dummy  = xt::zeros<std::complex<double>>({1});
    boozer_residual_impl<Array, 0>(G, iota, B, dummy, dummy, xphi, xtheta, dummy, dummy, dummy, res, dummy, dummy, 0, weight_inv_modB);
    return res;
}

std::tuple<std::complex<double>, Array> boozer_residual_ds(std::complex<double> G, std::complex<double> iota, Array& B, Array& dB_dx, Array& xphi, Array& xtheta, Array& dx_ds, Array& dxphi_ds, Array& dxtheta_ds, bool weight_inv_modB){
    size_t ndofs = dx_ds.shape(3);
    
    std::complex<double> res = 0.;
    Array dres  = xt::zeros<std::complex<double>>({ndofs+2});
    Array dummy  = xt::zeros<std::complex<double>>({1});
    boozer_residual_impl<Array, 1>(G, iota, B, dB_dx, dummy, xphi, xtheta, dx_ds, dxphi_ds, dxtheta_ds, res, dres, dummy, ndofs, weight_inv_modB);
    auto tup = std::make_tuple(res, dres);
    return tup;
}

std::tuple<std::complex<double>, Array, Array> boozer_residual_ds2(std::complex<double> G, std::complex<double> iota, Array& B, Array& dB_dx, Array& d2B_dx2, Array& xphi, Array& xtheta, Array& dx_ds, Array& dxphi_ds, Array& dxtheta_ds, bool weight_inv_modB){
    size_t ndofs = dx_ds.shape(3);

    std::complex<double> res = 0.;
    Array dres  = xt::zeros<std::complex<double>>({ndofs+2});
    Array d2res = xt::zeros<std::complex<double>>({ndofs+2, ndofs+2});
    boozer_residual_impl<Array, 2>(G, iota, B, dB_dx, d2B_dx2, xphi, xtheta, dx_ds, dxphi_ds, dxtheta_ds, res, dres, d2res, ndofs, weight_inv_modB);
    auto tup = std::make_tuple(res, dres, d2res);
    return tup;
}

