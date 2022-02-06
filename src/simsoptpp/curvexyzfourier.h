#pragma once

#include "curve.h"

template<class Array>
class CurveXYZFourier : public Curve<Array> {
    /*
       CurveXYZFourier is a curve that is represented in cartesian
       coordinates using the following Fourier series: 

           x(phi) = \sum_{m=0}^{order} x_{c,m}cos(m*phi) + \sum_{m=1}^order x_{s,m}sin(m*phi)
           y(phi) = \sum_{m=0}^{order} y_{c,m}cos(m*phi) + \sum_{m=1}^order y_{s,m}sin(m*phi)
           z(phi) = \sum_{m=0}^{order} z_{c,m}cos(m*phi) + \sum_{m=1}^order z_{s,m}sin(m*phi)

       The dofs are stored in the order 

           [x_{c,0},x_{s,1},x_{c,1},...,x_{s,order},x_{c,order},y_{c,0},y_{s,1},y_{c,1},...]

       */
    public:
        using Curve<Array>::quadpoints;
        using Curve<Array>::numquadpoints;
        using Curve<Array>::check_the_persistent_cache;
        vector<vector<double>> dofs;
        const int order;

        CurveXYZFourier(int _numquadpoints, int _order) : Curve<Array>(_numquadpoints), order(_order) {
            dofs = vector<vector<double>> {
                vector<double>(2*order+1, 0.), 
                vector<double>(2*order+1, 0.), 
                vector<double>(2*order+1, 0.)
            };
        }

        CurveXYZFourier(vector<double> _quadpoints, int _order) : Curve<Array>(_quadpoints), order(_order) {
            dofs = vector<vector<double>> {
                vector<double>(2*order+1, 0.), 
                vector<double>(2*order+1, 0.), 
                vector<double>(2*order+1, 0.)
            };
        }

        CurveXYZFourier(Array _quadpoints, int _order) : Curve<Array>(_quadpoints), order(_order) {
            dofs = vector<vector<double>> {
                vector<double>(2*order+1, 0.), 
                vector<double>(2*order+1, 0.), 
                vector<double>(2*order+1, 0.)
            };
        }

        inline int num_dofs() override {
            return 3*(2*order+1);
        }

        void set_dofs_impl(const vector<double>& _dofs) override {
            int counter = 0;
            for (int i = 0; i < 3; ++i) {
                dofs[i][0] = _dofs[counter++];
                for (int j = 1; j < order+1; ++j) {
                    dofs[i][2*j-1] = _dofs[counter++];
                    dofs[i][2*j] = _dofs[counter++];
                }
            }
        }

        vector<double> get_dofs() override {
            auto _dofs = vector<double>(num_dofs(), 0.);
            int counter = 0;
            for (int i = 0; i < 3; ++i) {
                _dofs[counter++] = dofs[i][0];
                for (int j = 1; j < order+1; ++j) {
                    _dofs[counter++] = dofs[i][2*j-1];
                    _dofs[counter++] = dofs[i][2*j];
                }
            }
            return _dofs;
        }

        Array& dgamma_by_dcoeff() override {
            return check_the_persistent_cache("dgamma_by_dcoeff", {numquadpoints, 3, num_dofs()}, [this](Array& A) { return dgamma_by_dcoeff_impl(A);});
        }
        Array& dgammadash_by_dcoeff() override {
            return check_the_persistent_cache("dgammadash_by_dcoeff", {numquadpoints, 3, num_dofs()}, [this](Array& A) { return dgammadash_by_dcoeff_impl(A);});
        }
        Array& dgammadashdash_by_dcoeff() override {
            return check_the_persistent_cache("dgammadashdash_by_dcoeff", {numquadpoints, 3, num_dofs()}, [this](Array& A) { return dgammadashdash_by_dcoeff_impl(A);});
        }
        Array& dgammadashdashdash_by_dcoeff() override {
            return check_the_persistent_cache("dgammadashdashdash_by_dcoeff", {numquadpoints, 3, num_dofs()}, [this](Array& A) { return dgammadashdashdash_by_dcoeff_impl(A);});
        }

        void gamma_impl(Array& data, Array& quadpoints) override;
        void gammadash_impl(Array& data) override;
        void gammadashdash_impl(Array& data) override;
        void gammadashdashdash_impl(Array& data) override;
        void dgamma_by_dcoeff_impl(Array& data) override;
        void dgammadash_by_dcoeff_impl(Array& data) override;
        void dgammadashdash_by_dcoeff_impl(Array& data) override;
        void dgammadashdashdash_by_dcoeff_impl(Array& data) override;
        //Array dgamma_by_dcoeff_vjp_impl(Array& v) override;
        //Array dgammadash_by_dcoeff_vjp_impl(Array& v) override;
};
