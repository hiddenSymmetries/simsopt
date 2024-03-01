#pragma once

#include "curve.h"

template<class Array>
class CurveRZFourier : public Curve<Array> {
    /*
       CurveRZFourier is a curve that is represented in cylindrical
       coordinates using the following Fourier series: 

           r(phi) = \sum_{n=0}^{order} x_{c,n}cos(n*nfp*phi) + \sum_{n=1}^order x_{s,n}sin(n*nfp*phi)
           z(phi) = \sum_{n=0}^{order} z_{c,n}cos(n*nfp*phi) + \sum_{n=1}^order z_{s,n}sin(n*nfp*phi)

       If stellsym = true, then the sin terms for r and the cos terms for z are zero.

       For the stellsym = False case, the dofs are stored in the order 

           [r_{c,0},...,r_{c,order},r_{s,1},...,r_{s,order},z_{c,0},....]

       or in the stellsym = true case they are stored 

           [r_{c,0},...,r_{c,order},z_{s,1},...,z_{s,order}]
       */
    public:
        const int order;
        const int nfp;
        const bool stellsym;
        using Curve<Array>::quadpoints;
        using Curve<Array>::numquadpoints;
        using Curve<Array>::check_the_persistent_cache;

        Array rc;
        Array rs;
        Array zc;
        Array zs;

        CurveRZFourier(int _numquadpoints, int _order, int _nfp, bool _stellsym) : Curve<Array>(_numquadpoints), order(_order), nfp(_nfp), stellsym(_stellsym) {
            rc = xt::zeros<double>({order + 1});
            rs = xt::zeros<double>({order});
            zc = xt::zeros<double>({order + 1});
            zs = xt::zeros<double>({order});
        }

        CurveRZFourier(vector<double> _quadpoints, int _order, int _nfp, bool _stellsym) : Curve<Array>(_quadpoints), order(_order), nfp(_nfp), stellsym(_stellsym) {
            rc = xt::zeros<double>({order + 1});
            rs = xt::zeros<double>({order});
            zc = xt::zeros<double>({order + 1});
            zs = xt::zeros<double>({order});
        }

        CurveRZFourier(Array _quadpoints, int _order, int _nfp, bool _stellsym) : Curve<Array>(_quadpoints), order(_order), nfp(_nfp), stellsym(_stellsym) {
            rc = xt::zeros<double>({order + 1});
            rs = xt::zeros<double>({order});
            zc = xt::zeros<double>({order + 1});
            zs = xt::zeros<double>({order});
        }

        inline int num_dofs() override {
            if(stellsym)
                return 2*order+1;
            else
                return 2*(2*order+1);
        }

        void set_dofs_impl(const vector<double>& dofs) override {
            int counter = 0;
            if(stellsym) {
                for (int i = 0; i < order + 1; ++i)
                    rc.data()[i] = dofs[counter++];
                for (int i = 0; i < order; ++i)
                    zs.data()[i] = dofs[counter++];
            } else {
                for (int i = 0; i < order + 1; ++i)
                    rc.data()[i] = dofs[counter++];
                for (int i = 0; i < order; ++i)
                    rs.data()[i] = dofs[counter++];
                for (int i = 0; i < order + 1; ++i)
                    zc.data()[i] = dofs[counter++];
                for (int i = 0; i < order; ++i)
                    zs.data()[i] = dofs[counter++];
            }
        }

        vector<double> get_dofs() override {
            auto res = vector<double>(num_dofs(), 0.);
            int counter = 0;
            if(stellsym) {
                for (int i = 0; i < order + 1; ++i)
                    res[counter++] = rc[i];
                for (int i = 0; i < order; ++i)
                    res[counter++] = zs[i];
            } else {
                for (int i = 0; i < order + 1; ++i)
                    res[counter++] = rc[i];
                for (int i = 0; i < order; ++i)
                    res[counter++] = rs[i];
                for (int i = 0; i < order + 1; ++i)
                    res[counter++] = zc[i];
                for (int i = 0; i < order; ++i)
                    res[counter++] = zs[i];
            }
            return res;
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

};
