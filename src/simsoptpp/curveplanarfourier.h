#pragma once

#include "curve.h"

template<class Array>
class CurvePlanarFourier : public Curve<Array> {
    /*
       CurvePlanarFourier is a curve that is represented as a plane rotated about the 
       x and y axis using the following Fourier series: 

           r(phi) = \sum_{n=0}^{order} x_{c,n}cos(n*nfp*phi) + \sum_{n=1}^order x_{s,n}sin(n*nfp*phi)
        
        with rotation about an axis and angle determine by a set of quarternions
            q = [cos(\theta/2), x * sin(\theta/2), y * sin(\theta/2), z * sin(\theta/2)]

       The dofs are stored in the order 

           [r_{c,0},...,r_{c,order},r_{s,1},...,r_{s,order},t_{x},t_{y}]

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
        Array q;
        Array center;

        CurvePlanarFourier(int _numquadpoints, int _order, int _nfp, bool _stellsym) : Curve<Array>(_numquadpoints), order(_order), nfp(_nfp), stellsym(_stellsym) {
            rc = xt::zeros<double>({order + 1});
            rs = xt::zeros<double>({order});
            q = xt::zeros<double>({4});
            center = xt::zeros<double>({3});
        }

        CurvePlanarFourier(vector<double> _quadpoints, int _order, int _nfp, bool _stellsym) : Curve<Array>(_quadpoints), order(_order), nfp(_nfp), stellsym(_stellsym) {
            rc = xt::zeros<double>({order + 1});
            rs = xt::zeros<double>({order});
            q = xt::zeros<double>({4});
            center = xt::zeros<double>({3});
        }

        CurvePlanarFourier(Array _quadpoints, int _order, int _nfp, bool _stellsym) : Curve<Array>(_quadpoints), order(_order), nfp(_nfp), stellsym(_stellsym) {
            rc = xt::zeros<double>({order + 1});
            rs = xt::zeros<double>({order});
            q = xt::zeros<double>({4});
            center = xt::zeros<double>({3});
        }

        inline int num_dofs() override {
            return (2*order+1)+7;
        }

        void set_dofs_impl(const vector<double>& dofs) override {
            int counter = 0;
            double s = 0;
            for (int i = 0; i < order + 1; ++i)
                rc.data()[i] = dofs[counter++];
            for (int i = 0; i < order; ++i)
                rs.data()[i] = dofs[counter++];
            for (int i = 0; i < 4; ++i){
                q.data()[i] = dofs[counter++];
                s += q[i] * q[i];
            }
            /* Converts to unit quaternion */
            if(s != 0) {
                for (int i = 0; i < 4; ++i)
                    q.data()[i] = q[i] / std::sqrt(s); 
            }
            else {
                q.data()[0] = 1;
            }
            for (int i = 0; i < 3; ++i){
                center.data()[i] = dofs[counter++];
            }
        }

        vector<double> get_dofs() override {
            auto res = vector<double>(num_dofs(), 0.);
            int counter = 0;
            for (int i = 0; i < order + 1; ++i)
                res[counter++] = rc[i];
            for (int i = 0; i < order; ++i)
                res[counter++] = rs[i];
            for (int i = 0; i < 4; ++i)
                res[counter++] = q[i];
            for (int i = 0; i < 3; ++i)
                res[counter++] = center[i];
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
