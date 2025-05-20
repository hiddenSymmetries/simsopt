#pragma once

#include "curve.h"

template<class Array>
class CurvePlanarFourier : public Curve<Array> {
    /*
        CurvePlanarFourier is a curve that is restricted to lie in a plane. In
        the plane, the curve is represented using a Fourier series in plane polar coordinates:

           r(\phi) = \sum_{m=0}^{\text{order}} r_{c,m}\cos(m \phi) + \sum_{m=1}^{\text{order}} r_{s,m}\sin(m \phi)
        
        The plane is rotated using a quaternion

            q = [q0, qi, qj, qk] = [\cos(\theta / 2), \hat{x}\sin(\theta / 2), \hat{y}\sin(\theta / 2), \hat{z}\sin(\theta / 2)]

        Details of the quaternion rotation can be found for example in pages
        575-576 of https://www.cis.upenn.edu/~cis5150/ws-book-Ib.pdf.

        The simsopt dofs for the quaternion need not generally have unit norm. The
        quaternion is normalized before being applied to the curve to prevent scaling the curve.

        A translation vector is used to specify the location of the center of the curve:

            c = [X, Y, Z]

        The dofs are stored in the order

           [r_{c,0},...,r_{c,order},r_{s,1},...,r_{s,order}, q0, qi, qj, qk, X, Y, Z]

       */
    public:
        const int order;
        using Curve<Array>::quadpoints;
        using Curve<Array>::numquadpoints;
        using Curve<Array>::check_the_persistent_cache;

        Array rc;
        Array rs;
        Array q;
        Array center;

        CurvePlanarFourier(int _numquadpoints, int _order) : Curve<Array>(_numquadpoints), order(_order) {
            rc = xt::zeros<double>({order + 1});
            rs = xt::zeros<double>({order});
            q = xt::zeros<double>({4});
            center = xt::zeros<double>({3});
        }

        CurvePlanarFourier(vector<double> _quadpoints, int _order) : Curve<Array>(_quadpoints), order(_order) {
            rc = xt::zeros<double>({order + 1});
            rs = xt::zeros<double>({order});
            q = xt::zeros<double>({4});
            center = xt::zeros<double>({3});
        }

        CurvePlanarFourier(Array _quadpoints, int _order) : Curve<Array>(_quadpoints), order(_order) {
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
            for (int i = 0; i < order + 1; ++i)
                rc.data()[i] = dofs[counter++];
            for (int i = 0; i < order; ++i)
                rs.data()[i] = dofs[counter++];
            for (int i = 0; i < 4; ++i)
                q.data()[i] = dofs[counter++];
            for (int i = 0; i < 3; ++i)
                center.data()[i] = dofs[counter++];
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

private:
        double inv_magnitude();

};
