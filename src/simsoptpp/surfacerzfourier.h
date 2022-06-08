#pragma once

#include "surface.h"

template<class Array>
class SurfaceRZFourier : public Surface<Array> {
    /*
       SurfaceRZFourier is a surface that is represented in cylindrical
       coordinates using the following Fourier series:

           r(theta, phi) = \sum_{m=0}^{mpol} \sum_{n=-ntor}^{ntor} [
               r_{c,m,n} \cos(m \theta - n nfp \phi)
               + r_{s,m,n} \sin(m \theta - n nfp \phi) ]

       and the same for z(theta, phi).

       Here, (r, phi, z) are standard cylindrical coordinates, and theta
       is any poloidal angle.

       Note that for m=0 we skip the n<0 term for the cos terms, and the n<=0
       for the sin terms.

       In addition, in the stellsym=True case, we skip the sin terms for r, and
       the cos terms for z.
       */

    public:
        using Surface<Array>::quadpoints_phi;
        using Surface<Array>::quadpoints_theta;
        using Surface<Array>::numquadpoints_phi;
        using Surface<Array>::numquadpoints_theta;
        Array rc;
        Array rs;
        Array zc;
        Array zs;
        int nfp;
        int mpol;
        int ntor;
        bool stellsym;

        SurfaceRZFourier(int _mpol, int _ntor, int _nfp, bool _stellsym, vector<double> _quadpoints_phi, vector<double> _quadpoints_theta)
            : Surface<Array>(_quadpoints_phi, _quadpoints_theta), mpol(_mpol), ntor(_ntor), nfp(_nfp), stellsym(_stellsym) {
                this->allocate();
            }

        void allocate() {
            rc = xt::zeros<double>({mpol+1, 2*ntor+1});
            rs = xt::zeros<double>({mpol+1, 2*ntor+1});
            zc = xt::zeros<double>({mpol+1, 2*ntor+1});
            zs = xt::zeros<double>({mpol+1, 2*ntor+1});
        }

        int num_dofs() override {
            if(stellsym)
                return 2*(mpol+1)*(2*ntor+1) - ntor - (ntor+1);
            else
                return 4*(mpol+1)*(2*ntor+1) - 2*ntor - 2*(ntor+1);
        }

        void set_dofs_impl(const vector<double>& dofs) override {
            int shift = (mpol+1)*(2*ntor+1);
            int counter = 0;
            if(stellsym) {
                for (int i = ntor; i < shift; ++i)
                    rc.data()[i] = dofs[counter++];
                for (int i = ntor+1; i < shift; ++i)
                    zs.data()[i] = dofs[counter++];

            } else {
                for (int i = ntor; i < shift; ++i)
                    rc.data()[i] = dofs[counter++];
                for (int i = ntor+1; i < shift; ++i)
                    rs.data()[i] = dofs[counter++];
                for (int i = ntor; i < shift; ++i)
                    zc.data()[i] = dofs[counter++];
                for (int i = ntor+1; i < shift; ++i)
                    zs.data()[i] = dofs[counter++];
            }
        }

        vector<double> get_dofs() override {
            auto res = vector<double>(num_dofs(), 0.);
            int shift = (mpol+1)*(2*ntor+1);
            int counter = 0;
            if(stellsym) {
                for (int i = ntor; i < shift; ++i)
                    res[counter++] = rc.data()[i];
                for (int i = ntor+1; i < shift; ++i)
                    res[counter++] = zs.data()[i];
            } else {
                for (int i = ntor; i < shift; ++i)
                    res[counter++] = rc.data()[i];
                for (int i = ntor+1; i < shift; ++i)
                    res[counter++] = rs.data()[i];
                for (int i = ntor; i < shift; ++i)
                    res[counter++] = zc.data()[i];
                for (int i = ntor+1; i < shift; ++i)
                    res[counter++] = zs.data()[i];
            }
            return res;
        }

        void gamma_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override;
        void gamma_lin(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override;
        void gammadash1_impl(Array& data) override;
        void gammadash2_impl(Array& data) override;
        void gammadash1dash1_impl(Array& data) override;
        void gammadash2dash2_impl(Array& data) override;
        void gammadash1dash2_impl(Array& data) override;

        void dgamma_by_dcoeff_impl(Array& data) override;
        void dgammadash1_by_dcoeff_impl(Array& data) override;
        void dgammadash2_by_dcoeff_impl(Array& data) override;
        void dgammadash1dash1_by_dcoeff_impl(Array& data) override;
        void dgammadash1dash2_by_dcoeff_impl(Array& data) override;
        void dgammadash2dash2_by_dcoeff_impl(Array& data) override;
        Array dgamma_by_dcoeff_vjp(Array& v) override;
        Array dgammadash1_by_dcoeff_vjp(Array& v) override;
        Array dgammadash2_by_dcoeff_vjp(Array& v) override;
};
