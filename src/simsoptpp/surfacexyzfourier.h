#pragma once

#include "surface.h"

template<class Array>
class SurfaceXYZFourier : public Surface<Array> {
    /*
       SurfaceXYZFourier is a surface that is represented in cartesian
       coordinates using the following Fourier series:

           \hat x(theta, phi) = \sum_{m=0}^{mpol} \sum_{n=-ntor}^{ntor} [
                 x_{c,m,n} \cos(m \theta - n nfp \phi)
               + x_{s,m,n} \sin(m \theta - n nfp \phi)
           ]

           \hat y(theta, phi) = \sum_{m=0}^{mpol} \sum_{n=-ntor}^{ntor} [
                 y_{c,m,n} \cos(m \theta - n nfp \phi)
               + y_{s,m,n} \sin(m \theta - n nfp \phi)
           ]

           x = \hat x * \cos(\phi) - \hat y * \sin(\phi)
           y = \hat x * \sin(\phi) + \hat y * \cos(\phi)

           z(theta, phi) = \sum_{m=0}^{mpol} \sum_{n=-ntor}^{ntor} [
               z_{c,m,n} \cos(m \theta - n nfp \phi)
               + z_{s,m,n} \sin(m \theta - n nfp \phi)
           ]

       Note that for m=0 we skip the n<0 term for the cos terms, and the n<=0
       for the sin terms.

       When enforcing stellarator symmetry, we set the

           x_{s,*,*}, y_{c,*,*} and z_{c,*,*}

       terms to zero.
       */

    public:
        using Surface<Array>::quadpoints_phi;
        using Surface<Array>::quadpoints_theta;
        using Surface<Array>::numquadpoints_phi;
        using Surface<Array>::numquadpoints_theta;
        Array xc;
        Array xs;
        Array yc;
        Array ys;
        Array zc;
        Array zs;
        int nfp;
        int mpol;
        int ntor;
        bool stellsym;

        SurfaceXYZFourier(int _mpol, int _ntor, int _nfp, bool _stellsym, vector<double> _quadpoints_phi, vector<double> _quadpoints_theta)
            : Surface<Array>(_quadpoints_phi, _quadpoints_theta), mpol(_mpol), ntor(_ntor), nfp(_nfp), stellsym(_stellsym) {
                xc = xt::zeros<double>({mpol+1, 2*ntor+1});
                xs = xt::zeros<double>({mpol+1, 2*ntor+1});
                yc = xt::zeros<double>({mpol+1, 2*ntor+1});
                ys = xt::zeros<double>({mpol+1, 2*ntor+1});
                zc = xt::zeros<double>({mpol+1, 2*ntor+1});
                zs = xt::zeros<double>({mpol+1, 2*ntor+1});
            }

        int num_dofs() override {
            if(stellsym)
                return 3*(mpol+1)*(2*ntor+1) - 1*ntor - 2*(ntor+1);
            else
                return 6*(mpol+1)*(2*ntor+1) - 3*ntor - 3*(ntor+1);
        }

        void set_dofs_impl(const vector<double>& dofs) override {
            int shift = (mpol+1)*(2*ntor+1);
            int counter = 0;
            if(stellsym) {
                for (int i = ntor; i < shift; ++i)
                    xc.data()[i] = dofs[counter++];
                for (int i = ntor+1; i < shift; ++i)
                    ys.data()[i] = dofs[counter++];
                for (int i = ntor+1; i < shift; ++i)
                    zs.data()[i] = dofs[counter++];

            } else {
                for (int i = ntor; i < shift; ++i)
                    xc.data()[i] = dofs[counter++];
                for (int i = ntor+1; i < shift; ++i)
                    xs.data()[i] = dofs[counter++];
                for (int i = ntor; i < shift; ++i)
                    yc.data()[i] = dofs[counter++];
                for (int i = ntor+1; i < shift; ++i)
                    ys.data()[i] = dofs[counter++];
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
                    res[counter++] = xc.data()[i];
                for (int i = ntor+1; i < shift; ++i)
                    res[counter++] = ys.data()[i];
                for (int i = ntor+1; i < shift; ++i)
                    res[counter++] = zs.data()[i];
            } else {
                for (int i = ntor; i < shift; ++i)
                    res[counter++] = xc.data()[i];
                for (int i = ntor+1; i < shift; ++i)
                    res[counter++] = xs.data()[i];
                for (int i = ntor; i < shift; ++i)
                    res[counter++] = yc.data()[i];
                for (int i = ntor+1; i < shift; ++i)
                    res[counter++] = ys.data()[i];
                for (int i = ntor; i < shift; ++i)
                    res[counter++] = zc.data()[i];
                for (int i = ntor+1; i < shift; ++i)
                    res[counter++] = zs.data()[i];
            }
            return res;
        }

        inline double get_coeff(int dim, bool cos, int m, int i) {
            if     (dim == 0 && cos)
                return xc(m, i);
            else if(dim == 0 && !cos)
                return xs(m, i);
            else if(dim == 1 && cos)
                return yc(m, i);
            else if(dim == 1 && !cos)
                return ys(m, i);
            else if(dim == 2 && cos)
                return zc(m, i);
            else
                return zs(m, i);
        }

        void gamma_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override;
        void gamma_lin(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override;
        void gammadash1_impl(Array& data) override;
        void gammadash2_impl(Array& data) override;
        void gammadash1dash1_impl(Array& data) override;
        void gammadash1dash2_impl(Array& data) override;
        void gammadash2dash2_impl(Array& data) override;

        void dgamma_by_dcoeff_impl(Array& data) override;
        void dgammadash1_by_dcoeff_impl(Array& data) override;
        void dgammadash2_by_dcoeff_impl(Array& data) override;
        void dgammadash1dash1_by_dcoeff_impl(Array& data) override;
        void dgammadash1dash2_by_dcoeff_impl(Array& data) override;
        void dgammadash2dash2_by_dcoeff_impl(Array& data) override;

};
