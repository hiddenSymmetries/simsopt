#pragma once

#include "curve.h"
#include "surface.h"

template <class Array>
class CurveCWS : public Curve<Array>
{
public:
    const shared_ptr<Surface<Array>> surface;
    const int order;
    const int nfp;
    const bool stellsym;
    using Curve<Array>::quadpoints;
    using Curve<Array>::numquadpoints;
    using Curve<Array>::check_the_persistent_cache;

    double phi_l;
    double theta_l;
    Array phi_s;
    Array phi_c;
    Array theta_c;
    Array theta_s;

    CurveCWS(shared_ptr<Surface<Array>> surface,
             int _numquadpoints,
             int _order,
             int _nfp,
             bool _stellsym) : surface(surface), Curve<Array>(_numquadpoints), order(_order), nfp(_nfp), stellsym(_stellsym)
    {
        phi_l = xt::zeros<double>({1});
        theta_l = xt::zeros<double>({1});
        phi_s = xt::zeros<double>({order});
        phi_c = xt::zeros<double>({order + 1});
        theta_s = xt::zeros<double>({order});
        theta_c = xt::zeros<double>({order + 1});
    }

    /*
    inline int num_dofs() override
    {
        if (stellsym)
            return 2 * order + 1;
        else
            return 2 * (2 * order + 1);
    }
    */

    void set_dofs_impl(const vector<double> &dofs) override
    {
        int counter = 0;

        theta_l = dofs[counter++];
        for (int i = 0; i < order + 1; ++i)
            theta_c.data()[i] = dofs[counter++];
        for (int i = 0; i < order; ++i)
            theta_s.data()[i] = dofs[counter++];
        phi_l = dofs[counter++];
        for (int i = 0; i < order + 1; ++i)
            phi_c.data()[i] = dofs[counter++];
        for (int i = 0; i < order; ++i)
            phi_s.data()[i] = dofs[counter++];
    }

    void gamma_impl(Array &data, Array &quadpoints) override;
};