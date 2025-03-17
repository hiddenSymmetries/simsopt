#pragma once

#include "curve.h"

template <class Array>
class CurveCWSFourier : public Curve<Array>
{
public:
    // Curve
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

    // SURFACE
    int mpol;
    int ntor;
    vector<double> idofs; // = vector<double>(num_dofs(), 0.);
    Array rc;
    Array rs;
    Array zc;
    Array zs;

    CurveCWSFourier(int _mpol, int _ntor, vector<double> _idofs, int _numquadpoints, int _order, int _nfp, bool _stellsym) : Curve<Array>(_numquadpoints), order(_order), nfp(_nfp), stellsym(_stellsym), mpol(_mpol), ntor(_ntor), idofs(_idofs)
    {
        phi_l = 0;
        theta_l = 0;
        phi_s = xt::zeros<double>({order});
        phi_c = xt::zeros<double>({order + 1});
        theta_s = xt::zeros<double>({order});
        theta_c = xt::zeros<double>({order + 1});

        rc = xt::zeros<double>({mpol + 1, 2 * ntor + 1});
        rs = xt::zeros<double>({mpol + 1, 2 * ntor + 1});
        zc = xt::zeros<double>({mpol + 1, 2 * ntor + 1});
        zs = xt::zeros<double>({mpol + 1, 2 * ntor + 1});
    }

    inline int num_dofs() override
    {
        return 2 * (2 * order + 1) + 2;
    }

    vector<double> get_dofs() override
    {
        auto res = vector<double>(num_dofs(), 0.);
        int counter = 0;

        res[counter++] = theta_l;
        for (int i = 0; i < order + 1; ++i)
            res[counter++] = theta_c.data()[i];
        for (int i = 0; i < order; ++i)
            res[counter++] = theta_s.data()[i];
        res[counter++] = phi_l;
        for (int i = 0; i < order + 1; ++i)
            res[counter++] = phi_c.data()[i];
        for (int i = 0; i < order; ++i)
            res[counter++] = phi_s.data()[i];

        return res;
    }

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

    void set_dofs_surface(vector<double> &dofs)
    {
        int shift = (mpol + 1) * (2 * ntor + 1);

        int counter = 0;
        if (stellsym)
        {
            for (int i = ntor; i < shift; ++i)
                rc.data()[i] = dofs[counter++];
            for (int i = ntor + 1; i < shift; ++i)
                zs.data()[i] = dofs[counter++];
        }
        else
        {
            for (int i = ntor; i < shift; ++i)
                rc.data()[i] = dofs[counter++];
            for (int i = ntor + 1; i < shift; ++i)
                rs.data()[i] = dofs[counter++];
            for (int i = ntor; i < shift; ++i)
                zc.data()[i] = dofs[counter++];
            for (int i = ntor + 1; i < shift; ++i)
                zs.data()[i] = dofs[counter++];
        }
    }

    int num_dofs_surface()
    {
        if (stellsym)
            return 2 * (mpol + 1) * (2 * ntor + 1) - ntor - (ntor + 1);
        else
            return 4 * (mpol + 1) * (2 * ntor + 1) - 2 * ntor - 2 * (ntor + 1);
    }

    vector<double> get_dofs_surface()
    {
        set_dofs_surface(idofs);
        auto res = vector<double>(num_dofs_surface(), 0.);
        int shift = (mpol + 1) * (2 * ntor + 1);
        int counter = 0;
        if (stellsym)
        {
            for (int i = ntor; i < shift; ++i)
                res[counter++] = rc.data()[i];
            for (int i = ntor + 1; i < shift; ++i)
                res[counter++] = zs.data()[i];
        }
        else
        {
            for (int i = ntor; i < shift; ++i)
                res[counter++] = rc.data()[i];
            for (int i = ntor + 1; i < shift; ++i)
                res[counter++] = rs.data()[i];
            for (int i = ntor; i < shift; ++i)
                res[counter++] = zc.data()[i];
            for (int i = ntor + 1; i < shift; ++i)
                res[counter++] = zs.data()[i];
        }
        return res;
    }

    Array &dgamma_by_dcoeff() override
    {
        return check_the_persistent_cache("dgamma_by_dcoeff", {numquadpoints, 3, num_dofs()}, [this](Array &A)
                                          { return dgamma_by_dcoeff_impl(A); });
    }

    Array &dgammadash_by_dcoeff() override
    {
        return check_the_persistent_cache("dgammadash_by_dcoeff", {numquadpoints, 3, num_dofs()}, [this](Array &A)
                                          { return dgammadash_by_dcoeff_impl(A); });
    }

    Array &dgammadashdash_by_dcoeff() override
    {
        return check_the_persistent_cache("dgammadashdash_by_dcoeff", {numquadpoints, 3, num_dofs()}, [this](Array &A)
                                          { return dgammadashdash_by_dcoeff_impl(A); });
    }
 
    Array &dgammadashdashdash_by_dcoeff() override
    {
        return check_the_persistent_cache("dgammadashdashdash_by_dcoeff", {numquadpoints, 3, num_dofs()}, [this](Array &A)
                                          { return dgammadashdashdash_by_dcoeff_impl(A); });
    }
    

    void gamma_impl(Array &data, Array &quadpoints) override;
    void gammadash_impl(Array &data) override;
    void gammadashdash_impl(Array &data) override;
    void gammadashdashdash_impl(Array &data) override;
    void dgamma_by_dcoeff_impl(Array &data) override;
    void dgammadash_by_dcoeff_impl(Array &data) override;
    void dgammadashdash_by_dcoeff_impl(Array &data) override;
    void dgammadashdashdash_by_dcoeff_impl(Array &data) override;
};