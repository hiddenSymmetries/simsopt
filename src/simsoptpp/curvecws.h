#pragma once

#include "curve.h"
#include "surfacerzfourier.h"
#include "curverzfourier.h"

template <class Array>
class CurveCWS : public Curve<Array>
{
public:
    const int order;
    const int nfp;
    const bool stellsym;
    using Curve<Array>::quadpoints;
    using Curve<Array>::numquadpoints;
    using Curve<Array>::check_the_persistent_cache;

    Array phi_l;
    Array theta_l;
    Array phi_s;
    Array phi_c;
    Array theta_c;
    Array theta_s;
    const int CWSorder;

    // int CWSnumquadpoints;
    // Array CWSquadpoints;

    Array rc;
    Array rs;
    Array zc;
    Array zs;

    // ODEIO FAZER CONSTRUTORES AAAAAAAAAAAAAAAAA
    /* CurveCWS(int _mpol, int _ntor, int _nfp, bool _stellsym, vector<double> _quadpoints_phi, vector<double> _quadpoints_theta, int _numquadpoints, int _order)
    {
        SurfaceRZFourier CWS = SurfaceRZFourier(int _mpol, int _ntor, int _nfp, bool _stellsym, vector<double> _quadpoints_phi, vector<double> _quadpoints_theta);
        CurveRZFourier CurveRZ(int _numquadpoints, int _order, int _nfp, bool _stellsym);
    } */

    CurveCWS(int _CWSorder,
             int _numquadpoints,
             int _order,
             int _nfp,
             bool _stellsym) : CWSorder(_CWSorder), Curve<Array>(_numquadpoints), order(_order), nfp(_nfp), stellsym(_stellsym)
    {
        rc = xt::zeros<double>({order + 1});
        rs = xt::zeros<double>({order});
        zc = xt::zeros<double>({order + 1});
        zs = xt::zeros<double>({order});

        phi_l = xt::zeros<double>({2});
        theta_l = xt::zeros<double>({2});
        phi_s = xt::zeros<double>({CWSorder});
        phi_c = xt::zeros<double>({CWSorder + 1});
        theta_s = xt::zeros<double>({CWSorder});
        theta_c = xt::zeros<double>({CWSorder + 1});
        /*
        CWSnumquadpoints = _CWSnumquadpoints;
        CWSquadpoints = xt::zeros<double>({_CWSnumquadpoints});
        for (int i = 0; i < CWSnumquadpoints; ++i)
        {
            CWSquadpoints[i] = (double(i)) / CWSnumquadpoints;
        }
        */
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

        for (int i = 0; i < 2; ++i)
            theta_l.data()[i] = dofs[counter++];
        for (int i = 0; i < CWSorder + 1; ++i)
            theta_c.data()[i] = dofs[counter++];
        for (int i = 0; i < CWSorder; ++i)
            theta_s.data()[i] = dofs[counter++];
        for (int i = 0; i < 2; ++1)
            phi_l.data()[i] = dofs[counter++];
        for (int i = 0; i < CWSorder + 1; ++i)
            phi_c.data()[i] = dofs[counter++];
        for (int i = 0; i < CWSorder; ++i)
            phi_s.data()[i] = dofs[counter++];
    }

    void gamma_impl(Array &data, Array &quadpoints) override;
};