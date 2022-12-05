#include "curvecws.h"

template <class Array>
void CurveCWS<Array>::gamma_impl(Array &data, Array &quadpoints)
{
    CurveCWS<Array>::set_dofs_surface(res);
    int numquadpoints = quadpoints.size();

    data *= 0;

    double pphi = 0;
    double ptheta = 0;

    double r = 0;
    double z = 0;

    for (int k = 0; k < numquadpoints; ++k)
    {
        double CWSt = 2 * M_PI * quadpoints[k];

        for (int i = 0; i < order + 1; ++i)
        {
            pphi += phi_c[i] * cos(i * CWSt);
            ptheta += theta_c[i] * cos(i * CWSt);
            data
        }
        for (int i = 1; i < order + 1; ++i)
        {
            pphi += phi_s[i - 1] * sin(i * CWSt);
            ptheta += theta_s[i - 1] * sin(i * CWSt);
        }

        pphi += phi_l[0] * CWSt;
        ptheta += theta_l[0] * CWSt;

        // SURFACE
        for (int m = 0; m <= mpol; ++m)
        {
            for (int i = 0; i < 2 * ntor + 1; ++i)
            {
                r += rc(m, i) * cos(m * ptheta - nfp * i * pphi) + rs(m, i) * sin(m * ptheta - nfp * i * pphi);
                z += zc(m, i) * cos(m * ptheta - nfp * i * pphi) + zs(m, i) * sin(m * ptheta - nfp * i * pphi);
            }
        }

        data(k, 0) = r * cos(pphi);
        data(k, 1) = r * sin(pphi);
        data(k, 2) = z;
    }

    /* data *= 0;
    for (int k = 0; k < numquadpoints; ++k)
    {
        double phi = 2 * M_PI * quadpoints[k];
        for (int i = 0; i < order + 1; ++i)
        {
            data(k, 0) += rc[i] * cos(nfp * i * phi) * cos(phi);
            data(k, 1) += rc[i] * cos(nfp * i * phi) * sin(phi);
        }
        for (int i = 1; i < order + 1; ++i)
        {
            data(k, 2) += zs[i - 1] * sin(nfp * i * phi);
        }
    }
    if (!stellsym)
    {
        for (int k = 0; k < numquadpoints; ++k)
        {
            double phi = 2 * M_PI * quadpoints[k];
            for (int i = 1; i < order + 1; ++i)
            {
                data(k, 0) += rs[i - 1] * sin(nfp * i * phi) * cos(phi);
                data(k, 1) += rs[i - 1] * sin(nfp * i * phi) * sin(phi);
            }
            for (int i = 0; i < order + 1; ++i)
            {
                data(k, 2) += zc[i] * cos(nfp * i * phi);
            }
        }
    } */
}