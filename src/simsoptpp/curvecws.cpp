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

#pragma omp parallel for
    for (int k = 0; k < numquadpoints; ++k)
    {
        double CWSt = 2 * M_PI * quadpoints[k];

        for (int i = 0; i < order + 1; ++i)
        {
            pphi += phi_c[i] * cos(i * CWSt);
            ptheta += theta_c[i] * cos(i * CWSt);
        }
        for (int i = 1; i < order + 1; ++i)
        {
            pphi += phi_s[i - 1] * sin(i * CWSt);
            ptheta += theta_s[i - 1] * sin(i * CWSt);
        }

        pphi += phi_l * CWSt;
        ptheta += theta_l * CWSt;

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
};  