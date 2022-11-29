#include "curvecws.h"

template <class Array>
void CurveCWS<Array>::gamma_impl(Array &data, Array &quadpoints)
{
    int numquadpoints = quadpoints.size();

    Array pphi;
    Array ptheta;
    pphi *= 0;
    ptheta *= 0;

    for (int k = 0; k < numquadpoints; ++k)
    {
        double CWSt = 2 * M_PI * quadpoints[k];

        for (int i = 0; i < CWSorder + 1; ++i)
        {
            pphi(k) += phi_c[i] * cos(i * CWSt);
            ptheta(k) += theta_c[i] * cos(i * CWSt);
        }
        for (int i = 1; i < CWSorder + 1; ++i)
        {
            pphi(k) += phi_s[i - 1] * sin(i * CWSt);
            ptheta(k) += theta_s[i - 1] * sin(i * CWSt);
        }

        pphi(k) += phi_l[0] * CWSt + phi_l[i];
        ptheta(k) += theta_l[0] * CWSt + theta_l[i]
    }

    data *= 0;
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
    }
}