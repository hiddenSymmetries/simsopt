#include "curvecws.h"

template <class Array>
void CurveCWS<Array>::gamma_impl(Array &data, Array &quadpoints)
{
    CurveCWS<Array>::set_dofs_surface(idofs);
    int numquadpoints = quadpoints.size();

    data *= 0;

#pragma omp parallel for
    for (int k = 0; k < numquadpoints; ++k)
    {
        double pphi = 0;
        double ptheta = 0;

        double r = 0;
        double z = 0;
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
                r += rc(m, i) * cos(m * ptheta - nfp * (i - ntor) * pphi);
                z += zs(m, i) * sin(m * ptheta - nfp * (i - ntor) * pphi);
            }
        }
        if (!stellsym)
        {
            for (int m = 0; m <= mpol; ++m)
            {
                for (int i = 0; i < 2 * ntor + 1; ++i)
                {
                    r += rs(m, i) * sin(m * ptheta - nfp * (i - ntor) * pphi);
                    z += zc(m, i) * cos(m * ptheta - nfp * (i - ntor) * pphi);
                }
            }
        }

        data(k, 0) = r * cos(pphi);
        data(k, 1) = r * sin(pphi);
        data(k, 2) = z;
    }
};

template <class Array>
void CurveCWS<Array>::gammadash_impl(Array &data)
{
    CurveCWS<Array>::set_dofs_surface(idofs);

    data *= 0;

#pragma omp parallel for
    for (int k = 0; k < numquadpoints; ++k)
    {
        double pphi = 0;
        double ptheta = 0;
        double dpphi = 0;
        double dptheta = 0;

        double r = 0;
        double z = 0;

        double dr = 0;
        double dz = 0;

        double CWSt = 2 * M_PI * quadpoints[k];

        // Termos com Cossenos e as suas derivas
        for (int i = 0; i < order + 1; ++i)
        {
            pphi += phi_c[i] * cos(i * CWSt);
            ptheta += theta_c[i] * cos(i * CWSt);

            dpphi += -phi_c[i] * i * sin(i * CWSt);
            dptheta += -theta_c[i] * i * sin(i * CWSt);
        }
        // Termos com Senos e as suas derivas
        for (int i = 1; i < order + 1; ++i)
        {
            pphi += phi_s[i - 1] * sin(i * CWSt);
            ptheta += theta_s[i - 1] * sin(i * CWSt);

            dpphi += phi_s[i - 1] * i * cos(i * CWSt);
            dptheta += theta_s[i - 1] * i * cos(i * CWSt);
        }

        pphi += phi_l * CWSt;
        ptheta += theta_l * CWSt;
        dpphi += phi_l;
        dptheta += theta_l;

        // SURFACE
        for (int m = 0; m <= mpol; ++m)
        {
            for (int i = 0; i < 2 * ntor + 1; ++i)
            {
                r += rc(m, i) * cos(m * ptheta - nfp * i * pphi);
                z += zs(m, i) * sin(m * ptheta - nfp * i * pphi);

                dr += -rc(m, i) * sin(m * ptheta - nfp * i * pphi) * (m * dptheta - nfp * i * dpphi);
                dz += zs(m, i) * cos(m * ptheta - nfp * i * pphi) * (m * dptheta - nfp * i * dpphi);
            }
        }
        if (!stellsym)
        {
            for (int m = 0; m <= mpol; ++m)
            {
                for (int i = 0; i < 2 * ntor + 1; ++i)
                {
                    r += rs(m, i) * sin(m * ptheta - nfp * i * pphi);
                    z += zc(m, i) * cos(m * ptheta - nfp * i * pphi);

                    dr += rs(m, i) * cos(m * ptheta - nfp * i * pphi) * (m * dptheta - nfp * i * dpphi);
                    dz += -zc(m, i) * sin(m * ptheta - nfp * i * pphi) * (m * dptheta - nfp * i * dpphi);
                }
            }
        }

        data(k, 0) = dr * cos(pphi) - r * sin(pphi) * dpphi;
        data(k, 1) = dr * sin(pphi) + r * cos(pphi) * dpphi;
        data(k, 2) = dz;
    }
};

template <class Array>
void CurveCWS<Array>::gammadashdash_impl(Array &data)
{
    CurveCWS<Array>::set_dofs_surface(idofs);

    data *= 0;

#pragma omp parallel for
    for (int k = 0; k < numquadpoints; ++k)
    {
        double pphi = 0;
        double ptheta = 0;
        double dpphi = 0;
        double dptheta = 0;
        double ddpphi = 0;
        double ddptheta = 0;

        double r = 0;
        double z = 0;
        double dr = 0;
        double dz = 0;
        double ddr = 0;
        double ddz = 0;

        double CWSt = 2 * M_PI * quadpoints[k];

        // Termos com Cossenos e as suas derivas
        for (int i = 0; i < order + 1; ++i)
        {
            pphi += phi_c[i] * cos(i * CWSt);
            ptheta += theta_c[i] * cos(i * CWSt);

            dpphi += -phi_c[i] * i * sin(i * CWSt);
            dptheta += -theta_c[i] * i * sin(i * CWSt);

            ddpphi += -phi_c[i] * pow(i, 2) * cos(i * CWSt);
            ddptheta += -theta_c[i] * pow(i, 2) * cos(i * CWSt);
        }
        // Termos com Senos e as suas derivas
        for (int i = 1; i < order + 1; ++i)
        {
            pphi += phi_s[i - 1] * sin(i * CWSt);
            ptheta += theta_s[i - 1] * sin(i * CWSt);

            dpphi += phi_s[i - 1] * i * cos(i * CWSt);
            dptheta += theta_s[i - 1] * i * cos(i * CWSt);

            ddpphi += -phi_s[i - 1] * pow(i, 2) * sin(i * CWSt);
            ddptheta += -theta_s[i - 1] * pow(i, 2) * sin(i * CWSt);
        }

        pphi += phi_l * CWSt;
        ptheta += theta_l * CWSt;
        dpphi += phi_l;
        dptheta += theta_l;

        // SURFACE
        for (int m = 0; m <= mpol; ++m)
        {
            for (int i = 0; i < 2 * ntor + 1; ++i)
            {
                r += rc(m, i) * cos(m * ptheta - nfp * i * pphi);
                z += zs(m, i) * sin(m * ptheta - nfp * i * pphi);

                dr += -rc(m, i) * sin(m * ptheta - nfp * i * pphi) * (m * dptheta - nfp * i * dpphi);
                dz += zs(m, i) * cos(m * ptheta - nfp * i * pphi) * (m * dptheta - nfp * i * dpphi);

                ddr += -rc(m, i) * cos(m * ptheta - nfp * i * pphi) * pow((m * dptheta - nfp * i * dpphi), 2) - rc(m, i) * sin(m * ptheta - nfp * i * pphi) * (m * ddptheta - nfp * i * ddpphi);
                ddz += -zs(m, i) * sin(m * ptheta - nfp * i * pphi) * pow((m * dptheta - nfp * i * dpphi), 2) + zs(m, i) * cos(m * ptheta - nfp * i * pphi) * (m * ddptheta - nfp * i * ddpphi);
            }
        }
        if (!stellsym)
        {
            for (int m = 0; m <= mpol; ++m)
            {
                for (int i = 0; i < 2 * ntor + 1; ++i)
                {
                    r += rs(m, i) * sin(m * ptheta - nfp * i * pphi);
                    z += zc(m, i) * cos(m * ptheta - nfp * i * pphi);

                    dr += rs(m, i) * cos(m * ptheta - nfp * i * pphi) * (m * dptheta - nfp * i * dpphi);
                    dz += -zc(m, i) * sin(m * ptheta - nfp * i * pphi) * (m * dptheta - nfp * i * dpphi);

                    ddr += -rs(m, i) * sin(m * ptheta - nfp * i * pphi) * pow((m * dptheta - nfp * i * dpphi), 2) + rs(m, i) * cos(m * ptheta - nfp * i * pphi) * (m * ddptheta - nfp * i * ddpphi);
                    ddz += -zc(m, i) * cos(m * ptheta - nfp * i * pphi) * pow((m * dptheta - nfp * i * dpphi), 2) - zc(m, i) * sin(m * ptheta - nfp * i * pphi) * (m * ddptheta - nfp * i * ddpphi);
                }
            }
        }

        data(k, 0) = ddr * cos(pphi) - 2 * (dr * sin(pphi) * dpphi) - r * cos(pphi) * pow(dpphi, 2) - r * sin(pphi) * ddpphi;
        data(k, 1) = ddr * sin(pphi) + 2 * (dr * cos(pphi) * dpphi) - r * sin(pphi) * pow(dpphi, 2) + r * cos(pphi) * ddpphi;
        data(k, 2) = ddz;
    }
};

template <class Array>
void CurveCWS<Array>::gammadashdashdash_impl(Array &data)
{
    CurveCWS<Array>::set_dofs_surface(idofs);

    data *= 0;

#pragma omp parallel for
    for (int k = 0; k < numquadpoints; ++k)
    {
        double pphi = 0;
        double ptheta = 0;
        double dpphi = 0;
        double dptheta = 0;
        double ddpphi = 0;
        double ddptheta = 0;
        double ddd_pphi = 0;
        double ddd_ptheta = 0;

        double r = 0;
        double z = 0;
        double dr = 0;
        double dz = 0;
        double ddr = 0;
        double ddz = 0;
        double ddd_r = 0;
        double ddd_z = 0;

        double CWSt = 2 * M_PI * quadpoints[k];

        // Termos com Cossenos e as suas derivas
        for (int i = 0; i < order + 1; ++i)
        {
            pphi += phi_c[i] * cos(i * CWSt);
            ptheta += theta_c[i] * cos(i * CWSt);

            dpphi += -phi_c[i] * i * sin(i * CWSt);
            dptheta += -theta_c[i] * i * sin(i * CWSt);

            ddpphi += -phi_c[i] * pow(i, 2) * cos(i * CWSt);
            ddptheta += -theta_c[i] * pow(i, 2) * cos(i * CWSt);

            ddd_pphi += phi_c[i] * pow(i, 3) * sin(i * CWSt);
            ddd_ptheta += theta_c[i] * pow(i, 3) * sin(i * CWSt);
        }
        // Termos com Senos e as suas derivas
        for (int i = 1; i < order + 1; ++i)
        {
            pphi += phi_s[i - 1] * sin(i * CWSt);
            ptheta += theta_s[i - 1] * sin(i * CWSt);

            dpphi += phi_s[i - 1] * i * cos(i * CWSt);
            dptheta += theta_s[i - 1] * i * cos(i * CWSt);

            ddpphi += -phi_s[i - 1] * pow(i, 2) * sin(i * CWSt);
            ddptheta += -theta_s[i - 1] * pow(i, 2) * sin(i * CWSt);

            ddd_pphi += -phi_s[i - 1] * pow(i, 3) * cos(i * CWSt);
            ddd_ptheta += -theta_s[i - 1] * pow(i, 3) * cos(i * CWSt);
        }

        pphi += phi_l * CWSt;
        ptheta += theta_l * CWSt;
        dpphi += phi_l;
        dptheta += theta_l;

        // SURFACE
        for (int m = 0; m <= mpol; ++m)
        {
            for (int i = 0; i < 2 * ntor + 1; ++i)
            {
                r += rc(m, i) * cos(m * ptheta - nfp * i * pphi);
                z += zs(m, i) * sin(m * ptheta - nfp * i * pphi);

                dr += -rc(m, i) * sin(m * ptheta - nfp * i * pphi) * (m * dptheta - nfp * i * dpphi);
                dz += zs(m, i) * cos(m * ptheta - nfp * i * pphi) * (m * dptheta - nfp * i * dpphi);

                ddr += -rc(m, i) * cos(m * ptheta - nfp * i * pphi) * pow((m * dptheta - nfp * i * dpphi), 2) - rc(m, i) * sin(m * ptheta - nfp * i * pphi) * (m * ddptheta - nfp * i * ddpphi);
                ddz += -zs(m, i) * sin(m * ptheta - nfp * i * pphi) * pow((m * dptheta - nfp * i * dpphi), 2) + zs(m, i) * cos(m * ptheta - nfp * i * pphi) * (m * ddptheta - nfp * i * ddpphi);

                ddd_r += rc(m, i) * sin(m * ptheta - nfp * i * pphi) * pow((m * dptheta - nfp * i * dpphi), 3) - rc(m, i) * cos(m * ptheta - nfp * i * pphi) * 2 * (m * dptheta - nfp * i * dpphi) * (m * ddptheta - nfp * i * ddpphi) - rc(m, i) * cos(m * ptheta - nfp * i * pphi) * (m * dptheta - nfp * i * dpphi) * (m * ddptheta - nfp * i * ddpphi) - rc(m, i) * sin(m * ptheta - nfp * i * pphi) * (m * ddd_ptheta - nfp * i * ddd_pphi);
                ddd_z += -zs(m, i) * cos(m * ptheta - nfp * i * pphi) * pow((m * dptheta - nfp * i * dpphi), 3) - zs(m, i) * sin(m * ptheta - nfp * i * pphi) * 2 * (m * dptheta - nfp * i * dpphi) * (m * ddptheta - nfp * i * ddpphi) - zs(m, i) * sin(m * ptheta - nfp * i * pphi) * (m * dptheta - nfp * i * dpphi) * (m * ddptheta - nfp * i * ddpphi) + zs(m, i) * cos(m * ptheta - nfp * i * pphi) * (m * ddd_ptheta - nfp * i * ddd_pphi);
            }
        }
        if (!stellsym)
        {
            for (int m = 0; m <= mpol; ++m)
            {
                for (int i = 0; i < 2 * ntor + 1; ++i)
                {
                    r += rs(m, i) * sin(m * ptheta - nfp * i * pphi);
                    z += zc(m, i) * cos(m * ptheta - nfp * i * pphi);

                    dr += rs(m, i) * cos(m * ptheta - nfp * i * pphi) * (m * dptheta - nfp * i * dpphi);
                    dz += -zc(m, i) * sin(m * ptheta - nfp * i * pphi) * (m * dptheta - nfp * i * dpphi);

                    ddr += -rs(m, i) * sin(m * ptheta - nfp * i * pphi) * pow((m * dptheta - nfp * i * dpphi), 2) + rs(m, i) * cos(m * ptheta - nfp * i * pphi) * (m * ddptheta - nfp * i * ddpphi);
                    ddz += -zc(m, i) * cos(m * ptheta - nfp * i * pphi) * pow((m * dptheta - nfp * i * dpphi), 2) - zc(m, i) * sin(m * ptheta - nfp * i * pphi) * (m * ddptheta - nfp * i * ddpphi);

                    ddd_r += -rs(m, i) * cos(m * ptheta - nfp * i * pphi) * pow((m * dptheta - nfp * i * dpphi), 3) - rs(m, i) * sin(m * ptheta - nfp * i * pphi) * 2 * (m * dptheta - nfp * i * dpphi) * (m * ddptheta - nfp * i * ddpphi) - rs(m, i) * sin(m * ptheta - nfp * i * pphi) * (m * dptheta - nfp * i * dpphi) * (m * ddptheta - nfp * i * ddpphi) + rs(m, i) * cos(m * ptheta - nfp * i * pphi) * (m * ddd_ptheta - nfp * i * ddd_pphi);
                    ddd_z += zc(m, i) * sin(m * ptheta - nfp * i * pphi) * pow((m * dptheta - nfp * i * dpphi), 3) - zc(m, i) * cos(m * ptheta - nfp * i * pphi) * 2 * (m * dptheta - nfp * i * dpphi) * (m * ddptheta - nfp * i * ddpphi) - zc(m, i) * cos(m * ptheta - nfp * i * pphi) * (m * dptheta - nfp * i * dpphi) * (m * ddptheta - nfp * i * ddpphi) - zc(m, i) * sin(m * ptheta - nfp * i * pphi) * (m * ddd_ptheta - nfp * i * ddd_pphi);
                }
            }
        }

        data(k, 0) = ddd_r * cos(pphi) - ddr * sin(pphi) * dpphi - 2 * (ddr * sin(pphi) * dpphi) - 2 * (dr * cos(pphi) * pow(dpphi, 2)) - 2 * (dr * sin(pphi) * ddpphi) - dr * cos(pphi) * pow(dpphi, 2) + r * sin(pphi) * pow(dpphi, 3) - r * cos(pphi) * 2 * dpphi * ddpphi - dr * sin(pphi) * ddpphi - r * cos(pphi) * dpphi * ddpphi - r * sin(pphi) * ddd_pphi;
        data(k, 1) = ddd_r * sin(pphi) + ddr * cos(pphi) * dpphi + 2 * (ddr * cos(pphi) * dpphi) - 2 * (dr * sin(pphi) * pow(dpphi, 2)) + 2 * (dr * cos(pphi) * ddpphi) - dr * sin(pphi) * pow(dpphi, 2) - r * cos(pphi) * pow(dpphi, 3) - r * sin(pphi) * 2 * dpphi * ddpphi + dr * cos(pphi) * ddpphi - r * sin(pphi) * dpphi * ddpphi + r * cos(pphi) * ddd_pphi;
        data(k, 2) = ddd_z;
    }
};

#include "xtensor-python/pyarray.hpp" // Numpy bindings
typedef xt::pyarray<double> Array;
template class CurveCWS<Array>;