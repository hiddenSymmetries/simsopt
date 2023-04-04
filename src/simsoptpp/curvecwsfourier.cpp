#include "curvecwsfourier.h"

template <class Array>
void CurveCWSFourier<Array>::gamma_impl(Array &data, Array &quadpoints)
{
    CurveCWSFourier<Array>::set_dofs_surface(idofs);
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
                int n = i - ntor;
                r += rc(m, i) * cos(m * ptheta - nfp * n * pphi);
                z += zs(m, i) * sin(m * ptheta - nfp * n * pphi);
            }
        }
        if (!stellsym)
        {
            for (int m = 0; m <= mpol; ++m)
            {
                for (int i = 0; i < 2 * ntor + 1; ++i)
                {
                    int n = i - ntor;
                    r += rs(m, i) * sin(m * ptheta - nfp * n * pphi);
                    z += zc(m, i) * cos(m * ptheta - nfp * n * pphi);
                }
            }
        }

        data(k, 0) = r * cos(pphi);
        data(k, 1) = r * sin(pphi);
        data(k, 2) = z;
    }
};

template <class Array>
void CurveCWSFourier<Array>::gammadash_impl(Array &data)
{
    CurveCWSFourier<Array>::set_dofs_surface(idofs);

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
                int n = i - ntor;
                r += rc(m, i) * cos(m * ptheta - nfp * n * pphi);
                z += zs(m, i) * sin(m * ptheta - nfp * n * pphi);

                dr += -rc(m, i) * sin(m * ptheta - nfp * n * pphi) * (m * dptheta - nfp * n * dpphi);
                dz += zs(m, i) * cos(m * ptheta - nfp * n * pphi) * (m * dptheta - nfp * n * dpphi);
            }
        }
        if (!stellsym)
        {
            for (int m = 0; m <= mpol; ++m)
            {
                for (int i = 0; i < 2 * ntor + 1; ++i)
                {
                    int n = i - ntor;
                    r += rs(m, i) * sin(m * ptheta - nfp * n * pphi);
                    z += zc(m, i) * cos(m * ptheta - nfp * n * pphi);

                    dr += rs(m, i) * cos(m * ptheta - nfp * n * pphi) * (m * dptheta - nfp * n * dpphi);
                    dz += -zc(m, i) * sin(m * ptheta - nfp * n * pphi) * (m * dptheta - nfp * n * dpphi);
                }
            }
        }

        data(k, 0) = dr * cos(pphi) - r * sin(pphi) * dpphi;
        data(k, 1) = dr * sin(pphi) + r * cos(pphi) * dpphi;
        data(k, 2) = dz;
    }
    data *= (2 * M_PI);
};

template <class Array>
void CurveCWSFourier<Array>::gammadashdash_impl(Array &data)
{
    CurveCWSFourier<Array>::set_dofs_surface(idofs);

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
                int n = i - ntor;
                r += rc(m, i) * cos(m * ptheta - nfp * n * pphi);
                z += zs(m, i) * sin(m * ptheta - nfp * n * pphi);

                dr += -rc(m, i) * sin(m * ptheta - nfp * n * pphi) * (m * dptheta - nfp * n * dpphi);
                dz += zs(m, i) * cos(m * ptheta - nfp * n * pphi) * (m * dptheta - nfp * n * dpphi);

                ddr += -rc(m, i) * cos(m * ptheta - nfp * n * pphi) * pow((m * dptheta - nfp * n * dpphi), 2) - rc(m, i) * sin(m * ptheta - nfp * n * pphi) * (m * ddptheta - nfp * n * ddpphi);
                ddz += -zs(m, i) * sin(m * ptheta - nfp * n * pphi) * pow((m * dptheta - nfp * n * dpphi), 2) + zs(m, i) * cos(m * ptheta - nfp * n * pphi) * (m * ddptheta - nfp * n * ddpphi);
            }
        }
        if (!stellsym)
        {
            for (int m = 0; m <= mpol; ++m)
            {
                for (int i = 0; i < 2 * ntor + 1; ++i)
                {
                    int n = i - ntor;
                    r += rs(m, i) * sin(m * ptheta - nfp * n * pphi);
                    z += zc(m, i) * cos(m * ptheta - nfp * n * pphi);

                    dr += rs(m, i) * cos(m * ptheta - nfp * n * pphi) * (m * dptheta - nfp * n * dpphi);
                    dz += -zc(m, i) * sin(m * ptheta - nfp * n * pphi) * (m * dptheta - nfp * n * dpphi);

                    ddr += -rs(m, i) * sin(m * ptheta - nfp * n * pphi) * pow((m * dptheta - nfp * n * dpphi), 2) + rs(m, i) * cos(m * ptheta - nfp * n * pphi) * (m * ddptheta - nfp * n * ddpphi);
                    ddz += -zc(m, i) * cos(m * ptheta - nfp * n * pphi) * pow((m * dptheta - nfp * n * dpphi), 2) - zc(m, i) * sin(m * ptheta - nfp * n * pphi) * (m * ddptheta - nfp * n * ddpphi);
                }
            }
        }

        data(k, 0) = ddr * cos(pphi) - 2 * (dr * sin(pphi) * dpphi) - r * cos(pphi) * pow(dpphi, 2) - r * sin(pphi) * ddpphi;
        data(k, 1) = ddr * sin(pphi) + 2 * (dr * cos(pphi) * dpphi) - r * sin(pphi) * pow(dpphi, 2) + r * cos(pphi) * ddpphi;
        data(k, 2) = ddz;
    }
    data *= 2 * M_PI * 2 * M_PI;
};

template <class Array>
void CurveCWSFourier<Array>::gammadashdashdash_impl(Array &data)
{
    CurveCWSFourier<Array>::set_dofs_surface(idofs);

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
                int n = i - ntor;
                r += rc(m, i) * cos(m * ptheta - nfp * n * pphi);
                z += zs(m, i) * sin(m * ptheta - nfp * n * pphi);

                dr += -rc(m, i) * sin(m * ptheta - nfp * n * pphi) * (m * dptheta - nfp * n * dpphi);
                dz += zs(m, i) * cos(m * ptheta - nfp * n * pphi) * (m * dptheta - nfp * n * dpphi);

                ddr += -rc(m, i) * cos(m * ptheta - nfp * n * pphi) * pow((m * dptheta - nfp * n * dpphi), 2) - rc(m, i) * sin(m * ptheta - nfp * n * pphi) * (m * ddptheta - nfp * n * ddpphi);
                ddz += -zs(m, i) * sin(m * ptheta - nfp * n * pphi) * pow((m * dptheta - nfp * n * dpphi), 2) + zs(m, i) * cos(m * ptheta - nfp * n * pphi) * (m * ddptheta - nfp * n * ddpphi);

                ddd_r += rc(m, i) * sin(m * ptheta - nfp * n * pphi) * pow((m * dptheta - nfp * n * dpphi), 3) - rc(m, i) * cos(m * ptheta - nfp * n * pphi) * 2 * (m * dptheta - nfp * n * dpphi) * (m * ddptheta - nfp * n * ddpphi) - rc(m, i) * cos(m * ptheta - nfp * n * pphi) * (m * dptheta - nfp * n * dpphi) * (m * ddptheta - nfp * n * ddpphi) - rc(m, i) * sin(m * ptheta - nfp * n * pphi) * (m * ddd_ptheta - nfp * n * ddd_pphi);
                ddd_z += -zs(m, i) * cos(m * ptheta - nfp * n * pphi) * pow((m * dptheta - nfp * n * dpphi), 3) - zs(m, i) * sin(m * ptheta - nfp * n * pphi) * 2 * (m * dptheta - nfp * n * dpphi) * (m * ddptheta - nfp * n * ddpphi) - zs(m, i) * sin(m * ptheta - nfp * n * pphi) * (m * dptheta - nfp * n * dpphi) * (m * ddptheta - nfp * n * ddpphi) + zs(m, i) * cos(m * ptheta - nfp * n * pphi) * (m * ddd_ptheta - nfp * n * ddd_pphi);
            }
        }
        if (!stellsym)
        {
            for (int m = 0; m <= mpol; ++m)
            {
                for (int i = 0; i < 2 * ntor + 1; ++i)
                {
                    int n = i - ntor;
                    r += rs(m, i) * sin(m * ptheta - nfp * n * pphi);
                    z += zc(m, i) * cos(m * ptheta - nfp * n * pphi);

                    dr += rs(m, i) * cos(m * ptheta - nfp * n * pphi) * (m * dptheta - nfp * n * dpphi);
                    dz += -zc(m, i) * sin(m * ptheta - nfp * n * pphi) * (m * dptheta - nfp * n * dpphi);

                    ddr += -rs(m, i) * sin(m * ptheta - nfp * n * pphi) * pow((m * dptheta - nfp * n * dpphi), 2) + rs(m, i) * cos(m * ptheta - nfp * n * pphi) * (m * ddptheta - nfp * n * ddpphi);
                    ddz += -zc(m, i) * cos(m * ptheta - nfp * n * pphi) * pow((m * dptheta - nfp * n * dpphi), 2) - zc(m, i) * sin(m * ptheta - nfp * n * pphi) * (m * ddptheta - nfp * n * ddpphi);

                    ddd_r += -rs(m, i) * cos(m * ptheta - nfp * n * pphi) * pow((m * dptheta - nfp * n * dpphi), 3) - rs(m, i) * sin(m * ptheta - nfp * n * pphi) * 2 * (m * dptheta - nfp * n * dpphi) * (m * ddptheta - nfp * n * ddpphi) - rs(m, i) * sin(m * ptheta - nfp * n * pphi) * (m * dptheta - nfp * n * dpphi) * (m * ddptheta - nfp * n * ddpphi) + rs(m, i) * cos(m * ptheta - nfp * n * pphi) * (m * ddd_ptheta - nfp * n * ddd_pphi);
                    ddd_z += zc(m, i) * sin(m * ptheta - nfp * n * pphi) * pow((m * dptheta - nfp * n * dpphi), 3) - zc(m, i) * cos(m * ptheta - nfp * n * pphi) * 2 * (m * dptheta - nfp * n * dpphi) * (m * ddptheta - nfp * n * ddpphi) - zc(m, i) * cos(m * ptheta - nfp * n * pphi) * (m * dptheta - nfp * n * dpphi) * (m * ddptheta - nfp * n * ddpphi) - zc(m, i) * sin(m * ptheta - nfp * n * pphi) * (m * ddd_ptheta - nfp * n * ddd_pphi);
                }
            }
        }

        data(k, 0) = ddd_r * cos(pphi) - ddr * sin(pphi) * dpphi - 2 * (ddr * sin(pphi) * dpphi) - 2 * (dr * cos(pphi) * pow(dpphi, 2)) - 2 * (dr * sin(pphi) * ddpphi) - dr * cos(pphi) * pow(dpphi, 2) + r * sin(pphi) * pow(dpphi, 3) - r * cos(pphi) * 2 * dpphi * ddpphi - dr * sin(pphi) * ddpphi - r * cos(pphi) * dpphi * ddpphi - r * sin(pphi) * ddd_pphi;
        data(k, 1) = ddd_r * sin(pphi) + ddr * cos(pphi) * dpphi + 2 * (ddr * cos(pphi) * dpphi) - 2 * (dr * sin(pphi) * pow(dpphi, 2)) + 2 * (dr * cos(pphi) * ddpphi) - dr * sin(pphi) * pow(dpphi, 2) - r * cos(pphi) * pow(dpphi, 3) - r * sin(pphi) * 2 * dpphi * ddpphi + dr * cos(pphi) * ddpphi - r * sin(pphi) * dpphi * ddpphi + r * cos(pphi) * ddd_pphi;
        data(k, 2) = ddd_z;
    }
    data *= 2*M_PI*2*M_PI*2*M_PI;

};

template <class Array>
void CurveCWSFourier<Array>::dgamma_by_dcoeff_impl(Array &data)
{

    CurveCWSFourier<Array>::set_dofs_surface(idofs);
    data *= 0;

    for (int k = 0; k < numquadpoints; ++k)
    {
        double CWSt = 2 * M_PI * quadpoints[k];

        double pphi = 0;
        double ptheta = 0;

        Array phi_array = xt::zeros<double>({2 * (order + 1)});
        Array theta_array = xt::zeros<double>({2 * (order + 1)});

        Array r_array = xt::zeros<double>({4 * (order + 1)});
        Array z_array = xt::zeros<double>({4 * (order + 1)});

        int counter = 0;

        double r_aux1 = 0;
        double z_aux1 = 0;
        double r_aux2 = 0;
        double z_aux2 = 0;

        phi_array[counter] = CWSt;
        theta_array[counter] = CWSt;

        counter++;

        for (int i = 0; i < order + 1; ++i)
        {
            phi_array[counter] = cos(i * CWSt);
            theta_array[counter] = cos(i * CWSt);
            counter++;

            pphi += phi_c[i] * cos(i * CWSt);
            ptheta += theta_c[i] * cos(i * CWSt);
        }

        for (int i = 1; i < order + 1; ++i)
        {
            phi_array[counter] = sin(i * CWSt);
            theta_array[counter] = sin(i * CWSt);
            counter++;

            pphi += phi_s[i - 1] * sin(i * CWSt);
            ptheta += theta_s[i - 1] * sin(i * CWSt);
        }

        pphi += phi_l * CWSt;
        ptheta += theta_l * CWSt;

// SURFACE
#pragma omp parallel for
        for (int i = 0; i < counter; ++i)
        {
            r_aux1 = 0;
            z_aux1 = 0;
            r_aux2 = 0;
            z_aux2 = 0;

            for (int m = 0; m <= mpol; ++m)
            {
                for (int j = 0; j < 2 * ntor + 1; ++j)
                {
                    int n = j - ntor;

                    r_aux1 += -rc(m, j) * sin(m * ptheta - nfp * n * pphi) * (m * theta_array[i] - nfp * n * pphi);
                    z_aux1 += zs(m, j) * cos(m * ptheta - nfp * n * pphi) * (m * theta_array[i] - nfp * n * pphi);

                    r_aux2 += -rc(m, j) * sin(m * ptheta - nfp * n * pphi) * (m * ptheta - nfp * n * phi_array[i]);
                    z_aux2 += zs(m, j) * cos(m * ptheta - nfp * n * pphi) * (m * ptheta - nfp * n * phi_array[i]);

                    if (!stellsym)
                    {
                        r_aux1 += rs(m, j) * cos(m * ptheta - nfp * n * pphi) * (m * theta_array[i] - nfp * n * pphi);
                        z_aux1 += -zc(m, j) * sin(m * ptheta - nfp * n * pphi) * (m * theta_array[i] - nfp * n * pphi);

                        r_aux2 += rs(m, j) * cos(m * ptheta - nfp * n * pphi) * (m * ptheta - nfp * n * phi_array[i]);
                        z_aux2 += -zc(m, j) * sin(m * ptheta - nfp * n * pphi) * (m * ptheta - nfp * n * phi_array[i]);
                    }
                }
            }
            r_array[i] = r_aux1;
            z_array[i] = z_aux1;
            r_array[i + counter] = r_aux2;
            z_array[i + counter] = z_aux2;
        }

        for (int p = 0; p < counter; p++)
        {
            data(k, 0, p) = r_array[p] * cos(pphi);
            data(k, 1, p) = r_array[p] * sin(pphi);
            data(k, 2, p) = z_array[p];

            data(k, 0, p + counter) = -r_array[p + counter] * sin(pphi) * phi_array[p];
            data(k, 1, p + counter) = r_array[p + counter] * cos(pphi) * phi_array[p];
            data(k, 2, p + counter) = z_array[p + counter];
        }
    }
};

    /*
    void CurveCWSFourier<Array>::dgammadash_by_dcoeff_impl(Array &data)
    {
        {
            CurveCWSFourier<Array>::set_dofs_surface(idofs);

            data *= 0;

    #pragma omp parallel for
            for (int k = 0; k < numquadpoints; ++k)
            {
                double CWSt = 2 * M_PI * quadpoints[k];

                double pphi = 0;
                double ptheta = 0;
                double dpphi = 0;
                double dptheta = 0;

                double r = 0;
                double z = 0;

                double dr = 0;
                double dz = 0;

                Array phi_array = xt::zeros<double>({2 * (order + 1)});
                Array theta_array = xt::zeros<double>({2 * (order + 1)});

                Array r_array = xt::zeros<double>({4 * (order + 1)});
                Array z_array = xt::zeros<double>({4 * (order + 1)});

                int counter = 0;

                phi_array[counter] = CWSt;
                phi_array[counter] = CWSt;

                // Termos com Cossenos e as suas derivas
                for (int i = 0; i < order + 1; ++i)
                {
                    phi_array[counter] = cos(i * CWSt);
                    theta_array[counter] = cos(i * CWSt);
                    counter++;

                    pphi += phi_c[i] * cos(i * CWSt);
                    ptheta += theta_c[i] * cos(i * CWSt);

                    dpphi += -phi_c[i] * i * sin(i * CWSt);
                    dptheta += -theta_c[i] * i * sin(i * CWSt);
                }
                // Termos com Senos e as suas derivas
                for (int i = 1; i < order + 1; ++i)
                {
                    phi_array[counter] = sin(i * CWSt);
                    theta_array[counter] = sin(i * CWSt);
                    counter++;

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
                        int n = i - ntor;
                        r += rc(m, i) * cos(m * ptheta - nfp * n * pphi);
                        z += zs(m, i) * sin(m * ptheta - nfp * n * pphi);

                        dr += -rc(m, i) * sin(m * ptheta - nfp * n * pphi) * (m * dptheta - nfp * n * dpphi);
                        dz += zs(m, i) * cos(m * ptheta - nfp * n * pphi) * (m * dptheta - nfp * n * dpphi);
                    }
                }
                if (!stellsym)
                {
                    for (int m = 0; m <= mpol; ++m)
                    {
                        for (int i = 0; i < 2 * ntor + 1; ++i)
                        {
                            int n = i - ntor;
                            r += rs(m, i) * sin(m * ptheta - nfp * n * pphi);
                            z += zc(m, i) * cos(m * ptheta - nfp * n * pphi);

                            dr += rs(m, i) * cos(m * ptheta - nfp * n * pphi) * (m * dptheta - nfp * n * dpphi);
                            dz += -zc(m, i) * sin(m * ptheta - nfp * n * pphi) * (m * dptheta - nfp * n * dpphi);
                        }
                    }
                }

                data(k, 0) = dr * cos(pphi) - r * sin(pphi) * dpphi;
                data(k, 1) = dr * sin(pphi) + r * cos(pphi) * dpphi;
                data(k, 2) = dz;
            }
        };
    };
    */

#include "xtensor-python/pyarray.hpp" // Numpy bindings
typedef xt::pyarray<double> Array;
template class CurveCWSFourier<Array>;