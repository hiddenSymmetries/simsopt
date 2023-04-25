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
        double phi = 0;
        double theta = 0;

        double r = 0;
        double z = 0;
        double CWSt = 2 * M_PI * quadpoints[k];

        for (int i = 0; i < order + 1; ++i)
        {
            phi += phi_c[i] * cos(i * CWSt);
            theta += theta_c[i] * cos(i * CWSt);

            // This creates an error in Y of the order of 10e-310...
            // phi += phi_s[i] * sin((i + 1) * CWSt);
            // theta += theta_s[i] * sin((i + 1) * CWSt);
        }

        for (int i = 1; i < order + 1; ++i)
        {
            phi += phi_s[i - 1] * sin(i * CWSt);
            theta += theta_s[i - 1] * sin(i * CWSt);
        }

        phi += phi_l * CWSt;
        theta += theta_l * CWSt;

        // SURFACE
        for (int m = 0; m <= mpol; ++m)
        {
            for (int i = 0; i < 2 * ntor + 1; ++i)
            {
                int n = i - ntor;
                r += rc(m, i) * cos(m * theta - nfp * n * phi);
                z += zs(m, i) * sin(m * theta - nfp * n * phi);
                if (!stellsym)
                {
                    int n = i - ntor;
                    r += rs(m, i) * sin(m * theta - nfp * n * phi);
                    z += zc(m, i) * cos(m * theta - nfp * n * phi);
                }
            }
        }
        data(k, 0) = r * cos(phi);
        data(k, 1) = r * sin(phi);
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
        double phi = 0;
        double theta = 0;
        double dphi = 0;
        double dtheta_dt = 0;

        double r = 0;

        double dr = 0;
        double dz = 0;

        double CWSt = 2 * M_PI * quadpoints[k];

        // Termos com Cossenos e as suas derivas
        for (int i = 0; i < order + 1; ++i)
        {
            phi += phi_c[i] * cos(i * CWSt);
            theta += theta_c[i] * cos(i * CWSt);

            dphi += -phi_c[i] * i * sin(i * CWSt);
            dtheta_dt += -theta_c[i] * i * sin(i * CWSt);
        }
        // Termos com Senos e as suas derivas
        for (int i = 1; i < order + 1; ++i)
        {
            phi += phi_s[i - 1] * sin(i * CWSt);
            theta += theta_s[i - 1] * sin(i * CWSt);

            dphi += phi_s[i - 1] * i * cos(i * CWSt);
            dtheta_dt += theta_s[i - 1] * i * cos(i * CWSt);
        }

        phi += phi_l * CWSt;
        theta += theta_l * CWSt;
        dphi += phi_l;
        dtheta_dt += theta_l;

        // SURFACE
        for (int m = 0; m <= mpol; ++m)
        {
            for (int i = 0; i < 2 * ntor + 1; ++i)
            {
                int n = i - ntor;
                r += rc(m, i) * cos(m * theta - nfp * n * phi);
                dr += -rc(m, i) * sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi);
                dz += zs(m, i) * cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi);

                if (!stellsym)
                {
                    int n = i - ntor;
                    r += rs(m, i) * sin(m * theta - nfp * n * phi);
                    dr += rs(m, i) * cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi);
                    dz += -zc(m, i) * sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi);
                }
            }
        }

        data(k, 0) = dr * cos(phi) - r * sin(phi) * dphi;
        data(k, 1) = dr * sin(phi) + r * cos(phi) * dphi;
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
        double phi = 0;
        double theta = 0;
        double dphi = 0;
        double dtheta_dt = 0;
        double d2phi_dt2 = 0;
        double d2theta_dt2 = 0;

        double r = 0;
        double dr = 0;
        double ddr = 0;
        double ddz = 0;

        double CWSt = 2 * M_PI * quadpoints[k];

        // Termos com Cossenos e as suas derivas
        for (int i = 0; i < order + 1; ++i)
        {
            phi += phi_c[i] * cos(i * CWSt);
            theta += theta_c[i] * cos(i * CWSt);

            dphi += -phi_c[i] * i * sin(i * CWSt);
            dtheta_dt += -theta_c[i] * i * sin(i * CWSt);

            d2phi_dt2 += -phi_c[i] * pow(i, 2) * cos(i * CWSt);
            d2theta_dt2 += -theta_c[i] * pow(i, 2) * cos(i * CWSt);
        }
        // Termos com Senos e as suas derivas
        for (int i = 1; i < order + 1; ++i)
        {
            phi += phi_s[i - 1] * sin(i * CWSt);
            theta += theta_s[i - 1] * sin(i * CWSt);

            dphi += phi_s[i - 1] * i * cos(i * CWSt);
            dtheta_dt += theta_s[i - 1] * i * cos(i * CWSt);

            d2phi_dt2 += -phi_s[i - 1] * pow(i, 2) * sin(i * CWSt);
            d2theta_dt2 += -theta_s[i - 1] * pow(i, 2) * sin(i * CWSt);
        }

        phi += phi_l * CWSt;
        theta += theta_l * CWSt;
        dphi += phi_l;
        dtheta_dt += theta_l;

        // SURFACE
        for (int m = 0; m <= mpol; ++m)
        {
            for (int i = 0; i < 2 * ntor + 1; ++i)
            {
                int n = i - ntor;
                r += rc(m, i) * cos(m * theta - nfp * n * phi);
                dr += -rc(m, i) * sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi);
                ddr += -rc(m, i) * cos(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi), 2) - rc(m, i) * sin(m * theta - nfp * n * phi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2);
                ddz += -zs(m, i) * sin(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi), 2) + zs(m, i) * cos(m * theta - nfp * n * phi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2);

                if (!stellsym)
                {
                    int n = i - ntor;
                    r += rs(m, i) * sin(m * theta - nfp * n * phi);
                    dr += rs(m, i) * cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi);
                    ddr += -rs(m, i) * sin(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi), 2) + rs(m, i) * cos(m * theta - nfp * n * phi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2);
                    ddz += -zc(m, i) * cos(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi), 2) - zc(m, i) * sin(m * theta - nfp * n * phi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2);
                }
            }
        }

        data(k, 0) = ddr * cos(phi) - 2 * (dr * sin(phi) * dphi) - r * (cos(phi) * pow(dphi, 2) + sin(phi) * d2phi_dt2);
        data(k, 1) = ddr * sin(phi) + 2 * (dr * cos(phi) * dphi) - r * (sin(phi) * pow(dphi, 2) - cos(phi) * d2phi_dt2);
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
        double phi = 0;
        double theta = 0;
        double dphi = 0;
        double dtheta_dt = 0;
        double d2phi_dt2 = 0;
        double d2theta_dt2 = 0;
        double d3phi_dt3 = 0;
        double d3theta_dt3 = 0;

        double r = 0;
        double dr = 0;
        double ddr = 0;
        double ddd_r = 0;
        double ddd_z = 0;

        double CWSt = 2 * M_PI * quadpoints[k];

        // Termos com Cossenos e as suas derivas
        for (int i = 0; i < order + 1; ++i)
        {
            phi += phi_c[i] * cos(i * CWSt);
            theta += theta_c[i] * cos(i * CWSt);

            dphi += -phi_c[i] * i * sin(i * CWSt);
            dtheta_dt += -theta_c[i] * i * sin(i * CWSt);

            d2phi_dt2 += -phi_c[i] * pow(i, 2) * cos(i * CWSt);
            d2theta_dt2 += -theta_c[i] * pow(i, 2) * cos(i * CWSt);

            d3phi_dt3 += phi_c[i] * pow(i, 3) * sin(i * CWSt);
            d3theta_dt3 += theta_c[i] * pow(i, 3) * sin(i * CWSt);
        }
        // Termos com Senos e as suas derivas
        for (int i = 1; i < order + 1; ++i)
        {
            phi += phi_s[i - 1] * sin(i * CWSt);
            theta += theta_s[i - 1] * sin(i * CWSt);

            dphi += phi_s[i - 1] * i * cos(i * CWSt);
            dtheta_dt += theta_s[i - 1] * i * cos(i * CWSt);

            d2phi_dt2 += -phi_s[i - 1] * pow(i, 2) * sin(i * CWSt);
            d2theta_dt2 += -theta_s[i - 1] * pow(i, 2) * sin(i * CWSt);

            d3phi_dt3 += -phi_s[i - 1] * pow(i, 3) * cos(i * CWSt);
            d3theta_dt3 += -theta_s[i - 1] * pow(i, 3) * cos(i * CWSt);
        }

        phi += phi_l * CWSt;
        theta += theta_l * CWSt;
        dphi += phi_l;
        dtheta_dt += theta_l;

        // SURFACE
        for (int m = 0; m <= mpol; ++m)
        {
            for (int i = 0; i < 2 * ntor + 1; ++i)
            {
                int n = i - ntor;
                r += rc(m, i) * cos(m * theta - nfp * n * phi);
                dr += -rc(m, i) * sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi);
                ddr += -rc(m, i) * cos(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi), 2) - rc(m, i) * sin(m * theta - nfp * n * phi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2);
                ddd_r += rc(m, i) * sin(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi), 3) - rc(m, i) * cos(m * theta - nfp * n * phi) * 2 * (m * dtheta_dt - nfp * n * dphi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2) - rc(m, i) * cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2) - rc(m, i) * sin(m * theta - nfp * n * phi) * (m * d3theta_dt3 - nfp * n * d3phi_dt3);
                ddd_z += -zs(m, i) * cos(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi), 3) - zs(m, i) * sin(m * theta - nfp * n * phi) * 2 * (m * dtheta_dt - nfp * n * dphi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2) - zs(m, i) * sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2) + zs(m, i) * cos(m * theta - nfp * n * phi) * (m * d3theta_dt3 - nfp * n * d3phi_dt3);

                if (!stellsym)
                {
                    int n = i - ntor;
                    r += rs(m, i) * sin(m * theta - nfp * n * phi);
                    dr += rs(m, i) * cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi);
                    ddr += -rs(m, i) * sin(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi), 2) + rs(m, i) * cos(m * theta - nfp * n * phi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2);
                    ddd_r += -rs(m, i) * cos(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi), 3) - rs(m, i) * sin(m * theta - nfp * n * phi) * 2 * (m * dtheta_dt - nfp * n * dphi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2) - rs(m, i) * sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2) + rs(m, i) * cos(m * theta - nfp * n * phi) * (m * d3theta_dt3 - nfp * n * d3phi_dt3);
                    ddd_z += zc(m, i) * sin(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi), 3) - zc(m, i) * cos(m * theta - nfp * n * phi) * 2 * (m * dtheta_dt - nfp * n * dphi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2) - zc(m, i) * cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2) - zc(m, i) * sin(m * theta - nfp * n * phi) * (m * d3theta_dt3 - nfp * n * d3phi_dt3);
                }
            }
        }

        data(k, 0) = ddd_r * cos(phi) - ddr * sin(phi) * dphi - 2 * (ddr * sin(phi) * dphi) - 2 * (dr * cos(phi) * pow(dphi, 2)) - 2 * (dr * sin(phi) * d2phi_dt2) - dr * cos(phi) * pow(dphi, 2) + r * sin(phi) * pow(dphi, 3) - r * cos(phi) * 2 * dphi * d2phi_dt2 - dr * sin(phi) * d2phi_dt2 - r * cos(phi) * dphi * d2phi_dt2 - r * sin(phi) * d3phi_dt3;
        data(k, 1) = ddd_r * sin(phi) + ddr * cos(phi) * dphi + 2 * (ddr * cos(phi) * dphi) - 2 * (dr * sin(phi) * pow(dphi, 2)) + 2 * (dr * cos(phi) * d2phi_dt2) - dr * sin(phi) * pow(dphi, 2) - r * cos(phi) * pow(dphi, 3) - r * sin(phi) * 2 * dphi * d2phi_dt2 + dr * cos(phi) * d2phi_dt2 - r * sin(phi) * dphi * d2phi_dt2 + r * cos(phi) * d3phi_dt3;
        data(k, 2) = ddd_z;
    }
    data *= 2 * M_PI * 2 * M_PI * 2 * M_PI;
};

template <class Array>
void CurveCWSFourier<Array>::dgamma_by_dcoeff_impl(Array &data)
{

    CurveCWSFourier<Array>::set_dofs_surface(idofs);
    data *= 0;

    for (int k = 0; k < numquadpoints; ++k)
    {
        double CWSt = 2 * M_PI * quadpoints[k];

        double phi = 0;
        double theta = 0;
        Array phi_array = xt::zeros<double>({2 * (order + 1)});
        Array theta_array = xt::zeros<double>({2 * (order + 1)});

        double r = 0;
        Array r_array = xt::zeros<double>({4 * (order + 1)});
        Array z_array = xt::zeros<double>({4 * (order + 1)});
        double r_aux1 = 0;
        double z_aux1 = 0;
        double r_aux2 = 0;
        double z_aux2 = 0;

        int counter = 0;

        phi_array[counter] = CWSt;
        theta_array[counter] = CWSt;

        counter++;

        for (int i = 0; i < order + 1; ++i)
        {
            phi_array[counter] = cos(i * CWSt);
            theta_array[counter] = cos(i * CWSt);
            counter++;

            phi += phi_c[i] * cos(i * CWSt);
            theta += theta_c[i] * cos(i * CWSt);
        }

        for (int i = 1; i < order + 1; ++i)
        {
            phi_array[counter] = sin(i * CWSt);
            theta_array[counter] = sin(i * CWSt);
            counter++;

            phi += phi_s[i - 1] * sin(i * CWSt);
            theta += theta_s[i - 1] * sin(i * CWSt);
        }

        phi += phi_l * CWSt;
        theta += theta_l * CWSt;

// SURFACE
#pragma omp parallel for
        for (int i = 0; i < counter; ++i)
        {
            r = 0;
            r_aux1 = 0;
            z_aux1 = 0;
            r_aux2 = 0;
            z_aux2 = 0;

            for (int m = 0; m <= mpol; ++m)
            {
                for (int j = 0; j < 2 * ntor + 1; ++j)
                {
                    int n = j - ntor;

                    r += rc(m, j) * cos(m * theta - nfp * n * phi);

                    r_aux1 += -rc(m, j) * sin(m * theta - nfp * n * phi) * (m * theta_array[i]);
                    r_aux2 += -rc(m, j) * sin(m * theta - nfp * n * phi) * (-nfp * n * phi_array[i]);

                    z_aux1 += zs(m, j) * cos(m * theta - nfp * n * phi) * (m * theta_array[i]);
                    z_aux2 += zs(m, j) * cos(m * theta - nfp * n * phi) * (-nfp * n * phi_array[i]);

                    if (!stellsym)
                    {
                        r_aux1 += rs(m, j) * cos(m * theta - nfp * n * phi) * (m * theta_array[i]);
                        r_aux2 += rs(m, j) * cos(m * theta - nfp * n * phi) * (-nfp * n * phi_array[i]);

                        z_aux1 += -zc(m, j) * sin(m * theta - nfp * n * phi) * (m * theta_array[i]);
                        z_aux2 += -zc(m, j) * sin(m * theta - nfp * n * phi) * (-nfp * n * phi_array[i]);
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
            data(k, 0, p) = r_array[p] * cos(phi);
            data(k, 1, p) = r_array[p] * sin(phi);
            data(k, 2, p) = z_array[p];

            data(k, 0, p + counter) = r_array[p + counter] * cos(phi) - r * sin(phi) * phi_array[p];
            data(k, 1, p + counter) = r_array[p + counter] * sin(phi) + r * cos(phi) * phi_array[p];
            data(k, 2, p + counter) = z_array[p + counter];
        }
    }
};

template <class Array>
void CurveCWSFourier<Array>::dgammadash_by_dcoeff_impl(Array &data)
{
    CurveCWSFourier<Array>::set_dofs_surface(idofs);
    data *= 0;

    for (int k = 0; k < numquadpoints; ++k)
    {
        double CWSt = 2 * M_PI * quadpoints[k];

        double phi = 0;
        double theta = 0;
        Array phi_array = xt::zeros<double>({2 * (order + 1)});
        Array theta_array = xt::zeros<double>({2 * (order + 1)});

        double dphi = 0;
        double dtheta_dt = 0;
        Array dphi_array = xt::zeros<double>({2 * (order + 1)});
        Array dtheta_array = xt::zeros<double>({2 * (order + 1)});

        double r = 0;
        double dr = 0;
        Array r_array = xt::zeros<double>({4 * (order + 1)});
        Array dr_array = xt::zeros<double>({4 * (order + 1)});
        Array dz_array = xt::zeros<double>({4 * (order + 1)});

        double r_aux1 = 0;
        double r_aux2 = 0;
        double dr_aux1 = 0;
        double dr_aux2 = 0;
        double dz_aux1 = 0;
        double dz_aux2 = 0;

        int counter = 0;

        theta_array[counter] = CWSt;
        phi_array[counter] = CWSt;
        dtheta_array[counter] = 0;
        dphi_array[counter] = 0;

        // Termos com Cossenos e as suas derivas
        for (int i = 0; i < order + 1; ++i)
        {
            phi_array[counter] = cos(i * CWSt);
            theta_array[counter] = cos(i * CWSt);

            dtheta_array[counter] = -i * sin(i * CWSt);
            dphi_array[counter] = -i * sin(i * CWSt);

            counter++;

            phi += phi_c[i] * cos(i * CWSt);
            theta += theta_c[i] * cos(i * CWSt);

            dphi += -phi_c[i] * i * sin(i * CWSt);
            dtheta_dt += -theta_c[i] * i * sin(i * CWSt);
        }
        // Termos com Senos e as suas derivas
        for (int i = 1; i < order + 1; ++i)
        {
            phi_array[counter] = sin(i * CWSt);
            theta_array[counter] = sin(i * CWSt);

            dtheta_array[counter] = i * cos(i * CWSt);
            dphi_array[counter] = i * cos(i * CWSt);

            counter++;

            phi += phi_s[i - 1] * sin(i * CWSt);
            theta += theta_s[i - 1] * sin(i * CWSt);

            dphi += phi_s[i - 1] * i * cos(i * CWSt);
            dtheta_dt += theta_s[i - 1] * i * cos(i * CWSt);
        }

        phi += phi_l * CWSt;
        theta += theta_l * CWSt;
        dphi += phi_l;
        dtheta_dt += theta_l;

// SURFACE
#pragma omp parallel for
        for (int i = 0; i < counter; ++i)
        {
            r = 0;
            dr = 0;
            r_aux1 = 0;
            r_aux2 = 0;
            dr_aux1 = 0;
            dr_aux2 = 0;
            dz_aux1 = 0;
            dz_aux2 = 0;

            for (int m = 0; m <= mpol; ++m)
            {
                for (int j = 0; j < 2 * ntor + 1; ++j)
                {
                    int n = j - ntor;

                    r += rc(m, j) * cos(m * theta - nfp * n * phi);
                    dr += -rc(m, j) * sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi);

                    r_aux1 += -rc(m, j) * sin(m * theta - nfp * n * phi) * (m * theta_array[i]);
                    r_aux2 += -rc(m, j) * sin(m * theta - nfp * n * phi) * (-nfp * n * phi_array[i]);

                    dr_aux1 += -rc(m, j) * (cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi) * (m * theta_array[i]) + sin(m * theta - nfp * n * phi) * (m * dtheta_array[i]));
                    dr_aux2 += -rc(m, j) * (cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi) * (-nfp * n * phi_array[i]) + sin(m * theta - nfp * n * phi) * (-nfp * n * phi_array[i]));

                    dz_aux1 += zs(m, j) * (sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi) * (m * theta_array[i]) + cos(m * theta - nfp * n * phi) * (m * dtheta_array[i]));
                    dz_aux2 += zs(m, j) * (sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi) * (-nfp * n * phi_array[i]) + cos(m * theta - nfp * n * phi) * (-nfp * n * phi_array[i]));

                    if (!stellsym)
                    {
                        r += rs(m, j) * sin(m * theta - nfp * n * phi);
                        dr += rs(m, j) * cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi);

                        r_aux1 += rs(m, j) * cos(m * theta - nfp * n * phi) * (m * theta_array[i]);
                        r_aux2 += rs(m, j) * cos(m * theta - nfp * n * phi) * (-nfp * n * phi_array[i]);

                        dr_aux1 += rs(m, j) * (sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi) * (m * theta_array[i]) + cos(m * theta - nfp * n * phi) * (m * dtheta_array[i]));
                        dr_aux2 += rs(m, j) * (sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi) * (-nfp * n * phi_array[i]) + cos(m * theta - nfp * n * phi) * (-nfp * n * phi_array[i]));

                        dz_aux1 += -zc(m, j) * (cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi) * (m * theta_array[i]) + sin(m * theta - nfp * n * phi) * (m * dtheta_array[i]));
                        dz_aux2 += -zc(m, j) * (cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi) * (-nfp * n * phi_array[i]) + sin(m * theta - nfp * n * phi) * (-nfp * n * phi_array[i]));
                    }
                }
            }
            r_array[i] = r_aux1;
            r_array[i + counter] = r_aux2;

            dr_array[i] = dr_aux1;
            dr_array[i + counter] = dr_aux2;

            dz_array[i] = dz_aux1;
            dz_array[i + counter] = dz_aux2;
        }

        for (int p = 0; p < counter; p++)
        {
            data(k, 0, p) = dr_array[p] * cos(phi) - r_array[p] * sin(phi) * dphi;
            data(k, 1, p) = dr_array[p] * sin(phi) + r_array[p] * cos(phi) * dphi;
            data(k, 2, p) = dz_array[p];

            data(k, 0, p + counter) = dr_array[p + counter] * cos(phi) - (dr * sin(phi) + r * cos(phi) * dphi) * phi_array[p + counter] - r_array[p + counter] * sin(phi) * dphi - r * sin(phi) * dphi_array[p + counter];
            data(k, 1, p + counter) = dr_array[p + counter] * sin(phi) + (dr * cos(phi) - r * sin(phi) * dphi) * phi_array[p + counter] + r_array[p + counter] * cos(phi) * dphi + r * cos(phi) * dphi_array[p + counter];
            data(k, 2, p + counter) = dz_array[p + counter];
        }
    }
    data *= (2 * M_PI);
};

template <class Array>
void CurveCWSFourier<Array>::dgammadashdash_by_dcoeff_impl(Array &data)
{

    CurveCWSFourier<Array>::set_dofs_surface(idofs);

    data *= 0;

    for (int k = 0; k < numquadpoints; ++k)
    {
        double CWSt = 2 * M_PI * quadpoints[k];

        double phi = 0;
        double theta = 0;
        Array phi_array = xt::zeros<double>({2 * (order + 1)});
        Array theta_array = xt::zeros<double>({2 * (order + 1)});

        double dphi = 0;
        double dtheta_dt = 0;
        Array dphi_array = xt::zeros<double>({2 * (order + 1)});
        Array dtheta_array = xt::zeros<double>({2 * (order + 1)});

        double d2phi_dt2 = 0;
        double d2theta_dt2 = 0;
        Array ddphi_array = xt::zeros<double>({2 * (order + 1)});
        Array ddtheta_dtt_array = xt::zeros<double>({2 * (order + 1)});

        double r = 0;
        double dr = 0;
        double ddr = 0;
        Array r_array = xt::zeros<double>({4 * (order + 1)});
        Array dr_array = xt::zeros<double>({4 * (order + 1)});
        Array ddr_array = xt::zeros<double>({4 * (order + 1)});
        Array ddz_array = xt::zeros<double>({4 * (order + 1)});

        double r_aux1 = 0;
        double r_aux2 = 0;
        double dr_aux1 = 0;
        double dr_aux2 = 0;
        double ddr_aux1 = 0;
        double ddr_aux2 = 0;
        double ddz_aux1 = 0;
        double ddz_aux2 = 0;

        int counter = 0;

        theta_array[counter] = CWSt;
        phi_array[counter] = CWSt;
        dtheta_array[counter] = 0;
        dphi_array[counter] = 0;
        ddtheta_dtt_array[counter] = 0;
        ddphi_array[counter] = 0;

        // Termos com Cossenos e as suas derivas
        for (int i = 0; i < order + 1; ++i)
        {
            phi_array[counter] = cos(i * CWSt);
            theta_array[counter] = cos(i * CWSt);

            dtheta_array[counter] = -i * sin(i * CWSt);
            dphi_array[counter] = -i * sin(i * CWSt);

            ddtheta_dtt_array[counter] = -pow(i, 2) * cos(i * CWSt);
            ddphi_array[counter] = -pow(i, 2) * cos(i * CWSt);

            counter++;

            phi += phi_c[i] * cos(i * CWSt);
            theta += theta_c[i] * cos(i * CWSt);

            dphi += -phi_c[i] * i * sin(i * CWSt);
            dtheta_dt += -theta_c[i] * i * sin(i * CWSt);

            d2phi_dt2 += -phi_c[i] * pow(i, 2) * cos(i * CWSt);
            d2theta_dt2 += -theta_c[i] * pow(i, 2) * cos(i * CWSt);
        }
        // Termos com Senos e as suas derivas
        for (int i = 1; i < order + 1; ++i)
        {
            phi_array[counter] = sin(i * CWSt);
            theta_array[counter] = sin(i * CWSt);

            dtheta_array[counter] = i * cos(i * CWSt);
            dphi_array[counter] = i * cos(i * CWSt);

            ddtheta_dtt_array[counter] = -pow(i, 2) * sin(i * CWSt);
            ddphi_array[counter] = -pow(i, 2) * sin(i * CWSt);

            counter++;

            phi += phi_s[i - 1] * sin(i * CWSt);
            theta += theta_s[i - 1] * sin(i * CWSt);

            dphi += phi_s[i - 1] * i * cos(i * CWSt);
            dtheta_dt += theta_s[i - 1] * i * cos(i * CWSt);

            d2phi_dt2 += -phi_s[i - 1] * pow(i, 2) * sin(i * CWSt);
            d2theta_dt2 += -theta_s[i - 1] * pow(i, 2) * sin(i * CWSt);
        }

        phi += phi_l * CWSt;
        theta += theta_l * CWSt;
        dphi += phi_l;
        dtheta_dt += theta_l;

// SURFACE
#pragma omp parallel for
        for (int i = 0; i < counter; ++i)
        {
            r = 0;
            dr = 0;
            ddr = 0;

            r_aux1 = 0;
            r_aux2 = 0;

            dr_aux1 = 0;
            dr_aux2 = 0;

            ddr_aux1 = 0;
            ddr_aux2 = 0;

            ddz_aux1 = 0;
            ddz_aux2 = 0;

            for (int m = 0; m <= mpol; ++m)
            {
                for (int j = 0; j < 2 * ntor + 1; ++j)
                {
                    int n = j - ntor;

                    r += rc(m, j) * cos(m * theta - nfp * n * phi);
                    dr += -rc(m, j) * sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi);
                    ddr += -rc(m, i) * cos(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi), 2) - rc(m, i) * sin(m * theta - nfp * n * phi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2);

                    r_aux1 += -rc(m, j) * sin(m * theta - nfp * n * phi) * (m * theta_array[i]);
                    r_aux2 += -rc(m, j) * sin(m * theta - nfp * n * phi) * (-nfp * n * phi_array[i]);

                    dr_aux1 += -rc(m, j) * (cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi) * (m * theta_array[i]) + sin(m * theta - nfp * n * phi) * (m * dtheta_array[i]));
                    dr_aux2 += -rc(m, j) * (cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi) * (-nfp * n * phi_array[i]) + sin(m * theta - nfp * n * phi) * (-nfp * n * phi_array[i]));

                    ddr_aux1 += -rc(m, j) * ((-sin(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi), 2) + cos(m * theta - nfp * n * phi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2)) * (m * theta_array[i]) + cos(m * theta - nfp * n * phi) * 2 * (m * dtheta_dt - nfp * n * dphi) * (m * dtheta_array[i]) + sin(m * theta - nfp * n * phi) * (m * ddtheta_dtt_array[i]));
                    ddr_aux2 += -rc(m, j) * ((-sin(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi), 2) + cos(m * theta - nfp * n * phi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2)) * (-nfp * n * phi_array[i]) + cos(m * theta - nfp * n * phi) * 2 * (m * dtheta_dt - nfp * n * dphi) * (-nfp * n * dphi_array[i]) + sin(m * theta - nfp * n * phi) * (-nfp * n * ddphi_array[i]));

                    ddz_aux1 += zs(m, j) * ((-cos(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi), 2) - sin(m * theta - nfp * n * phi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2) * (m * theta_array[i])) - 2 * sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi) * (m * dtheta_array[i]) + cos(m * theta - nfp * n * phi) * (m * ddtheta_dtt_array[i]));
                    ddz_aux2 += zs(m, j) * ((-cos(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi), 2) - sin(m * theta - nfp * n * phi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2) * (-nfp * n * phi_array[i])) - 2 * sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi) * (-nfp * n * dphi_array[i]) + cos(m * theta - nfp * n * phi) * (-nfp * n * ddphi_array[i]));
                    // STELLSYM IS MISSING FOR NOW
                }
            }
            r_array[i] = r_aux1;
            r_array[i + counter] = r_aux2;

            dr_array[i] = dr_aux1;
            dr_array[i + counter] = dr_aux2;

            ddr_array[i] = ddr_aux1;
            ddr_array[i + counter] = ddr_aux2;

            ddz_array[i] = ddz_aux1;
            ddz_array[i + counter] = ddz_aux2;
        }

        for (int p = 0; p < counter; p++)
        {
            data(k, 0, p) = ddr_array[p] * cos(phi) - 2 * (dr_array[p] * sin(phi) * dphi) - r_array[p] * (cos(phi) * pow(dphi, 2) + sin(phi) * d2phi_dt2);
            data(k, 1, p) = ddr_array[p] * sin(phi) + 2 * (dr_array[p] * cos(phi) * dphi) - r_array[p] * (sin(phi) * pow(dphi, 2) - cos(phi) * d2phi_dt2);
            data(k, 2, p) = ddz_array[p];

            data(k, 0, p + counter) = ddr_array[p + counter] * cos(phi) - 2 * dr_array[p + counter] * sin(phi) * dphi - r * sin(phi) * ddphi_array[p + counter] + (-sin(phi) * d2phi_dt2 - cos(phi) * pow(dphi, 2)) * r_array[p + counter] + (-2 * dr * sin(phi) - 2 * r * dphi * cos(phi)) * dphi_array[p + counter] + (sin(phi) * (-ddr + r * pow(dphi, 2)) + cos(phi) * (-2 * dr * dphi - r * d2phi_dt2)) * phi_array[p + counter];
            data(k, 1, p + counter) = ddr_array[p + counter] * sin(phi) + 2 * dr_array[p + counter] * cos(phi) * dphi + r * cos(phi) * ddphi_array[p + counter] + (cos(phi) * d2phi_dt2 - sin(phi) * pow(dphi, 2)) * r_array[p + counter] + (2 * dr * cos(phi) - 2 * r * dphi * sin(phi)) * dphi_array[p + counter] + (cos(phi) * (ddr - r * pow(dphi, 2)) + sin(phi) * (-2 * dr * dphi - r * d2phi_dt2)) * phi_array[p + counter];
            data(k, 2, p + counter) = ddz_array[p + counter];
        }
    }
    data *= 2 * M_PI * 2 * M_PI;
};
#include "xtensor-python/pyarray.hpp" // Numpy bindings
typedef xt::pyarray<double> Array;
template class CurveCWSFourier<Array>;