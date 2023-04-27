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
        double dphi_dt = 0;
        double dtheta_dt = 0;

        double r = 0;

        double dr_dt = 0;
        double dz_dt = 0;

        double CWSt = 2 * M_PI * quadpoints[k];

        // Termos com Cossenos e as suas derivas
        for (int i = 0; i < order + 1; ++i)
        {
            phi += phi_c[i] * cos(i * CWSt);
            theta += theta_c[i] * cos(i * CWSt);

            dphi_dt += -phi_c[i] * i * sin(i * CWSt);
            dtheta_dt += -theta_c[i] * i * sin(i * CWSt);
        }
        // Termos com Senos e as suas derivas
        for (int i = 1; i < order + 1; ++i)
        {
            phi += phi_s[i - 1] * sin(i * CWSt);
            theta += theta_s[i - 1] * sin(i * CWSt);

            dphi_dt += phi_s[i - 1] * i * cos(i * CWSt);
            dtheta_dt += theta_s[i - 1] * i * cos(i * CWSt);
        }

        phi += phi_l * CWSt;
        theta += theta_l * CWSt;
        dphi_dt += phi_l;
        dtheta_dt += theta_l;

        // SURFACE
        for (int m = 0; m <= mpol; ++m)
        {
            for (int i = 0; i < 2 * ntor + 1; ++i)
            {
                int n = i - ntor;
                r += rc(m, i) * cos(m * theta - nfp * n * phi);
                dr_dt += -rc(m, i) * sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt);
                dz_dt += zs(m, i) * cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt);

                if (!stellsym)
                {
                    r += rs(m, i) * sin(m * theta - nfp * n * phi);
                    dr_dt += rs(m, i) * cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt);
                    dz_dt += -zc(m, i) * sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt);
                }
            }
        }

        data(k, 0) = dr_dt * cos(phi) - r * sin(phi) * dphi_dt;
        data(k, 1) = dr_dt * sin(phi) + r * cos(phi) * dphi_dt;
        data(k, 2) = dz_dt;
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
        double dphi_dt = 0;
        double dtheta_dt = 0;
        double d2phi_dt2 = 0;
        double d2theta_dt2 = 0;

        double r = 0;
        double dr_dt = 0;
        double d2r_dt2 = 0;
        double d2z_dt2 = 0;

        double CWSt = 2 * M_PI * quadpoints[k];

        // Termos com Cossenos e as suas derivas
        for (int i = 0; i < order + 1; ++i)
        {
            phi += phi_c[i] * cos(i * CWSt);
            theta += theta_c[i] * cos(i * CWSt);

            dphi_dt += -phi_c[i] * i * sin(i * CWSt);
            dtheta_dt += -theta_c[i] * i * sin(i * CWSt);

            d2phi_dt2 += -phi_c[i] * pow(i, 2) * cos(i * CWSt);
            d2theta_dt2 += -theta_c[i] * pow(i, 2) * cos(i * CWSt);
        }
        // Termos com Senos e as suas derivas
        for (int i = 1; i < order + 1; ++i)
        {
            phi += phi_s[i - 1] * sin(i * CWSt);
            theta += theta_s[i - 1] * sin(i * CWSt);

            dphi_dt += phi_s[i - 1] * i * cos(i * CWSt);
            dtheta_dt += theta_s[i - 1] * i * cos(i * CWSt);

            d2phi_dt2 += -phi_s[i - 1] * pow(i, 2) * sin(i * CWSt);
            d2theta_dt2 += -theta_s[i - 1] * pow(i, 2) * sin(i * CWSt);
        }

        phi += phi_l * CWSt;
        theta += theta_l * CWSt;
        dphi_dt += phi_l;
        dtheta_dt += theta_l;

        // SURFACE
        for (int m = 0; m <= mpol; ++m)
        {
            for (int i = 0; i < 2 * ntor + 1; ++i)
            {
                int n = i - ntor;
                r += rc(m, i) * cos(m * theta - nfp * n * phi);
                dr_dt += -rc(m, i) * sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt);
                d2r_dt2 += -rc(m, i) * cos(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi_dt), 2) - rc(m, i) * sin(m * theta - nfp * n * phi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2);
                d2z_dt2 += -zs(m, i) * sin(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi_dt), 2) + zs(m, i) * cos(m * theta - nfp * n * phi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2);

                if (!stellsym)
                {
                    r += rs(m, i) * sin(m * theta - nfp * n * phi);
                    dr_dt += rs(m, i) * cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt);
                    d2r_dt2 += -rs(m, i) * sin(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi_dt), 2) + rs(m, i) * cos(m * theta - nfp * n * phi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2);
                    d2z_dt2 += -zc(m, i) * cos(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi_dt), 2) - zc(m, i) * sin(m * theta - nfp * n * phi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2);
                }
            }
        }

        data(k, 0) = d2r_dt2 * cos(phi) - 2 * (dr_dt * sin(phi) * dphi_dt) - r * (cos(phi) * pow(dphi_dt, 2) + sin(phi) * d2phi_dt2);
        data(k, 1) = d2r_dt2 * sin(phi) + 2 * (dr_dt * cos(phi) * dphi_dt) - r * (sin(phi) * pow(dphi_dt, 2) - cos(phi) * d2phi_dt2);
        data(k, 2) = d2z_dt2;
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
        double dphi_dt = 0;
        double dtheta_dt = 0;
        double d2phi_dt2 = 0;
        double d2theta_dt2 = 0;
        double d3phi_dt3 = 0;
        double d3theta_dt3 = 0;

        double r = 0;
        double dr_dt = 0;
        double d2r_dt2 = 0;
        double d3r_dt3 = 0;
        double d3z_dt3 = 0;

        double CWSt = 2 * M_PI * quadpoints[k];

        // Termos com Cossenos e as suas derivas
        for (int i = 0; i < order + 1; ++i)
        {
            phi += phi_c[i] * cos(i * CWSt);
            theta += theta_c[i] * cos(i * CWSt);

            dphi_dt += -phi_c[i] * i * sin(i * CWSt);
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

            dphi_dt += phi_s[i - 1] * i * cos(i * CWSt);
            dtheta_dt += theta_s[i - 1] * i * cos(i * CWSt);

            d2phi_dt2 += -phi_s[i - 1] * pow(i, 2) * sin(i * CWSt);
            d2theta_dt2 += -theta_s[i - 1] * pow(i, 2) * sin(i * CWSt);

            d3phi_dt3 += -phi_s[i - 1] * pow(i, 3) * cos(i * CWSt);
            d3theta_dt3 += -theta_s[i - 1] * pow(i, 3) * cos(i * CWSt);
        }

        phi += phi_l * CWSt;
        theta += theta_l * CWSt;
        dphi_dt += phi_l;
        dtheta_dt += theta_l;

        // SURFACE
        for (int m = 0; m <= mpol; ++m)
        {
            for (int i = 0; i < 2 * ntor + 1; ++i)
            {
                int n = i - ntor;
                r += rc(m, i) * cos(m * theta - nfp * n * phi);
                dr_dt += -rc(m, i) * sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt);
                d2r_dt2 += -rc(m, i) * cos(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi_dt), 2) - rc(m, i) * sin(m * theta - nfp * n * phi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2);
                d3r_dt3 += rc(m, i) * sin(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi_dt), 3) - rc(m, i) * cos(m * theta - nfp * n * phi) * 2 * (m * dtheta_dt - nfp * n * dphi_dt) * (m * d2theta_dt2 - nfp * n * d2phi_dt2) - rc(m, i) * cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt) * (m * d2theta_dt2 - nfp * n * d2phi_dt2) - rc(m, i) * sin(m * theta - nfp * n * phi) * (m * d3theta_dt3 - nfp * n * d3phi_dt3);
                d3z_dt3 += -zs(m, i) * cos(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi_dt), 3) - zs(m, i) * sin(m * theta - nfp * n * phi) * 2 * (m * dtheta_dt - nfp * n * dphi_dt) * (m * d2theta_dt2 - nfp * n * d2phi_dt2) - zs(m, i) * sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt) * (m * d2theta_dt2 - nfp * n * d2phi_dt2) + zs(m, i) * cos(m * theta - nfp * n * phi) * (m * d3theta_dt3 - nfp * n * d3phi_dt3);

                if (!stellsym)
                {
                    r += rs(m, i) * sin(m * theta - nfp * n * phi);
                    dr_dt += rs(m, i) * cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt);
                    d2r_dt2 += -rs(m, i) * sin(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi_dt), 2) + rs(m, i) * cos(m * theta - nfp * n * phi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2);
                    d3r_dt3 += -rs(m, i) * cos(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi_dt), 3) - rs(m, i) * sin(m * theta - nfp * n * phi) * 2 * (m * dtheta_dt - nfp * n * dphi_dt) * (m * d2theta_dt2 - nfp * n * d2phi_dt2) - rs(m, i) * sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt) * (m * d2theta_dt2 - nfp * n * d2phi_dt2) + rs(m, i) * cos(m * theta - nfp * n * phi) * (m * d3theta_dt3 - nfp * n * d3phi_dt3);
                    d3z_dt3 += zc(m, i) * sin(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi_dt), 3) - zc(m, i) * cos(m * theta - nfp * n * phi) * 2 * (m * dtheta_dt - nfp * n * dphi_dt) * (m * d2theta_dt2 - nfp * n * d2phi_dt2) - zc(m, i) * cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt) * (m * d2theta_dt2 - nfp * n * d2phi_dt2) - zc(m, i) * sin(m * theta - nfp * n * phi) * (m * d3theta_dt3 - nfp * n * d3phi_dt3);
                }
            }
        }

        data(k, 0) = d3r_dt3 * cos(phi) - d2r_dt2 * sin(phi) * dphi_dt - 2 * (d2r_dt2 * sin(phi) * dphi_dt) - 2 * (dr_dt * cos(phi) * pow(dphi_dt, 2)) - 2 * (dr_dt * sin(phi) * d2phi_dt2) - dr_dt * cos(phi) * pow(dphi_dt, 2) + r * sin(phi) * pow(dphi_dt, 3) - r * cos(phi) * 2 * dphi_dt * d2phi_dt2 - dr_dt * sin(phi) * d2phi_dt2 - r * cos(phi) * dphi_dt * d2phi_dt2 - r * sin(phi) * d3phi_dt3;
        data(k, 1) = d3r_dt3 * sin(phi) + d2r_dt2 * cos(phi) * dphi_dt + 2 * (d2r_dt2 * cos(phi) * dphi_dt) - 2 * (dr_dt * sin(phi) * pow(dphi_dt, 2)) + 2 * (dr_dt * cos(phi) * d2phi_dt2) - dr_dt * sin(phi) * pow(dphi_dt, 2) - r * cos(phi) * pow(dphi_dt, 3) - r * sin(phi) * 2 * dphi_dt * d2phi_dt2 + dr_dt * cos(phi) * d2phi_dt2 - r * sin(phi) * dphi_dt * d2phi_dt2 + r * cos(phi) * d3phi_dt3;
        data(k, 2) = d3z_dt3;
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
        Array dphi_by_dphicoeff = xt::zeros<double>({2 * (order + 1)});
        Array dtheta_by_dthetacoeff = xt::zeros<double>({2 * (order + 1)});

        double r = 0;
        Array dr_dcoeff = xt::zeros<double>({4 * (order + 1)});
        Array dz_dcoeff = xt::zeros<double>({4 * (order + 1)});

        double dr_dthetacoeff = 0;
        double dz_dthetacoeff = 0;
        double dr_dphicoeff = 0;
        double dz_dphicoeff = 0;

        int counter = 0;

        dphi_by_dphicoeff[counter] = CWSt;
        dtheta_by_dthetacoeff[counter] = CWSt;

        counter++;

        for (int i = 0; i < order + 1; ++i)
        {
            dphi_by_dphicoeff[counter] = cos(i * CWSt);
            dtheta_by_dthetacoeff[counter] = cos(i * CWSt);
            counter++;

            phi += phi_c[i] * cos(i * CWSt);
            theta += theta_c[i] * cos(i * CWSt);
        }

        for (int i = 1; i < order + 1; ++i)
        {
            dphi_by_dphicoeff[counter] = sin(i * CWSt);
            dtheta_by_dthetacoeff[counter] = sin(i * CWSt);
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
            dr_dthetacoeff = 0;
            dz_dthetacoeff = 0;
            dr_dphicoeff = 0;
            dz_dphicoeff = 0;

            for (int m = 0; m <= mpol; ++m)
            {
                for (int j = 0; j < 2 * ntor + 1; ++j)
                {
                    int n = j - ntor;

                    r += rc(m, j) * cos(m * theta - nfp * n * phi);

                    dr_dthetacoeff += -rc(m, j) * sin(m * theta - nfp * n * phi) * (m * dtheta_by_dthetacoeff[i]);
                    dr_dphicoeff += -rc(m, j) * sin(m * theta - nfp * n * phi) * (-nfp * n * dphi_by_dphicoeff[i]);

                    dz_dthetacoeff += zs(m, j) * cos(m * theta - nfp * n * phi) * (m * dtheta_by_dthetacoeff[i]);
                    dz_dphicoeff += zs(m, j) * cos(m * theta - nfp * n * phi) * (-nfp * n * dphi_by_dphicoeff[i]);

                    if (!stellsym)
                    {
                        dr_dthetacoeff += rs(m, j) * cos(m * theta - nfp * n * phi) * (m * dtheta_by_dthetacoeff[i]);
                        dr_dphicoeff += rs(m, j) * cos(m * theta - nfp * n * phi) * (-nfp * n * dphi_by_dphicoeff[i]);

                        dz_dthetacoeff += -zc(m, j) * sin(m * theta - nfp * n * phi) * (m * dtheta_by_dthetacoeff[i]);
                        dz_dphicoeff += -zc(m, j) * sin(m * theta - nfp * n * phi) * (-nfp * n * dphi_by_dphicoeff[i]);
                    }
                }
            }
            dr_dcoeff[i] = dr_dthetacoeff;
            dz_dcoeff[i] = dz_dthetacoeff;
            dr_dcoeff[i + counter] = dr_dphicoeff;
            dz_dcoeff[i + counter] = dz_dphicoeff;
        }

        for (int p = 0; p < counter; p++)
        {
            data(k, 0, p) = dr_dcoeff[p] * cos(phi);
            data(k, 1, p) = dr_dcoeff[p] * sin(phi);
            data(k, 2, p) = dz_dcoeff[p];

            data(k, 0, p + counter) = dr_dcoeff[p + counter] * cos(phi) - r * sin(phi) * dphi_by_dphicoeff[p];
            data(k, 1, p + counter) = dr_dcoeff[p + counter] * sin(phi) + r * cos(phi) * dphi_by_dphicoeff[p];
            data(k, 2, p + counter) = dz_dcoeff[p + counter];
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
        Array dphi_by_dphicoeff = xt::zeros<double>({2 * (order + 1)});
        Array dtheta_by_dthetacoeff = xt::zeros<double>({2 * (order + 1)});

        double dphi_dt = 0;
        double dtheta_dt = 0;
        Array d2phi_by_dphicoeffdt = xt::zeros<double>({2 * (order + 1)});
        Array d2theta_by_dthetacoeffdt = xt::zeros<double>({2 * (order + 1)});

        double r = 0;
        double dr_dt = 0;

        Array dr_dcoeff = xt::zeros<double>({4 * (order + 1)});
        Array d2r_dcoeffdt = xt::zeros<double>({4 * (order + 1)});

        Array d2z_dcoeffdt = xt::zeros<double>({4 * (order + 1)});

        // AUX VECTORS
        double dr_dthetacoeff = 0;
        double dr_dphicoeff = 0;
        double d2r_dthetacoeffdt = 0;
        double d2r_dphicoeffdt = 0;
        double d2z_dthetacoeffdt = 0;
        double d2z_dphicoeffdt = 0;

        int counter = 0;

        dtheta_by_dthetacoeff[counter] = CWSt;
        dphi_by_dphicoeff[counter] = CWSt;
        d2theta_by_dthetacoeffdt[counter] = 0;
        d2phi_by_dphicoeffdt[counter] = 0;

        counter++;

        // Termos com Cossenos e as suas derivadas
        for (int i = 0; i < order + 1; ++i)
        {
            dtheta_by_dthetacoeff[counter] = cos(i * CWSt);
            dphi_by_dphicoeff[counter] = cos(i * CWSt);

            d2theta_by_dthetacoeffdt[counter] = -i * sin(i * CWSt);
            d2phi_by_dphicoeffdt[counter] = -i * sin(i * CWSt);

            counter++;

            phi += phi_c[i] * cos(i * CWSt);
            theta += theta_c[i] * cos(i * CWSt);

            dphi_dt += -phi_c[i] * i * sin(i * CWSt);
            dtheta_dt += -theta_c[i] * i * sin(i * CWSt);
        }
        // Termos com Senos e as suas derivadas
        for (int i = 1; i < order + 1; ++i)
        {
            dphi_by_dphicoeff[counter] = sin(i * CWSt);
            dtheta_by_dthetacoeff[counter] = sin(i * CWSt);

            d2theta_by_dthetacoeffdt[counter] = i * cos(i * CWSt);
            d2phi_by_dphicoeffdt[counter] = i * cos(i * CWSt);

            counter++;

            phi += phi_s[i - 1] * sin(i * CWSt);
            theta += theta_s[i - 1] * sin(i * CWSt);

            dphi_dt += phi_s[i - 1] * i * cos(i * CWSt);
            dtheta_dt += theta_s[i - 1] * i * cos(i * CWSt);
        }

        phi += phi_l * CWSt;
        theta += theta_l * CWSt;
        dphi_dt += phi_l;
        dtheta_dt += theta_l;

// SURFACE
#pragma omp parallel for
        for (int i = 0; i < counter; ++i)
        {
            r = 0;
            dr_dt = 0;
            dr_dthetacoeff = 0;
            dr_dphicoeff = 0;
            d2r_dthetacoeffdt = 0;
            d2r_dphicoeffdt = 0;
            d2z_dthetacoeffdt = 0;
            d2z_dphicoeffdt = 0;

            for (int m = 0; m <= mpol; ++m)
            {
                for (int j = 0; j < 2 * ntor + 1; ++j)
                {
                    int n = j - ntor;

                    r += rc(m, j) * cos(m * theta - nfp * n * phi);
                    dr_dt += -rc(m, j) * sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt);

                    dr_dthetacoeff += -rc(m, j) * sin(m * theta - nfp * n * phi) * (m * dtheta_by_dthetacoeff[i]);
                    dr_dphicoeff += -rc(m, j) * sin(m * theta - nfp * n * phi) * (-nfp * n * dphi_by_dphicoeff[i]);

                    d2r_dthetacoeffdt += -rc(m, j) * (cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt) * (m * dtheta_by_dthetacoeff[i]) + sin(m * theta - nfp * n * phi) * (m * d2theta_by_dthetacoeffdt[i]));
                    d2r_dphicoeffdt += -rc(m, j) * (cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt) * (-nfp * n * dphi_by_dphicoeff[i]) + sin(m * theta - nfp * n * phi) * (-nfp * n * dphi_by_dphicoeff[i]));

                    d2z_dthetacoeffdt += zs(m, j) * (sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt) * (m * dtheta_by_dthetacoeff[i]) + cos(m * theta - nfp * n * phi) * (m * d2theta_by_dthetacoeffdt[i]));
                    d2z_dphicoeffdt += zs(m, j) * (sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt) * (-nfp * n * dphi_by_dphicoeff[i]) + cos(m * theta - nfp * n * phi) * (-nfp * n * dphi_by_dphicoeff[i]));

                    if (!stellsym)
                    {
                        r += rs(m, j) * sin(m * theta - nfp * n * phi);
                        dr_dt += rs(m, j) * cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt);

                        dr_dthetacoeff += rs(m, j) * cos(m * theta - nfp * n * phi) * (m * dtheta_by_dthetacoeff[i]);
                        dr_dphicoeff += rs(m, j) * cos(m * theta - nfp * n * phi) * (-nfp * n * dphi_by_dphicoeff[i]);

                        d2r_dthetacoeffdt += rs(m, j) * (sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt) * (m * dtheta_by_dthetacoeff[i]) + cos(m * theta - nfp * n * phi) * (m * d2theta_by_dthetacoeffdt[i]));
                        d2r_dphicoeffdt += rs(m, j) * (sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt) * (-nfp * n * dphi_by_dphicoeff[i]) + cos(m * theta - nfp * n * phi) * (-nfp * n * dphi_by_dphicoeff[i]));

                        d2z_dthetacoeffdt += -zc(m, j) * (cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt) * (m * dtheta_by_dthetacoeff[i]) + sin(m * theta - nfp * n * phi) * (m * d2theta_by_dthetacoeffdt[i]));
                        d2z_dphicoeffdt += -zc(m, j) * (cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt) * (-nfp * n * dphi_by_dphicoeff[i]) + sin(m * theta - nfp * n * phi) * (-nfp * n * dphi_by_dphicoeff[i]));
                    }
                }
            }
            dr_dcoeff[i] = dr_dthetacoeff;
            dr_dcoeff[i + counter] = dr_dphicoeff;

            d2r_dcoeffdt[i] = d2r_dthetacoeffdt;
            d2r_dcoeffdt[i + counter] = d2r_dphicoeffdt;

            d2z_dcoeffdt[i] = d2z_dthetacoeffdt;
            d2z_dcoeffdt[i + counter] = d2z_dphicoeffdt;
        }

        for (int p = 0; p < counter; p++)
        {
            data(k, 0, p) = d2r_dcoeffdt[p] * cos(phi) - dr_dcoeff[p] * sin(phi) * dphi_dt;
            data(k, 1, p) = d2r_dcoeffdt[p] * sin(phi) + dr_dcoeff[p] * cos(phi) * dphi_dt;
            data(k, 2, p) = d2z_dcoeffdt[p];

            data(k, 0, p + counter) = d2r_dcoeffdt[p + counter] * cos(phi) - (dr_dt * sin(phi) + r * cos(phi) * dphi_dt) * dphi_by_dphicoeff[p + counter] - dr_dcoeff[p + counter] * sin(phi) * dphi_dt - r * sin(phi) * d2phi_by_dphicoeffdt[p + counter];
            data(k, 1, p + counter) = d2r_dcoeffdt[p + counter] * sin(phi) + (dr_dt * cos(phi) - r * sin(phi) * dphi_dt) * dphi_by_dphicoeff[p + counter] + dr_dcoeff[p + counter] * cos(phi) * dphi_dt + r * cos(phi) * d2phi_by_dphicoeffdt[p + counter];
            data(k, 2, p + counter) = d2z_dcoeffdt[p + counter];
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
        Array dphi_by_dphicoeff = xt::zeros<double>({2 * (order + 1)});
        Array dtheta_by_dthetacoeff = xt::zeros<double>({2 * (order + 1)});

        double dphi_dt = 0;
        double dtheta_dt = 0;
        Array d2phi_by_dphicoeffdt = xt::zeros<double>({2 * (order + 1)});
        Array d2theta_by_dthetacoeffdt = xt::zeros<double>({2 * (order + 1)});

        double d2phi_dt2 = 0;
        double d2theta_dt2 = 0;
        Array d3phi_by_dphicoeffdt2 = xt::zeros<double>({2 * (order + 1)});
        Array d3theta_by_dthetacoeffdt2 = xt::zeros<double>({2 * (order + 1)});

        double r = 0;
        double dr_dt = 0;
        double d2r_dt2 = 0;

        Array dr_dcoeff = xt::zeros<double>({4 * (order + 1)});
        Array d2r_dcoeffdt = xt::zeros<double>({4 * (order + 1)});
        Array d3r_dcoeffdt2 = xt::zeros<double>({4 * (order + 1)});
        Array d3z_dcoeffdt2 = xt::zeros<double>({4 * (order + 1)});

        double dr_dthetacoeff = 0;
        double dr_dphicoeff = 0;
        double d2r_dthetacoeffdt = 0;
        double d2r_dphicoeffdt = 0;
        double d3r_dthetacoeffdt2 = 0;
        double d3r_dphicoeffdt2 = 0;
        double d3z_dthetacoeffdt2 = 0;
        double d3z_dphicoeffdt2 = 0;

        int counter = 0;

        dtheta_by_dthetacoeff[counter] = CWSt;
        dphi_by_dphicoeff[counter] = CWSt;
        d2theta_by_dthetacoeffdt[counter] = 0;
        d2phi_by_dphicoeffdt[counter] = 0;
        d3theta_by_dthetacoeffdt2[counter] = 0;
        d3phi_by_dphicoeffdt2[counter] = 0;

        counter++;

        // Termos com Cossenos e as suas derivas
        for (int i = 0; i < order + 1; ++i)
        {
            dphi_by_dphicoeff[counter] = cos(i * CWSt);
            dtheta_by_dthetacoeff[counter] = cos(i * CWSt);

            d2theta_by_dthetacoeffdt[counter] = -i * sin(i * CWSt);
            d2phi_by_dphicoeffdt[counter] = -i * sin(i * CWSt);

            d3theta_by_dthetacoeffdt2[counter] = -pow(i, 2) * cos(i * CWSt);
            d3phi_by_dphicoeffdt2[counter] = -pow(i, 2) * cos(i * CWSt);

            counter++;

            phi += phi_c[i] * cos(i * CWSt);
            theta += theta_c[i] * cos(i * CWSt);

            dphi_dt += -phi_c[i] * i * sin(i * CWSt);
            dtheta_dt += -theta_c[i] * i * sin(i * CWSt);

            d2phi_dt2 += -phi_c[i] * pow(i, 2) * cos(i * CWSt);
            d2theta_dt2 += -theta_c[i] * pow(i, 2) * cos(i * CWSt);
        }
        // Termos com Senos e as suas derivas
        for (int i = 1; i < order + 1; ++i)
        {
            dphi_by_dphicoeff[counter] = sin(i * CWSt);
            dtheta_by_dthetacoeff[counter] = sin(i * CWSt);

            d2theta_by_dthetacoeffdt[counter] = i * cos(i * CWSt);
            d2phi_by_dphicoeffdt[counter] = i * cos(i * CWSt);

            d3theta_by_dthetacoeffdt2[counter] = -pow(i, 2) * sin(i * CWSt);
            d3phi_by_dphicoeffdt2[counter] = -pow(i, 2) * sin(i * CWSt);

            counter++;

            phi += phi_s[i - 1] * sin(i * CWSt);
            theta += theta_s[i - 1] * sin(i * CWSt);

            dphi_dt += phi_s[i - 1] * i * cos(i * CWSt);
            dtheta_dt += theta_s[i - 1] * i * cos(i * CWSt);

            d2phi_dt2 += -phi_s[i - 1] * pow(i, 2) * sin(i * CWSt);
            d2theta_dt2 += -theta_s[i - 1] * pow(i, 2) * sin(i * CWSt);
        }

        phi += phi_l * CWSt;
        theta += theta_l * CWSt;
        dphi_dt += phi_l;
        dtheta_dt += theta_l;

// SURFACE
#pragma omp parallel for
        for (int i = 0; i < counter; ++i)
        {
            r = 0;
            dr_dt = 0;
            d2r_dt2 = 0;

            dr_dthetacoeff = 0;
            dr_dphicoeff = 0;

            d2r_dthetacoeffdt = 0;
            d2r_dphicoeffdt = 0;

            d3r_dthetacoeffdt2 = 0;
            d3r_dphicoeffdt2 = 0;

            d3z_dthetacoeffdt2 = 0;
            d3z_dphicoeffdt2 = 0;

            for (int m = 0; m <= mpol; ++m)
            {
                for (int j = 0; j < 2 * ntor + 1; ++j)
                {
                    int n = j - ntor;

                    r += rc(m, j) * cos(m * theta - nfp * n * phi);
                    dr_dt += -rc(m, j) * sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt);
                    d2r_dt2 += -rc(m, i) * cos(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi_dt), 2) - rc(m, i) * sin(m * theta - nfp * n * phi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2);

                    dr_dthetacoeff += -rc(m, j) * sin(m * theta - nfp * n * phi) * (m * dtheta_by_dthetacoeff[i]);
                    dr_dphicoeff += -rc(m, j) * sin(m * theta - nfp * n * phi) * (-nfp * n * dphi_by_dphicoeff[i]);

                    d2r_dthetacoeffdt += -rc(m, j) * (cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt) * (m * dtheta_by_dthetacoeff[i]) + sin(m * theta - nfp * n * phi) * (m * d2theta_by_dthetacoeffdt[i]));
                    d2r_dphicoeffdt += -rc(m, j) * (cos(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt) * (-nfp * n * dphi_by_dphicoeff[i]) + sin(m * theta - nfp * n * phi) * (-nfp * n * dphi_by_dphicoeff[i]));

                    d3r_dthetacoeffdt2 += -rc(m, j) * ((-sin(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi_dt), 2) + cos(m * theta - nfp * n * phi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2)) * (m * dtheta_by_dthetacoeff[i]) + cos(m * theta - nfp * n * phi) * 2 * (m * dtheta_dt - nfp * n * dphi_dt) * (m * d2theta_by_dthetacoeffdt[i]) + sin(m * theta - nfp * n * phi) * (m * d3theta_by_dthetacoeffdt2[i]));
                    d3r_dphicoeffdt2 += -rc(m, j) * ((-sin(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi_dt), 2) + cos(m * theta - nfp * n * phi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2)) * (-nfp * n * dphi_by_dphicoeff[i]) + cos(m * theta - nfp * n * phi) * 2 * (m * dtheta_dt - nfp * n * dphi_dt) * (-nfp * n * d2phi_by_dphicoeffdt[i]) + sin(m * theta - nfp * n * phi) * (-nfp * n * d3phi_by_dphicoeffdt2[i]));

                    d3z_dthetacoeffdt2 += zs(m, j) * ((-cos(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi_dt), 2) - sin(m * theta - nfp * n * phi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2) * (m * dtheta_by_dthetacoeff[i])) - 2 * sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt) * (m * d2theta_by_dthetacoeffdt[i]) + cos(m * theta - nfp * n * phi) * (m * d3theta_by_dthetacoeffdt2[i]));
                    d3z_dphicoeffdt2 += zs(m, j) * ((-cos(m * theta - nfp * n * phi) * pow((m * dtheta_dt - nfp * n * dphi_dt), 2) - sin(m * theta - nfp * n * phi) * (m * d2theta_dt2 - nfp * n * d2phi_dt2) * (-nfp * n * dphi_by_dphicoeff[i])) - 2 * sin(m * theta - nfp * n * phi) * (m * dtheta_dt - nfp * n * dphi_dt) * (-nfp * n * d2phi_by_dphicoeffdt[i]) + cos(m * theta - nfp * n * phi) * (-nfp * n * d3phi_by_dphicoeffdt2[i]));
                    // STELLSYM IS MISSING FOR NOW
                }
            }
            dr_dcoeff[i] = dr_dthetacoeff;
            dr_dcoeff[i + counter] = dr_dphicoeff;

            d2r_dcoeffdt[i] = d2r_dthetacoeffdt;
            d2r_dcoeffdt[i + counter] = d2r_dphicoeffdt;

            d3r_dcoeffdt2[i] = d3r_dthetacoeffdt2;
            d3r_dcoeffdt2[i + counter] = d3r_dphicoeffdt2;

            d3z_dcoeffdt2[i] = d3z_dthetacoeffdt2;
            d3z_dcoeffdt2[i + counter] = d3z_dphicoeffdt2;
        }

        for (int p = 0; p < counter; p++)
        {
            data(k, 0, p) = d3r_dcoeffdt2[p] * cos(phi) - 2 * (d2r_dcoeffdt[p] * sin(phi) * dphi_dt) - dr_dcoeff[p] * (cos(phi) * pow(dphi_dt, 2) + sin(phi) * d2phi_dt2);
            data(k, 1, p) = d3r_dcoeffdt2[p] * sin(phi) + 2 * (d2r_dcoeffdt[p] * cos(phi) * dphi_dt) - dr_dcoeff[p] * (sin(phi) * pow(dphi_dt, 2) - cos(phi) * d2phi_dt2);
            data(k, 2, p) = d3z_dcoeffdt2[p];

            data(k, 0, p + counter) = d3r_dcoeffdt2[p + counter] * cos(phi) - 2 * d2r_dcoeffdt[p + counter] * sin(phi) * dphi_dt - r * sin(phi) * d3phi_by_dphicoeffdt2[p + counter] + (-sin(phi) * d2phi_dt2 - cos(phi) * pow(dphi_dt, 2)) * dr_dcoeff[p + counter] + (-2 * dr_dt * sin(phi) - 2 * r * dphi_dt * cos(phi)) * d2phi_by_dphicoeffdt[p + counter] + (sin(phi) * (-d2r_dt2 + r * pow(dphi_dt, 2)) + cos(phi) * (-2 * dr_dt * dphi_dt - r * d2phi_dt2)) * dphi_by_dphicoeff[p + counter];
            data(k, 1, p + counter) = d3r_dcoeffdt2[p + counter] * sin(phi) + 2 * d2r_dcoeffdt[p + counter] * cos(phi) * dphi_dt + r * cos(phi) * d3phi_by_dphicoeffdt2[p + counter] + (cos(phi) * d2phi_dt2 - sin(phi) * pow(dphi_dt, 2)) * dr_dcoeff[p + counter] + (2 * dr_dt * cos(phi) - 2 * r * dphi_dt * sin(phi)) * d2phi_by_dphicoeffdt[p + counter] + (cos(phi) * (d2r_dt2 - r * pow(dphi_dt, 2)) + sin(phi) * (-2 * dr_dt * dphi_dt - r * d2phi_dt2)) * dphi_by_dphicoeff[p + counter];
            data(k, 2, p + counter) = d3z_dcoeffdt2[p + counter];
        }
    }
    data *= 2 * M_PI * 2 * M_PI;
};
#include "xtensor-python/pyarray.hpp" // Numpy bindings
typedef xt::pyarray<double> Array;
template class CurveCWSFourier<Array>;