#pragma once

#include "surface.cpp"

template<class Array>
class SurfaceXYZTensorFourier : public Surface<Array> {
    /*
       SurfaceXYZTensorFourier is a surface that is represented in cartesian
       coordinates using the following Fourier series: 

       \hat x(theta, phi) = 
           \sum_{i=0}^{2*mpol} \sum_{j=0}^{2*ntor} x_{ij} w_i(\theta)*v_j(\phi)
       \hat y(theta, phi) = 
           \sum_{i=0}^{2*mpol} \sum_{j=0}^{2*ntor} y_{ij} w_i(\theta)*v_j(\phi)

       x = \hat x * \cos(\phi) - \hat y * \sin(\phi)
       y = \hat x * \sin(\phi) + \hat y * \cos(\phi)

       z(theta, phi) = 
           \sum_{i=0}^{2*mpol} \sum_{j=0}^{2*ntor} z_{ij} w_i(\theta)*v_j(\phi)

       where the basis functions {v_j} are given by

        {1, cos(1*nfp*\phi), ..., cos(ntor*nfp*phi), sin(1*nfp*phi), ..., sin(ntor*nfp*phi)}
       
       and {w_i} are given by

        {1, cos(1*\theta), ..., cos(ntor*theta), sin(1*theta), ..., sin(ntor*theta)}

       When enforcing stellarator symmetry, ...
       */

    public:
        using Surface<Array>::quadpoints_phi;
        using Surface<Array>::quadpoints_theta;
        using Surface<Array>::numquadpoints_phi;
        using Surface<Array>::numquadpoints_theta;
        Array x;
        Array y;
        Array z;
        int nfp;
        int mpol;
        int ntor;
        bool stellsym;

        SurfaceXYZTensorFourier(int _mpol, int _ntor, int _nfp, bool _stellsym, vector<double> _quadpoints_phi, vector<double> _quadpoints_theta)
            : Surface<Array>(_quadpoints_phi, _quadpoints_theta), mpol(_mpol), ntor(_ntor), nfp(_nfp), stellsym(_stellsym) {
                x = xt::zeros<double>({2*mpol+1, 2*ntor+1});
                y = xt::zeros<double>({2*mpol+1, 2*ntor+1});
                z = xt::zeros<double>({2*mpol+1, 2*ntor+1});
            }

        SurfaceXYZTensorFourier(int _mpol, int _ntor, int _nfp, bool _stellsym, int _numquadpoints_phi, int _numquadpoints_theta)
            : Surface<Array>(_numquadpoints_phi, _numquadpoints_theta), mpol(_mpol), ntor(_ntor), nfp(_nfp), stellsym(_stellsym) {
                x = xt::zeros<double>({2*mpol+1, 2*ntor+1});
                y = xt::zeros<double>({2*mpol+1, 2*ntor+1});
                z = xt::zeros<double>({2*mpol+1, 2*ntor+1});
            }



        int num_dofs() override {
            if(stellsym)
                return (ntor+1)*(mpol+1)+ ntor*mpol + 2*(ntor+1)*mpol + 2*ntor*(mpol+1);
            else
                return 3 * (2*mpol+1) * (2*ntor+1);
        }

        void set_dofs_impl(const vector<double>& dofs) override {
            int counter = 0;
            for (int m = 0; m <= 2*mpol; ++m) {
                for (int n = 0; n <= 2*ntor; ++n) {
                    if(skip(0, m, n)) continue;
                    x(m, n) = dofs[counter++];
                }
            }
            for (int m = 0; m <= 2*mpol; ++m) {
                for (int n = 0; n <= 2*ntor; ++n) {
                    if(skip(1, m, n)) continue;
                    y(m, n) = dofs[counter++];
                }
            }
            for (int m = 0; m <= 2*mpol; ++m) {
                for (int n = 0; n <= 2*ntor; ++n) {
                    if(skip(2, m, n)) continue;
                    z(m, n) = dofs[counter++];
                }
            }
        }

        vector<double> get_dofs() override {
            auto res = vector<double>(num_dofs(), 0.);
            int counter = 0;
            for (int m = 0; m <= 2*mpol; ++m) {
                for (int n = 0; n <= 2*ntor; ++n) {
                    if(skip(0, m, n)) continue;
                    res[counter++] = x(m, n);
                }
            }
            for (int m = 0; m <= 2*mpol; ++m) {
                for (int n = 0; n <= 2*ntor; ++n) {
                    if(skip(1, m, n)) continue;
                    res[counter++] = y(m, n);
                }
            }
            for (int m = 0; m <= 2*mpol; ++m) {
                for (int n = 0; n <= 2*ntor; ++n) {
                    if(skip(2, m, n)) continue;
                    res[counter++] = z(m, n);
                }
            }
            return res;
        }

        inline double basis_fun_phi(int n, double phi){
            if(n <= ntor)
                return cos(nfp*n*phi);
            else
                return sin(nfp*(n-ntor)*phi);
        }

        inline double basis_fun_phi_dash(int n, double phi){
            if(n <= ntor)
                return -nfp*n*sin(nfp*n*phi);
            else
                return nfp*(n-ntor)*cos(nfp*(n-ntor)*phi);
        }

        inline double basis_fun_theta(int m, double theta){
            if(m <= mpol)
                return cos(m*theta);
            else
                return sin((m-mpol)*theta);
        }

        inline double basis_fun_theta_dash(int m, double theta){
            if(m <= mpol)
                return -m*sin(m*theta);
            else
                return (m-mpol)*cos((m-mpol)*theta);
        }

        inline bool skip(int dim, int m, int n){
            if (!stellsym)
                return false;
            if (dim == 0)
                return (n <= ntor && m >  mpol) || (n >  ntor && m <= mpol);
            else if(dim == 1)
                return (n <= ntor && m <= mpol) || (n >  ntor && m >  mpol);
            else
                return (n <= ntor && m <= mpol) || (n >  ntor && m >  mpol);
        }

        inline double get_coeff(int dim, int m, int n) {
            if(skip(dim, m, n))
                return 0.;
            if (dim == 0){
                return this->x(m, n);
            }
            else if (dim == 1) {
                return this->y(m, n);
            }
            else {
                return this->z(m, n);
            }
        }

        void gamma_impl(Array& data) override {
            data *= 0.;
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    for (int m = 0; m <= 2*mpol; ++m) {
                        for (int n = 0; n <= 2*ntor; ++n) {
                            double wivj = basis_fun_theta(m, theta) * basis_fun_phi(n, phi);
                            double xhat = get_coeff(0, m, n) * wivj;
                            double yhat = get_coeff(1, m, n) * wivj;
                            double x = xhat * cos(phi) - yhat * sin(phi);
                            double y = xhat * sin(phi) + yhat * cos(phi);
                            double z = get_coeff(2, m, n) * wivj;
                            data(k1, k2, 0) += x;
                            data(k1, k2, 1) += y;
                            data(k1, k2, 2) += z;
                        }
                    }
                }
            }
        }

        void gammadash1_impl(Array& data) override {
            data *= 0.;
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    for (int m = 0; m <= 2*mpol; ++m) {
                        for (int n = 0; n <= 2*ntor; ++n) {
                            double wivj = basis_fun_theta(m, theta) * basis_fun_phi(n, phi);
                            double wivjdash = basis_fun_theta(m, theta) * basis_fun_phi_dash(n, phi);
                            double xhat = get_coeff(0, m, n) * wivj;
                            double yhat = get_coeff(1, m, n) * wivj;
                            double xhatdash = get_coeff(0, m, n) * (wivjdash);
                            double yhatdash = get_coeff(1, m, n) * (wivjdash);
                            double xdash = xhatdash * cos(phi) - yhatdash * sin(phi) - xhat * sin(phi) - yhat * cos(phi);
                            double ydash = xhatdash * sin(phi) + yhatdash * cos(phi) + xhat * cos(phi) - yhat * sin(phi);
                            double zdash = get_coeff(2, m, n) * wivjdash;
                            data(k1, k2, 0) += 2*M_PI*xdash;
                            data(k1, k2, 1) += 2*M_PI*ydash;
                            data(k1, k2, 2) += 2*M_PI*zdash;
                        }
                    }
                }
            }
        }

        void gammadash2_impl(Array& data) override {
            data *= 0.;
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    for (int m = 0; m <= 2*mpol; ++m) {
                        for (int n = 0; n <= 2*ntor; ++n) {
                            double wivj = basis_fun_theta(m, theta) * basis_fun_phi(n, phi);
                            double wivjdash = basis_fun_theta_dash(m, theta) * basis_fun_phi(n, phi);
                            double xhatdash = get_coeff(0, m, n) * (wivjdash);
                            double yhatdash = get_coeff(1, m, n) * (wivjdash);
                            double xdash = xhatdash * cos(phi) - yhatdash * sin(phi);
                            double ydash = xhatdash * sin(phi) + yhatdash * cos(phi);
                            double zdash = get_coeff(2, m, n) * wivjdash;
                            data(k1, k2, 0) += 2*M_PI*xdash;
                            data(k1, k2, 1) += 2*M_PI*ydash;
                            data(k1, k2, 2) += 2*M_PI*zdash;
                        }
                    }
                }
            }
        }

        void dgamma_by_dcoeff_impl(Array& data) override {
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    int counter = 0;
                    for (int d = 0; d < 3; ++d) {
                        for (int m = 0; m <= 2*mpol; ++m) {
                            for (int n = 0; n <= 2*ntor; ++n) {
                                if(skip(d, m, n)) continue;
                                double wivj = basis_fun_theta(m, theta) * basis_fun_phi(n, phi);
                                if(d==0) {
                                    double dxhat = wivj;
                                    double dyhat = 0;
                                    double dx = dxhat * cos(phi) - dyhat * sin(phi);
                                    double dy = dxhat * sin(phi) + dyhat * cos(phi);
                                    data(k1, k2, 0, counter) = dx;
                                    data(k1, k2, 1, counter) = dy;
                                }else if(d==1) {
                                    double dxhat = 0;
                                    double dyhat = wivj;
                                    double dx = dxhat * cos(phi) - dyhat * sin(phi);
                                    double dy = dxhat * sin(phi) + dyhat * cos(phi);
                                    data(k1, k2, 0, counter) = dx;
                                    data(k1, k2, 1, counter) = dy;
                                }else {
                                    double dz = wivj;
                                    data(k1, k2, 2, counter) = dz;
                                }
                                counter++;
                            }
                        }
                    }
                }
            }
        }

        void dgammadash1_by_dcoeff_impl(Array& data) override {
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    int counter = 0;
                    for (int d = 0; d < 3; ++d) {
                        for (int m = 0; m <= 2*mpol; ++m) {
                            for (int n = 0; n <= 2*ntor; ++n) {
                                if(skip(d, m, n)) continue;
                                double wivj = basis_fun_theta(m, theta) * basis_fun_phi(n, phi);
                                double wivjdash = basis_fun_theta(m, theta) * basis_fun_phi_dash(n, phi);
                                if(d==0) {
                                    double dxhat = wivj;
                                    double dyhat = 0.;
                                    double dxhatdash = wivjdash;
                                    double dyhatdash = 0.;
                                    double dxdash = dxhatdash * cos(phi) - dyhatdash * sin(phi) - dxhat * sin(phi) - dyhat * cos(phi);
                                    double dydash = dxhatdash * sin(phi) + dyhatdash * cos(phi) + dxhat * cos(phi) - dyhat * sin(phi);
                                    data(k1, k2, 0, counter) = 2*M_PI*dxdash;
                                    data(k1, k2, 1, counter) = 2*M_PI*dydash;
                                }else if(d==1) {
                                    double dxhat = 0.;
                                    double dyhat = wivj;
                                    double dxhatdash = 0.;
                                    double dyhatdash = wivjdash;
                                    double dxdash = dxhatdash * cos(phi) - dyhatdash * sin(phi) - dxhat * sin(phi) - dyhat * cos(phi);
                                    double dydash = dxhatdash * sin(phi) + dyhatdash * cos(phi) + dxhat * cos(phi) - dyhat * sin(phi);
                                    data(k1, k2, 0, counter) = 2*M_PI*dxdash;
                                    data(k1, k2, 1, counter) = 2*M_PI*dydash;
                                }else {
                                    double dzdash = wivjdash;
                                    data(k1, k2, 2, counter) = 2*M_PI*dzdash;;
                                }
                                counter++;
                            }
                        }
                    }
                }
            }
        }

        void dgammadash2_by_dcoeff_impl(Array& data) override {
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    int counter = 0;
                    for (int d = 0; d < 3; ++d) {
                        for (int m = 0; m <= 2*mpol; ++m) {
                            for (int n = 0; n <= 2*ntor; ++n) {
                                if(skip(d, m, n)) continue;
                                double wivj = basis_fun_theta(m, theta) * basis_fun_phi(n, phi);
                                double wivjdash = basis_fun_theta_dash(m, theta) * basis_fun_phi(n, phi);
                                if(d==0) {
                                    double dxhat = wivj;
                                    double dyhat = 0.;
                                    double dxhatdash = wivjdash;
                                    double dyhatdash = 0.;
                                    double dxdash = dxhatdash * cos(phi) - dyhatdash * sin(phi);
                                    double dydash = dxhatdash * sin(phi) + dyhatdash * cos(phi);
                                    data(k1, k2, 0, counter) = 2*M_PI*dxdash;
                                    data(k1, k2, 1, counter) = 2*M_PI*dydash;
                                }else if(d==1) {
                                    double dxhat = 0.;
                                    double dyhat = wivj;
                                    double dxhatdash = 0.;
                                    double dyhatdash = wivjdash;
                                    double dxdash = dxhatdash * cos(phi) - dyhatdash * sin(phi);
                                    double dydash = dxhatdash * sin(phi) + dyhatdash * cos(phi);
                                    data(k1, k2, 0, counter) = 2*M_PI*dxdash;
                                    data(k1, k2, 1, counter) = 2*M_PI*dydash;
                                }else {
                                    double dzdash = wivjdash;
                                    data(k1, k2, 2, counter) = 2*M_PI*dzdash;;
                                }
                                counter++;
                            }
                        }
                    }
                }
            }
        }

};
