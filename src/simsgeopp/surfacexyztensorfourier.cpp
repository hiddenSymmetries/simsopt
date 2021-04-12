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
        std::vector<bool> clamped_dims;

        SurfaceXYZTensorFourier(int _mpol, int _ntor, int _nfp, bool _stellsym, std::vector<bool> _clamped_dims, vector<double> _quadpoints_phi, vector<double> _quadpoints_theta)
            : Surface<Array>(_quadpoints_phi, _quadpoints_theta), mpol(_mpol), ntor(_ntor), nfp(_nfp), stellsym(_stellsym), clamped_dims(_clamped_dims) {
                x = xt::zeros<double>({2*mpol+1, 2*ntor+1});
                y = xt::zeros<double>({2*mpol+1, 2*ntor+1});
                z = xt::zeros<double>({2*mpol+1, 2*ntor+1});
            }

        SurfaceXYZTensorFourier(int _mpol, int _ntor, int _nfp, bool _stellsym, std::vector<bool> _clamped_dims, int _numquadpoints_phi, int _numquadpoints_theta)
            : Surface<Array>(_numquadpoints_phi, _numquadpoints_theta), mpol(_mpol), ntor(_ntor), nfp(_nfp), stellsym(_stellsym), clamped_dims(_clamped_dims) {
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

        inline double bc_enforcer_fun(int dim, int n, double phi, int m, double theta){
            if(n<=ntor && m<=mpol)
                return clamped_dims[dim] ? pow(sin(nfp*phi/2), 2) + pow(sin(theta/2), 2) : 1.;
            else
                return 1;
        }

        inline double bc_enforcer_dphi_fun(int dim, int n, double phi, int m, double theta){
            if(n<=ntor && m<=mpol)
                return clamped_dims[dim] ? nfp*cos(nfp*phi/2)*sin(nfp*phi/2) : 0.;
            else
                return 0;
        }

        inline double bc_enforcer_dtheta_fun(int dim, int n, double phi, int m, double theta){
            if(n<=ntor && m<=mpol)
                return clamped_dims[dim] ? cos(theta/2)*sin(theta/2) : 0.;
            else
                return 0;
        }

        inline double basis_fun(int dim, int n, double phi, int m, double theta){
            double bc_enforcer = bc_enforcer_fun(dim, n, phi, m, theta);
            //double bc_enforcer =  (dim > 1) ? (phi*(2*M_PI-phi) + theta*(2*M_PI-theta)) : 1.;
            return basis_fun_phi(n, phi) * basis_fun_theta(m, theta) * bc_enforcer;
        }

        inline double basis_fun_dphi(int dim, int n, double phi, int m, double theta){
            double bc_enforcer =  bc_enforcer_fun(dim, n, phi, m, theta);
            double bc_enforcer_dphi = bc_enforcer_dphi_fun(dim, n, phi, m, theta);
            //double bc_enforcer =  (dim > 1) ? (phi*(2*M_PI-phi) + theta*(2*M_PI-theta)) : 1.;
            //double bc_enforcer_dphi =  (dim > 1) ? (2*M_PI-2*phi) : 0.;
            return basis_fun_phi_dash(n, phi) * basis_fun_theta(m, theta) * bc_enforcer + basis_fun_phi(n, phi) * basis_fun_theta(m, theta) * bc_enforcer_dphi;
        }

        inline double basis_fun_dtheta(int dim, int n, double phi, int m, double theta){
            double bc_enforcer =  bc_enforcer_fun(dim, n, phi, m, theta);
            double bc_enforcer_dtheta = bc_enforcer_dtheta_fun(dim, n, phi, m, theta);
            //double bc_enforcer =  (dim > 1) ? (phi*(2*M_PI-phi) + theta*(2*M_PI-theta)) : 1.;
            //double bc_enforcer_dtheta =  (dim > 1) ? (2*M_PI-2*theta) : 0.;
            return basis_fun_phi(n, phi) * basis_fun_theta_dash(m, theta) * bc_enforcer + basis_fun_phi(n, phi) * basis_fun_theta(m, theta) * bc_enforcer_dtheta;
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

        void gamma_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override {
            int numquadpoints_phi = quadpoints_phi.size();
            int numquadpoints_theta = quadpoints_theta.size();
            data *= 0.;
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    for (int m = 0; m <= 2*mpol; ++m) {
                        for (int n = 0; n <= 2*ntor; ++n) {
                            double xhat = get_coeff(0, m, n) * basis_fun(0, n, phi, m, theta);
                            double yhat = get_coeff(1, m, n) * basis_fun(1, n, phi, m, theta);
                            double x = xhat * cos(phi) - yhat * sin(phi);
                            double y = xhat * sin(phi) + yhat * cos(phi);
                            //double x = xhat;
                            //double y = yhat;
                            double z = get_coeff(2, m, n) * basis_fun(2, n, phi, m, theta);
                            data(k1, k2, 0) += x;
                            data(k1, k2, 1) += y;
                            data(k1, k2, 2) += z;
                        }
                    }
                }
            }
        }
        void gamma_lin(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override {
            int numquadpoints_phi = quadpoints_phi.size();
            data *= 0.;
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                double theta  = 2*M_PI*quadpoints_theta[k1];
                for (int m = 0; m <= 2*mpol; ++m) {
                    for (int n = 0; n <= 2*ntor; ++n) {
                        double xhat = get_coeff(0, m, n) * basis_fun(0, n, phi, m, theta);
                        double yhat = get_coeff(1, m, n) * basis_fun(1, n, phi, m, theta);
                        double x = xhat * cos(phi) - yhat * sin(phi);
                        double y = xhat * sin(phi) + yhat * cos(phi);
                        //double x = xhat;
                        //double y = yhat;
                        double z = get_coeff(2, m, n) * basis_fun(2, n, phi, m, theta);
                        data(k1, 0) += x;
                        data(k1, 1) += y;
                        data(k1, 2) += z;
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
                            double xhat = get_coeff(0, m, n) * basis_fun(0, n, phi, m, theta);
                            double yhat = get_coeff(1, m, n) * basis_fun(1, n, phi, m, theta);
                            double xhatdash = get_coeff(0, m, n) * basis_fun_dphi(0, n, phi, m, theta);
                            double yhatdash = get_coeff(1, m, n) * basis_fun_dphi(1, n, phi, m, theta);
                            double xdash = xhatdash * cos(phi) - yhatdash * sin(phi) - xhat * sin(phi) - yhat * cos(phi);
                            double ydash = xhatdash * sin(phi) + yhatdash * cos(phi) + xhat * cos(phi) - yhat * sin(phi);
                            //double xdash = xhatdash;
                            //double ydash = yhatdash;
                            double zdash = get_coeff(2, m, n) * basis_fun_dphi(2, n, phi, m, theta);
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
                            double xhatdash = get_coeff(0, m, n) * basis_fun_dtheta(0, n, phi, m, theta);
                            double yhatdash = get_coeff(1, m, n) * basis_fun_dtheta(1, n, phi, m, theta);
                            double xdash = xhatdash * cos(phi) - yhatdash * sin(phi);
                            double ydash = xhatdash * sin(phi) + yhatdash * cos(phi);
                            //double xdash = xhatdash;
                            //double ydash = yhatdash;
                            double zdash = get_coeff(2, m, n) * basis_fun_dtheta(2, n, phi, m, theta);;
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
                                double wivj = basis_fun(d, n, phi, m, theta);
                                if(d==0) {
                                    double dxhat = wivj;
                                    double dyhat = 0;
                                    double dx = dxhat * cos(phi) - dyhat * sin(phi);
                                    double dy = dxhat * sin(phi) + dyhat * cos(phi);
                                    //double dx = dxhat;
                                    //double dy = dyhat;
                                    data(k1, k2, 0, counter) = dx;
                                    data(k1, k2, 1, counter) = dy;
                                }else if(d==1) {
                                    double dxhat = 0;
                                    double dyhat = wivj;
                                    double dx = dxhat * cos(phi) - dyhat * sin(phi);
                                    double dy = dxhat * sin(phi) + dyhat * cos(phi);
                                    //double dx = dxhat;
                                    //double dy = dyhat;
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
                                double wivj = basis_fun(d, n, phi, m, theta);
                                double wivjdash = basis_fun_dphi(d, n, phi, m, theta);
                                if(d==0) {
                                    double dxhat = wivj;
                                    double dyhat = 0.;
                                    double dxhatdash = wivjdash;
                                    double dyhatdash = 0.;
                                    double dxdash = dxhatdash * cos(phi) - dyhatdash * sin(phi) - dxhat * sin(phi) - dyhat * cos(phi);
                                    double dydash = dxhatdash * sin(phi) + dyhatdash * cos(phi) + dxhat * cos(phi) - dyhat * sin(phi);
                                    //double dxdash = dxhatdash;
                                    //double dydash = dyhatdash;
                                    data(k1, k2, 0, counter) = 2*M_PI*dxdash;
                                    data(k1, k2, 1, counter) = 2*M_PI*dydash;
                                }else if(d==1) {
                                    double dxhat = 0.;
                                    double dyhat = wivj;
                                    double dxhatdash = 0.;
                                    double dyhatdash = wivjdash;
                                    double dxdash = dxhatdash * cos(phi) - dyhatdash * sin(phi) - dxhat * sin(phi) - dyhat * cos(phi);
                                    double dydash = dxhatdash * sin(phi) + dyhatdash * cos(phi) + dxhat * cos(phi) - dyhat * sin(phi);
                                    //double dxdash = dxhatdash;
                                    //double dydash = dyhatdash;
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
                                double wivj = basis_fun(d, n, phi, m, theta);
                                double wivjdash = basis_fun_dtheta(d, n, phi, m, theta);
                                if(d==0) {
                                    double dxhat = wivj;
                                    double dyhat = 0.;
                                    double dxhatdash = wivjdash;
                                    double dyhatdash = 0.;
                                    double dxdash = dxhatdash * cos(phi) - dyhatdash * sin(phi);
                                    double dydash = dxhatdash * sin(phi) + dyhatdash * cos(phi);
                                    //double dxdash = dxhatdash;
                                    //double dydash = dyhatdash;
                                    data(k1, k2, 0, counter) = 2*M_PI*dxdash;
                                    data(k1, k2, 1, counter) = 2*M_PI*dydash;
                                }else if(d==1) {
                                    double dxhat = 0.;
                                    double dyhat = wivj;
                                    double dxhatdash = 0.;
                                    double dyhatdash = wivjdash;
                                    double dxdash = dxhatdash * cos(phi) - dyhatdash * sin(phi);
                                    double dydash = dxhatdash * sin(phi) + dyhatdash * cos(phi);
                                    //double dxdash = dxhatdash;
                                    //double dydash = dyhatdash;
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
