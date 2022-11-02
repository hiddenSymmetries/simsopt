#pragma once

#include "surface.h"

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

        SurfaceXYZTensorFourier(int mpol, int ntor, int nfp, bool stellsym, std::vector<bool> clamped_dims, vector<double> quadpoints_phi, vector<double> quadpoints_theta);

        int num_dofs() override ;

        void set_dofs_impl(const vector<double>& dofs) override ;

        vector<double> get_dofs() override ;

        void gamma_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override ;

        void gamma_lin(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override ;


        void gammadash1_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override ;

        void gammadash2_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override ;

        void gammadash2dash2_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override ;

        void gammadash1dash2_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override ;

        void gammadash1dash1_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override ;

        void dgammadash1dash1_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override ;

        void dgammadash1dash2_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override ;

        void dgammadash2dash2_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override ;

        void dgamma_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override ;

        void dgammadash1_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override ;

        void dgammadash2_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override ;

    private:

        Array cache_quadpoints_phi;
        Array cache_quadpoints_theta;
        Array cache_basis_fun_phi;
        Array cache_basis_fun_phi_dash;
        Array cache_basis_fun_phi_dashdash;
        Array cache_basis_fun_theta;
        Array cache_basis_fun_theta_dash;
        Array cache_basis_fun_theta_dashdash;
        Array cache_enforcer;
        Array cache_enforcer_dphi;
        Array cache_enforcer_dtheta;
        Array cache_enforcer_dphidphi;
        Array cache_enforcer_dthetadtheta;

        void build_cache(Array& quadpoints_phi, Array& quadpoints_theta);
        void rebuild_cache(Array& quadpoints_phi, Array& quadpoints_theta);

        inline bool apply_bc_enforcer(int dim, int n, int m) {
            return (clamped_dims[dim] && n<=ntor && m<=mpol);
        }

        inline double bc_enforcer_fun(int dim, int n, double phi, int m, double theta){
            if(apply_bc_enforcer(dim, n, m))
                return pow(sin(nfp*phi/2), 2) + pow(sin(theta/2), 2);
            else
                return 1;
        }

        inline double bc_enforcer_dphi_fun(int dim, int n, double phi, int m, double theta){
            if(apply_bc_enforcer(dim, n, m))
                return nfp*cos(nfp*phi/2)*sin(nfp*phi/2);
            else
                return 0;
        }

        inline double bc_enforcer_dphidphi_fun(int dim, int n, double phi, int m, double theta){
            if(apply_bc_enforcer(dim, n, m))
                return (nfp*nfp/2)*(pow(cos(nfp*phi/2),2) - pow(sin(nfp*phi/2),2));
            else
                return 0;
        }

        inline double bc_enforcer_dtheta_fun(int dim, int n, double phi, int m, double theta){
            if(apply_bc_enforcer(dim, n, m))
                return cos(theta/2)*sin(theta/2);
            else
                return 0;
        }

        inline double bc_enforcer_dthetadtheta_fun(int dim, int n, double phi, int m, double theta){
            if(apply_bc_enforcer(dim, n, m))
                return (1/2)*(pow(cos(theta/2),2) - pow(sin(theta/2),2));
            else
                return 0;
        }

        inline double basis_fun(int dim, int n, int phiidx, int m, int thetaidx){
            double fun = cache_basis_fun_phi(phiidx, n)*cache_basis_fun_theta(thetaidx, m);
            if(apply_bc_enforcer(dim, n, m))
                fun *= cache_enforcer(phiidx, thetaidx);
            return fun;
        }

        inline double basis_fun_dphi(int dim, int n, int phiidx, int m, int thetaidx){
            double fun_dphi = cache_basis_fun_phi_dash(phiidx, n)*cache_basis_fun_theta(thetaidx, m);
            if(apply_bc_enforcer(dim, n, m)){
                double fun = cache_basis_fun_phi(phiidx, n)*cache_basis_fun_theta(thetaidx, m);
                fun_dphi = fun_dphi*cache_enforcer(phiidx, thetaidx) + fun*cache_enforcer_dphi(phiidx, thetaidx);
            }
            return fun_dphi;
        }

        inline double basis_fun_dphidphi(int dim, int n, int phiidx, int m, int thetaidx){
            double fun_dphidphi = cache_basis_fun_phi_dashdash(phiidx, n)*cache_basis_fun_theta(thetaidx, m);
            if(apply_bc_enforcer(dim, n, m)){
                double fun_dphi = cache_basis_fun_phi_dash(phiidx, n)*cache_basis_fun_theta(thetaidx, m);
                double fun = cache_basis_fun_phi(phiidx, n)*cache_basis_fun_theta(thetaidx, m);
                fun_dphidphi = fun_dphidphi*cache_enforcer(phiidx, thetaidx) + 2*fun_dphi*cache_enforcer_dphi(phiidx, thetaidx) \
                            +  fun*cache_enforcer_dphidphi(phiidx, thetaidx);
            }
            return fun_dphidphi;
        }

        inline double basis_fun_dtheta(int dim, int n, int phiidx, int m, int thetaidx){
            double fun_dtheta = cache_basis_fun_phi(phiidx, n)*cache_basis_fun_theta_dash(thetaidx, m);
            if(apply_bc_enforcer(dim, n, m)){
                double fun = cache_basis_fun_phi(phiidx, n)*cache_basis_fun_theta(thetaidx, m);
                fun_dtheta = fun_dtheta*cache_enforcer(phiidx, thetaidx) + fun*cache_enforcer_dtheta(phiidx, thetaidx);
            }
            return fun_dtheta;
        }

        inline double basis_fun_dthetadphi(int dim, int n, int phiidx, int m, int thetaidx){
            double fun_dthetadphi = cache_basis_fun_phi_dash(phiidx, n)*cache_basis_fun_theta_dash(thetaidx, m);
            if(apply_bc_enforcer(dim, n, m)){
                double fun_dtheta = cache_basis_fun_phi(phiidx, n)*cache_basis_fun_theta_dash(thetaidx, m);
                double fun_dphi = cache_basis_fun_phi_dash(phiidx, n)*cache_basis_fun_theta(thetaidx, m);
                fun_dthetadphi = fun_dthetadphi*cache_enforcer(phiidx, thetaidx) \
                                + fun_dtheta*cache_enforcer_dphi(phiidx, thetaidx) \
                                + fun_dphi*cache_enforcer_dtheta(phiidx, thetaidx);
            }
            return fun_dthetadphi;
        }

        inline double basis_fun_dthetadtheta(int dim, int n, int phiidx, int m, int thetaidx){
          double fun_dthetadtheta = cache_basis_fun_phi(phiidx, n)*cache_basis_fun_theta_dashdash(thetaidx, m);
          if(apply_bc_enforcer(dim, n, m)){
              double fun = cache_basis_fun_phi(phiidx, n)*cache_basis_fun_theta(thetaidx, m);
              double fun_dtheta = cache_basis_fun_phi(phiidx, n)*cache_basis_fun_theta_dash(thetaidx, m);
              fun_dthetadtheta = fun_dthetadtheta*cache_enforcer(phiidx, thetaidx) \
                              + 2*fun_dtheta*cache_enforcer_dtheta(phiidx, thetaidx) \
                              + fun*cache_enforcer_dthetadtheta(phiidx, thetaidx);
          }
          return fun_dthetadtheta;
        }

        inline double basis_fun(int dim, int n, double phi, int m, double theta){
            double bc_enforcer = bc_enforcer_fun(dim, n, phi, m, theta);
            return basis_fun_phi(n, phi) * basis_fun_theta(m, theta) * bc_enforcer;
        }

        inline double basis_fun_dphi(int dim, int n, double phi, int m, double theta){
            double bc_enforcer =  bc_enforcer_fun(dim, n, phi, m, theta);
            double bc_enforcer_dphi = bc_enforcer_dphi_fun(dim, n, phi, m, theta);
            return basis_fun_phi_dash(n, phi) * basis_fun_theta(m, theta) * bc_enforcer + basis_fun_phi(n, phi) * basis_fun_theta(m, theta) * bc_enforcer_dphi;
        }

        inline double basis_fun_dtheta(int dim, int n, double phi, int m, double theta){
            double bc_enforcer =  bc_enforcer_fun(dim, n, phi, m, theta);
            double bc_enforcer_dtheta = bc_enforcer_dtheta_fun(dim, n, phi, m, theta);
            return basis_fun_phi(n, phi) * basis_fun_theta_dash(m, theta) * bc_enforcer + basis_fun_phi(n, phi) * basis_fun_theta(m, theta) * bc_enforcer_dtheta;
        }

        inline double basis_fun_dthetadtheta(int dim, int n, double phi, int m, double theta){
            double bc_enforcer =  bc_enforcer_fun(dim, n, phi, m, theta);
            double bc_enforcer_dtheta = bc_enforcer_dtheta_fun(dim, n, phi, m, theta);
            double bc_enforcer_dthetadtheta = bc_enforcer_dthetadtheta_fun(dim, n, phi, m, theta);
            return basis_fun_phi(n, phi) * (basis_fun_theta_dashdash(m, theta) * bc_enforcer \
                 + 2* basis_fun_theta_dash(m, theta) * bc_enforcer_dtheta \
                 +    basis_fun_theta(m, theta) * bc_enforcer_dthetadtheta);
        }

        inline double basis_fun_dphidphi(int dim, int n, double phi, int m, double theta){
            double bc_enforcer =  bc_enforcer_fun(dim, n, phi, m, theta);
            double bc_enforcer_dphi = bc_enforcer_dphi_fun(dim, n, phi, m, theta);
            double bc_enforcer_dphidphi = bc_enforcer_dphidphi_fun(dim, n, phi, m, theta);
            return basis_fun_theta(m, theta) * (basis_fun_phi_dashdash(n, phi) * bc_enforcer \
                  + 2*basis_fun_phi_dash(n, phi)*bc_enforcer_dphi \
                  + basis_fun_phi(n,phi) * bc_enforcer_dphidphi);
        }

        inline double basis_fun_dthetadphi(int dim, int n, double phi, int m, double theta){
            double bc_enforcer =  bc_enforcer_fun(dim, n, phi, m, theta);
            double bc_enforcer_dtheta = bc_enforcer_dtheta_fun(dim, n, phi, m, theta);
            double bc_enforcer_dphi = bc_enforcer_dphi_fun(dim, n, phi, m, theta);
            return basis_fun_phi_dash(n, phi) * basis_fun_theta_dash(m, theta) * bc_enforcer \
                 +  basis_fun_phi(n, phi) * basis_fun_theta_dash(m, theta) * bc_enforcer_dphi \
                 +  basis_fun_phi_dash(n, phi) * basis_fun_theta(m, theta) * bc_enforcer_dtheta;
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

        inline double basis_fun_phi_dashdash(int n, double phi){
            if(n <= ntor)
                return -nfp*n*nfp*n*cos(nfp*n*phi);
            else
                return -nfp*(n-ntor)*nfp*(n-ntor)*sin(nfp*(n-ntor)*phi);
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

        inline double basis_fun_theta_dashdash(int m, double theta){
            if(m <= mpol)
                return -m*m*cos(m*theta);
            else
                return -(m-mpol)*(m-mpol)*sin((m-mpol)*theta);
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
};
