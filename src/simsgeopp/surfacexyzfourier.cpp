#pragma once

#include "surface.cpp"

template<class Array>
class SurfaceXYZFourier : public Surface<Array> {
    /*
       SurfaceXYZFourier is a surface that is represented in cartesian
       coordinates using the following Fourier series:

           \hat x(theta, phi) = \sum_{m=0}^{mpol} \sum_{n=-ntor}^{ntor} [
                 x_{c,m,n} \cos(m \theta - n nfp \phi)
               + x_{s,m,n} \sin(m \theta - n nfp \phi)
           ]

           \hat y(theta, phi) = \sum_{m=0}^{mpol} \sum_{n=-ntor}^{ntor} [
                 y_{c,m,n} \cos(m \theta - n nfp \phi)
               + y_{s,m,n} \sin(m \theta - n nfp \phi)
           ]

           x = \hat x * \cos(\phi) - \hat y * \sin(\phi)
           y = \hat x * \sin(\phi) + \hat y * \cos(\phi)

           z(theta, phi) = \sum_{m=0}^{mpol} \sum_{n=-ntor}^{ntor} [
               z_{c,m,n} \cos(m \theta - n nfp \phi)
               + z_{s,m,n} \sin(m \theta - n nfp \phi)
           ]

       Note that for m=0 we skip the n<0 term for the cos terms, and the n<=0
       for the sin terms.

       When enforcing stellarator symmetry, we set the

           x_{s,*,*}, y_{c,*,*} and z_{c,*,*}

       terms to zero.
       */

    public:
        using Surface<Array>::quadpoints_phi;
        using Surface<Array>::quadpoints_theta;
        using Surface<Array>::numquadpoints_phi;
        using Surface<Array>::numquadpoints_theta;
        Array xc;
        Array xs;
        Array yc;
        Array ys;
        Array zc;
        Array zs;
        int nfp;
        int mpol;
        int ntor;
        bool stellsym;

        SurfaceXYZFourier(int _mpol, int _ntor, int _nfp, bool _stellsym, vector<double> _quadpoints_phi, vector<double> _quadpoints_theta)
            : Surface<Array>(_quadpoints_phi, _quadpoints_theta), mpol(_mpol), ntor(_ntor), nfp(_nfp), stellsym(_stellsym) {
                xc = xt::zeros<double>({mpol+1, 2*ntor+1});
                xs = xt::zeros<double>({mpol+1, 2*ntor+1});
                yc = xt::zeros<double>({mpol+1, 2*ntor+1});
                ys = xt::zeros<double>({mpol+1, 2*ntor+1});
                zc = xt::zeros<double>({mpol+1, 2*ntor+1});
                zs = xt::zeros<double>({mpol+1, 2*ntor+1});
            }

        SurfaceXYZFourier(int _mpol, int _ntor, int _nfp, bool _stellsym, int _numquadpoints_phi, int _numquadpoints_theta)
            : Surface<Array>(_numquadpoints_phi, _numquadpoints_theta), mpol(_mpol), ntor(_ntor), nfp(_nfp), stellsym(_stellsym) {
                xc = xt::zeros<double>({mpol+1, 2*ntor+1});
                xs = xt::zeros<double>({mpol+1, 2*ntor+1});
                yc = xt::zeros<double>({mpol+1, 2*ntor+1});
                ys = xt::zeros<double>({mpol+1, 2*ntor+1});
                zc = xt::zeros<double>({mpol+1, 2*ntor+1});
                zs = xt::zeros<double>({mpol+1, 2*ntor+1});
            }



        int num_dofs() override {
            if(stellsym)
                return 3*(mpol+1)*(2*ntor+1) - 1*ntor - 2*(ntor+1);
            else
                return 6*(mpol+1)*(2*ntor+1) - 3*ntor - 3*(ntor+1);
        }

        void set_dofs_impl(const vector<double>& dofs) override {
            int shift = (mpol+1)*(2*ntor+1);
            int counter = 0;
            if(stellsym) {
                for (int i = ntor; i < shift; ++i)
                    xc.data()[i] = dofs[counter++];
                for (int i = ntor+1; i < shift; ++i)
                    ys.data()[i] = dofs[counter++];
                for (int i = ntor+1; i < shift; ++i)
                    zs.data()[i] = dofs[counter++];

            } else {
                for (int i = ntor; i < shift; ++i)
                    xc.data()[i] = dofs[counter++];
                for (int i = ntor+1; i < shift; ++i)
                    xs.data()[i] = dofs[counter++];
                for (int i = ntor; i < shift; ++i)
                    yc.data()[i] = dofs[counter++];
                for (int i = ntor+1; i < shift; ++i)
                    ys.data()[i] = dofs[counter++];
                for (int i = ntor; i < shift; ++i)
                    zc.data()[i] = dofs[counter++];
                for (int i = ntor+1; i < shift; ++i)
                    zs.data()[i] = dofs[counter++];
            }
        }

        vector<double> get_dofs() override {
            auto res = vector<double>(num_dofs(), 0.);
            int shift = (mpol+1)*(2*ntor+1);
            int counter = 0;
            if(stellsym) {
                for (int i = ntor; i < shift; ++i)
                    res[counter++] = xc.data()[i];
                for (int i = ntor+1; i < shift; ++i)
                    res[counter++] = ys.data()[i];
                for (int i = ntor+1; i < shift; ++i)
                    res[counter++] = zs.data()[i];
            } else {
                for (int i = ntor; i < shift; ++i)
                    res[counter++] = xc.data()[i];
                for (int i = ntor+1; i < shift; ++i)
                    res[counter++] = xs.data()[i];
                for (int i = ntor; i < shift; ++i)
                    res[counter++] = yc.data()[i];
                for (int i = ntor+1; i < shift; ++i)
                    res[counter++] = ys.data()[i];
                for (int i = ntor; i < shift; ++i)
                    res[counter++] = zc.data()[i];
                for (int i = ntor+1; i < shift; ++i)
                    res[counter++] = zs.data()[i];
            }
            return res;
        }

        inline double get_coeff(int dim, bool cos, int m, int i) {
            if     (dim == 0 && cos)
                return xc(m, i);
            else if(dim == 0 && !cos)
                return xs(m, i);
            else if(dim == 1 && cos)
                return yc(m, i);
            else if(dim == 1 && !cos)
                return ys(m, i);
            else if(dim == 2 && cos)
                return zc(m, i);
            else
                return zs(m, i);
        }

        void gamma_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override {
            int numquadpoints_phi = quadpoints_phi.size();
            int numquadpoints_theta = quadpoints_theta.size();
            

            data *= 0.;
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    for (int m = 0; m <= mpol; ++m) {
                        for (int i = 0; i < 2*ntor+1; ++i) {
                            int n  = i - ntor;
                            double xhat = get_coeff(0, true, m, i) * cos(m*theta-n*nfp*phi) + get_coeff(0, false, m, i) * sin(m*theta-n*nfp*phi);
                            double yhat = get_coeff(1, true, m, i) * cos(m*theta-n*nfp*phi) + get_coeff(1, false, m, i) * sin(m*theta-n*nfp*phi);
                            double x = xhat * cos(phi) - yhat * sin(phi);
                            double y = xhat * sin(phi) + yhat * cos(phi);
                            double z = get_coeff(2, true , m, i) * cos(m*theta-n*nfp*phi) + get_coeff(2, false, m, i) * sin(m*theta-n*nfp*phi);
                            data(k1, k2, 0) += x;
                            data(k1, k2, 1) += y;
                            data(k1, k2, 2) += z;
                        }
                    }
                }
            }
        }

        void gamma_lin(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override {
            int numquadpoints = quadpoints_phi.size();
            data *= 0.;
            for (int k1 = 0; k1 < numquadpoints; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                double theta  = 2*M_PI*quadpoints_theta[k1];
                for (int m = 0; m <= mpol; ++m) {
                    for (int i = 0; i < 2*ntor+1; ++i) {
                        int n  = i - ntor;
                        double xhat = get_coeff(0, true, m, i) * cos(m*theta-n*nfp*phi) + get_coeff(0, false, m, i) * sin(m*theta-n*nfp*phi);
                        double yhat = get_coeff(1, true, m, i) * cos(m*theta-n*nfp*phi) + get_coeff(1, false, m, i) * sin(m*theta-n*nfp*phi);
                        double x = xhat * cos(phi) - yhat * sin(phi);
                        double y = xhat * sin(phi) + yhat * cos(phi);
                        double z = get_coeff(2, true , m, i) * cos(m*theta-n*nfp*phi) + get_coeff(2, false, m, i) * sin(m*theta-n*nfp*phi);
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
                    for (int m = 0; m <= mpol; ++m) {
                        for (int i = 0; i < 2*ntor+1; ++i) {
                            int n  = i - ntor;
                            double xhat = get_coeff(0, true, m, i) * cos(m*theta-n*nfp*phi) + get_coeff(0, false, m, i) * sin(m*theta-n*nfp*phi);
                            double yhat = get_coeff(1, true, m, i) * cos(m*theta-n*nfp*phi) + get_coeff(1, false, m, i) * sin(m*theta-n*nfp*phi);
                            double xhatdash = get_coeff(0, true, m, i) * (n*nfp)*sin(m*theta-n*nfp*phi) + get_coeff(0, false, m, i) * (-n*nfp)*cos(m*theta-n*nfp*phi);
                            double yhatdash = get_coeff(1, true, m, i) * (n*nfp)*sin(m*theta-n*nfp*phi) + get_coeff(1, false, m, i) * (-n*nfp)*cos(m*theta-n*nfp*phi);
                            double xdash = xhatdash * cos(phi) - yhatdash * sin(phi) - xhat * sin(phi) - yhat * cos(phi);
                            double ydash = xhatdash * sin(phi) + yhatdash * cos(phi) + xhat * cos(phi) - yhat * sin(phi);
                            double zdash = get_coeff(2, true , m, i) * (n*nfp)*sin(m*theta-n*nfp*phi) + get_coeff(2, false, m, i) * (-n*nfp)*cos(m*theta-n*nfp*phi);
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
                    for (int m = 0; m <= mpol; ++m) {
                        for (int i = 0; i < 2*ntor+1; ++i) {
                            int n  = i - ntor;
                            double xhatdash = get_coeff(0, true, m, i) * (-m)* sin(m*theta-n*nfp*phi) + get_coeff(0, false, m, i) * m * cos(m*theta-n*nfp*phi);
                            double yhatdash = get_coeff(1, true, m, i) * (-m)* sin(m*theta-n*nfp*phi) + get_coeff(1, false, m, i) * m * cos(m*theta-n*nfp*phi);
                            double xdash = xhatdash * cos(phi) - yhatdash * sin(phi);
                            double ydash = xhatdash * sin(phi) + yhatdash * cos(phi);
                            double zdash = get_coeff(2, true , m, i) * (-m) * sin(m*theta-n*nfp*phi) + get_coeff(2, false, m, i) * m * cos(m*theta-n*nfp*phi);
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
                        for (int m = 0; m <= mpol; ++m) {
                            for (int n = -ntor; n <= ntor; ++n) {
                                if(m==0 && n<0) continue;
                                if(d == 0) {
                                    data(k1, k2, 0, counter) = cos(m*theta-n*nfp*phi) * cos(phi);
                                    data(k1, k2, 1, counter) = cos(m*theta-n*nfp*phi) * sin(phi);
                                }else if(d == 1) {
                                    if(stellsym)
                                        continue;
                                    data(k1, k2, 0, counter) = -cos(m*theta-n*nfp*phi) * sin(phi);
                                    data(k1, k2, 1, counter) =  cos(m*theta-n*nfp*phi) * cos(phi);
                                }
                                else if(d == 2) {
                                    if(stellsym)
                                        continue;
                                    data(k1, k2, 2, counter) =  cos(m*theta-n*nfp*phi);
                                }
                                counter++;
                            }
                        }
                        for (int m = 0; m <= mpol; ++m) {
                            for (int n = -ntor; n <= ntor; ++n) {
                                if(m==0 && n<=0) continue;
                                if(d == 0) {
                                    if(stellsym)
                                        continue;
                                    data(k1, k2, 0, counter) = sin(m*theta-n*nfp*phi) * cos(phi);
                                    data(k1, k2, 1, counter) = sin(m*theta-n*nfp*phi) * sin(phi);
                                }else if(d == 1) {
                                    data(k1, k2, 0, counter) = -sin(m*theta-n*nfp*phi) * sin(phi);
                                    data(k1, k2, 1, counter) =  sin(m*theta-n*nfp*phi) * cos(phi);
                                }
                                else if(d == 2) {
                                    data(k1, k2, 2, counter) =  sin(m*theta-n*nfp*phi);
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
                        for (int m = 0; m <= mpol; ++m) {
                            for (int n = -ntor; n <= ntor; ++n) {
                                if(m==0 && n<0) continue;
                                if(d == 0) {
                                    data(k1, k2, 0, counter) = (n*nfp)*sin(m*theta-n*nfp*phi) * cos(phi) - cos(m*theta-n*nfp*phi) * sin(phi);
                                    data(k1, k2, 1, counter) = (n*nfp)*sin(m*theta-n*nfp*phi) * sin(phi) + cos(m*theta-n*nfp*phi) * cos(phi);
                                }else if(d == 1) {
                                    if(stellsym)
                                        continue;
                                    data(k1, k2, 0, counter) = -(n*nfp)*sin(m*theta-n*nfp*phi) * sin(phi) - cos(m*theta-n*nfp*phi) * cos(phi);
                                    data(k1, k2, 1, counter) =  (n*nfp)*sin(m*theta-n*nfp*phi) * cos(phi) - cos(m*theta-n*nfp*phi) * sin(phi);
                                }
                                else if(d == 2) {
                                    if(stellsym)
                                        continue;
                                    data(k1, k2, 2, counter) =  (n*nfp)*sin(m*theta-n*nfp*phi);
                                }
                                counter++;
                            }
                        }
                        for (int m = 0; m <= mpol; ++m) {
                            for (int n = -ntor; n <= ntor; ++n) {
                                if(m==0 && n<=0) continue;
                                if(d == 0) {
                                    if(stellsym)
                                        continue;
                                    data(k1, k2, 0, counter) = -(n*nfp)*cos(m*theta-n*nfp*phi) * cos(phi) - sin(m*theta-n*nfp*phi) * sin(phi);
                                    data(k1, k2, 1, counter) = -(n*nfp)*cos(m*theta-n*nfp*phi) * sin(phi) + sin(m*theta-n*nfp*phi) * cos(phi);
                                }else if(d == 1) {
                                    data(k1, k2, 0, counter) = (n*nfp)*cos(m*theta-n*nfp*phi) * sin(phi)  - sin(m*theta-n*nfp*phi) * cos(phi);
                                    data(k1, k2, 1, counter) = (-n*nfp)*cos(m*theta-n*nfp*phi) * cos(phi) - sin(m*theta-n*nfp*phi) * sin(phi);
                                }
                                else if(d == 2) {
                                    data(k1, k2, 2, counter) = (-n*nfp)*cos(m*theta-n*nfp*phi);
                                }
                                counter++;
                            }
                        }
                    }
                }
            }
            data *= 2 * M_PI;
        }

        void dgammadash2_by_dcoeff_impl(Array& data) override {
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    int counter = 0;
                    for (int d = 0; d < 3; ++d) {
                        for (int m = 0; m <= mpol; ++m) {
                            for (int n = -ntor; n <= ntor; ++n) {
                                if(m==0 && n<0) continue;
                                if(d == 0) {
                                    data(k1, k2, 0, counter) = (-m)* sin(m*theta-n*nfp*phi) * cos(phi);
                                    data(k1, k2, 1, counter) = (-m)* sin(m*theta-n*nfp*phi) * sin(phi);
                                }else if(d == 1) {
                                    if(stellsym)
                                        continue;
                                    data(k1, k2, 0, counter) = (-m)* sin(m*theta-n*nfp*phi) * (-1) * sin(phi);
                                    data(k1, k2, 1, counter) = (-m)* sin(m*theta-n*nfp*phi) * cos(phi);
                                }
                                else if(d == 2) {
                                    if(stellsym)
                                        continue;
                                    data(k1, k2, 2, counter) = (-m) * sin(m*theta-n*nfp*phi);
                                }
                                counter++;
                            }
                        }
                        for (int m = 0; m <= mpol; ++m) {
                            for (int n = -ntor; n <= ntor; ++n) {
                                if(m==0 && n<=0) continue;
                                if(d == 0) {
                                    if(stellsym)
                                        continue;
                                    data(k1, k2, 0, counter) = m * cos(m*theta-n*nfp*phi) * cos(phi);
                                    data(k1, k2, 1, counter) = m * cos(m*theta-n*nfp*phi) * sin(phi);
                                }else if(d == 1) {
                                    data(k1, k2, 0, counter) = m * cos(m*theta-n*nfp*phi) * (-1) * sin(phi);
                                    data(k1, k2, 1, counter) = m * cos(m*theta-n*nfp*phi) * cos(phi);
                                }
                                else if(d == 2) {
                                    data(k1, k2, 2, counter) = m * cos(m*theta-n*nfp*phi);
                                }
                                counter++;
                            }
                        }
                    }
                }
            }
            data *= 2*M_PI;
        }

};
