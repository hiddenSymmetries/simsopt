#pragma once

#include "surface.cpp"

template<class Array>
class SurfaceRZFourier : public Surface<Array> {
    /*
       SurfaceRZFourier is a surface that is represented in cylindrical
       coordinates using the following Fourier series: 

           r(theta, phi) = \sum_{m=0}^{mpol} \sum_{n=-ntor}^{ntor} [
               r_{c,m,n} \cos(m \theta - n nfp \phi)
               + r_{s,m,n} \sin(m \theta - n nfp \phi) ]

       and the same for z(theta, phi).

       Here, (r, phi, z) are standard cylindrical coordinates, and theta
       is any poloidal angle.

       Note that for m=0 we skip the n<0 term for the cos terms, and the n<=0
       for the sin terms.
       
       In addition, in the stellsym=True case, we skip the sin terms for r, and
       the cos terms for z.
       */

    public:
        using Surface<Array>::quadpoints_phi;
        using Surface<Array>::quadpoints_theta;
        using Surface<Array>::numquadpoints_phi;
        using Surface<Array>::numquadpoints_theta;
        Array rc;
        Array rs;
        Array zc;
        Array zs;
        int nfp;
        int mpol;
        int ntor;
        bool stellsym;

        SurfaceRZFourier(int _mpol, int _ntor, int _nfp, bool _stellsym, vector<double> _quadpoints_phi, vector<double> _quadpoints_theta)
            : Surface<Array>(_quadpoints_phi, _quadpoints_theta), mpol(_mpol), ntor(_ntor), nfp(_nfp), stellsym(_stellsym) {
                this->allocate();
            }

        SurfaceRZFourier(int _mpol, int _ntor, int _nfp, bool _stellsym, int _numquadpoints_phi, int _numquadpoints_theta)
            : Surface<Array>(_numquadpoints_phi, _numquadpoints_theta), mpol(_mpol), ntor(_ntor), nfp(_nfp), stellsym(_stellsym) {
                this->allocate();
            }

        void allocate() {
            rc = xt::zeros<double>({mpol+1, 2*ntor+1});
            rs = xt::zeros<double>({mpol+1, 2*ntor+1});
            zc = xt::zeros<double>({mpol+1, 2*ntor+1});
            zs = xt::zeros<double>({mpol+1, 2*ntor+1});
        }

        int num_dofs() override {
            if(stellsym)
                return 2*(mpol+1)*(2*ntor+1) - ntor - (ntor+1);
            else
                return 4*(mpol+1)*(2*ntor+1) - 2*ntor - 2*(ntor+1);
        }

        void set_dofs_impl(const vector<double>& dofs) override {
            int shift = (mpol+1)*(2*ntor+1);
            int counter = 0;
            if(stellsym) {
                for (int i = ntor; i < shift; ++i)
                    rc.data()[i] = dofs[counter++];
                for (int i = ntor+1; i < shift; ++i)
                    zs.data()[i] = dofs[counter++];

            } else {
                for (int i = ntor; i < shift; ++i)
                    rc.data()[i] = dofs[counter++];
                for (int i = ntor+1; i < shift; ++i)
                    rs.data()[i] = dofs[counter++];
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
                    res[counter++] = rc.data()[i];
                for (int i = ntor+1; i < shift; ++i)
                    res[counter++] = zs.data()[i];
            } else {
                for (int i = ntor; i < shift; ++i)
                    res[counter++] = rc.data()[i];
                for (int i = ntor+1; i < shift; ++i)
                    res[counter++] = rs.data()[i];
                for (int i = ntor; i < shift; ++i)
                    res[counter++] = zc.data()[i];
                for (int i = ntor+1; i < shift; ++i)
                    res[counter++] = zs.data()[i];
            }
            return res;
        }
        
        void gamma_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override {
            int numquadpoints_phi = quadpoints_phi.size();
            int numquadpoints_theta = quadpoints_theta.size();

            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    double r = 0;
                    double z = 0;
                    for (int m = 0; m <= mpol; ++m) {
                        for (int i = 0; i < 2*ntor+1; ++i) {
                            int n  = i - ntor;
                            r += rc(m, i) * cos(m*theta-n*nfp*phi);
                            if(!stellsym) {
                                r += rs(m, i) * sin(m*theta-n*nfp*phi);
                                z += zc(m, i) * cos(m*theta-n*nfp*phi);
                            }
                            z += zs(m, i) * sin(m*theta-n*nfp*phi);
                        }
                    }
                    data(k1, k2, 0) = r * cos(phi);
                    data(k1, k2, 1) = r * sin(phi);
                    data(k1, k2, 2) = z;
                }
            }
        }


        void gamma_lin(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override {
            int numquadpoints = quadpoints_phi.size();

            for (int k1 = 0; k1 < numquadpoints; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                double theta  = 2*M_PI*quadpoints_theta[k1];
                double r = 0;
                double z = 0;
                for (int m = 0; m <= mpol; ++m) {
                    for (int i = 0; i < 2*ntor+1; ++i) {
                        int n  = i - ntor;
                        r += rc(m, i) * cos(m*theta-n*nfp*phi);
                        if(!stellsym) {
                            r += rs(m, i) * sin(m*theta-n*nfp*phi);
                            z += zc(m, i) * cos(m*theta-n*nfp*phi);
                        }
                        z += zs(m, i) * sin(m*theta-n*nfp*phi);
                    }
                }
                data(k1, 0) = r * cos(phi);
                data(k1, 1) = r * sin(phi);
                data(k1, 2) = z;
            }
        }





        void gammadash1_impl(Array& data) override {
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    double r = 0;
                    double rd = 0;
                    double zd = 0;
                    for (int i = 0; i < 2*ntor+1; ++i) {
                        int n  = i - ntor;
                        for (int m = 0; m <= mpol; ++m) {
                            r  += rc(m, i) * cos(m*theta-n*nfp*phi);
                            rd += rc(m, i) * (n*nfp) * sin(m*theta-n*nfp*phi);
                            if(!stellsym) {
                                r  += rs(m, i) * sin(m*theta-n*nfp*phi);
                                rd += rs(m, i) * (-n*nfp)*cos(m*theta-n*nfp*phi);
                                zd += zc(m, i) * (n*nfp)*sin(m*theta-n*nfp*phi);
                            }
                            zd += zs(m, i) * (-n*nfp)*cos(m*theta-n*nfp*phi);
                        }
                    }
                    data(k1, k2, 0) = 2*M_PI*(rd * cos(phi) - r * sin(phi));
                    data(k1, k2, 1) = 2*M_PI*(rd * sin(phi) + r * cos(phi));
                    data(k1, k2, 2) = 2*M_PI*zd;
                }
            }
        }
        void gammadash2_impl(Array& data) override {
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    double rd = 0;
                    double zd = 0;
                    for (int i = 0; i < 2*ntor+1; ++i) {
                        int n  = i - ntor;
                        for (int m = 0; m <= mpol; ++m) {
                            rd += rc(m, i) * (-m) * sin(m*theta-n*nfp*phi);
                            if(!stellsym) {
                                rd += rs(m, i) * m * cos(m*theta-n*nfp*phi);
                                zd += zc(m, i) * (-m) * sin(m*theta-n*nfp*phi);
                            }
                            zd += zs(m, i) * m * cos(m*theta-n*nfp*phi);
                        }
                    }
                    data(k1, k2, 0) = 2*M_PI*rd*cos(phi);
                    data(k1, k2, 1) = 2*M_PI*rd*sin(phi);
                    data(k1, k2, 2) = 2*M_PI*zd;
                }
            }
        }

        void dgamma_by_dcoeff_impl(Array& data) override {
            for (int k1 = 0; k1 < numquadpoints_phi; ++k1) {
                double phi  = 2*M_PI*quadpoints_phi[k1];
                for (int k2 = 0; k2 < numquadpoints_theta; ++k2) {
                    double theta  = 2*M_PI*quadpoints_theta[k2];
                    int counter = 0;
                    for (int m = 0; m <= mpol; ++m) {
                        for (int n = -ntor; n <= ntor; ++n) {
                            if(m==0 && n<0) continue;
                            data(k1, k2, 0, counter) = cos(m*theta-n*nfp*phi) * cos(phi);
                            data(k1, k2, 1, counter) = cos(m*theta-n*nfp*phi) * sin(phi);
                            data(k1, k2, 2, counter) = 0;
                            counter++;
                        }
                    }
                    if(!stellsym) {
                        for (int m = 0; m <= mpol; ++m) {
                            for (int n = -ntor; n <= ntor; ++n) {
                                if(m==0 && n<=0) continue;
                                data(k1, k2, 0, counter) = sin(m*theta-n*nfp*phi) * cos(phi);
                                data(k1, k2, 1, counter) = sin(m*theta-n*nfp*phi) * sin(phi);
                                data(k1, k2, 2, counter) = 0;
                                counter++;
                            }
                        }
                        for (int m = 0; m <= mpol; ++m) {
                            for (int n = -ntor; n <= ntor; ++n) {
                                if(m==0 && n<0) continue;
                                data(k1, k2, 0, counter) = 0;
                                data(k1, k2, 1, counter) = 0;
                                data(k1, k2, 2, counter) = cos(m*theta-n*nfp*phi);
                                counter++;
                            }
                        }
                    }
                    for (int m = 0; m <= mpol; ++m) {
                        for (int n = -ntor; n <= ntor; ++n) {
                            if(m==0 && n<=0) continue;
                            data(k1, k2, 0, counter) = 0;
                            data(k1, k2, 1, counter) = 0;
                            data(k1, k2, 2, counter) = sin(m*theta-n*nfp*phi);
                            counter++;
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
                    for (int m = 0; m <= mpol; ++m) {
                        for (int n = -ntor; n <= ntor; ++n) {
                            if(m==0 && n<0) continue;
                            data(k1, k2, 0, counter) = 2*M_PI*((n*nfp) * sin(m*theta-n*nfp*phi) * cos(phi) - cos(m*theta-n*nfp*phi) * sin(phi));
                            data(k1, k2, 1, counter) = 2*M_PI*((n*nfp) * sin(m*theta-n*nfp*phi) * sin(phi) + cos(m*theta-n*nfp*phi) * cos(phi));
                            data(k1, k2, 2, counter) = 0;
                            counter++;
                        }
                    }
                    if(!stellsym) {
                        for (int m = 0; m <= mpol; ++m) {
                            for (int n = -ntor; n <= ntor; ++n) {
                                if(m==0 && n<=0) continue;
                                data(k1, k2, 0, counter) = 2*M_PI*((-n*nfp)*cos(m*theta-n*nfp*phi) * cos(phi) - sin(m*theta-n*nfp*phi) * sin(phi));
                                data(k1, k2, 1, counter) = 2*M_PI*((-n*nfp)*cos(m*theta-n*nfp*phi) * sin(phi) + sin(m*theta-n*nfp*phi) * cos(phi));
                                data(k1, k2, 2, counter) = 0;
                                counter++;
                            }
                        }
                        for (int m = 0; m <= mpol; ++m) {
                            for (int n = -ntor; n <= ntor; ++n) {
                                if(m==0 && n<0) continue;
                                data(k1, k2, 0, counter) = 0;
                                data(k1, k2, 1, counter) = 0;
                                data(k1, k2, 2, counter) = 2*M_PI*(n*nfp)*sin(m*theta-n*nfp*phi);
                                counter++;
                            }
                        }
                    }
                    for (int m = 0; m <= mpol; ++m) {
                        for (int n = -ntor; n <= ntor; ++n) {
                            if(m==0 && n<=0) continue;
                            data(k1, k2, 0, counter) = 0;
                            data(k1, k2, 1, counter) = 0;
                            data(k1, k2, 2, counter) = 2*M_PI*(-n*nfp)*cos(m*theta-n*nfp*phi);
                            counter++;
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
                    for (int m = 0; m <= mpol; ++m) {
                        for (int n = -ntor; n <= ntor; ++n) {
                            if(m==0 && n<0) continue;
                            data(k1, k2, 0, counter) = 2*M_PI*(-m) * sin(m*theta-n*nfp*phi)*cos(phi);
                            data(k1, k2, 1, counter) = 2*M_PI*(-m) * sin(m*theta-n*nfp*phi)*sin(phi);
                            data(k1, k2, 2, counter) = 0;
                            counter++;
                        }
                    }
                    if(!stellsym) {
                        for (int m = 0; m <= mpol; ++m) {
                            for (int n = -ntor; n <= ntor; ++n) {
                                if(m==0 && n<=0) continue;
                                data(k1, k2, 0, counter) = 2*M_PI*m * cos(m*theta-n*nfp*phi)*cos(phi);
                                data(k1, k2, 1, counter) = 2*M_PI*m * cos(m*theta-n*nfp*phi)*sin(phi);
                                data(k1, k2, 2, counter) = 0;
                                counter++;
                            }
                        }
                        for (int m = 0; m <= mpol; ++m) {
                            for (int n = -ntor; n <= ntor; ++n) {
                                if(m==0 && n<0) continue;
                                data(k1, k2, 0, counter) = 0;
                                data(k1, k2, 1, counter) = 0;
                                data(k1, k2, 2, counter) = 2*M_PI*(-m) * sin(m*theta-n*nfp*phi);
                                counter++;
                            }
                        }
                    }
                    for (int m = 0; m <= mpol; ++m) {
                        for (int n = -ntor; n <= ntor; ++n) {
                            if(m==0 && n<=0) continue;
                            data(k1, k2, 0, counter) = 0;
                            data(k1, k2, 1, counter) = 0;
                            data(k1, k2, 2, counter) = 2*M_PI*m * cos(m*theta-n*nfp*phi);
                            counter++;
                        }
                    }
                }
            }
        }

};
