#include "surfacerzfourier.h"

template<class Array>
void SurfaceRZFourier<Array>::gamma_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
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


template<class Array>
void SurfaceRZFourier<Array>::gamma_lin(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) {
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





template<class Array>
void SurfaceRZFourier<Array>::gammadash1_impl(Array& data) {
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
template<class Array>
void SurfaceRZFourier<Array>::gammadash2_impl(Array& data) {
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

template<class Array>
void SurfaceRZFourier<Array>::dgamma_by_dcoeff_impl(Array& data) {
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

template<class Array>
void SurfaceRZFourier<Array>::dgammadash1_by_dcoeff_impl(Array& data) {
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

template<class Array>
void SurfaceRZFourier<Array>::dgammadash2_by_dcoeff_impl(Array& data) {
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

#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;
template class SurfaceRZFourier<Array>;
