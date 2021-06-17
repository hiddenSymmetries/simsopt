#include "curverzfourier.h"


template<class Array>
void CurveRZFourier<Array>::gamma_impl(Array& data, Array& quadpoints) {
    int numquadpoints = quadpoints.size();
    data *= 0;
    for (int k = 0; k < numquadpoints; ++k) {
        double phi = 2 * M_PI * quadpoints[k];
        for (int i = 0; i < order+1; ++i) {
            data(k, 0) += rc[i] * cos(nfp*i*phi) * cos(phi);
            data(k, 1) += rc[i] * cos(nfp*i*phi) * sin(phi);
        }
        for (int i = 1; i < order+1; ++i) {
            data(k, 2) += zs[i-1] * sin(nfp*i*phi);
        }
    }
    if(!stellsym){
        for (int k = 0; k < numquadpoints; ++k) {
            double phi = 2 * M_PI * quadpoints[k];
            for (int i = 1; i < order+1; ++i) {
                data(k, 0) += rs[i-1] * sin(nfp*i*phi) * cos(phi);
                data(k, 1) += rs[i-1] * sin(nfp*i*phi) * sin(phi);
            }
            for (int i = 0; i < order+1; ++i) {
                data(k, 2) += zc[i] * cos(nfp*i*phi);
            }
        }
    }
}

template<class Array>
void CurveRZFourier<Array>::gammadash_impl(Array& data) {
    data *= 0;
    for (int k = 0; k < numquadpoints; ++k) {
        double phi = 2 * M_PI * quadpoints[k];
        for (int i = 0; i < order+1; ++i) {
            data(k, 0) += rc[i] * ( -(i*nfp) * sin(nfp*i*phi) * cos(phi) - cos(nfp*i*phi) * sin(phi));
            data(k, 1) += rc[i] * ( -(i*nfp) * sin(nfp*i*phi) * sin(phi) + cos(nfp*i*phi) * cos(phi));
        }
        for (int i = 1; i < order+1; ++i) {
            data(k, 2) += zs[i-1] * (nfp*i) * cos(nfp*i*phi);
        }
    }
    if(!stellsym){
        for (int k = 0; k < numquadpoints; ++k) {
            double phi = 2 * M_PI * quadpoints[k];
            for (int i = 1; i < order+1; ++i) {
                data(k, 0) += rs[i-1] * ( (i*nfp) * cos(nfp*i*phi) * cos(phi) - sin(nfp*i*phi) * sin(phi));
                data(k, 1) += rs[i-1] * ( (i*nfp) * cos(nfp*i*phi) * sin(phi) + sin(nfp*i*phi) * cos(phi));
            }
            for (int i = 0; i < order+1; ++i) {
                data(k, 2) -= zc[i] * (nfp*i) * sin(nfp*i*phi);
            }
        }
    }
    data *= (2*M_PI);
}

template<class Array>
void CurveRZFourier<Array>::gammadashdash_impl(Array& data) {
    data *= 0;
    for (int k = 0; k < numquadpoints; ++k) {
        double phi = 2 * M_PI * quadpoints[k];
        for (int i = 0; i < order+1; ++i) {
            data(k, 0) += rc[i] * (+2*(nfp*i)*sin(nfp*i*phi)*sin(phi)-(pow(nfp*i, 2)+1)*cos(nfp*i*phi)*cos(phi));
            data(k, 1) += rc[i] * (-2*(nfp*i)*sin(nfp*i*phi)*cos(phi)-(pow(nfp*i, 2)+1)*cos(nfp*i*phi)*sin(phi));
        }
        for (int i = 1; i < order+1; ++i) {
            data(k, 2) -= zs[i-1] * pow(nfp*i, 2)*sin(nfp*i*phi);
        }
    }
    if(!stellsym){
        for (int k = 0; k < numquadpoints; ++k) {
            double phi = 2 * M_PI * quadpoints[k];
            for (int i = 1; i < order+1; ++i) {
                data(k, 0) += rs[i-1] * (-(pow(nfp*i,2)+1)*sin(nfp*i*phi)*cos(phi) - 2*(i*nfp)*cos(nfp*i*phi)*sin(phi));
                data(k, 1) += rs[i-1] * (-(pow(nfp*i,2)+1)*sin(nfp*i*phi)*sin(phi) + 2*(i*nfp)*cos(nfp*i*phi)*cos(phi));
            }
            for (int i = 0; i < order+1; ++i) {
                data(k, 2) -= zc[i] * pow(nfp*i, 2)*cos(nfp*i*phi);
            }
        }
    }
    data *= 2*M_PI*2*M_PI;
}

template<class Array>
void CurveRZFourier<Array>::gammadashdashdash_impl(Array& data) {
    data *= 0;
    for (int k = 0; k < numquadpoints; ++k) {
        double phi = 2 * M_PI * quadpoints[k];
        for (int i = 0; i < order+1; ++i) {
            data(k, 0) += rc[i]*(
                    +(3*pow(nfp*i, 2) + 1)*cos(nfp*i*phi)*sin(phi)
                    +(pow(nfp*i, 2) + 3)*(nfp*i)*sin(nfp*i*phi)*cos(phi)
                    );
            data(k, 1) += rc[i]*(
                    +(pow(nfp*i, 2) + 3)*(nfp*i)*sin(nfp*i*phi)*sin(phi)
                    -(3*pow(nfp*i, 2) + 1)*cos(nfp*i*phi)*cos(phi)
                    );
        }
        for (int i = 1; i < order+1; ++i) {
            data(k, 2) -= zs[i-1] * pow(nfp*i, 3) * cos(nfp*i*phi);
        }
    }
    if(!stellsym){
        for (int k = 0; k < numquadpoints; ++k) {
            double phi = 2 * M_PI * quadpoints[k];
            for (int i = 1; i < order+1; ++i) {
                data(k, 0) += rs[i-1]*(
                        -(pow(nfp*i,2)+3) * (nfp*i) * cos(nfp*i*phi)*cos(phi)
                        +(3*pow(nfp*i,2)+1) * sin(nfp*i*phi)*sin(phi)
                        );
                data(k, 1) += rs[i-1]*(
                        -(pow(nfp*i,2)+3)*(nfp*i)*cos(nfp*i*phi)*sin(phi) 
                        -(3*pow(nfp*i,2)+1)*sin(nfp*i*phi)*cos(phi) 
                        );
            }
            for (int i = 0; i < order+1; ++i) {
                data(k, 2) += zc[i] * pow(nfp*i, 3) * sin(nfp*i*phi);
            }
        }
    }
    data *= 2*M_PI*2*M_PI*2*M_PI;
}

template<class Array>
void CurveRZFourier<Array>::dgamma_by_dcoeff_impl(Array& data) {
    for (int k = 0; k < numquadpoints; ++k) {
        double phi = 2 * M_PI * quadpoints[k];
        int counter = 0;
        for (int i = 0; i < order+1; ++i) {
            data(k, 0, counter) = cos(nfp*i*phi) * cos(phi);
            data(k, 1, counter) = cos(nfp*i*phi) * sin(phi);
            counter++;
        }
        if(!stellsym){
            for (int i = 1; i < order+1; ++i) {
                data(k, 0, counter) = sin(nfp*i*phi) * cos(phi);
                data(k, 1, counter) = sin(nfp*i*phi) * sin(phi);
                counter++;
            }
            for (int i = 0; i < order+1; ++i) {
                data(k, 2, counter) = cos(nfp*i*phi);
                counter++;
            }
        }
        for (int i = 1; i < order+1; ++i) {
            data(k, 2, counter) = sin(nfp*i*phi); counter++;
        }
    }
}

template<class Array>
void CurveRZFourier<Array>::dgammadash_by_dcoeff_impl(Array& data) {
    for (int k = 0; k < numquadpoints; ++k) {
        double phi = 2 * M_PI * quadpoints[k];
        int counter = 0;
        for (int i = 0; i < order+1; ++i) {
            data(k, 0, counter) = ( -(i*nfp) * sin(nfp*i*phi) * cos(phi) - cos(nfp*i*phi) * sin(phi));
            data(k, 1, counter) = ( -(i*nfp) * sin(nfp*i*phi) * sin(phi) + cos(nfp*i*phi) * cos(phi));
            counter++;
        }
        if(!stellsym){
            for (int i = 1; i < order+1; ++i) {
                data(k, 0, counter) = ( (i*nfp) * cos(nfp*i*phi) * cos(phi) - sin(nfp*i*phi) * sin(phi));
                data(k, 1, counter) = ( (i*nfp) * cos(nfp*i*phi) * sin(phi) + sin(nfp*i*phi) * cos(phi));
                counter++;
            }
            for (int i = 0; i < order+1; ++i) {
                data(k, 2, counter) = -(nfp*i) * sin(nfp*i*phi);
                counter++;
            }
        }
        for (int i = 1; i < order+1; ++i) {
            data(k, 2, counter) = (nfp*i) * cos(nfp*i*phi);
            counter++;
        }
    }
    data *= (2*M_PI);
}

template<class Array>
void CurveRZFourier<Array>::dgammadashdash_by_dcoeff_impl(Array& data) {
    for (int k = 0; k < numquadpoints; ++k) {
        double phi = 2 * M_PI * quadpoints[k];
        int counter = 0;
        for (int i = 0; i < order+1; ++i) {
            data(k, 0, counter) = (+2*(nfp*i)*sin(nfp*i*phi)*sin(phi)-(pow(nfp*i, 2)+1)*cos(nfp*i*phi)*cos(phi));
            data(k, 1, counter) = (-2*(nfp*i)*sin(nfp*i*phi)*cos(phi)-(pow(nfp*i, 2)+1)*cos(nfp*i*phi)*sin(phi));
            counter++;
        }
        if(!stellsym){
            for (int i = 1; i < order+1; ++i) {
                data(k, 0, counter) = (-(pow(nfp*i,2)+1)*sin(nfp*i*phi)*cos(phi) - 2*(i*nfp)*cos(nfp*i*phi)*sin(phi));
                data(k, 1, counter) = (-(pow(nfp*i,2)+1)*sin(nfp*i*phi)*sin(phi) + 2*(i*nfp)*cos(nfp*i*phi)*cos(phi));
                counter++;
            }
            for (int i = 0; i < order+1; ++i) {
                data(k, 2, counter) = -pow(nfp*i, 2)*cos(nfp*i*phi);
                counter++;
            }
        }
        for (int i = 1; i < order+1; ++i) {
            data(k, 2, counter) = -pow(nfp*i, 2)*sin(nfp*i*phi);
            counter++;
        }
    }
    data *= 2*M_PI*2*M_PI;
}

template<class Array>
void CurveRZFourier<Array>::dgammadashdashdash_by_dcoeff_impl(Array& data) {
    for (int k = 0; k < numquadpoints; ++k) {
        double phi = 2 * M_PI * quadpoints[k];
        int counter = 0;
        for (int i = 0; i < order+1; ++i) {
            data(k, 0, counter) = (
                    +(3*pow(nfp*i, 2) + 1)*cos(nfp*i*phi)*sin(phi)
                    +(pow(nfp*i, 2) + 3)*(nfp*i)*sin(nfp*i*phi)*cos(phi)
                    );
            data(k, 1, counter) = (
                    +(pow(nfp*i, 2) + 3)*(nfp*i)*sin(nfp*i*phi)*sin(phi)
                    -(3*pow(nfp*i, 2) + 1)*cos(nfp*i*phi)*cos(phi)
                    );
            counter++;
        }
        if(!stellsym){
            for (int i = 1; i < order+1; ++i) {
                data(k, 0, counter) = (
                        -(pow(nfp*i,2)+3) * (nfp*i) * cos(nfp*i*phi)*cos(phi)
                        +(3*pow(nfp*i,2)+1) * sin(nfp*i*phi)*sin(phi)
                        );
                data(k, 1, counter) = (
                        -(pow(nfp*i,2)+3)*(nfp*i)*cos(nfp*i*phi)*sin(phi) 
                        -(3*pow(nfp*i,2)+1)*sin(nfp*i*phi)*cos(phi) 
                        );
                counter++;
            }
            for (int i = 0; i < order+1; ++i) {
                data(k, 2, counter) = pow(nfp*i, 3) * sin(nfp*i*phi);
                counter++;
            }
        }

        for (int i = 1; i < order+1; ++i) {
            data(k, 2, counter) = -pow(nfp*i, 3) * cos(nfp*i*phi);
            counter++;
        }
    }
    data *= 2*M_PI*2*M_PI*2*M_PI;
}

#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;
template class CurveRZFourier<Array>;
