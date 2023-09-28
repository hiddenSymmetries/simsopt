#include "curveplanarfourier.h"


template<class Array>
void CurvePlanarFourier<Array>::gamma_impl(Array& data, Array& quadpoints) {
    int numquadpoints = quadpoints.size();
    data *= 0;

    /* Converts q dofs to unit quaternion */
    Array q_norm = xt::zeros<double>({4});
    double s = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if(s != 0) {
        for (int i = 0; i < 4; ++i)
            q_norm[i] = q[i] / std::sqrt(s); 
    }
    else {
        q_norm[0] = 1;
    }


    for (int k = 0; k < numquadpoints; ++k) {
        double phi = 2 * M_PI * quadpoints[k];
        for (int i = 0; i < order+1; ++i) {
            data(k, 0) += rc[i] * cos(i*phi) * cos(phi);
            data(k, 1) += rc[i] * cos(i*phi) * sin(phi);
        }
    }
    for (int k = 0; k < numquadpoints; ++k) {
        double phi = 2 * M_PI * quadpoints[k];
        for (int i = 1; i < order+1; ++i) {
            data(k, 0) += rs[i-1] * sin(i*phi) * cos(phi);
            data(k, 1) += rs[i-1] * sin(i*phi) * sin(phi);
        }
    }
    for (int m = 0; m < numquadpoints; ++m) {
        double i;
        double j;
        double k;
        i = data(m, 0);
        j = data(m, 1);
        k = data(m, 2);

        data(m, 0) = (i - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k) + center[0];
        data(m, 1) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i + j - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k) + center[1];
        data(m, 2) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j + k - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k) + center[2];
    }
}

template<class Array>
void CurvePlanarFourier<Array>::gammadash_impl(Array& data) {
    data *= 0;

    /* Converts q dofs to unit quaternion */
    Array q_norm = xt::zeros<double>({4});
    double s = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if(s != 0) {
        for (int i = 0; i < 4; ++i)
            q_norm[i] = q[i] / std::sqrt(s); 
    }
    else {
        q_norm[0] = 1;
    }

    for (int k = 0; k < numquadpoints; ++k) {
        double phi = 2 * M_PI * quadpoints[k];
        for (int i = 0; i < order+1; ++i) {
            data(k, 0) += rc[i] * ( -(i) * sin(i*phi) * cos(phi) - cos(i*phi) * sin(phi));
            data(k, 1) += rc[i] * ( -(i) * sin(i*phi) * sin(phi) + cos(i*phi) * cos(phi));
        }
    }
    for (int k = 0; k < numquadpoints; ++k) {
        double phi = 2 * M_PI * quadpoints[k];
        for (int i = 1; i < order+1; ++i) {
            data(k, 0) += rs[i-1] * ( (i) * cos(i*phi) * cos(phi) - sin(i*phi) * sin(phi));
            data(k, 1) += rs[i-1] * ( (i) * cos(i*phi) * sin(phi) + sin(i*phi) * cos(phi));
        }
    }
    data *= (2*M_PI);
    for (int m = 0; m < numquadpoints; ++m) {
        Array i = xt::zeros<double>({1});
        i[0] = data(m, 0);
        Array j = xt::zeros<double>({1});
        j[0] = data(m, 1);
        Array k = xt::zeros<double>({1});
        k[0] = data(m, 2);

        data(m, 0) = (i[0] - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i[0] + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j[0] + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k[0]);
        data(m, 1) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i[0] + j[0] - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j[0] + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k[0]);
        data(m, 2) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i[0] + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j[0] + k[0] - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k[0]);
    }
    
}

template<class Array>
void CurvePlanarFourier<Array>::gammadashdash_impl(Array& data) {
    data *= 0;

    /* Converts q dofs to unit quaternion */
    Array q_norm = xt::zeros<double>({4});
    double s = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if(s != 0) {
        for (int i = 0; i < 4; ++i)
            q_norm[i] = q[i] / std::sqrt(s); 
    }
    else {
        q_norm[0] = 1;
    }

    for (int k = 0; k < numquadpoints; ++k) {
        double phi = 2 * M_PI * quadpoints[k];
        for (int i = 0; i < order+1; ++i) {
            data(k, 0) += rc[i] * (+2*(i)*sin(i*phi)*sin(phi)-(pow(i, 2)+1)*cos(i*phi)*cos(phi));
            data(k, 1) += rc[i] * (-2*(i)*sin(i*phi)*cos(phi)-(pow(i, 2)+1)*cos(i*phi)*sin(phi));
        }
        data(k, 2) = 0;
    }
    for (int k = 0; k < numquadpoints; ++k) {
        double phi = 2 * M_PI * quadpoints[k];
        for (int i = 1; i < order+1; ++i) {
            data(k, 0) += rs[i-1] * (-(pow(i,2)+1)*sin(i*phi)*cos(phi) - 2*(i)*cos(i*phi)*sin(phi));
            data(k, 1) += rs[i-1] * (-(pow(i,2)+1)*sin(i*phi)*sin(phi) + 2*(i)*cos(i*phi)*cos(phi));
        }
        data(k, 2) = 0;
    }
    data *= 2*M_PI*2*M_PI;
    for (int m = 0; m < numquadpoints; ++m) {
        Array i = xt::zeros<double>({1});
        i[0] = data(m, 0);
        Array j = xt::zeros<double>({1});
        j[0] = data(m, 1);
        Array k = xt::zeros<double>({1});
        k[0] = data(m, 2);

        data(m, 0) = (i[0] - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i[0] + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j[0] + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k[0]);
        data(m, 1) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i[0] + j[0] - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j[0] + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k[0]);
        data(m, 2) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i[0] + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j[0] + k[0] - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k[0]);

    }
}

template<class Array>
void CurvePlanarFourier<Array>::gammadashdashdash_impl(Array& data) {
    data *= 0;

    /* Converts q dofs to unit quaternion */
    Array q_norm = xt::zeros<double>({4});
    double s = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if(s != 0) {
        for (int i = 0; i < 4; ++i)
            q_norm[i] = q[i] / std::sqrt(s); 
    }
    else {
        q_norm[0] = 1;
    }

    for (int k = 0; k < numquadpoints; ++k) {
        double phi = 2 * M_PI * quadpoints[k];
        for (int i = 0; i < order+1; ++i) {
            data(k, 0) += rc[i]*(
                    +(3*pow(i, 2) + 1)*cos(i*phi)*sin(phi)
                    +(pow(i, 2) + 3)*(i)*sin(i*phi)*cos(phi)
                    );
            data(k, 1) += rc[i]*(
                    +(pow(i, 2) + 3)*(i)*sin(i*phi)*sin(phi)
                    -(3*pow(i, 2) + 1)*cos(i*phi)*cos(phi)
                    );
        }
    }
    for (int k = 0; k < numquadpoints; ++k) {
        double phi = 2 * M_PI * quadpoints[k];
        for (int i = 1; i < order+1; ++i) {
            data(k, 0) += rs[i-1]*(
                    -(pow(i,2)+3) * (i) * cos(i*phi)*cos(phi)
                    +(3*pow(i,2)+1) * sin(i*phi)*sin(phi)
                    );
            data(k, 1) += rs[i-1]*(
                    -(pow(i,2)+3)*(i)*cos(i*phi)*sin(phi) 
                    -(3*pow(i,2)+1)*sin(i*phi)*cos(phi) 
                    );
        }
    }
    data *= 2*M_PI*2*M_PI*2*M_PI;
    for (int m = 0; m < numquadpoints; ++m) {
        Array i = xt::zeros<double>({1});
        i[0] = data(m, 0);
        Array j = xt::zeros<double>({1});
        j[0] = data(m, 1);
        Array k = xt::zeros<double>({1});
        k[0] = data(m, 2);

        data(m, 0) = (i[0] - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i[0] + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j[0] + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k[0]);
        data(m, 1) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i[0] + j[0] - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j[0] + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k[0]);
        data(m, 2) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i[0] + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j[0] + k[0] - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k[0]);

    }
}

template<class Array>
void CurvePlanarFourier<Array>::dgamma_by_dcoeff_impl(Array& data) {
    data *= 0;
    
    /* Converts q dofs to unit quaternion */
    Array q_norm = xt::zeros<double>({4});
    double s = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if(s != 0) {
        for (int i = 0; i < 4; ++i)
            q_norm[i] = q[i] / std::sqrt(s); 
    }
    else {
        q_norm[0] = 1;
    }

    for (int m = 0; m < numquadpoints; ++m) {
        double phi = 2 * M_PI * quadpoints[m];
        int counter = 0;
        Array i = xt::zeros<double>({1});
        Array j = xt::zeros<double>({1});
        Array k = xt::zeros<double>({1});

        for (int n = 0; n < order+1; ++n) {
            i[0] = cos(n*phi) * cos(phi);
            j[0] = cos(n*phi) * sin(phi);
            k[0] = 0;

            data(m, 0, counter) = (i[0] - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i[0] + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j[0] + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k[0]);
            data(m, 1, counter) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i[0] + j[0] - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j[0] + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k[0]);
            data(m, 2, counter) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i[0] + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j[0] + k[0] - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k[0]);

            counter++;
        }
        
        for (int n = 1; n < order+1; ++n) {
            i[0] = sin(n*phi) * cos(phi);
            j[0] = sin(n*phi) * sin(phi);
            k[0] = 0;

            data(m, 0, counter) = (i[0] - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i[0] + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j[0] + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k[0]);
            data(m, 1, counter) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i[0] + j[0] - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j[0] + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k[0]);
            data(m, 2, counter) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i[0] + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j[0] + k[0] - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k[0]);

            counter++;
        }

        i[0] = 0;
        j[0] = 0;
        k[0] = 0;

        for (int n = 0; n < order+1; ++n) {
            i[0] += rc[n] * cos(n*phi) * cos(phi);
            j[0] += rc[n] * cos(n*phi) * sin(phi);
        }
        for (int n = 1; n < order+1; ++n) {
            i[0] += rs[n-1] * sin(n*phi) * cos(phi);
            j[0] += rs[n-1] * sin(n*phi) * sin(phi);
        }
        
        data(m, 0, counter) = (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j[0] * q_norm[3]) 
                            * (1 / std::sqrt(s) - q[0] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[2]) 
                            * (- q[1] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[1]) 
                            * (- q[2] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j[0] * q_norm[0]) 
                            * (- q[3] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 1, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i[0] * q_norm[3]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (1 / std::sqrt(s) - q[0] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[2]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * q_norm[1])
                            * (- q[1] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i[0] * q_norm[1]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (- q[2] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i[0] * q_norm[0]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * q_norm[3])
                            * (- q[3] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 2, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i[0] * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j[0] * q_norm[1])
                            * (1 / std::sqrt(s) - q[0] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[0])
                            * (- q[1] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 2 * i[0] * q_norm[0]
                            - 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[3])
                            * (- q[2] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (2 * i[0] * q_norm[1]
                            + 4 * i[0] * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j[0] * q_norm[2])
                            * (- q[3] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));
        counter++;

        data(m, 0, counter) = (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0] 
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j[0] * q_norm[3]) 
                            * (- q[0] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[2]) 
                            * (1 / std::sqrt(s) - q[1] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[1]) 
                            * (- q[2] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j[0] * q_norm[0])
                            * (- q[3] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 1, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i[0] * q_norm[3]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (- q[0] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[2]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * q_norm[1])
                            * (1 / std::sqrt(s) - q[1] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i[0] * q_norm[1]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (- q[2] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i[0] * q_norm[0]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * q_norm[3])
                            * (- q[3] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 2, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i[0] * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j[0] * q_norm[1])
                            * (- q[0] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[0])
                            * (1 / std::sqrt(s) - q[1] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 2 * i[0] * q_norm[0]
                            - 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[3])
                            * (- q[2] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (2 * i[0] * q_norm[1]
                            + 4 * i[0] * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j[0] * q_norm[2])
                            * (- q[3] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));
        counter++;

        data(m, 0, counter) = (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0] 
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j[0] * q_norm[3]) 
                            * (- q[0] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            + 
                            (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[2]) 
                            * (- q[1] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[1]) 
                            * (1 / std::sqrt(s) - q[2] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j[0] * q_norm[0]) 
                            * (- q[3] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 1, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i[0] * q_norm[3]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (- q[0] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[2]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * q_norm[1])
                            * (- q[1] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i[0] * q_norm[1]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (1 / std::sqrt(s) - q[2] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i[0] * q_norm[0]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * q_norm[3])
                            * (- q[3] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 2, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i[0] * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j[0] * q_norm[1])
                            * (- q[0] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[0])
                            * (- q[1] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 2 * i[0] * q_norm[0]
                            - 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[3])
                            * (1 / std::sqrt(s) - q[2] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (2 * i[0] * q_norm[1]
                            + 4 * i[0] * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j[0] * q_norm[2])
                            * (- q[3] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));
        counter++;

        data(m, 0, counter) = (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0] 
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j[0] * q_norm[3]) 
                            * (- q[0] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            + (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[2]) 
                            * (- q[1] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[1]) 
                            * (- q[2] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +                            
                            (- 4 * i[0] * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j[0] * q_norm[0]) 
                            * (1 / std::sqrt(s) - q[3] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 1, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i[0] * q_norm[3]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (- q[0] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[2]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * q_norm[1])
                            * (- q[1] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i[0] * q_norm[1]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (- q[2] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i[0] * q_norm[0]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * q_norm[3])
                            * (1 / std::sqrt(s) - q[3] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 2, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i[0] * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j[0] * q_norm[1])
                            * (- q[0] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[0])
                            * (- q[1] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 2 * i[0] * q_norm[0]
                            - 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[3])
                            * (- q[2] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (2 * i[0] * q_norm[1]
                            + 4 * i[0] * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j[0] * q_norm[2])
                            * (1 / std::sqrt(s) - q[3] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));
        counter++;


        for (int i = 0; i < 3; ++i) {
            data(m, 0, counter) = 0;
            data(m, 1, counter) = 0;
            data(m, 2, counter) = 0;
            data(m, i, counter) = 1;

            counter++;
        }
    }
}

template<class Array>
void CurvePlanarFourier<Array>::dgammadash_by_dcoeff_impl(Array& data) {
    data *= 0;

    /* Converts q dofs to unit quaternion */
    Array q_norm = xt::zeros<double>({4});
    double s = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if(s != 0) {
        for (int i = 0; i < 4; ++i)
            q_norm[i] = q[i] / std::sqrt(s); 
    }
    else {
        q_norm[0] = 1;
    }

    for (int m = 0; m < numquadpoints; ++m) {
        double phi = 2 * M_PI * quadpoints[m];
        int counter = 0;
        Array i = xt::zeros<double>({1});
        Array j = xt::zeros<double>({1});
        Array k = xt::zeros<double>({1});
        for (int n = 0; n < order+1; ++n) {
            i[0] = ( -(n) * sin(n*phi) * cos(phi) - cos(n*phi) * sin(phi));
            j[0] = ( -(n) * sin(n*phi) * sin(phi) + cos(n*phi) * cos(phi));
            k[0] = 0;

            i[0] *= (2*M_PI);
            j[0] *= (2*M_PI);

            data(m, 0, counter) = (i[0] - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i[0] + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j[0] + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k[0]);
            data(m, 1, counter) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i[0] + j[0] - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j[0] + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k[0]);
            data(m, 2, counter) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i[0] + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j[0] + k[0] - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k[0]);
            

            counter++;
        }
        
        for (int n = 1; n < order+1; ++n) {
            i[0] = ( (n) * cos(n*phi) * cos(phi) - sin(n*phi) * sin(phi));
            j[0] = ( (n) * cos(n*phi) * sin(phi) + sin(n*phi) * cos(phi));
            k[0] = 0;

            i[0] *= (2*M_PI);
            j[0] *= (2*M_PI);

            data(m, 0, counter) = (i[0] - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i[0] + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j[0] + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k[0]);
            data(m, 1, counter) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i[0] + j[0] - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j[0] + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k[0]);
            data(m, 2, counter) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i[0] + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j[0] + k[0] - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k[0]);

            counter++;
            
        }
        
        i[0] = 0;
        j[0] = 0;
        k[0] = 0;
        for (int n = 0; n < order+1; ++n) {
            i[0] += rc[n] * ( -(n) * sin(n*phi) * cos(phi) - cos(n*phi) * sin(phi)) * 2 * M_PI;
            j[0] += rc[n] * ( -(n) * sin(n*phi) * sin(phi) + cos(n*phi) * cos(phi)) * 2 * M_PI;
        }
        for (int n = 1; n < order+1; ++n) {
                i[0] += rs[n-1] * ( (n) * cos(n*phi) * cos(phi) - sin(n*phi) * sin(phi)) * 2 * M_PI;
                j[0] += rs[n-1] * ( (n) * cos(n*phi) * sin(phi) + sin(n*phi) * cos(phi)) * 2 * M_PI;
        }

        data(m, 0, counter) = (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j[0] * q_norm[3]) 
                            * (1 / std::sqrt(s) - q[0] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[2]) 
                            * (- q[1] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[1]) 
                            * (- q[2] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j[0] * q_norm[0]) 
                            * (- q[3] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 1, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i[0] * q_norm[3]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (1 / std::sqrt(s) - q[0] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[2]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * q_norm[1])
                            * (- q[1] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i[0] * q_norm[1]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (- q[2] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i[0] * q_norm[0]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * q_norm[3])
                            * (- q[3] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 2, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i[0] * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j[0] * q_norm[1])
                            * (1 / std::sqrt(s) - q[0] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[0])
                            * (- q[1] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 2 * i[0] * q_norm[0]
                            - 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[3])
                            * (- q[2] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (2 * i[0] * q_norm[1]
                            + 4 * i[0] * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j[0] * q_norm[2])
                            * (- q[3] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));
        counter++;

        data(m, 0, counter) = (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0] 
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j[0] * q_norm[3]) 
                            * (- q[0] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[2]) 
                            * (1 / std::sqrt(s) - q[1] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[1]) 
                            * (- q[2] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j[0] * q_norm[0])
                            * (- q[3] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 1, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i[0] * q_norm[3]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (- q[0] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[2]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * q_norm[1])
                            * (1 / std::sqrt(s) - q[1] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i[0] * q_norm[1]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (- q[2] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i[0] * q_norm[0]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * q_norm[3])
                            * (- q[3] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 2, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i[0] * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j[0] * q_norm[1])
                            * (- q[0] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[0])
                            * (1 / std::sqrt(s) - q[1] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 2 * i[0] * q_norm[0]
                            - 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[3])
                            * (- q[2] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (2 * i[0] * q_norm[1]
                            + 4 * i[0] * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j[0] * q_norm[2])
                            * (- q[3] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));
        counter++;

        data(m, 0, counter) = (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0] 
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j[0] * q_norm[3]) 
                            * (- q[0] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            + 
                            (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[2]) 
                            * (- q[1] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[1]) 
                            * (1 / std::sqrt(s) - q[2] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j[0] * q_norm[0]) 
                            * (- q[3] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 1, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i[0] * q_norm[3]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (- q[0] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[2]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * q_norm[1])
                            * (- q[1] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i[0] * q_norm[1]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (1 / std::sqrt(s) - q[2] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i[0] * q_norm[0]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * q_norm[3])
                            * (- q[3] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 2, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i[0] * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j[0] * q_norm[1])
                            * (- q[0] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[0])
                            * (- q[1] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 2 * i[0] * q_norm[0]
                            - 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[3])
                            * (1 / std::sqrt(s) - q[2] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (2 * i[0] * q_norm[1]
                            + 4 * i[0] * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j[0] * q_norm[2])
                            * (- q[3] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));
        counter++;

        data(m, 0, counter) = (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0] 
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j[0] * q_norm[3]) 
                            * (- q[0] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            + (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[2]) 
                            * (- q[1] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[1]) 
                            * (- q[2] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +                            
                            (- 4 * i[0] * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j[0] * q_norm[0]) 
                            * (1 / std::sqrt(s) - q[3] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 1, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i[0] * q_norm[3]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (- q[0] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[2]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * q_norm[1])
                            * (- q[1] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i[0] * q_norm[1]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (- q[2] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i[0] * q_norm[0]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * q_norm[3])
                            * (1 / std::sqrt(s) - q[3] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 2, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i[0] * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j[0] * q_norm[1])
                            * (- q[0] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[0])
                            * (- q[1] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 2 * i[0] * q_norm[0]
                            - 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[3])
                            * (- q[2] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (2 * i[0] * q_norm[1]
                            + 4 * i[0] * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j[0] * q_norm[2])
                            * (1 / std::sqrt(s) - q[3] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));
        counter++;

        for (int i = 0; i < 2; ++i) {
            data(m, 0, counter) = 0;
            data(m, 1, counter) = 0;
            data(m, 2, counter) = 0;
            
            counter++;
        }
        
    }
}

template<class Array>
void CurvePlanarFourier<Array>::dgammadashdash_by_dcoeff_impl(Array& data) {
    data *= 0;

    /* Converts q dofs to unit quaternion */
    Array q_norm = xt::zeros<double>({4});
    double s = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if(s != 0) {
        for (int i = 0; i < 4; ++i)
            q_norm[i] = q[i] / std::sqrt(s); 
    }
    else {
        q_norm[0] = 1;
    }

    for (int m = 0; m < numquadpoints; ++m) {
        double phi = 2 * M_PI * quadpoints[m];
        int counter = 0;
        Array i = xt::zeros<double>({1});
        Array j = xt::zeros<double>({1});
        Array k = xt::zeros<double>({1});
        for (int n = 0; n < order+1; ++n) {
            i[0] = (+2*(n)*sin(n*phi)*sin(phi)-(pow(n, 2)+1)*cos(n*phi)*cos(phi));
            j[0] = (-2*(n)*sin(n*phi)*cos(phi)-(pow(n, 2)+1)*cos(n*phi)*sin(phi));
            k[0] = 0;

            i[0] *= 2*M_PI*2*M_PI;
            j[0] *= 2*M_PI*2*M_PI;
            k[0] *= 2*M_PI*2*M_PI;

            data(m, 0, counter) = (i[0] - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i[0] + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j[0] + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k[0]);
            data(m, 1, counter) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i[0] + j[0] - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j[0] + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k[0]);
            data(m, 2, counter) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i[0] + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j[0] + k[0] - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k[0]);
            
            counter++;
        }
        
        for (int n = 1; n < order+1; ++n) {
            i[0] = (-(pow(n,2)+1)*sin(n*phi)*cos(phi) - 2*(n)*cos(n*phi)*sin(phi));
            j[0] = (-(pow(n,2)+1)*sin(n*phi)*sin(phi) + 2*(n)*cos(n*phi)*cos(phi));
            k[0] = 0;

            i[0] *= 2*M_PI*2*M_PI;
            j[0] *= 2*M_PI*2*M_PI;
            k[0] *= 2*M_PI*2*M_PI;

            data(m, 0, counter) = (i[0] - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i[0] + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j[0] + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k[0]);
            data(m, 1, counter) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i[0] + j[0] - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j[0] + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k[0]);
            data(m, 2, counter) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i[0] + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j[0] + k[0] - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k[0]);
            
            counter++;
        }
        
        i[0] = 0;
        j[0] = 0;
        k[0] = 0;
        for (int n = 0; n < order+1; ++n) {
            i[0] += rc[n] * (+2*(n)*sin(n*phi)*sin(phi)-(pow(n, 2)+1)*cos(n*phi)*cos(phi));
            j[0] += rc[n] * (-2*(n)*sin(n*phi)*cos(phi)-(pow(n, 2)+1)*cos(n*phi)*sin(phi));
        }
        for (int n = 1; n < order+1; ++n) {
            i[0] += rs[n-1] * (-(pow(n,2)+1)*sin(n*phi)*cos(phi) - 2*(n)*cos(n*phi)*sin(phi));
            j[0] += rs[n-1] * (-(pow(n,2)+1)*sin(n*phi)*sin(phi) + 2*(n)*cos(n*phi)*cos(phi));
        }
        i[0] *= 2*M_PI*2*M_PI;
        j[0] *= 2*M_PI*2*M_PI;
        k[0] *= 2*M_PI*2*M_PI;
        
        data(m, 0, counter) = (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j[0] * q_norm[3]) 
                            * (1 / std::sqrt(s) - q[0] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[2]) 
                            * (- q[1] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[1]) 
                            * (- q[2] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j[0] * q_norm[0]) 
                            * (- q[3] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 1, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i[0] * q_norm[3]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (1 / std::sqrt(s) - q[0] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[2]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * q_norm[1])
                            * (- q[1] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i[0] * q_norm[1]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (- q[2] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i[0] * q_norm[0]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * q_norm[3])
                            * (- q[3] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 2, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i[0] * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j[0] * q_norm[1])
                            * (1 / std::sqrt(s) - q[0] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[0])
                            * (- q[1] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 2 * i[0] * q_norm[0]
                            - 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[3])
                            * (- q[2] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (2 * i[0] * q_norm[1]
                            + 4 * i[0] * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j[0] * q_norm[2])
                            * (- q[3] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));
        counter++;

        data(m, 0, counter) = (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0] 
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j[0] * q_norm[3]) 
                            * (- q[0] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[2]) 
                            * (1 / std::sqrt(s) - q[1] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[1]) 
                            * (- q[2] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j[0] * q_norm[0])
                            * (- q[3] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 1, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i[0] * q_norm[3]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (- q[0] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[2]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * q_norm[1])
                            * (1 / std::sqrt(s) - q[1] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i[0] * q_norm[1]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (- q[2] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i[0] * q_norm[0]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * q_norm[3])
                            * (- q[3] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 2, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i[0] * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j[0] * q_norm[1])
                            * (- q[0] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[0])
                            * (1 / std::sqrt(s) - q[1] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 2 * i[0] * q_norm[0]
                            - 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[3])
                            * (- q[2] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (2 * i[0] * q_norm[1]
                            + 4 * i[0] * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j[0] * q_norm[2])
                            * (- q[3] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));
        counter++;

        data(m, 0, counter) = (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0] 
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j[0] * q_norm[3]) 
                            * (- q[0] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            + 
                            (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[2]) 
                            * (- q[1] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[1]) 
                            * (1 / std::sqrt(s) - q[2] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j[0] * q_norm[0]) 
                            * (- q[3] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 1, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i[0] * q_norm[3]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (- q[0] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[2]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * q_norm[1])
                            * (- q[1] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i[0] * q_norm[1]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (1 / std::sqrt(s) - q[2] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i[0] * q_norm[0]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * q_norm[3])
                            * (- q[3] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 2, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i[0] * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j[0] * q_norm[1])
                            * (- q[0] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[0])
                            * (- q[1] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 2 * i[0] * q_norm[0]
                            - 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[3])
                            * (1 / std::sqrt(s) - q[2] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (2 * i[0] * q_norm[1]
                            + 4 * i[0] * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j[0] * q_norm[2])
                            * (- q[3] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));
        counter++;

        data(m, 0, counter) = (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0] 
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j[0] * q_norm[3]) 
                            * (- q[0] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            + (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[2]) 
                            * (- q[1] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[1]) 
                            * (- q[2] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +                            
                            (- 4 * i[0] * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j[0] * q_norm[0]) 
                            * (1 / std::sqrt(s) - q[3] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 1, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i[0] * q_norm[3]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (- q[0] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[2]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * q_norm[1])
                            * (- q[1] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i[0] * q_norm[1]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (- q[2] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i[0] * q_norm[0]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * q_norm[3])
                            * (1 / std::sqrt(s) - q[3] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 2, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i[0] * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j[0] * q_norm[1])
                            * (- q[0] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[0])
                            * (- q[1] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 2 * i[0] * q_norm[0]
                            - 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[3])
                            * (- q[2] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (2 * i[0] * q_norm[1]
                            + 4 * i[0] * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j[0] * q_norm[2])
                            * (1 / std::sqrt(s) - q[3] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));
        counter++;

        for (int i = 0; i < 3; ++i) {
            data(m, 0, counter) = 0;
            data(m, 1, counter) = 0;
            data(m, 2, counter) = 0;
            
            counter++;
        }
    }
}

template<class Array>
void CurvePlanarFourier<Array>::dgammadashdashdash_by_dcoeff_impl(Array& data) {
    data *= 0;

    /* Converts q dofs to unit quaternion */
    Array q_norm = xt::zeros<double>({4});
    double s = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if(s != 0) {
        for (int i = 0; i < 4; ++i)
            q_norm[i] = q[i] / std::sqrt(s); 
    }
    else {
        q_norm[0] = 1;
    }

    for (int m = 0; m < numquadpoints; ++m) {
        double phi = 2 * M_PI * quadpoints[m];
        int counter = 0;
        Array i = xt::zeros<double>({1});
        Array j = xt::zeros<double>({1});
        Array k = xt::zeros<double>({1});
        for (int n = 0; n < order+1; ++n) {
            i[0] = (
                    +(3*pow(n, 2) + 1)*cos(n*phi)*sin(phi)
                    +(pow(n, 2) + 3)*(n)*sin(n*phi)*cos(phi)
                    );
            j[0] = (
                    +(pow(n, 2) + 3)*(n)*sin(n*phi)*sin(phi)
                    -(3*pow(n, 2) + 1)*cos(n*phi)*cos(phi)
                    );
            k[0] = 0;

            i[0] *= 2*M_PI*2*M_PI*2*M_PI;
            j[0] *= 2*M_PI*2*M_PI*2*M_PI;
            k[0] *= 2*M_PI*2*M_PI*2*M_PI;

            data(m, 0, counter) = (i[0] - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i[0] + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j[0] + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k[0]);
            data(m, 1, counter) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i[0] + j[0] - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j[0] + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k[0]);
            data(m, 2, counter) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i[0] + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j[0] + k[0] - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k[0]);

            counter++;
        }
        
        for (int n = 1; n < order+1; ++n) {
            i[0] = (
                    -(pow(n,2)+3) * (n) * cos(n*phi)*cos(phi)
                    +(3*pow(n,2)+1) * sin(n*phi)*sin(phi)
                    );
            j[0] = (
                    -(pow(n,2)+3)*(n)*cos(n*phi)*sin(phi) 
                    -(3*pow(n,2)+1)*sin(n*phi)*cos(phi) 
                    );
            k[0] = 0;

            i[0] *= 2*M_PI*2*M_PI*2*M_PI;
            j[0] *= 2*M_PI*2*M_PI*2*M_PI;
            k[0] *= 2*M_PI*2*M_PI*2*M_PI;

            data(m, 0, counter) = (i[0] - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i[0] + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j[0] + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k[0]);
            data(m, 1, counter) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i[0] + j[0] - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j[0] + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k[0]);
            data(m, 2, counter) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i[0] + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j[0] + k[0] - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k[0]);

            counter++;
        }
        
        i[0] = 0;
        j[0] = 0;
        k[0] = 0;
        for (int n = 0; n < order+1; ++n) {
            i[0] += rc[n]*(
                    +(3*pow(n, 2) + 1)*cos(n*phi)*sin(phi)
                    +(pow(n, 2) + 3)*(n)*sin(n*phi)*cos(phi)
                    );
            j[0] += rc[n]*(
                    +(pow(n, 2) + 3)*(n)*sin(n*phi)*sin(phi)
                    -(3*pow(n, 2) + 1)*cos(n*phi)*cos(phi)
                    );
        }
        for (int n = 1; n < order+1; ++n) {
            i[0] += rs[n-1]*(
                    -(pow(n,2)+3) * (n) * cos(n*phi)*cos(phi)
                    +(3*pow(n,2)+1) * sin(n*phi)*sin(phi)
                    );
            j[0] += rs[n-1]*(
                    -(pow(n,2)+3)*(n)*cos(n*phi)*sin(phi) 
                    -(3*pow(n,2)+1)*sin(n*phi)*cos(phi) 
                    );
        }
        i[0] *= 2*M_PI*2*M_PI*2*M_PI;
        j[0] *= 2*M_PI*2*M_PI*2*M_PI;
        k[0] *= 2*M_PI*2*M_PI*2*M_PI;
        
        data(m, 0, counter) = (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j[0] * q_norm[3]) 
                            * (1 / std::sqrt(s) - q[0] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[2]) 
                            * (- q[1] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[1]) 
                            * (- q[2] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j[0] * q_norm[0]) 
                            * (- q[3] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 1, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i[0] * q_norm[3]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (1 / std::sqrt(s) - q[0] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[2]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * q_norm[1])
                            * (- q[1] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i[0] * q_norm[1]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (- q[2] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i[0] * q_norm[0]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * q_norm[3])
                            * (- q[3] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 2, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i[0] * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j[0] * q_norm[1])
                            * (1 / std::sqrt(s) - q[0] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[0])
                            * (- q[1] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 2 * i[0] * q_norm[0]
                            - 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[3])
                            * (- q[2] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (2 * i[0] * q_norm[1]
                            + 4 * i[0] * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j[0] * q_norm[2])
                            * (- q[3] * q[0] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));
        counter++;

        data(m, 0, counter) = (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0] 
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j[0] * q_norm[3]) 
                            * (- q[0] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[2]) 
                            * (1 / std::sqrt(s) - q[1] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[1]) 
                            * (- q[2] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j[0] * q_norm[0])
                            * (- q[3] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 1, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i[0] * q_norm[3]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (- q[0] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[2]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * q_norm[1])
                            * (1 / std::sqrt(s) - q[1] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i[0] * q_norm[1]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (- q[2] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i[0] * q_norm[0]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * q_norm[3])
                            * (- q[3] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 2, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i[0] * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j[0] * q_norm[1])
                            * (- q[0] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[0])
                            * (1 / std::sqrt(s) - q[1] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 2 * i[0] * q_norm[0]
                            - 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[3])
                            * (- q[2] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (2 * i[0] * q_norm[1]
                            + 4 * i[0] * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j[0] * q_norm[2])
                            * (- q[3] * q[1] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));
        counter++;

        data(m, 0, counter) = (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0] 
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j[0] * q_norm[3]) 
                            * (- q[0] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            + 
                            (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[2]) 
                            * (- q[1] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[1]) 
                            * (1 / std::sqrt(s) - q[2] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j[0] * q_norm[0]) 
                            * (- q[3] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 1, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i[0] * q_norm[3]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (- q[0] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[2]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * q_norm[1])
                            * (- q[1] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i[0] * q_norm[1]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (1 / std::sqrt(s) - q[2] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i[0] * q_norm[0]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * q_norm[3])
                            * (- q[3] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 2, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i[0] * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j[0] * q_norm[1])
                            * (- q[0] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[0])
                            * (- q[1] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 2 * i[0] * q_norm[0]
                            - 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[3])
                            * (1 / std::sqrt(s) - q[2] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (2 * i[0] * q_norm[1]
                            + 4 * i[0] * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j[0] * q_norm[2])
                            * (- q[3] * q[2] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));
        counter++;

        data(m, 0, counter) = (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0] 
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j[0] * q_norm[3]) 
                            * (- q[0] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            + (4 * i[0] * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[2]) 
                            * (- q[1] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[1]) 
                            * (- q[2] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +                            
                            (- 4 * i[0] * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j[0] * q_norm[0]) 
                            * (1 / std::sqrt(s) - q[3] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 1, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i[0] * q_norm[3]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (- q[0] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[2]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j[0] * q_norm[1])
                            * (- q[1] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i[0] * q_norm[1]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (- q[2] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i[0] * q_norm[0]
                            + 4 * j[0] * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * q_norm[3])
                            * (1 / std::sqrt(s) - q[3] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));


        data(m, 2, counter) = (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i[0] * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j[0] * q_norm[1])
                            * (- q[0] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i[0] * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j[0] * q_norm[0])
                            * (- q[1] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (- 2 * i[0] * q_norm[0]
                            - 4 * i[0] * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j[0] * q_norm[3])
                            * (- q[2] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)))
                            +
                            (2 * i[0] * q_norm[1]
                            + 4 * i[0] * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j[0] * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j[0] * q_norm[2])
                            * (1 / std::sqrt(s) - q[3] * q[3] / (std::sqrt(s) * std::sqrt(s) * std::sqrt(s)));
        counter++;

        for (int i = 0; i < 3; ++i) {
            data(m, 0, counter) = 0;
            data(m, 1, counter) = 0;
            data(m, 2, counter) = 0;

            counter++;
        }
    }
}



#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;
template class CurvePlanarFourier<Array>;
