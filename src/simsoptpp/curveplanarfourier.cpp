#include "curveplanarfourier.h"

template<class Array>
double CurvePlanarFourier<Array>::inv_magnitude() {
    double s = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    if(s != 0) {
        return 1 / std::sqrt(s);
    } else {
        return 1;
    }

}

template<class Array>
void CurvePlanarFourier<Array>::gamma_impl(Array& data, Array& quadpoints) {
    double sinphi, cosphi, siniphi, cosiphi;
    int numquadpoints = quadpoints.size();
    data *= 0;

    Array q_norm = q * inv_magnitude();


    for (int k = 0; k < numquadpoints; ++k) {
        double phi = 2 * M_PI * quadpoints[k];
        sinphi = sin(phi);
        cosphi = cos(phi);
        data(k, 0) = rc[0] * cosphi;
        data(k, 1) = rc[0] * sinphi;
        for (int i = 1; i < order+1; ++i) {
            siniphi = sin(i * phi);
            cosiphi = cos(i * phi);
            data(k, 0) += (rc[i] * cosiphi + rs[i-1] * siniphi) * cosphi;
            data(k, 1) += (rc[i] * cosiphi + rs[i-1] * siniphi) * sinphi;
        }
    }
    for (int m = 0; m < numquadpoints; ++m) {
        double i = data(m, 0);
        double j = data(m, 1);
        double k = data(m, 2);

        /* Performs quaternion based rotation, see https://www.cis.upenn.edu/~cis5150/ws-book-Ib.pdf page 575, 576 for details regarding this rotation*/
        data(m, 0) = (i - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k) + center[0];
        data(m, 1) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i + j - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k) + center[1];
        data(m, 2) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j + k - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k) + center[2];
    }
}

template<class Array>
void CurvePlanarFourier<Array>::gammadash_impl(Array& data) {
    data *= 0;

    double inv_sqrt_s = inv_magnitude();
    Array q_norm = q * inv_sqrt_s;

    for (int k = 0; k < numquadpoints; ++k) {
        double phi = 2 * M_PI * quadpoints[k];
        for (int i = 0; i < order+1; ++i) {
            data(k, 0) += rc[i] * ( -(i) * sin(i*phi) * cos(phi) - cos(i*phi) * sin(phi));
            data(k, 1) += rc[i] * ( -(i) * sin(i*phi) * sin(phi) + cos(i*phi) * cos(phi));
        }
        for (int i = 1; i < order+1; ++i) {
            data(k, 0) += rs[i-1] * ( (i) * cos(i*phi) * cos(phi) - sin(i*phi) * sin(phi));
            data(k, 1) += rs[i-1] * ( (i) * cos(i*phi) * sin(phi) + sin(i*phi) * cos(phi));
        }
    }
        
    data *= (2*M_PI);
    for (int m = 0; m < numquadpoints; ++m) {
        double i = data(m, 0);
        double j = data(m, 1);
        double k = data(m, 2);

        /* Performs quaternion based rotation, see https://www.cis.upenn.edu/~cis5150/ws-book-Ib.pdf page 575, 576 for details regarding this rotation*/
        data(m, 0) = (i - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k);
        data(m, 1) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i + j - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k);
        data(m, 2) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j + k - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k);
    }
    
}

template<class Array>
void CurvePlanarFourier<Array>::gammadashdash_impl(Array& data) {
    data *= 0;

    Array q_norm = q * inv_magnitude();

    for (int k = 0; k < numquadpoints; ++k) {
        double phi = 2 * M_PI * quadpoints[k];
        for (int i = 0; i < order+1; ++i) {
            data(k, 0) += rc[i] * (+2*(i)*sin(i*phi)*sin(phi)-(i*i+1)*cos(i*phi)*cos(phi));
            data(k, 1) += rc[i] * (-2*(i)*sin(i*phi)*cos(phi)-(i*i+1)*cos(i*phi)*sin(phi));
        }
        for (int i = 1; i < order+1; ++i) {
            data(k, 0) += rs[i-1] * (-(i*i+1)*sin(i*phi)*cos(phi) - 2*(i)*cos(i*phi)*sin(phi));
            data(k, 1) += rs[i-1] * (-(i*i+1)*sin(i*phi)*sin(phi) + 2*(i)*cos(i*phi)*cos(phi));
        }
    }
    data *= 2*M_PI*2*M_PI;
    for (int m = 0; m < numquadpoints; ++m) {
        double i = data(m, 0);
        double j = data(m, 1);
        double k = data(m, 2);

        /* Performs quaternion based rotation*/
        data(m, 0) = (i - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k);
        data(m, 1) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i + j - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k);
        data(m, 2) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j + k - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k);

    }
}

template<class Array>
void CurvePlanarFourier<Array>::gammadashdashdash_impl(Array& data) {
    data *= 0;

    Array q_norm = q * inv_magnitude();

    for (int k = 0; k < numquadpoints; ++k) {
        double phi = 2 * M_PI * quadpoints[k];
        for (int i = 0; i < order+1; ++i) {
            data(k, 0) += rc[i]*(
                    +(3*i*i + 1)*cos(i*phi)*sin(phi)
                    +(i*i + 3)*(i)*sin(i*phi)*cos(phi)
                    );
            data(k, 1) += rc[i]*(
                    +(i*i + 3)*(i)*sin(i*phi)*sin(phi)
                    -(3*i*i + 1)*cos(i*phi)*cos(phi)
                    );
        }
        for (int i = 1; i < order+1; ++i) {
            data(k, 0) += rs[i-1]*(
                    -(i*i+3) * (i) * cos(i*phi)*cos(phi)
                    +(3*i*i+1) * sin(i*phi)*sin(phi)
                    );
            data(k, 1) += rs[i-1]*(
                    -(i*i+3)*(i)*cos(i*phi)*sin(phi) 
                    -(3*i*i+1)*sin(i*phi)*cos(phi) 
                    );
        }
    }
    data *= 2*M_PI*2*M_PI*2*M_PI;
    for (int m = 0; m < numquadpoints; ++m) {
        double i = data(m, 0);
        double j = data(m, 1);
        double k = data(m, 2);

        /* Performs quaternion based rotation*/
        data(m, 0) = (i - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k);
        data(m, 1) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i + j - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k);
        data(m, 2) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j + k - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k);

    }
}

template<class Array>
void CurvePlanarFourier<Array>::dgamma_by_dcoeff_impl(Array& data) {
    data *= 0;
    
    Array q_norm = q * inv_magnitude();

    for (int m = 0; m < numquadpoints; ++m) {
        double phi = 2 * M_PI * quadpoints[m];
        int counter = 0;
        double i;
        double j;
        double k;

        for (int n = 0; n < order+1; ++n) {
            i = cos(n*phi) * cos(phi);
            j = cos(n*phi) * sin(phi);
            k = 0;

            data(m, 0, counter) = (i - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k);
            data(m, 1, counter) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i + j - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k);
            data(m, 2, counter) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j + k - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k);

            counter++;
        }
        
        for (int n = 1; n < order+1; ++n) {
            i = sin(n*phi) * cos(phi);
            j = sin(n*phi) * sin(phi);
            k = 0;

            data(m, 0, counter) = (i - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k);
            data(m, 1, counter) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i + j - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k);
            data(m, 2, counter) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j + k - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k);

            counter++;
        }

        i = 0;
        j = 0;
        k = 0;

        for (int n = 0; n < order+1; ++n) {
            i += rc[n] * cos(n*phi) * cos(phi);
            j += rc[n] * cos(n*phi) * sin(phi);
        }
        for (int n = 1; n < order+1; ++n) {
            i += rs[n-1] * sin(n*phi) * cos(phi);
            j += rs[n-1] * sin(n*phi) * sin(phi);
        }
        double inv_sqrt_s = inv_magnitude();
        
        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j * q_norm[3]) 
                            * (inv_sqrt_s - q[0] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[2]) 
                            * (- q[1] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[1]) 
                            * (- q[2] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j * q_norm[0]) 
                            * (- q[3] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 1, counter) = (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i * q_norm[3]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (inv_sqrt_s - q[0] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[2]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * q_norm[1])
                            * (- q[1] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i * q_norm[1]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (- q[2] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i * q_norm[0]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j * q_norm[3])
                            * (- q[3] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 2, counter) = (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j * q_norm[1])
                            * (inv_sqrt_s - q[0] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[0])
                            * (- q[1] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 2 * i * q_norm[0]
                            - 4 * i * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[3])
                            * (- q[2] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (2 * i * q_norm[1]
                            + 4 * i * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j * q_norm[2])
                            * (- q[3] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);
        counter++;

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0] 
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j * q_norm[3]) 
                            * (- q[0] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[2]) 
                            * (inv_sqrt_s - q[1] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[1]) 
                            * (- q[2] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j * q_norm[0])
                            * (- q[3] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 1, counter) = (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i * q_norm[3]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (- q[0] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[2]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * q_norm[1])
                            * (inv_sqrt_s - q[1] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i * q_norm[1]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (- q[2] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i * q_norm[0]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j * q_norm[3])
                            * (- q[3] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 2, counter) = (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j * q_norm[1])
                            * (- q[0] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[0])
                            * (inv_sqrt_s - q[1] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 2 * i * q_norm[0]
                            - 4 * i * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[3])
                            * (- q[2] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (2 * i * q_norm[1]
                            + 4 * i * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j * q_norm[2])
                            * (- q[3] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);
        counter++;

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0] 
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j * q_norm[3]) 
                            * (- q[0] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            + 
                            (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[2]) 
                            * (- q[1] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[1]) 
                            * (inv_sqrt_s - q[2] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j * q_norm[0]) 
                            * (- q[3] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 1, counter) = (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i * q_norm[3]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (- q[0] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[2]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * q_norm[1])
                            * (- q[1] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i * q_norm[1]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (inv_sqrt_s - q[2] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i * q_norm[0]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j * q_norm[3])
                            * (- q[3] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 2, counter) = (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j * q_norm[1])
                            * (- q[0] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[0])
                            * (- q[1] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 2 * i * q_norm[0]
                            - 4 * i * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[3])
                            * (inv_sqrt_s - q[2] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (2 * i * q_norm[1]
                            + 4 * i * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j * q_norm[2])
                            * (- q[3] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);
        counter++;

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0] 
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j * q_norm[3]) 
                            * (- q[0] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            + (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[2]) 
                            * (- q[1] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[1]) 
                            * (- q[2] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +                            
                            (- 4 * i * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j * q_norm[0]) 
                            * (inv_sqrt_s - q[3] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 1, counter) = (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i * q_norm[3]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (- q[0] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[2]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * q_norm[1])
                            * (- q[1] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i * q_norm[1]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (- q[2] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i * q_norm[0]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j * q_norm[3])
                            * (inv_sqrt_s - q[3] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 2, counter) = (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j * q_norm[1])
                            * (- q[0] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[0])
                            * (- q[1] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 2 * i * q_norm[0]
                            - 4 * i * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[3])
                            * (- q[2] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (2 * i * q_norm[1]
                            + 4 * i * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j * q_norm[2])
                            * (inv_sqrt_s - q[3] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);
        counter++;


        data(m, 0, counter) = 0;
        data(m, 1, counter) = 0;
        data(m, 2, counter) = 0;
        for (int i = 0; i < 3; ++i) {
            data(m, i, counter) = 1;
            counter++;
        }
    }
}

template<class Array>
void CurvePlanarFourier<Array>::dgammadash_by_dcoeff_impl(Array& data) {
    data *= 0;

    Array q_norm = q * inv_magnitude();

    for (int m = 0; m < numquadpoints; ++m) {
        double phi = 2 * M_PI * quadpoints[m];
        int counter = 0;
        double i;
        double j;
        double k;
        for (int n = 0; n < order+1; ++n) {
            i = ( -(n) * sin(n*phi) * cos(phi) - cos(n*phi) * sin(phi));
            j = ( -(n) * sin(n*phi) * sin(phi) + cos(n*phi) * cos(phi));
            k = 0;

            i *= (2*M_PI);
            j *= (2*M_PI);

            data(m, 0, counter) = (i - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k);
            data(m, 1, counter) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i + j - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k);
            data(m, 2, counter) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j + k - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k);
            

            counter++;
        }
        
        for (int n = 1; n < order+1; ++n) {
            i = ( (n) * cos(n*phi) * cos(phi) - sin(n*phi) * sin(phi));
            j = ( (n) * cos(n*phi) * sin(phi) + sin(n*phi) * cos(phi));
            k = 0;

            i *= (2*M_PI);
            j *= (2*M_PI);

            data(m, 0, counter) = (i - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k);
            data(m, 1, counter) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i + j - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k);
            data(m, 2, counter) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j + k - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k);

            counter++;
            
        }
        
        i = 0;
        j = 0;
        k = 0;
        for (int n = 0; n < order+1; ++n) {
            i += rc[n] * ( -(n) * sin(n*phi) * cos(phi) - cos(n*phi) * sin(phi)) * 2 * M_PI;
            j += rc[n] * ( -(n) * sin(n*phi) * sin(phi) + cos(n*phi) * cos(phi)) * 2 * M_PI;
        }
        for (int n = 1; n < order+1; ++n) {
                i += rs[n-1] * ( (n) * cos(n*phi) * cos(phi) - sin(n*phi) * sin(phi)) * 2 * M_PI;
                j += rs[n-1] * ( (n) * cos(n*phi) * sin(phi) + sin(n*phi) * cos(phi)) * 2 * M_PI;
        }
        double inv_sqrt_s = inv_magnitude();

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j * q_norm[3]) 
                            * (inv_sqrt_s - q[0] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[2]) 
                            * (- q[1] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[1]) 
                            * (- q[2] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j * q_norm[0]) 
                            * (- q[3] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 1, counter) = (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i * q_norm[3]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (inv_sqrt_s - q[0] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[2]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * q_norm[1])
                            * (- q[1] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i * q_norm[1]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (- q[2] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i * q_norm[0]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j * q_norm[3])
                            * (- q[3] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 2, counter) = (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j * q_norm[1])
                            * (inv_sqrt_s - q[0] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[0])
                            * (- q[1] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 2 * i * q_norm[0]
                            - 4 * i * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[3])
                            * (- q[2] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (2 * i * q_norm[1]
                            + 4 * i * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j * q_norm[2])
                            * (- q[3] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);
        counter++;

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0] 
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j * q_norm[3]) 
                            * (- q[0] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[2]) 
                            * (inv_sqrt_s - q[1] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[1]) 
                            * (- q[2] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j * q_norm[0])
                            * (- q[3] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 1, counter) = (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i * q_norm[3]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (- q[0] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[2]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * q_norm[1])
                            * (inv_sqrt_s - q[1] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i * q_norm[1]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (- q[2] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i * q_norm[0]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j * q_norm[3])
                            * (- q[3] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 2, counter) = (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j * q_norm[1])
                            * (- q[0] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[0])
                            * (inv_sqrt_s - q[1] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 2 * i * q_norm[0]
                            - 4 * i * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[3])
                            * (- q[2] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (2 * i * q_norm[1]
                            + 4 * i * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j * q_norm[2])
                            * (- q[3] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);
        counter++;

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0] 
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j * q_norm[3]) 
                            * (- q[0] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            + 
                            (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[2]) 
                            * (- q[1] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[1]) 
                            * (inv_sqrt_s - q[2] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j * q_norm[0]) 
                            * (- q[3] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 1, counter) = (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i * q_norm[3]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (- q[0] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[2]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * q_norm[1])
                            * (- q[1] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i * q_norm[1]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (inv_sqrt_s - q[2] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i * q_norm[0]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j * q_norm[3])
                            * (- q[3] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 2, counter) = (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j * q_norm[1])
                            * (- q[0] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[0])
                            * (- q[1] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 2 * i * q_norm[0]
                            - 4 * i * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[3])
                            * (inv_sqrt_s - q[2] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (2 * i * q_norm[1]
                            + 4 * i * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j * q_norm[2])
                            * (- q[3] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);
        counter++;

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0] 
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j * q_norm[3]) 
                            * (- q[0] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            + (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[2]) 
                            * (- q[1] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[1]) 
                            * (- q[2] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +                            
                            (- 4 * i * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j * q_norm[0]) 
                            * (inv_sqrt_s - q[3] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 1, counter) = (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i * q_norm[3]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (- q[0] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[2]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * q_norm[1])
                            * (- q[1] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i * q_norm[1]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (- q[2] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i * q_norm[0]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j * q_norm[3])
                            * (inv_sqrt_s - q[3] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 2, counter) = (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j * q_norm[1])
                            * (- q[0] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[0])
                            * (- q[1] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 2 * i * q_norm[0]
                            - 4 * i * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[3])
                            * (- q[2] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (2 * i * q_norm[1]
                            + 4 * i * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j * q_norm[2])
                            * (inv_sqrt_s - q[3] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);
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

    Array q_norm = q * inv_magnitude();

    for (int m = 0; m < numquadpoints; ++m) {
        double phi = 2 * M_PI * quadpoints[m];
        int counter = 0;
        double i;
        double j;
        double k;
        for (int n = 0; n < order+1; ++n) {
            i = (+2*(n)*sin(n*phi)*sin(phi)-(n*n+1)*cos(n*phi)*cos(phi));
            j = (-2*(n)*sin(n*phi)*cos(phi)-(n*n+1)*cos(n*phi)*sin(phi));
            k = 0;

            i *= 2*M_PI*2*M_PI;
            j *= 2*M_PI*2*M_PI;
            k *= 2*M_PI*2*M_PI;

            data(m, 0, counter) = (i - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k);
            data(m, 1, counter) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i + j - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k);
            data(m, 2, counter) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j + k - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k);
            
            counter++;
        }
        
        for (int n = 1; n < order+1; ++n) {
            i = (-(n*n+1)*sin(n*phi)*cos(phi) - 2*(n)*cos(n*phi)*sin(phi));
            j = (-(n*n+1)*sin(n*phi)*sin(phi) + 2*(n)*cos(n*phi)*cos(phi));
            k = 0;

            i *= 2*M_PI*2*M_PI;
            j *= 2*M_PI*2*M_PI;
            k *= 2*M_PI*2*M_PI;

            data(m, 0, counter) = (i - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k);
            data(m, 1, counter) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i + j - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k);
            data(m, 2, counter) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j + k - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k);
            
            counter++;
        }
        
        i = 0;
        j = 0;
        k = 0;
        for (int n = 0; n < order+1; ++n) {
            i += rc[n] * (+2*(n)*sin(n*phi)*sin(phi)-(n*n+1)*cos(n*phi)*cos(phi));
            j += rc[n] * (-2*(n)*sin(n*phi)*cos(phi)-(n*n+1)*cos(n*phi)*sin(phi));
        }
        for (int n = 1; n < order+1; ++n) {
            i += rs[n-1] * (-(n*n+1)*sin(n*phi)*cos(phi) - 2*(n)*cos(n*phi)*sin(phi));
            j += rs[n-1] * (-(n*n+1)*sin(n*phi)*sin(phi) + 2*(n)*cos(n*phi)*cos(phi));
        }
        i *= 2*M_PI*2*M_PI;
        j *= 2*M_PI*2*M_PI;
        k *= 2*M_PI*2*M_PI;
        double inv_sqrt_s = inv_magnitude();
        
        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j * q_norm[3]) 
                            * (inv_sqrt_s - q[0] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[2]) 
                            * (- q[1] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[1]) 
                            * (- q[2] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j * q_norm[0]) 
                            * (- q[3] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 1, counter) = (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i * q_norm[3]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (inv_sqrt_s - q[0] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[2]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * q_norm[1])
                            * (- q[1] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i * q_norm[1]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (- q[2] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i * q_norm[0]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j * q_norm[3])
                            * (- q[3] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 2, counter) = (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j * q_norm[1])
                            * (inv_sqrt_s - q[0] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[0])
                            * (- q[1] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 2 * i * q_norm[0]
                            - 4 * i * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[3])
                            * (- q[2] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (2 * i * q_norm[1]
                            + 4 * i * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j * q_norm[2])
                            * (- q[3] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);
        counter++;

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0] 
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j * q_norm[3]) 
                            * (- q[0] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[2]) 
                            * (inv_sqrt_s - q[1] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[1]) 
                            * (- q[2] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j * q_norm[0])
                            * (- q[3] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 1, counter) = (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i * q_norm[3]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (- q[0] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[2]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * q_norm[1])
                            * (inv_sqrt_s - q[1] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i * q_norm[1]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (- q[2] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i * q_norm[0]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j * q_norm[3])
                            * (- q[3] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 2, counter) = (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j * q_norm[1])
                            * (- q[0] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[0])
                            * (inv_sqrt_s - q[1] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 2 * i * q_norm[0]
                            - 4 * i * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[3])
                            * (- q[2] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (2 * i * q_norm[1]
                            + 4 * i * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j * q_norm[2])
                            * (- q[3] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);
        counter++;

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0] 
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j * q_norm[3]) 
                            * (- q[0] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            + 
                            (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[2]) 
                            * (- q[1] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[1]) 
                            * (inv_sqrt_s - q[2] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j * q_norm[0]) 
                            * (- q[3] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 1, counter) = (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i * q_norm[3]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (- q[0] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[2]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * q_norm[1])
                            * (- q[1] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i * q_norm[1]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (inv_sqrt_s - q[2] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i * q_norm[0]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j * q_norm[3])
                            * (- q[3] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 2, counter) = (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j * q_norm[1])
                            * (- q[0] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[0])
                            * (- q[1] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 2 * i * q_norm[0]
                            - 4 * i * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[3])
                            * (inv_sqrt_s - q[2] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (2 * i * q_norm[1]
                            + 4 * i * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j * q_norm[2])
                            * (- q[3] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);
        counter++;

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0] 
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j * q_norm[3]) 
                            * (- q[0] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            + (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[2]) 
                            * (- q[1] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[1]) 
                            * (- q[2] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +                            
                            (- 4 * i * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j * q_norm[0]) 
                            * (inv_sqrt_s - q[3] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 1, counter) = (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i * q_norm[3]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (- q[0] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[2]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * q_norm[1])
                            * (- q[1] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i * q_norm[1]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (- q[2] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i * q_norm[0]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j * q_norm[3])
                            * (inv_sqrt_s - q[3] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 2, counter) = (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j * q_norm[1])
                            * (- q[0] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[0])
                            * (- q[1] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 2 * i * q_norm[0]
                            - 4 * i * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[3])
                            * (- q[2] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (2 * i * q_norm[1]
                            + 4 * i * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j * q_norm[2])
                            * (inv_sqrt_s - q[3] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);
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

    Array q_norm = q * inv_magnitude();

    for (int m = 0; m < numquadpoints; ++m) {
        double phi = 2 * M_PI * quadpoints[m];
        int counter = 0;
        double i;
        double j;
        double k;
        for (int n = 0; n < order+1; ++n) {
            i = (
                    +(3*n*n + 1)*cos(n*phi)*sin(phi)
                    +(n*n + 3)*(n)*sin(n*phi)*cos(phi)
                    );
            j = (
                    +(n*n + 3)*(n)*sin(n*phi)*sin(phi)
                    -(3*n*n + 1)*cos(n*phi)*cos(phi)
                    );
            k = 0;

            i *= 2*M_PI*2*M_PI*2*M_PI;
            j *= 2*M_PI*2*M_PI*2*M_PI;
            k *= 2*M_PI*2*M_PI*2*M_PI;

            data(m, 0, counter) = (i - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k);
            data(m, 1, counter) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i + j - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k);
            data(m, 2, counter) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j + k - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k);

            counter++;
        }
        
        for (int n = 1; n < order+1; ++n) {
            i = (
                    -(n*n+3) * (n) * cos(n*phi)*cos(phi)
                    +(3*n*n+1) * sin(n*phi)*sin(phi)
                    );
            j = (
                    -(n*n+3)*(n)*cos(n*phi)*sin(phi) 
                    -(3*n*n+1)*sin(n*phi)*cos(phi) 
                    );
            k = 0;

            i *= 2*M_PI*2*M_PI*2*M_PI;
            j *= 2*M_PI*2*M_PI*2*M_PI;
            k *= 2*M_PI*2*M_PI*2*M_PI;

            data(m, 0, counter) = (i - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k);
            data(m, 1, counter) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i + j - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k);
            data(m, 2, counter) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j + k - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k);

            counter++;
        }
        
        i = 0;
        j = 0;
        k = 0;
        for (int n = 0; n < order+1; ++n) {
            i += rc[n]*(
                    +(3*n*n + 1)*cos(n*phi)*sin(phi)
                    +(n*n + 3)*(n)*sin(n*phi)*cos(phi)
                    );
            j += rc[n]*(
                    +(n*n + 3)*(n)*sin(n*phi)*sin(phi)
                    -(3*n*n + 1)*cos(n*phi)*cos(phi)
                    );
        }
        for (int n = 1; n < order+1; ++n) {
            i += rs[n-1]*(
                    -(n*n+3) * (n) * cos(n*phi)*cos(phi)
                    +(3*n*n+1) * sin(n*phi)*sin(phi)
                    );
            j += rs[n-1]*(
                    -(n*n+3)*(n)*cos(n*phi)*sin(phi) 
                    -(3*n*n+1)*sin(n*phi)*cos(phi) 
                    );
        }
        i *= 2*M_PI*2*M_PI*2*M_PI;
        j *= 2*M_PI*2*M_PI*2*M_PI;
        k *= 2*M_PI*2*M_PI*2*M_PI;
        double inv_sqrt_s = inv_magnitude();
        
        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j * q_norm[3]) 
                            * (inv_sqrt_s - q[0] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[2]) 
                            * (- q[1] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[1]) 
                            * (- q[2] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j * q_norm[0]) 
                            * (- q[3] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 1, counter) = (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i * q_norm[3]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (inv_sqrt_s - q[0] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[2]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * q_norm[1])
                            * (- q[1] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i * q_norm[1]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (- q[2] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i * q_norm[0]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j * q_norm[3])
                            * (- q[3] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 2, counter) = (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j * q_norm[1])
                            * (inv_sqrt_s - q[0] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[0])
                            * (- q[1] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 2 * i * q_norm[0]
                            - 4 * i * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[3])
                            * (- q[2] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (2 * i * q_norm[1]
                            + 4 * i * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j * q_norm[2])
                            * (- q[3] * q[0] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);
        counter++;

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0] 
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j * q_norm[3]) 
                            * (- q[0] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[2]) 
                            * (inv_sqrt_s - q[1] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[1]) 
                            * (- q[2] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j * q_norm[0])
                            * (- q[3] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 1, counter) = (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i * q_norm[3]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (- q[0] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[2]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * q_norm[1])
                            * (inv_sqrt_s - q[1] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i * q_norm[1]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (- q[2] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i * q_norm[0]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j * q_norm[3])
                            * (- q[3] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 2, counter) = (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j * q_norm[1])
                            * (- q[0] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[0])
                            * (inv_sqrt_s - q[1] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 2 * i * q_norm[0]
                            - 4 * i * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[3])
                            * (- q[2] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (2 * i * q_norm[1]
                            + 4 * i * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j * q_norm[2])
                            * (- q[3] * q[1] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);
        counter++;

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0] 
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j * q_norm[3]) 
                            * (- q[0] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            + 
                            (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[2]) 
                            * (- q[1] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[1]) 
                            * (inv_sqrt_s - q[2] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j * q_norm[0]) 
                            * (- q[3] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 1, counter) = (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i * q_norm[3]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (- q[0] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[2]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * q_norm[1])
                            * (- q[1] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i * q_norm[1]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (inv_sqrt_s - q[2] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i * q_norm[0]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j * q_norm[3])
                            * (- q[3] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 2, counter) = (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j * q_norm[1])
                            * (- q[0] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[0])
                            * (- q[1] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 2 * i * q_norm[0]
                            - 4 * i * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[3])
                            * (inv_sqrt_s - q[2] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (2 * i * q_norm[1]
                            + 4 * i * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j * q_norm[2])
                            * (- q[3] * q[2] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);
        counter++;

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0] 
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0]
                            - 2 * j * q_norm[3]) 
                            * (- q[0] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            + (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[2]) 
                            * (- q[1] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[0] * q_norm[0] + q_norm[1] * q_norm[1]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[1]) 
                            * (- q[2] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +                            
                            (- 4 * i * (q_norm[1] * q_norm[1] + q_norm[0] * q_norm[0]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3]
                            - 2 * j * q_norm[0]) 
                            * (inv_sqrt_s - q[3] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 1, counter) = (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[0]
                            + 2 * i * q_norm[3]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0])
                            * (- q[0] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[2]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[1]
                            - 4 * j * q_norm[1])
                            * (- q[1] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[2]
                            + 2 * i * q_norm[1]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2])
                            * (- q[2] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * q_norm[3]
                            + 2 * i * q_norm[0]
                            + 4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[3]
                            - 4 * j * q_norm[3])
                            * (inv_sqrt_s - q[3] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);


        data(m, 2, counter) = (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[0]
                            - 2 * i * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[0]
                            + 2 * j * q_norm[1])
                            * (- q[0] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 4 * i * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * q_norm[1]
                            + 2 * i * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[1]
                            + 2 * j * q_norm[0])
                            * (- q[1] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (- 2 * i * q_norm[0]
                            - 4 * i * (q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[2]
                            + 2 * j * q_norm[3])
                            * (- q[2] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s)
                            +
                            (2 * i * q_norm[1]
                            + 4 * i * (q_norm[2] * q_norm[0] - q_norm[1] * q_norm[3]) * q_norm[3]
                            - 4 * j * (q_norm[1] * q_norm[0] + q_norm[2] * q_norm[3]) * q_norm[3]
                            + 2 * j * q_norm[2])
                            * (inv_sqrt_s - q[3] * q[3] * inv_sqrt_s * inv_sqrt_s * inv_sqrt_s);
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
