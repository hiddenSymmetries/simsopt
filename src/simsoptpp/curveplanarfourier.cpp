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
        double cosphi = cos(phi);
        double sinphi = sin(phi);
        data(k, 0) = rc[0] * cosphi;
        data(k, 1) = rc[0] * sinphi;
        for (int i = 1; i < order+1; ++i) {
            cosiphi = cos(i*phi);
            siniphi = sin(i*phi);
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

    double cosiphi, siniphi;
    for (int k = 0; k < numquadpoints; ++k) {
        double phi = 2 * M_PI * quadpoints[k];
        double cosphi = cos(phi);
        double sinphi = sin(phi);
        data(k, 0) = rc[0] * (-sinphi);
        data(k, 1) = rc[0] * (cosphi);
        for (int i = 1; i < order+1; ++i) {
            cosiphi = cos(i*phi);
            siniphi = sin(i*phi);
            data(k, 0) += rc[i] * ( -(i) * siniphi * cosphi - cosiphi * sinphi) 
                + rs[i-1] * ( (i) * cosiphi * cosphi - siniphi * sinphi);
            data(k, 1) += rc[i] * ( -(i) * siniphi * sinphi + cosiphi * cosphi)
                + rs[i-1] * ( (i) * cosiphi * sinphi + siniphi * cosphi);
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

    double cosiphi, siniphi;
    for (int k = 0; k < numquadpoints; ++k) {
        double phi = 2 * M_PI * quadpoints[k];
        double cosphi = cos(phi);
        double sinphi = sin(phi);
        data(k, 0) = rc[0] * (-cosphi);
        data(k, 1) = rc[0] * (-sinphi);
        for (int i = 1; i < order+1; ++i) {
            cosiphi = cos(i*phi);
            siniphi = sin(i*phi);
            data(k, 0) += rc[i] * (+2*(i)*siniphi*sinphi-(i*i+1)*cosiphi*cosphi)
                + rs[i-1] * (-(i*i+1)*siniphi*cosphi - 2*(i)*cosiphi*sinphi);
            data(k, 1) += rc[i] * (-2*(i)*siniphi*cosphi-(i*i+1)*cosiphi*sinphi)
                + rs[i-1] * (-(i*i+1)*siniphi*sinphi + 2*(i)*cosiphi*cosphi);
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

    double cosiphi, siniphi;
    for (int k = 0; k < numquadpoints; ++k) {
        double phi = 2 * M_PI * quadpoints[k];
        double cosphi = cos(phi);
        double sinphi = sin(phi);

        data(k, 0) = rc[0]*(+sinphi);
        data(k, 1) = rc[0]*(-cosphi);
        for (int i = 1; i < order+1; ++i) {
            cosiphi = cos(i*phi);
            siniphi = sin(i*phi);
            data(k, 0) += rc[i]*(
                    +(3*i*i + 1)*cosiphi*sinphi
                    +(i*i + 3)*(i)*siniphi*cosphi
                    ) + rs[i-1]*(
                    -(i*i+3) * (i) * cosiphi*cosphi
                    +(3*i*i+1) * siniphi*sinphi
                    );
            data(k, 1) += rc[i]*(
                    +(i*i + 3)*(i)*siniphi*sinphi
                    -(3*i*i + 1)*cosiphi*cosphi
                    ) + rs[i-1]*(
                    -(i*i+3)*(i)*cosiphi*sinphi 
                    -(3*i*i+1)*siniphi*cosphi 
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

    double cosnphi, sinnphi;
    for (int m = 0; m < numquadpoints; ++m) {
        double phi = 2 * M_PI * quadpoints[m];
        int counter = 0;
        double i;
        double j;
        double k;

        double cosphi = cos(phi);
        double sinphi = sin(phi);
        double cosnphi = 0;
        double sinnphi = 0;

        for (int n = 0; n < order+1; ++n) {
            cosnphi = cos(n * phi);
            i = cosnphi * cosphi;
            j = cosnphi * sinphi;
            k = 0;

            data(m, 0, counter) = (i - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k);
            data(m, 1, counter) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i + j - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k);
            data(m, 2, counter) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j + k - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k);

            counter++;
        }
        
        for (int n = 1; n < order+1; ++n) {
            sinnphi = sin(n * phi);
            i = sinnphi * cosphi;
            j = sinnphi * sinphi;
            k = 0;

            data(m, 0, counter) = (i - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k);
            data(m, 1, counter) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i + j - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k);
            data(m, 2, counter) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j + k - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k);

            counter++;
        }

        // i and j represent X0 and Y0 here before applying rotation
        i = rc[0] * cosphi;
        j = rc[0] * sinphi;
        k = 0;

        for (int n = 1; n < order+1; ++n) {
            cosnphi = cos(n * phi);
            sinnphi = sin(n * phi);
            i += (rc[n] * cosnphi + rs[n-1] * sinnphi) * cosphi;
            j += (rc[n] * cosnphi + rs[n-1] * sinnphi) * sinphi;
        }
        double inv_sqrt_s = inv_magnitude();
        
        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0]
                            - 4 * j * ((q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0] + 0.5 * q_norm[3]))
                            * inv_sqrt_s;

        data(m, 1, counter) = (4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0]
                            - 4 * i * ((q_norm[1] * q_norm[2] + q_norm[0] * q_norm[3]) * q_norm[0] - 0.5 * q_norm[3]))
                            * inv_sqrt_s;

        data(m, 2, counter) = (- 4 * i * ((q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[0] + 0.5 * q_norm[2])
                            - 4 * j * ((q_norm[2] * q_norm[3] + q_norm[0] * q_norm[1]) * q_norm[0] - 0.5 * q_norm[1]))
                            * inv_sqrt_s;
        counter++;

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                    - 4 * j * ((q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1] - 0.5 * q_norm[2]))
                    * inv_sqrt_s;

        data(m, 1, counter) = (4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3] - 1.0) * q_norm[1]
                            - 4 * i * ((q_norm[1] * q_norm[2] + q_norm[0] * q_norm[3]) * q_norm[1] - 0.5 * q_norm[2]))
                            * inv_sqrt_s;

        data(m, 2, counter) = (- 4 * i * ((q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[1] - 0.5 * q_norm[3])
                            - 4 * j * ((q_norm[2] * q_norm[3] + q_norm[0] * q_norm[1]) * q_norm[1] - 0.5 * q_norm[0]))
                            * inv_sqrt_s;

        counter++;

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3] - 1.0) * q_norm[2]
            - 4 * j * ((q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2] - 0.5 * q_norm[1]))
            * inv_sqrt_s;

        data(m, 1, counter) = (4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2]
                            - 4 * i * ((q_norm[1] * q_norm[2] + q_norm[0] * q_norm[3]) * q_norm[2] - 0.5 * q_norm[1]))
                            * inv_sqrt_s;

        data(m, 2, counter) = (- 4 * i * ((q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2] + 0.5 * q_norm[0])
                            - 4 * j * ((q_norm[2] * q_norm[3] + q_norm[0] * q_norm[1]) * q_norm[2] - 0.5 * q_norm[3]))
                            * inv_sqrt_s;

        counter++;

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3] - 1.0) * q_norm[3]
            - 4 * j * ((q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3] + 0.5 * q_norm[0]))
            * inv_sqrt_s;

        data(m, 1, counter) = (4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3] - 1.0) * q_norm[3]
                            - 4 * i * ((q_norm[1] * q_norm[2] + q_norm[0] * q_norm[3]) * q_norm[3] - 0.5 * q_norm[0]))
                            * inv_sqrt_s;

        data(m, 2, counter) = (- 4 * i * ((q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[3] - 0.5 * q_norm[1])
                            - 4 * j * ((q_norm[2] * q_norm[3] + q_norm[0] * q_norm[1]) * q_norm[3] - 0.5 * q_norm[2]))
                            * inv_sqrt_s;
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

    Array q_norm = q * inv_magnitude();

    double cosnphi, sinnphi;
    for (int m = 0; m < numquadpoints; ++m) {
        double phi = 2 * M_PI * quadpoints[m];
        double cosphi = cos(phi);
        double sinphi = sin(phi);
        int counter = 0;
        double i;
        double j;
        double k;
        for (int n = 0; n < order+1; ++n) {
            cosnphi = cos(n*phi);
            sinnphi = sin(n*phi);
            i = ( -(n) * sinnphi * cosphi - cosnphi * sinphi);
            j = ( -(n) * sinnphi * sinphi + cosnphi * cosphi);
            k = 0;

            i *= (2*M_PI);
            j *= (2*M_PI);

            data(m, 0, counter) = (i - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k);
            data(m, 1, counter) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i + j - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k);
            data(m, 2, counter) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j + k - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k);
            

            counter++;
        }
        
        for (int n = 1; n < order+1; ++n) {
            cosnphi = cos(n*phi);
            sinnphi = sin(n*phi);
            i = ( (n) * cosnphi * cosphi - sinnphi * sinphi);
            j = ( (n) * cosnphi * sinphi + sinnphi * cosphi);
            k = 0;

            i *= (2*M_PI);
            j *= (2*M_PI);

            data(m, 0, counter) = (i - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k);
            data(m, 1, counter) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i + j - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k);
            data(m, 2, counter) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j + k - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k);

            counter++;
            
        }
        
        i = rc[0] * (-sinphi);
        j = rc[0] * (cosphi);
        k = 0;
        for (int n = 1; n < order+1; ++n) {
            cosnphi = cos(n*phi);
            sinnphi = sin(n*phi);
            i += rc[n] * ( -(n) * sinnphi * cosphi - cosnphi * sinphi)
                + rs[n-1] * ( (n) * cosnphi * cosphi - sinnphi * sinphi);
            j += rc[n] * ( -(n) * sinnphi * sinphi + cosnphi * cosphi)
                + rs[n-1] * ( (n) * cosnphi * sinphi + sinnphi * cosphi);
        }
        i *= (2*M_PI);
        j *= (2*M_PI);
        double inv_sqrt_s = inv_magnitude();

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0]
                            - 4 * j * ((q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0] + 0.5 * q_norm[3]))
                            * inv_sqrt_s;

        data(m, 1, counter) = (4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0]
                            - 4 * i * ((q_norm[1] * q_norm[2] + q_norm[0] * q_norm[3]) * q_norm[0] - 0.5 * q_norm[3]))
                            * inv_sqrt_s;

        data(m, 2, counter) = (- 4 * i * ((q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[0] + 0.5 * q_norm[2])
                            - 4 * j * ((q_norm[2] * q_norm[3] + q_norm[0] * q_norm[1]) * q_norm[0] - 0.5 * q_norm[1]))
                            * inv_sqrt_s;
        counter++;

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                    - 4 * j * ((q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1] - 0.5 * q_norm[2]))
                    * inv_sqrt_s;

        data(m, 1, counter) = (4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3] - 1.0) * q_norm[1]
                            - 4 * i * ((q_norm[1] * q_norm[2] + q_norm[0] * q_norm[3]) * q_norm[1] - 0.5 * q_norm[2]))
                            * inv_sqrt_s;

        data(m, 2, counter) = (- 4 * i * ((q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[1] - 0.5 * q_norm[3])
                            - 4 * j * ((q_norm[2] * q_norm[3] + q_norm[0] * q_norm[1]) * q_norm[1] - 0.5 * q_norm[0]))
                            * inv_sqrt_s;

        counter++;

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3] - 1.0) * q_norm[2]
            - 4 * j * ((q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2] - 0.5 * q_norm[1]))
            * inv_sqrt_s;

        data(m, 1, counter) = (4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2]
                            - 4 * i * ((q_norm[1] * q_norm[2] + q_norm[0] * q_norm[3]) * q_norm[2] - 0.5 * q_norm[1]))
                            * inv_sqrt_s;

        data(m, 2, counter) = (- 4 * i * ((q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2] + 0.5 * q_norm[0])
                            - 4 * j * ((q_norm[2] * q_norm[3] + q_norm[0] * q_norm[1]) * q_norm[2] - 0.5 * q_norm[3]))
                            * inv_sqrt_s;

        counter++;

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3] - 1.0) * q_norm[3]
            - 4 * j * ((q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3] + 0.5 * q_norm[0]))
            * inv_sqrt_s;

        data(m, 1, counter) = (4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3] - 1.0) * q_norm[3]
                            - 4 * i * ((q_norm[1] * q_norm[2] + q_norm[0] * q_norm[3]) * q_norm[3] - 0.5 * q_norm[0]))
                            * inv_sqrt_s;

        data(m, 2, counter) = (- 4 * i * ((q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[3] - 0.5 * q_norm[1])
                            - 4 * j * ((q_norm[2] * q_norm[3] + q_norm[0] * q_norm[1]) * q_norm[3] - 0.5 * q_norm[2]))
                            * inv_sqrt_s;
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

    double cosnphi, sinnphi;
    for (int m = 0; m < numquadpoints; ++m) {
        double phi = 2 * M_PI * quadpoints[m];
        double cosphi = cos(phi);
        double sinphi = sin(phi);
        int counter = 0;
        double i;
        double j;
        double k;
        for (int n = 0; n < order+1; ++n) {
            cosnphi = cos(n*phi);
            sinnphi = sin(n*phi);
            i = (+2*(n)*sinnphi*sinphi-(n*n+1)*cosnphi*cosphi);
            j = (-2*(n)*sinnphi*cosphi-(n*n+1)*cosnphi*sinphi);
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
            cosnphi = cos(n*phi);
            sinnphi = sin(n*phi);
            i = (-(n*n+1)*sinnphi*cosphi - 2*(n)*cosnphi*sinphi);
            j = (-(n*n+1)*sinnphi*sinphi + 2*(n)*cosnphi*cosphi);
            k = 0;

            i *= 2*M_PI*2*M_PI;
            j *= 2*M_PI*2*M_PI;
            k *= 2*M_PI*2*M_PI;

            data(m, 0, counter) = (i - 2 * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * i + 2 * (q_norm[1] * q_norm[2] - q_norm[3] * q_norm[0]) * j + 2 * (q_norm[1] * q_norm[3] + q_norm[2] * q_norm[0]) * k);
            data(m, 1, counter) = (2 * (q_norm[1] * q_norm[2] + q_norm[3] * q_norm[0]) * i + j - 2 * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * j + 2 * (q_norm[2] * q_norm[3] - q_norm[1] * q_norm[0]) * k);
            data(m, 2, counter) = (2 * (q_norm[1] * q_norm[3] - q_norm[2] * q_norm[0]) * i + 2 * (q_norm[2] * q_norm[3] + q_norm[1] * q_norm[0]) * j + k - 2 * (q_norm[1] * q_norm[1] + q_norm[2] * q_norm[2]) * k);
            
            counter++;
        }
        
        i = rc[0] * (-cosphi);
        j = rc[0] * (-sinphi);
        k = 0;
        for (int n = 1; n < order+1; ++n) {
            cosnphi = cos(n*phi);
            sinnphi = sin(n*phi);
            i += rc[n] * (+2*(n)*sinnphi*sinphi-(n*n+1)*cosnphi*cosphi)
                + rs[n-1] * (-(n*n+1)*sinnphi*cosphi - 2*(n)*cosnphi*sinphi);
            j += rc[n] * (-2*(n)*sinnphi*cosphi-(n*n+1)*cosnphi*sinphi)
                + rs[n-1] * (-(n*n+1)*sinnphi*sinphi + 2*(n)*cosnphi*cosphi);
        }
        i *= 2*M_PI*2*M_PI;
        j *= 2*M_PI*2*M_PI;
        k *= 2*M_PI*2*M_PI;
        double inv_sqrt_s = inv_magnitude();
        
        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0]
                            - 4 * j * ((q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0] + 0.5 * q_norm[3]))
                            * inv_sqrt_s;

        data(m, 1, counter) = (4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0]
                            - 4 * i * ((q_norm[1] * q_norm[2] + q_norm[0] * q_norm[3]) * q_norm[0] - 0.5 * q_norm[3]))
                            * inv_sqrt_s;

        data(m, 2, counter) = (- 4 * i * ((q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[0] + 0.5 * q_norm[2])
                            - 4 * j * ((q_norm[2] * q_norm[3] + q_norm[0] * q_norm[1]) * q_norm[0] - 0.5 * q_norm[1]))
                            * inv_sqrt_s;
        counter++;

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                    - 4 * j * ((q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1] - 0.5 * q_norm[2]))
                    * inv_sqrt_s;

        data(m, 1, counter) = (4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3] - 1.0) * q_norm[1]
                            - 4 * i * ((q_norm[1] * q_norm[2] + q_norm[0] * q_norm[3]) * q_norm[1] - 0.5 * q_norm[2]))
                            * inv_sqrt_s;

        data(m, 2, counter) = (- 4 * i * ((q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[1] - 0.5 * q_norm[3])
                            - 4 * j * ((q_norm[2] * q_norm[3] + q_norm[0] * q_norm[1]) * q_norm[1] - 0.5 * q_norm[0]))
                            * inv_sqrt_s;

        counter++;

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3] - 1.0) * q_norm[2]
            - 4 * j * ((q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2] - 0.5 * q_norm[1]))
            * inv_sqrt_s;

        data(m, 1, counter) = (4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2]
                            - 4 * i * ((q_norm[1] * q_norm[2] + q_norm[0] * q_norm[3]) * q_norm[2] - 0.5 * q_norm[1]))
                            * inv_sqrt_s;

        data(m, 2, counter) = (- 4 * i * ((q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2] + 0.5 * q_norm[0])
                            - 4 * j * ((q_norm[2] * q_norm[3] + q_norm[0] * q_norm[1]) * q_norm[2] - 0.5 * q_norm[3]))
                            * inv_sqrt_s;

        counter++;

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3] - 1.0) * q_norm[3]
            - 4 * j * ((q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3] + 0.5 * q_norm[0]))
            * inv_sqrt_s;

        data(m, 1, counter) = (4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3] - 1.0) * q_norm[3]
                            - 4 * i * ((q_norm[1] * q_norm[2] + q_norm[0] * q_norm[3]) * q_norm[3] - 0.5 * q_norm[0]))
                            * inv_sqrt_s;

        data(m, 2, counter) = (- 4 * i * ((q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[3] - 0.5 * q_norm[1])
                            - 4 * j * ((q_norm[2] * q_norm[3] + q_norm[0] * q_norm[1]) * q_norm[3] - 0.5 * q_norm[2]))
                            * inv_sqrt_s;
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

    double cosnphi, sinnphi;
    for (int m = 0; m < numquadpoints; ++m) {
        double phi = 2 * M_PI * quadpoints[m];
        double cosphi = cos(phi);
        double sinphi = sin(phi);
        int counter = 0;
        double i;
        double j;
        double k;
        for (int n = 0; n < order+1; ++n) {
            cosnphi = cos(n*phi);
            sinnphi = sin(n*phi);
            i = (
                    +(3*n*n + 1)*cosnphi*sinphi
                    +(n*n + 3)*(n)*sinnphi*cosphi
                    );
            j = (
                    +(n*n + 3)*(n)*sinnphi*sinphi
                    -(3*n*n + 1)*cosnphi*cosphi
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
            cosnphi = cos(n*phi);
            sinnphi = sin(n*phi);
            i = (
                    -(n*n+3) * (n) * cosnphi*cosphi
                    +(3*n*n+1) * sinnphi*sinphi
                    );
            j = (
                    -(n*n+3)*(n)*cosnphi*sinphi 
                    -(3*n*n+1)*sinnphi*cosphi 
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
        
        i = rc[0]*(sinphi);
        j = rc[0]*(-cosphi);
        k = 0;
        for (int n = 1; n < order+1; ++n) {
            cosnphi = cos(n*phi);
            sinnphi = sin(n*phi);
            i += rc[n]*(
                    +(3*n*n + 1)*cosnphi*sinphi
                    +(n*n + 3)*(n)*sinnphi*cosphi
                    ) + rs[n-1]*(
                    -(n*n+3) * (n) * cosnphi*cosphi
                    +(3*n*n+1) * sinnphi*sinphi
                    );
            j += rc[n]*(
                    +(n*n + 3)*(n)*sinnphi*sinphi
                    -(3*n*n + 1)*cosnphi*cosphi
                    ) + rs[n-1]*(
                    -(n*n+3)*(n)*cosnphi*sinphi 
                    -(3*n*n+1)*sinnphi*cosphi 
                    );
        }
        i *= 2*M_PI*2*M_PI*2*M_PI;
        j *= 2*M_PI*2*M_PI*2*M_PI;
        k *= 2*M_PI*2*M_PI*2*M_PI;
        double inv_sqrt_s = inv_magnitude();
        
        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[0]
                            - 4 * j * ((q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[0] + 0.5 * q_norm[3]))
                            * inv_sqrt_s;

        data(m, 1, counter) = (4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[0]
                            - 4 * i * ((q_norm[1] * q_norm[2] + q_norm[0] * q_norm[3]) * q_norm[0] - 0.5 * q_norm[3]))
                            * inv_sqrt_s;

        data(m, 2, counter) = (- 4 * i * ((q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[0] + 0.5 * q_norm[2])
                            - 4 * j * ((q_norm[2] * q_norm[3] + q_norm[0] * q_norm[1]) * q_norm[0] - 0.5 * q_norm[1]))
                            * inv_sqrt_s;
        counter++;

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3]) * q_norm[1]
                    - 4 * j * ((q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[1] - 0.5 * q_norm[2]))
                    * inv_sqrt_s;

        data(m, 1, counter) = (4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3] - 1.0) * q_norm[1]
                            - 4 * i * ((q_norm[1] * q_norm[2] + q_norm[0] * q_norm[3]) * q_norm[1] - 0.5 * q_norm[2]))
                            * inv_sqrt_s;

        data(m, 2, counter) = (- 4 * i * ((q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[1] - 0.5 * q_norm[3])
                            - 4 * j * ((q_norm[2] * q_norm[3] + q_norm[0] * q_norm[1]) * q_norm[1] - 0.5 * q_norm[0]))
                            * inv_sqrt_s;

        counter++;

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3] - 1.0) * q_norm[2]
            - 4 * j * ((q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[2] - 0.5 * q_norm[1]))
            * inv_sqrt_s;

        data(m, 1, counter) = (4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3]) * q_norm[2]
                            - 4 * i * ((q_norm[1] * q_norm[2] + q_norm[0] * q_norm[3]) * q_norm[2] - 0.5 * q_norm[1]))
                            * inv_sqrt_s;

        data(m, 2, counter) = (- 4 * i * ((q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[2] + 0.5 * q_norm[0])
                            - 4 * j * ((q_norm[2] * q_norm[3] + q_norm[0] * q_norm[1]) * q_norm[2] - 0.5 * q_norm[3]))
                            * inv_sqrt_s;

        counter++;

        data(m, 0, counter) = (4 * i * (q_norm[2] * q_norm[2] + q_norm[3] * q_norm[3] - 1.0) * q_norm[3]
            - 4 * j * ((q_norm[1] * q_norm[2] - q_norm[0] * q_norm[3]) * q_norm[3] + 0.5 * q_norm[0]))
            * inv_sqrt_s;

        data(m, 1, counter) = (4 * j * (q_norm[1] * q_norm[1] + q_norm[3] * q_norm[3] - 1.0) * q_norm[3]
                            - 4 * i * ((q_norm[1] * q_norm[2] + q_norm[0] * q_norm[3]) * q_norm[3] - 0.5 * q_norm[0]))
                            * inv_sqrt_s;

        data(m, 2, counter) = (- 4 * i * ((q_norm[1] * q_norm[3] - q_norm[0] * q_norm[2]) * q_norm[3] - 0.5 * q_norm[1])
                            - 4 * j * ((q_norm[2] * q_norm[3] + q_norm[0] * q_norm[1]) * q_norm[3] - 0.5 * q_norm[2]))
                            * inv_sqrt_s;
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
