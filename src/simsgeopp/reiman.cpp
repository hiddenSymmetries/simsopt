#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "xtensor-python/pyarray.hpp"
typedef xt::pyarray<double> Array;
Array ReimanB(double& iota0, double& iota1, Array& k_theta, Array& epsilon, int& m0_symmetry, Array& points){
    int num_points = points.shape(0);
    int num_coeffs = k_theta.shape(0);
    Array B        = xt::zeros<double>({points.shape(0), points.shape(1)});
    double x,y,ZZ,RR,varphi,cosphi,sinphi,BR,BZ,Bphi;
    double R_axis = 1.0, theta, rmin, combo, combo1;
    for (int i = 0; i < num_points; ++i) {
        x      = points(i, 0);
        y      = points(i, 1);
        ZZ     = points(i, 2);
        RR     = sqrt(x*x+y*y);
        cosphi = x/RR;
        sinphi = y/RR;
        varphi = atan2(y,x);
        theta  = atan2(ZZ, RR - R_axis);
        rmin = sqrt(pow((RR-R_axis), 2.0) + pow(ZZ, 2.0));

        combo = iota0 + iota1*rmin*rmin;
        combo1 = 0.0;

        for (int ind=0; ind < num_coeffs; ++ind) {
            combo       -= k_theta[ind] * epsilon[ind] * pow(rmin, k_theta[ind] - 2) * cos(k_theta[ind]*theta - m0_symmetry*varphi);
            combo1      += k_theta[ind] * epsilon[ind] * pow(rmin, k_theta[ind] - 2) * sin(k_theta[ind]*theta - m0_symmetry*varphi);
        }

        BR   =  ( (RR - R_axis)/RR)*combo1 + (ZZ/RR)*combo ;
        BZ   = -( (RR - R_axis)/RR)*combo  + (ZZ/RR)*combo1;
        Bphi = -1.0;

        B(i,0) = BR*cosphi-Bphi*sinphi;
        B(i,1) = BR*sinphi+Bphi*cosphi;
        B(i,2) = BZ;
    }
    return B;
}

Array ReimandB(double& iota0, double& iota1, Array& k_theta, Array& epsilon, int& m0_symmetry, Array& points){
    int num_points = points.shape(0);
    int num_coeffs = k_theta.shape(0);
    Array dB       = xt::zeros<double>({points.shape(0), points.shape(1), points.shape(1)});
    double x,y,ZZ,RR,varphi,cosphi,sinphi,BR,BZ,Bphi,dRBR,dZBR,dphiBR,dRBZ,dZBZ,dphiBZ,dRBphi,dZBphi,dphiBphi;
    double R_axis = 1.0, theta, rmin, combo, combo1, dcombodR, dcombodZ, dcombodphi, dcombo1dR, dcombo1dZ, dcombo1dphi;
    for (int i = 0; i < num_points; ++i) {
        x      = points(i, 0);
        y      = points(i, 1);
        ZZ     = points(i, 2);
        RR     = sqrt(x*x+y*y);
        cosphi = x/RR;
        sinphi = y/RR;
        varphi = atan2(y,x);
        theta  = atan2(ZZ, RR - R_axis);
        rmin = sqrt(pow((RR-R_axis), 2.0) + pow(ZZ, 2.0));

        combo = iota0 + iota1*rmin*rmin;
        combo1 = 0.0;

        dcombodR = 2.0*iota1*(RR - R_axis);
        dcombodZ = 2.0*iota1*ZZ;
        dcombodphi  = 0.0;
        dcombo1dR   = 0.0;
        dcombo1dZ   = 0.0;
        dcombo1dphi = 0.0;

        for (int ind=0; ind < num_coeffs; ++ind) {
            combo       -= k_theta[ind] * epsilon[ind] * pow(rmin, k_theta[ind] - 2) * cos(k_theta[ind]*theta - m0_symmetry*varphi);
            combo1      += k_theta[ind] * epsilon[ind] * pow(rmin, k_theta[ind] - 2) * sin(k_theta[ind]*theta - m0_symmetry*varphi);
            dcombodR    -= k_theta[ind] * pow( rmin, k_theta[ind] - 4 ) * epsilon[ind] * ( k_theta[ind] * ZZ * sin( k_theta[ind]*theta - m0_symmetry*varphi )  +  (k_theta[ind] - 2) *  (RR - R_axis) * cos(k_theta[ind]*theta - m0_symmetry*varphi) ) ;
            dcombodZ    += pow( rmin, k_theta[ind] - 4 ) * epsilon[ind] * k_theta[ind] * ( k_theta[ind] * sin( k_theta[ind]*theta - m0_symmetry*varphi ) * ( RR - R_axis ) -  (k_theta[ind] - 2) * ZZ * cos(k_theta[ind]*theta - m0_symmetry * varphi) );
            dcombodphi  -= k_theta[ind] * epsilon[ind] * pow(rmin, k_theta[ind] - 2) * sin(k_theta[ind]*theta - m0_symmetry*varphi)*m0_symmetry;
            dcombo1dR   += k_theta[ind] * pow(rmin, k_theta[ind] - 4) * epsilon[ind] * ( - k_theta[ind]* ZZ * cos(k_theta[ind]*theta - m0_symmetry*varphi) + ( k_theta[ind] -2 ) * sin(k_theta[ind]*theta - m0_symmetry*varphi) * (RR - R_axis) ) ;
            dcombo1dZ   += k_theta[ind] * pow(rmin, k_theta[ind] - 4) * epsilon[ind] * ( k_theta[ind] * cos(k_theta[ind]*theta - m0_symmetry *varphi) * (RR - R_axis) +  (k_theta[ind] - 2) * sin(k_theta[ind]*theta - m0_symmetry*varphi) * ZZ ) ;
            dcombo1dphi -= k_theta[ind] * epsilon[ind] * pow(rmin, k_theta[ind] - 2) * cos(k_theta[ind]*theta - m0_symmetry*varphi)*m0_symmetry;
        }

        BR   =  ( (RR - R_axis)/RR)*combo1 + (ZZ/RR)*combo ;
        BZ   = -( (RR - R_axis)/RR)*combo  + (ZZ/RR)*combo1;
        Bphi = -1.0;

        dRBR     = ( - ZZ / pow( RR, 2.0) ) *  combo + (ZZ / RR) * dcombodR + combo1 * R_axis / pow(RR, 2.0) + dcombo1dR * (RR - R_axis) / RR;
        dZBR     = ( 1.0 / RR ) * combo + (ZZ / RR) * dcombodZ + dcombo1dZ * (RR - R_axis) / RR;
        dphiBR   = ( (RR - R_axis)/RR)*dcombo1dphi + (ZZ/RR)*dcombodphi;
        dRBZ     = ( - R_axis / pow( RR, 2.0) ) *  combo - ((RR - R_axis) / RR) * dcombodR - combo1 * ZZ / pow(RR, 2.0) + dcombo1dR * ZZ / RR;
        dZBZ     = - ((RR - R_axis) / RR) * dcombodZ + combo1 * ( 1.0 / RR ) + dcombo1dZ * ZZ / RR ;
        dphiBZ   = -( (RR - R_axis)/RR)*dcombodphi + (ZZ/RR)*dcombo1dphi;
        dRBphi   = 0.0;
        dZBphi   = 0.0;
        dphiBphi = 0.0;

        dB(i,0,0) = dRBR*cosphi*cosphi-(dphiBR-Bphi+dRBphi*RR)*cosphi*sinphi/RR+sinphi*sinphi*(dphiBphi+BR)/RR;
        dB(i,0,1) = sinphi*cosphi*(dRBR*RR-dphiBphi-BR)/RR+sinphi*sinphi*(Bphi-dphiBR)/RR+cosphi*cosphi*dRBphi;
        dB(i,0,2) = dRBZ*cosphi-dphiBZ*sinphi/RR;
        dB(i,1,0) = sinphi*cosphi*(dRBR*RR-dphiBphi-BR)/RR+cosphi*cosphi*(dphiBR-Bphi)/RR-sinphi*sinphi*dRBphi;
        dB(i,1,1) = dRBR*sinphi*sinphi+(dphiBR-Bphi+dRBphi*RR)*cosphi*sinphi/RR+cosphi*cosphi*(dphiBphi+BR)/RR;
        dB(i,1,2) = dRBZ*sinphi+dphiBZ*cosphi/RR;
        dB(i,2,0) = dZBR*cosphi-dZBphi*sinphi;
        dB(i,2,1) = dZBR*sinphi+dZBphi*cosphi;
        dB(i,2,2) = dZBZ;
    }
    return dB;
}