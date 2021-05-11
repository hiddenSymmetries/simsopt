#include "dommaschk.h"
#include <math.h>

double alpha(int m,int l) {
	double y;
    if (l < 0) {
		y=0;
	}
	else {
		y = pow(-1.0,l)/(tgamma(m+l+1)*tgamma(l+1)*pow(2.0,2*l+m));
    }
	return y;
}

double alphas(int m,int l) {
	double y;
	y = (2*l+m)*alpha(m,l);
	return y;
}

double beta(int m,int l) {
	double y;
    if (l < 0 || l >=m) {
		y=0;
	}
	else {
		y = tgamma(m-l)/(tgamma(l+1)*pow(2.0,2*l-m+1));
    }
	return y;
}

double betas(int m,int l) {
	double y;
	y = (2*l-m)*beta(m,l);
	return y;
}

double gamma1(int m,int l) {
	double y;
    if (l <= 0) {
		y=0;
	}
	else {
        double sumN = 0.0;
        for(int i = 1; i <= l; ++i)
        {
            sumN += 1.0/(i) + 1.0/(m+i);
        }
		y = (alpha(m,l)/2)*sumN;
    }
	return y;
}

double gammas(int m,int l) {
	double y;
	y = (2*l+m)*gamma1(m,l);
	return y;
}

double Dmn(int m, int n, double R, double Z) {
	double sumD=0.0, y=0.0;
	int j, k;
	for (k=0; k<=floor(n/2); k++) {
		sumD = 0.0;
		for (j=0;j<k+1;j++) {
			sumD += -(alpha(m,j)*(alphas(m,k-m-j)*log(R)+gammas(m,k-m-j)-alpha(m,k-m-j))-gamma1(m,j)*alphas(m,k-m-j)+alpha(m,j)*betas(m,k-j))*pow(R,(2*j+m)) + alphas(m,k-j)*beta(m,j)*pow(R,2*j-m);
		}
		y += (pow(Z,n-2*k)/tgamma(n-2*k+1))*sumD;
	}
	return y;
}

double Nmn(int m, int n, double R, double Z) {
	double sumN = 0.0, y = 0.0;
	int j, k;
	for (k=0;k<=floor(n/2); k++) {
		sumN = 0.0;
		for (j=0;j<k+1;j++) {
            sumN += +(alpha(m,j)*(alpha(m,k-m-j)*log(R)+gamma1(m,k-m-j))-gamma1(m,j)*alpha(m,k-m-j)+alpha(m,j)*beta(m,k-j))*pow(R,(2*j+m)) - alpha(m,k-j)*beta(m,j)*pow(R,2*j-m);
		}
		y += (pow(Z,n-2*k)/tgamma(n-2*k+1))*sumN;
	}
	return y;
}

double dRDmn(int m, int n, double R, double Z) {
	double sumD= 0.0, y=0.0;
	int j, k;
	for (k=0; k<=floor(n/2); k++) {
		sumD = 0.0;
		for (j=0;j<k+1;j++) {
			sumD += -(alpha(m,j)*(alphas(m,k-m-j)*log(R)+gammas(m,k-m-j)-alpha(m,k-m-j))-gamma1(m,j)*alphas(m,k-m-j)+alpha(m,j)*betas(m,k-j))*pow(R,(2*j+m-1))*(2*j+m) + alphas(m,k-j)*beta(m,j)*pow(R,2*j-m-1)*(2*j-m);
		}
		y += (pow(Z,n-2*k)/tgamma(n-2*k+1))*sumD;
	}
	return y;
}

double dZDmn(int m, int n, double R, double Z) {
	double sumD= 0.0, y=0.0;
	int j, k;
	for (k=0;k<=floor(n/2); k++) {
		sumD = 0.0;
		for (j=0; j<k+1;j++) {
			sumD += -(alpha(m,j)*(alphas(m,k-m-j)*log(R)+gammas(m,k-m-j)-alpha(m,k-m-j))-gamma1(m,j)*alphas(m,k-m-j)+alpha(m,j)*betas(m,k-j))*pow(R,(2*j+m)) + alphas(m,k-j)*beta(m,j)*pow(R,2*j-m);
		}
        if (n-2*k==0){
            y += 0;
        }
        else {
		    y += ((n-2*k)*pow(Z,n-2*k-1)/tgamma(n-2*k+1))*sumD;
        }
	}
	return y;
}

double dRRDmn(int m, int n, double R, double Z) {
	double sumD= 0.0, y=0.0;
	int j, k;
	for (k=0;k<=floor(n/2);k++) {
		sumD = 0.0;
		for (j=0; j<k+1; j++) {
			sumD += -(alpha(m,j)*(alphas(m,k-m-j)*log(R)+gammas(m,k-m-j)-alpha(m,k-m-j))-gamma1(m,j)*alphas(m,k-m-j)+alpha(m,j)*betas(m,k-j))*pow(R,(2*j+m-2))*(2*j+m)*(2*j+m-1) + alphas(m,k-j)*beta(m,j)*pow(R,2*j-m-2)*(2*j-m)*(2*j-m-1);
		}
		y += (pow(Z,n-2*k)/tgamma(n-2*k+1))*sumD;
	}
	return y;
}

double dZZDmn(int m, int n, double R, double Z) {
	double sumD= 0.0, y=0.0;
	int j, k;
	for (k=0;k<=floor(n/2);k++) {
		sumD = 0.0;
		for (j=0; j<k+1; j++) {
			sumD += -(alpha(m,j)*(alphas(m,k-m-j)*log(R)+gammas(m,k-m-j)-alpha(m,k-m-j))-gamma1(m,j)*alphas(m,k-m-j)+alpha(m,j)*betas(m,k-j))*pow(R,(2*j+m)) + alphas(m,k-j)*beta(m,j)*pow(R,2*j-m);
		}
        if (n-2*k==0 || n-2*k-1==0){
            y += 0;
        }
        else {
		    y += ((n-2*k-1)*pow(Z,n-2*k-2)/tgamma(n-2*k))*sumD;
        }
	}
	return y;
}

double dRZDmn(int m, int n, double R, double Z) {
	double sumD= 0.0, y=0.0;
	int j, k;
	for (k=0;k<=floor(n/2);k++) {
		sumD = 0.0;
		for (j=0; j<k+1; j++) {
			sumD += -(alpha(m,j)*(alphas(m,k-m-j)*log(R)+gammas(m,k-m-j)-alpha(m,k-m-j))-gamma1(m,j)*alphas(m,k-m-j)+alpha(m,j)*betas(m,k-j))*pow(R,(2*j+m-1))*(2*j+m) + alphas(m,k-j)*beta(m,j)*pow(R,2*j-m-1)*(2*j-m);
		}
        if (n-2*k==0){
            y += 0;
        }
        else {
		    y += (pow(Z,n-2*k-1)/tgamma(n-2*k))*sumD;
        }
	}
	return y;
}

double dRNmn(int m, int n, double R, double Z) {
	double sumN= 0.0, y=0.0;
	int j, k;
	for (k=0;k<=floor(n/2); k++) {
		sumN = 0.0;
		for (j=0; j<k+1; j++) {
			sumN += +(alpha(m,j)*(alpha(m,k-m-j)*log(R)+gamma1(m,k-m-j))-gamma1(m,j)*alpha(m,k-m-j)+alpha(m,j)*beta(m,k-j))*pow(R,(2*j+m-1))*(2*j+m) - alpha(m,k-j)*beta(m,j)*pow(R,2*j-m-1)*(2*j-m);
		}	
	y += (pow(Z,n-2*k)/tgamma(n-2*k+1))*sumN;
	}
	return y;
}

double dZNmn(int m, int n, double R, double Z) {
	double sumN= 0.0, y=0.0;
	int j, k;
	for (k=0;k<=floor(n/2);k++) {
		sumN = 0.0;
		for (j=0; j<k+1; j++) {
			sumN += +(alpha(m,j)*(alpha(m,k-m-j)*log(R)+gamma1(m,k-m-j))-gamma1(m,j)*alpha(m,k-m-j)+alpha(m,j)*beta(m,k-j))*pow(R,(2*j+m)) - alpha(m,k-j)*beta(m,j)*pow(R,2*j-m);
		}
        if (n-2*k==0 || n<0) {
            y += 0;
        }
        else {
		    y += ((n-2*k)*pow(Z,n-2*k-1)/tgamma(n-2*k+1))*sumN;
        }
	}
	return y;
}

double dRRNmn(int m, int n, double R, double Z) {
	double sumN= 0.0, y=0.0;
	int j, k;
	for (k=0; k< n/2 + 1; k++) {
		sumN = 0.0;
		for (j=0; j< k+1; j++) {
			sumN += +(alpha(m,j)*(alpha(m,k-m-j)*log(R)+gamma1(m,k-m-j))-gamma1(m,j)*alpha(m,k-m-j)+alpha(m,j)*beta(m,k-j))*pow(R,(2*j+m-2))*(2*j+m)*(2*j+m-1) - alpha(m,k-j)*beta(m,j)*pow(R,2*j-m-2)*(2*j-m)*(2*j-m-1);
		}
		y += (pow(Z,n-2*k)/tgamma(n-2*k+1))*sumN;
	}
	return y;
}

double dZZNmn(int m, int n, double R, double Z) {
	double sumN= 0.0, y=0.0;
	int j, k;
	for (k=0; k< n/2; k++) {
		sumN = 0.0;
		for (j =0; j<k+1; j++) {
			sumN += +(alpha(m,j)*(alpha(m,k-m-j)*log(R)+gamma1(m,k-m-j))-gamma1(m,j)*alpha(m,k-m-j)+alpha(m,j)*beta(m,k-j))*pow(R,(2*j+m)) - alpha(m,k-j)*beta(m,j)*pow(R,2*j-m);
		}
        if (n-2*k==0 || n-2*k-1==0 || n<0) {
            y += 0;
        }
        else {
		    y += ((n-2*k-1)*pow(Z,n-2*k-2)/tgamma(n-2*k))*sumN;
        }
	}
	return y;
}

double dRZNmn(int m, int n, double R, double Z) {
	double sumN= 0.0, y=0.0;
	int j, k;
	for (k=0;k<=floor(n/2); k++) {
		sumN = 0.0;
		for (j=0; j< k+1; j++) {
			sumN += +(alpha(m,j)*(alpha(m,k-m-j)*log(R)+gamma1(m,k-m-j))-gamma1(m,j)*alpha(m,k-m-j)+alpha(m,j)*beta(m,k-j))*pow(R,(2*j+m-1))*(2*j+m) - alpha(m,k-j)*beta(m,j)*pow(R,2*j-m-1)*(2*j-m);
		}
        if (n-2*k==0 || n<0) {
            y += 0;
        }
        else {
		    y += ((n-2*k)*pow(Z,n-2*k-1)/tgamma(n-2*k+1))*sumN;
        }
	}
	return y;
}

double Phi(int m, int nn, double R, double Z, double phi,double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (nn%2 == 0) {
		a = d = 0;
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1; 
		d = coeff2; 
		b = c = 0;
	}
	y = (a*cos(m*phi) + b*sin(m*phi))*Dmn(m,nn,R,Z) + (c*cos(m*phi) + d*sin(m*phi))*Nmn(m,nn-1,R,Z);
	return y;
}

double BR(int m, int n, double R, double Z, double phi, double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (n%2 == 0) {
		a = d = 0;
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1;
		d = coeff2; 
		b = c = 0;
	}
	y = (a*cos(m*phi) + b*sin(m*phi))*dRDmn(m,n,R,Z) + (c*cos(m*phi) + d*sin(m*phi))*dRNmn(m,n-1,R,Z);
	return y;
}

double BZ(int m, int n, double R, double Z, double phi, double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (n%2 == 0) {
		a = d = 0;
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1;
		d = coeff2; 
		b = c = 0;
	}
	y = (a*cos(m*phi) + b*sin(m*phi))*dZDmn(m,n,R,Z) + (c*cos(m*phi) + d*sin(m*phi))*dZNmn(m,n-1,R,Z);
	return y;
}

double Bphi(int m, int n, double R, double Z, double phi, double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (n%2 == 0) {
		a = d = 0;
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1;
		d = coeff2; 
		b = c = 0;
	}
	y = m*(-a*sin(m*phi) + b*cos(m*phi))*Dmn(m,n,R,Z)/R + m*(-c*sin(m*phi) + d*cos(m*phi))*Nmn(m,n-1,R,Z)/R;
	return y;
}

double dphiBR(int m, int n, double R, double Z, double phi, double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (n%2 == 0) {
		a = d = 0;
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1;
		d = coeff2; 
		b = c = 0;
	}
	y = m*((-a*sin(m*phi) + b*cos(m*phi))*dRDmn(m,n,R,Z) + (-c*sin(m*phi) + d*cos(m*phi))*dRNmn(m,n-1,R,Z));
	return y;
}

double dphiBZ(int m, int n, double R, double Z, double phi, double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (n%2 == 0) {
		a = d = 0;
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1;
		d = coeff2; 
		b = c = 0;
	}
	y = m*((-a*sin(m*phi) + b*cos(m*phi))*dZDmn(m,n,R,Z) + (-c*sin(m*phi) + d*cos(m*phi))*dZNmn(m,n-1,R,Z));
	return y;
}

double dphiBphi(int m, int n, double R, double Z, double phi, double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (n%2 == 0) {
		a = d = 0;
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1;
		d = coeff2; 
		b = c = 0;
	}
	y = m*(m*(-a*cos(m*phi) - b*sin(m*phi))*Dmn(m,n,R,Z)/R + m*(-c*cos(m*phi) - d*sin(m*phi))*Nmn(m,n-1,R,Z)/R);
	return y;
}

double dRBR(int m, int n, double R, double Z, double phi, double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (n%2 == 0) {
		a = d = 0;
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1;
		d = coeff2; 
		b = c = 0;
	}
	y = (a*cos(m*phi) + b*sin(m*phi))*dRRDmn(m,n,R,Z) + (c*cos(m*phi) + d*sin(m*phi))*dRRNmn(m,n-1,R,Z);
	return y;
}

double dZBZ(int m, int n, double R, double Z, double phi, double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (n%2 == 0) {
		a = d = 0;
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1;
		d = coeff2; 
		b = c = 0;
	}
	y = (a*cos(m*phi) + b*sin(m*phi))*dZZDmn(m,n,R,Z) + (c*cos(m*phi) + d*sin(m*phi))*dZZNmn(m,n-1,R,Z);
	return y;
}

double dRBZ(int m, int n, double R, double Z, double phi, double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (n%2 == 0) {
		a = d = 0;
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1;
		d = coeff2; 
		b = c = 0;
	}
	y = (a*cos(m*phi) + b*sin(m*phi))*dRZDmn(m,n,R,Z) + (c*cos(m*phi) + d*sin(m*phi))*dRZNmn(m,n-1,R,Z);
	return y;
}

double dZBR(int m, int n, double R, double Z, double phi, double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (n%2 == 0) {
		a = d = 0; 
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1;
		d = coeff2; 
		b = c = 0;
	}
	y = (a*cos(m*phi) + b*sin(m*phi))*dRZDmn(m,n,R,Z) + (c*cos(m*phi) + d*sin(m*phi))*dRZNmn(m,n-1,R,Z);
	return y;
}

double dRBphi(int m, int n, double R, double Z, double phi, double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (n%2 == 0) {
		a = d = 0;
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1; 
		d = coeff2;
		b = c = 0;
	}
	y =  m*(-a*sin(m*phi) + b*cos(m*phi))*dRDmn(m,n,R,Z)/R + m*(-c*sin(m*phi) + d*cos(m*phi))*dRNmn(m,n-1,R,Z)/R - m*(-a*sin(m*phi) + b*cos(m*phi))*Dmn(m,n,R,Z)/pow(R,2) - m*(-c*sin(m*phi) + d*cos(m*phi))*Nmn(m,n-1,R,Z)/pow(R,2);
	return y;
}

double dZBphi(int m, int n, double R, double Z, double phi, double coeff1, double coeff2) {
	double y;
	double a, b, c, d;
	if (n%2 == 0){
		a = d = 0;
		b = coeff1;
		c = coeff2;
	}
	else {
		a = coeff1; 
		d = coeff2;
		b = c = 0;
	}
	y = m*(-a*sin(m*phi) + b*cos(m*phi))*dZDmn(m,n,R,Z)/R + m*(-c*sin(m*phi) + d*cos(m*phi))*dZNmn(m,n-1,R,Z)/R;
	return y;
}

#include "xtensor-python/pyarray.hpp"
typedef xt::pyarray<double> Array;

Array DommaschkB(Array& mArray, Array& nArray, Array& coeffs, Array& points){
    int num_points = points.shape(0);
    int num_coeffs = coeffs.shape(0);
    Array B        = xt::zeros<double>({coeffs.shape(0), points.shape(0), points.shape(1)});
    double x,y,z,R,phi,cosphi,sinphi,coeff1,coeff2;
    int m,n;
    for (int j=0; j < num_coeffs; ++j) {
        m      = mArray(j);
        n      = nArray(j);
        coeff1 = coeffs(j,0);
        coeff2 = coeffs(j,1);
        #pragma omp parallel for
        for (int i = 0; i < num_points; ++i) {
            x    = points(i, 0);
            y    = points(i, 1);
            z    = points(i, 2);
            R    = sqrt(x*x+y*y);
            phi  = atan2(y,x);
            cosphi = x/R;
            sinphi = y/R;
            B(j,i,0) = BR(m,n,R,z,phi,coeff1,coeff2)*cosphi-Bphi(m,n,R,z,phi,coeff1,coeff2)*sinphi;
            B(j,i,1) = BR(m,n,R,z,phi,coeff1,coeff2)*sinphi+Bphi(m,n,R,z,phi,coeff1,coeff2)*cosphi;
            B(j,i,2) = BZ(m,n,R,z,phi,coeff1,coeff2);
        }
    }
    return B;
}

Array DommaschkdB(Array& mArray, Array& nArray, Array& coeffs, Array& points){
    int num_points = points.shape(0);
    int num_coeffs = coeffs.shape(0);
    Array dB       = xt::zeros<double>({coeffs.shape(0), points.shape(0), points.shape(1), points.shape(1)});
    double x,y,z,R,phi,cosphi,sinphi,coeff1,coeff2;
    int m,n;
    for (int j=0; j < num_coeffs; ++j) {
        m      = mArray(j);
        n      = nArray(j);
        coeff1 = coeffs(j,0);
        coeff2 = coeffs(j,1);
        #pragma omp parallel for
        for (int i = 0; i < num_points; ++i) {
            x    = points(i, 0);
            y    = points(i, 1);
            z    = points(i, 2);
            R    = sqrt(x*x+y*y);
            phi  = atan2(y,x);
            cosphi = x/R;
            sinphi = y/R;
            dB(j,i,0,0) = dRBR(m,n,R,z,phi,coeff1,coeff2)*cosphi*cosphi-(dphiBR(m,n,R,z,phi,coeff1,coeff2)-Bphi(m,n,R,z,phi,coeff1,coeff2)+dRBphi(m,n,R,z,phi,coeff1,coeff2)*R)*cosphi*sinphi/R+sinphi*sinphi*(dphiBphi(m,n,R,z,phi,coeff1,coeff2)+BR(m,n,R,z,phi,coeff1,coeff2))/R;
            dB(j,i,0,1) = sinphi*cosphi*(dRBR(m,n,R,z,phi,coeff1,coeff2)*R-dphiBphi(m,n,R,z,phi,coeff1,coeff2)-BR(m,n,R,z,phi,coeff1,coeff2))/R+sinphi*sinphi*(Bphi(m,n,R,z,phi,coeff1,coeff2)-dphiBR(m,n,R,z,phi,coeff1,coeff2))/R+cosphi*cosphi*dRBphi(m,n,R,z,phi,coeff1,coeff2);
            dB(j,i,0,2) = dRBZ(m,n,R,z,phi,coeff1,coeff2)*cosphi-dphiBZ(m,n,R,z,phi,coeff1,coeff2)*sinphi/R;
            dB(j,i,1,0) = sinphi*cosphi*(dRBR(m,n,R,z,phi,coeff1,coeff2)*R-dphiBphi(m,n,R,z,phi,coeff1,coeff2)-BR(m,n,R,z,phi,coeff1,coeff2))/R+cosphi*cosphi*(dphiBR(m,n,R,z,phi,coeff1,coeff2)-Bphi(m,n,R,z,phi,coeff1,coeff2))/R-sinphi*sinphi*dRBphi(m,n,R,z,phi,coeff1,coeff2);
            dB(j,i,1,1) = dRBR(m,n,R,z,phi,coeff1,coeff2)*sinphi*sinphi+(dphiBR(m,n,R,z,phi,coeff1,coeff2)-Bphi(m,n,R,z,phi,coeff1,coeff2)+dRBphi(m,n,R,z,phi,coeff1,coeff2)*R)*cosphi*sinphi/R+cosphi*cosphi*(dphiBphi(m,n,R,z,phi,coeff1,coeff2)+BR(m,n,R,z,phi,coeff1,coeff2))/R;
            dB(j,i,1,2) = dRBZ(m,n,R,z,phi,coeff1,coeff2)*sinphi+dphiBZ(m,n,R,z,phi,coeff1,coeff2)*cosphi/R;
            dB(j,i,2,0) = dZBR(m,n,R,z,phi,coeff1,coeff2)*cosphi-dZBphi(m,n,R,z,phi,coeff1,coeff2)*sinphi;
            dB(j,i,2,1) = dZBR(m,n,R,z,phi,coeff1,coeff2)*sinphi+dZBphi(m,n,R,z,phi,coeff1,coeff2)*cosphi;
            dB(j,i,2,2) = dZBZ(m,n,R,z,phi,coeff1,coeff2);
        }
    }
    return dB;
}
