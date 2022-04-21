#include "dipole_field.h"

// Calculate the B field at a set of evaluation points from N dipoles
Array dipole_field_B(Array& points, Array& m_points, Array& m) {
    int num_points = points.shape(0);
    int num_dipoles = m_points.shape(0);
    Array B = xt::zeros<double>({points.shape(0), points.shape(1)});
    double x, y, z, mx, my, mz, mpx, mpy, mpz, rx, ry, rz, rmag, rdotm;
//#pragma omp parallel for private(x, y, z, mx, my, mz, mpx, mpy, mpz, rx, ry, rz, rmag, rdotm)
    for (int i = 0; i < num_points; ++i) {
        x = points(i, 0);
        y = points(i, 1);
        z = points(i, 2);
	for (int j = 0; j < num_dipoles; ++j) {
            mpx = m_points(j, 0);
            mpy = m_points(j, 1);
            mpz = m_points(j, 2);
            mx = m(j, 0);
            my = m(j, 1);
            mz = m(j, 2);
            rx = x - mpx;
	    ry = y - mpy;
	    rz = z - mpz;
	    rmag = sqrt(rx * rx + ry * ry + rz * rz);
            rdotm = rx * mx + ry * my + rz * mz;
	    B(i, 0) += 3.0 * rdotm * rx / pow(rmag, 5) - mx / pow(rmag, 3);
            B(i, 1) += 3.0 * rdotm * ry / pow(rmag, 5) - my / pow(rmag, 3);
            B(i, 2) += 3.0 * rdotm * rz / pow(rmag, 5) - mz / pow(rmag, 3);
	} 
    }
    return B * 1e-7;
}

// Calculate the gradient of the B field at a set of evaluation points from N dipoles
Array dipole_field_dB(Array& points, Array& m_points, Array& m) {
    int num_points = points.shape(0);
    int num_dipoles = m_points.shape(0);
    Array dB = xt::zeros<double>({points.shape(0), points.shape(1), points.shape(1)});
    double x, y, z, mx, my, mz, mpx, mpy, mpz, rx, ry, rz, rmag, rdotm, r5;
#pragma omp parallel for private(x, y, z, mx, my, mz, mpx, mpy, mpz, rx, ry, rz, rmag, rdotm, r5)
    for (int i = 0; i < num_points; ++i) {
        x = points(i, 0);
        y = points(i, 1);
        z = points(i, 2);
	for (int j = 0; j < num_dipoles; ++j) {
            mpx = m_points(j, 0);
            mpy = m_points(j, 1);
            mpz = m_points(j, 2);
            mx = m(j, 0);
            my = m(j, 1);
            mz = m(j, 2);
            rx = x - mpx;
	    ry = y - mpy;
	    rz = z - mpz;
	    rmag = sqrt(rx * rx + ry * ry + rz * rz);
            rdotm = rx * mx + ry * my + rz * mz;
            r5 = 3.0 / pow(rmag, 5);
	    dB(i, 0, 0) += r5 * ((2 * mx * rx + rdotm) - 5 * rdotm / pow(rmag, 2) * rx * rx); 
	    dB(i, 0, 1) += r5 * ((mx * ry + my * rx) - 5 * rdotm / pow(rmag, 2) * rx * ry);
	    dB(i, 0, 2) += r5 * ((mx * rz + mz * rx) - 5 * rdotm / pow(rmag, 2) * rx * rz);
	    dB(i, 1, 1) += r5 * ((2 * my * ry + rdotm) - 5 * rdotm / pow(rmag, 2) * ry * ry); 
	    dB(i, 1, 2) += r5 * ((my * rz + mz * ry) - 5 * rdotm / pow(rmag, 2) * ry * rz);
	    dB(i, 2, 2) += r5 * ((2 * mz * rz + rdotm) - 5 * rdotm / pow(rmag, 2) * rz * rz); 
	}
        dB(i, 2, 1) = dB(i, 1, 2);
        dB(i, 2, 0) = dB(i, 0, 2);
        dB(i, 1, 0) = dB(i, 0, 1);
    }
    return dB * 1e-7;
}
