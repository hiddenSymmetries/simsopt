#include "biot_savart_impl.h"
#include "biot_savart_py.h"

void biot_savart(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis, vector<Array>& B, vector<Array>& dB_by_dX, vector<Array>& d2B_by_dXdX) {
    auto pointsx = AlignedPaddedVec(points.shape(0), 0);
    auto pointsy = AlignedPaddedVec(points.shape(0), 0);
    auto pointsz = AlignedPaddedVec(points.shape(0), 0);
    int num_points = points.shape(0);
    for (int i = 0; i < num_points; ++i) {
        pointsx[i] = points(i, 0);
        pointsy[i] = points(i, 1);
        pointsz[i] = points(i, 2);
    }
    int num_coils  = gammas.size();

    Array dummyjac = xt::zeros<double>({1, 1, 1});
    Array dummyhess = xt::zeros<double>({1, 1, 1, 1});

    int nderivs = 0;
    if(dB_by_dX.size() == num_coils) {
        nderivs = 1;
        if(d2B_by_dXdX.size() == num_coils) {
            nderivs = 2;
        }
    }

#pragma omp parallel for
    for(int i=0; i<num_coils; i++) {
        if(nderivs == 2)
            biot_savart_kernel<Array, 2>(pointsx, pointsy, pointsz, gammas[i], dgamma_by_dphis[i], B[i], dB_by_dX[i], d2B_by_dXdX[i]);
        else {
            if(nderivs == 1) 
                biot_savart_kernel<Array, 1>(pointsx, pointsy, pointsz, gammas[i], dgamma_by_dphis[i], B[i], dB_by_dX[i], dummyhess);
            else
                biot_savart_kernel<Array, 0>(pointsx, pointsy, pointsz, gammas[i], dgamma_by_dphis[i], B[i], dummyjac, dummyhess);
        }
    }
}

Array biot_savart_B(Array& points, vector<Array>& gammas, vector<Array>& dgamma_by_dphis, vector<double>& currents){
    auto dB_by_dXs = vector<Array>();
    auto d2B_by_dXdXs = vector<Array>();
    int num_coils = currents.size();
    auto Bs = vector<Array>(num_coils, Array());
    for (int i = 0; i < num_coils; ++i) {
        Bs[i] = xt::zeros<double>({points.shape(0), points.shape(1)});
    }
    biot_savart(points, gammas, dgamma_by_dphis, Bs, dB_by_dXs, d2B_by_dXdXs);
    Array B = xt::zeros<double>({points.shape(0), points.shape(1)});
    for (int i = 0; i < num_coils; ++i) {
        B += currents[i] * Bs[i];
    }
    return B;
}
