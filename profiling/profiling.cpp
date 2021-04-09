#define FORCE_IMPORT_ARRAY
#include "xtensor/xnpy.hpp"
#include "xtensor/xrandom.hpp"
#include "biot_savart.h"
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <stdint.h>
uint64_t rdtsc(){
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

void profile_biot_savart(int nsources, int ntargets, int nderivatives){ 
    xt::xarray<double> points         = xt::random::randn<double>({ntargets, 3});
    xt::xarray<double> gamma          = xt::random::randn<double>({nsources, 3});
    xt::xarray<double> dgamma_by_dphi = xt::random::randn<double>({nsources, 3});

    auto pointsx = vector_type(points.shape(0), 0);
    auto pointsy = vector_type(points.shape(0), 0);
    auto pointsz = vector_type(points.shape(0), 0);
    for (int i = 0; i < points.shape(0); ++i) {
        pointsx[i] = points(i, 0);
        pointsy[i] = points(i, 1);
        pointsz[i] = points(i, 2);
    }
    auto B = xt::xarray<double>::from_shape({points.shape(0), 3});
    auto dB_by_dX = xt::xarray<double>::from_shape({points.shape(0), 3, 3});
    auto d2B_by_dXdX = xt::xarray<double>::from_shape({points.shape(0), 3, 3, 3});
    int n = int(1e9/(nsources*ntargets));

    uint64_t tick = rdtsc();  // tick before
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; ++i) {
        if(nderivatives == 0)
            biot_savart_kernel<xt::xarray<double>, 0>(pointsx, pointsy, pointsz, gamma, dgamma_by_dphi, B, dB_by_dX, d2B_by_dXdX);
        else if(nderivatives == 1)
            biot_savart_kernel<xt::xarray<double>, 1>(pointsx, pointsy, pointsz, gamma, dgamma_by_dphi, B, dB_by_dX, d2B_by_dXdX);
        else
            biot_savart_kernel<xt::xarray<double>, 2>(pointsx, pointsy, pointsz, gamma, dgamma_by_dphi, B, dB_by_dX, d2B_by_dXdX);
        //if(i==0){
        //    std::cout << B(0, 0) << " " << B(8, 0) << std::endl;
        //    std::cout << dB_by_dX(0, 0, 0) << " " << dB_by_dX(8, 0, 0) << std::endl;
        //    std::cout << d2B_by_dXdX(0, 0, 0, 0) << " " << d2B_by_dXdX(8, 0, 0, 0) << std::endl;
        //}
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto clockcycles = rdtsc() - tick;
    double simdtime = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
    double interactions = points.shape(0) * gamma.shape(0) * n;
    std::cout << std::setw (10) << nsources*ntargets 
        << std::setw (13) << simdtime/n 
        << std::setw (19) << std::setprecision(5) << (interactions/(1e9 * simdtime/1000.)) << std::endl;
}


int main() {
    for(int nd=0; nd<3; nd++) {
        std::cout << "Number of derivatives: " << nd << std::endl;
        std::cout << "         N" << " Time (in ms)" << " Gigainteractions/s" << std::endl;
        for(int nst=100; nst<=10000; nst*=10)
            profile_biot_savart(nst, nst, nd);
    }
    return 0;
}
