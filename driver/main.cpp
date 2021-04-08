//#include <iostream>

//#include "fouriercurve.cpp"
//typedef xt::xarray<double> CppArray;

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


int main() {
    int N = 10000;
    xt::xarray<double> points         = xt::random::randn<double>({N, 3});
    xt::xarray<double> gamma          = xt::random::randn<double>({N, 3});
    xt::xarray<double> dgamma_by_dphi = xt::random::randn<double>({N, 3});
    int n = 10;

    auto pointsx = vector_type(points.shape(0), 0);
    auto pointsy = vector_type(points.shape(0), 0);
    auto pointsz = vector_type(points.shape(0), 0);
    for (int i = 0; i < points.shape(0); ++i) {
        pointsx[i] = points(i, 0);
        pointsy[i] = points(i, 1);
        pointsz[i] = points(i, 2);
    }
    uint64_t tick = rdtsc();  // tick before
    auto t1 = std::chrono::high_resolution_clock::now();
//#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        auto B = xt::xarray<double>::from_shape({points.shape(0), 3});
        xt::xarray<double> dB_by_dX = xt::xarray<double>::from_shape({points.shape(0), 3, 3});
        xt::xarray<double> d2B_by_dXdX = xt::xarray<double>::from_shape({points.shape(0), 3, 3, 3});
        biot_savart_kernel<xt::xarray<double>, 0>(pointsx, pointsy, pointsz, gamma, dgamma_by_dphi, B, dB_by_dX, d2B_by_dXdX);
        if(i==0){
            std::cout << std::setprecision(15);
            std::cout << B(0, 0) << " " << B(40, 2) << std::endl;
            std::cout << dB_by_dX(0, 0, 1) << " " << dB_by_dX(40, 2, 2) << std::endl;
            std::cout << d2B_by_dXdX(0, 0, 1, 2) << " " << d2B_by_dXdX(40, 2, 2, 1) << std::endl;
        }
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto clockcycles = rdtsc() - tick;
    double simdtime = std::chrono::duration_cast<std::chrono::milliseconds>( t2 - t1 ).count();
    std::cout << "Time: " << simdtime << " ms." << std::endl;
    double interactions = points.shape(0) * gamma.shape(0) * n;
    std::cout << "Gigainteractions per second:" << (interactions/(1e9 * simdtime/1000.)) << std::endl;
    std::cout << "Gigaclockcycles: "<< clockcycles/1e9 << std::endl;
    std::cout << "clockcycles per interaction:" << clockcycles/interactions << std::endl;


    return 0;
}
//int main() {
//    xt::xarray<double> points         = xt::load_npy<double>("points_200.npy");
//    xt::xarray<double> gamma          = xt::load_npy<double>("gamma_200.npy");
//    xt::xarray<double> dgamma_by_dphi = xt::load_npy<double>("dgamma_by_dphi_200.npy");
//    int n = 10000;

//    auto pointsx = vector_type(points.shape(0), 0);
//    auto pointsy = vector_type(points.shape(0), 0);
//    auto pointsz = vector_type(points.shape(0), 0);
//    for (int i = 0; i < points.shape(0); ++i) {
//        pointsx[i] = points(i, 0);
//        pointsy[i] = points(i, 1);
//        pointsz[i] = points(i, 2);
//    }
//    auto v = xt::xarray<double>::from_shape({gamma.shape(0), 3});
//    auto vgrad = xt::xarray<double>::from_shape({gamma.shape(0), 3, 3});
//    for (int i = 0; i < gamma.shape(0); ++i) {
//        for (int j = 0; j < 3; ++j) {
//            v(i, j) = 0.01 + i/100. -  j/3.;
//            for (int k = 0; k < 3; ++k) {
//                vgrad(i, j, k) = 0.01 + i/100. -  j/3. - k;
//            }
//        }
//    }
//    auto res_gamma = xt::xarray<double>::from_shape({points.shape(0), 3});
//    auto res_dgamma_by_dphi = xt::xarray<double>::from_shape({points.shape(0), 3});
//    auto res_grad_gamma = xt::xarray<double>::from_shape({points.shape(0), 3});
//    auto res_grad_dgamma_by_dphi = xt::xarray<double>::from_shape({points.shape(0), 3});
//    auto t5 = std::chrono::high_resolution_clock::now();
//    for (int i = 0; i < n; ++i) {
//        biot_savart_B_only_vjp_impl(pointsx, pointsy, pointsz, gamma, dgamma_by_dphi, v, res_gamma, res_dgamma_by_dphi, vgrad, res_grad_gamma, res_grad_dgamma_by_dphi);
//    }
//    auto t6 = std::chrono::high_resolution_clock::now();
//    double revtime = std::chrono::duration_cast<std::chrono::milliseconds>( t6 - t5 ).count();
//    std::cout << "Time: " << revtime << " ms." << std::endl;
//    return 1;
//}
