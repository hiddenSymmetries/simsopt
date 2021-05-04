#pragma once

#include <iostream>
#include <vector>

#include <tuple>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <fmt/core.h>

using Vec = std::vector<double>;

template<int degree>
double basis_fun(int idx, double x);

template<int degree>
class RegularGridInterpolant3D {
    private:
        int nx;
        int ny;
        int nz;
        int value_size;
        int padded_value_size;
        Vec vals;
        Vec vals_local;
        Vec xs;
        Vec ys;
        Vec zs;
        Vec xsmesh;
        Vec ysmesh;
        Vec zsmesh;

    public:
        RegularGridInterpolant3D(int nx, int ny, int nz, int value_size) : 
            nx(nx), ny(ny), nz(nz), value_size(value_size) {

            xsmesh = Vec(nx+1, 0.0);
            for (int i = 0; i <= nx; ++i)
                xsmesh[i] = double(i)/nx;
            xs = Vec(nx*degree+1, 0.0);
            for (int i = 0; i <= nx*degree; ++i)
                xs[i] = double(i)/(nx*degree);

            ysmesh = Vec(ny+1, 0.0);
            for (int i = 0; i <= ny; ++i)
                ysmesh[i] = double(i)/ny;
            ys = Vec(ny*degree+1, 0.0);
            for (int i = 0; i <= ny*degree; ++i)
                ys[i] = double(i)/(ny*degree);

            zsmesh = Vec(nz+1, 0.0);
            for (int i = 0; i <= nz; ++i)
                zsmesh[i] = double(i)/nz;
            zs = Vec(nz*degree+1, 0.0);
            for (int i = 0; i <= nz*degree; ++i)
                zs[i] = double(i)/(nz*degree);

            int simdcount = 4;
            padded_value_size = (value_size + simdcount) - (value_size % simdcount);
            int nnodes = (nx*degree+1)*(ny*degree+1)*(nz*degree+1);
            vals = Vec(nnodes*padded_value_size, 0.);
            vals_local = Vec((degree+1)*(degree+1)*(degree+1)*padded_value_size, 0.);
            fmt::print("Memory usage of interpolant={:E} bytes \n", double(vals.size()*sizeof(double)));
            fmt::print("{} function evaluations required\n", nnodes);

        }

        void interpolate(std::function<Vec(double, double, double)> &f);
        void interpolate_batch(std::function<Vec(Vec, Vec, Vec)> &f);

        inline int idx_dof(int i, int j, int k){
            return i*(ny*degree+1)*(nz*degree+1) + j*(nz*degree+1) + k;
        }

        inline int idx_mesh(int i, int j, int k){
            return i*(ny+1)*(nz+1) + j*(nz+1) + k;
        }

        inline int idx_dof_local(int i, int j, int k){
            return i*(degree+1)*(degree+1) + j*(degree+1) + k;
        } 

        Vec evaluate(double x, double y, double z);
        Vec evaluate_local(double x, double y, double z);
        std::pair<double, double> estimate_error(std::function<Vec(double, double, double)> &f, int samples);
};
