#pragma once

#include <iostream>
#include <vector>

#include <tuple>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <fmt/core.h>
#include "simdhelpers.h"

using simd_t = xs::simd_type<double>;
using AlignedVec = std::vector<double, aligned_padded_allocator<XSIMD_DEFAULT_ALIGNMENT>>;
using Vec = std::vector<double>;
using RangeTriplet = std::tuple<double, double, int>;

template<int degree>
double basis_fun(int idx, double x);

Vec linspace(double min, double max, int n, bool endpoint);

template<int degree>
class RegularGridInterpolant3D {
    private:
        int nx, ny, nz;
        double hx, hy, hz;
        double xmin, ymin, zmin;
        double xmax, ymax, zmax;
        int value_size;
        int padded_value_size;
        AlignedVec vals;
        //Vec vals_local;
        Vec xs;
        Vec ys;
        Vec zs;
        Vec xsmesh;
        Vec ysmesh;
        Vec zsmesh;
        std::vector<AlignedVec> all_local_vals;
        Vec sumi, sumj, sumk;
        static constexpr int simdcount = 4;

    public:

        RegularGridInterpolant3D(RangeTriplet xrange, RangeTriplet yrange, RangeTriplet zrange, int value_size) :
            xmin(std::get<0>(xrange)), xmax(std::get<1>(xrange)), nx(std::get<2>(xrange)),
            ymin(std::get<0>(yrange)), ymax(std::get<1>(yrange)), ny(std::get<2>(yrange)),
            zmin(std::get<0>(zrange)), zmax(std::get<1>(zrange)), nz(std::get<2>(zrange)), value_size(value_size)
        {
            hx = (xmax-xmin)/nx;
            hy = (ymax-ymin)/ny;
            hz = (zmax-zmin)/nz;

            xsmesh = linspace(xmin, xmax, nx+1, true);
            xs = linspace(xmin, xmax, nx*degree+1, true);
            ysmesh = linspace(ymin, ymax, ny+1, true);
            ys = linspace(ymin, ymax, ny*degree+1, true);
            zsmesh = linspace(zmin, zmax, nz+1, true);
            zs = linspace(zmin, zmax, nz*degree+1, true);

            padded_value_size = (value_size + simdcount) - (value_size % simdcount);
            int nsimdblocks = padded_value_size/simdcount;
            int nnodes = (nx*degree+1)*(ny*degree+1)*(nz*degree+1);
            vals = AlignedVec(nnodes*padded_value_size, 0.);
            //vals_local = Vec((degree+1)*(degree+1)*(degree+1)*padded_value_size, 0.);
            fmt::print("Memory usage of interpolant={:E} bytes \n", double(vals.size()*sizeof(double)));
            fmt::print("{} function evaluations required\n", nnodes);
            sumi = Vec(value_size, 0.);
            sumj = Vec(value_size, 0.);
            sumk = Vec(value_size, 0.);

        }

        RegularGridInterpolant3D(int nx, int ny, int nz, int value_size) : 
            RegularGridInterpolant3D({0., 1., nx}, {0., 1., ny}, {0., 1., nz}, value_size) {

        }

        void interpolate(std::function<Vec(double, double, double)> &f);
        void interpolate_batch(std::function<Vec(Vec, Vec, Vec)> &f);
        void build_local_vals();

        inline int idx_dof(int i, int j, int k){
            return i*(ny*degree+1)*(nz*degree+1) + j*(nz*degree+1) + k;
        }

        inline int idx_cell(int i, int j, int k){
            return i*ny*nz + j*nz + k;
        }

        inline int idx_mesh(int i, int j, int k){
            return i*(ny+1)*(nz+1) + j*(nz+1) + k;
        }

        inline int idx_dof_local(int i, int j, int k){
            return i*(degree+1)*(degree+1) + j*(degree+1) + k;
        } 

        Vec evaluate(double x, double y, double z);
        Vec evaluate_local(double x, double y, double z, int cell_idx);
        std::pair<double, double> estimate_error(std::function<Vec(double, double, double)> &f, int samples);
};
