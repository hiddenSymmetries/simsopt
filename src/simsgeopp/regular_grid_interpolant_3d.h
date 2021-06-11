#pragma once

#include <iostream>
#include <vector>
#include <functional>

#include <tuple>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <fmt/core.h>
#include "simdhelpers.h"

using simd_t = xs::simd_type<double>;
using AlignedVec = std::vector<double, aligned_padded_allocator<double, XSIMD_DEFAULT_ALIGNMENT>>;
using Vec = std::vector<double>;
using RangeTriplet = std::tuple<double, double, int>;

Vec linspace(double min, double max, int n, bool endpoint);


class InterpolationRule {
    public:
        const int degree;
        Vec nodes;
        Vec scalings;
        InterpolationRule(int degree) : degree(degree), nodes(degree+1, 0.), scalings(degree+1, 1.0) {
        }
        double basis_fun(int idx, double x) const {
            double res = scalings[idx];
            for(int i = 0; i < degree+1; ++i) {
                if(i == idx) continue;
                res *= (x-nodes[i]);
            }
            return res;
        }

        simd_t basis_fun(int idx, simd_t x) const {
            simd_t res(scalings[idx]);
            for(int i = 0; i < degree+1; ++i) {
                if(i == idx) continue;
                res *= (x-simd_t(nodes[i]));
            }
            return res;
        }
};

class UniformInterpolationRule : public InterpolationRule {
    public:
        using InterpolationRule::nodes;
        using InterpolationRule::degree;
        using InterpolationRule::scalings;
        UniformInterpolationRule(int degree) : InterpolationRule(degree) {
            double degreeinv = double(1.)/degree;
            for (int i = 0; i < degree+1; ++i) {
                nodes[i] = i*degreeinv;
            }
            for(int idx = 0; idx < degree+1; ++idx) {
                for(int i = 0; i < degree+1; ++i) {
                    if(i == idx) continue;
                    scalings[idx] *= 1./(nodes[idx]-nodes[i]);
                }
            }
        }
};

#include <fmt/ranges.h>

class ChebyshevInterpolationRule : public InterpolationRule {
    public:
        using InterpolationRule::nodes;
        using InterpolationRule::degree;
        using InterpolationRule::scalings;
        ChebyshevInterpolationRule(int degree) : InterpolationRule(degree) {
            double degreeinv = double(1.)/degree;
            for (int i = 0; i < degree+1; ++i) {
                nodes[i] = (-0.5)*std::cos(i*M_PI*degreeinv) + 0.5;
            }
            //fmt::print("Chebyshev nodes = {}\n", fmt::join(nodes, ", "));
            for(int idx = 0; idx < degree+1; ++idx) {
                for(int i = 0; i < degree+1; ++i) {
                    if(i == idx) continue;
                    scalings[idx] *= 1./(nodes[idx]-nodes[i]);
                }
            }
        }
};

template<class Array>
class RegularGridInterpolant3D {
    private:
        const int nx, ny, nz;
        const double xmin, ymin, zmin;
        const double xmax, ymax, zmax;
        double hx, hy, hz;
        const int value_size;
        int padded_value_size;
        AlignedVec vals;
        Vec xs;
        Vec ys;
        Vec zs;
        Vec xsmesh;
        Vec ysmesh;
        Vec zsmesh;
        AlignedVec all_local_vals;
        int local_vals_size;
        static const int simdcount = xsimd::simd_type<double>::size;
        const InterpolationRule rule;
        Vec pkxs, pkys, pkzs;

    public:
        bool extrapolate;

        RegularGridInterpolant3D(InterpolationRule rule, RangeTriplet xrange, RangeTriplet yrange, RangeTriplet zrange, int value_size, bool extrapolate) :
            rule(rule), 
            xmin(std::get<0>(xrange)), xmax(std::get<1>(xrange)), nx(std::get<2>(xrange)),
            ymin(std::get<0>(yrange)), ymax(std::get<1>(yrange)), ny(std::get<2>(yrange)),
            zmin(std::get<0>(zrange)), zmax(std::get<1>(zrange)), nz(std::get<2>(zrange)),
            value_size(value_size), extrapolate(extrapolate)
        {
            int degree = rule.degree;
            pkxs = Vec(degree+1, 0.);
            pkys = Vec(degree+1, 0.);
            pkzs = Vec(degree+1, 0.);
            hx = (xmax-xmin)/nx;
            hy = (ymax-ymin)/ny;
            hz = (zmax-zmin)/nz;

            xsmesh = linspace(xmin, xmax, nx+1, true);
            ysmesh = linspace(ymin, ymax, ny+1, true);
            zsmesh = linspace(zmin, zmax, nz+1, true);
            xs = Vec(nx*degree+1, 0.);
            ys = Vec(ny*degree+1, 0.);
            zs = Vec(nz*degree+1, 0.);
            int i, j;
            for (i = 0; i < nx; ++i) {
                for (j = 0; j < degree+1; ++j) {
                    xs[i*degree+j] = xsmesh[i] + rule.nodes[j]*hx;
                }
            }
            for (i = 0; i < ny; ++i) {
                for (j = 0; j < degree+1; ++j) {
                    ys[i*degree+j] = ysmesh[i] + rule.nodes[j]*hy;
                }
            }
            for (i = 0; i < nz; ++i) {
                for (j = 0; j < degree+1; ++j) {
                    zs[i*degree+j] = zsmesh[i] + rule.nodes[j]*hz;
                }
            }

            padded_value_size = (value_size + simdcount) - (value_size % simdcount);
            //int nsimdblocks = padded_value_size/simdcount;
            int nnodes = (nx*degree+1)*(ny*degree+1)*(nz*degree+1);
            vals = AlignedVec(nnodes*padded_value_size, 0.);
            local_vals_size = (degree+1)*(degree+1)*(degree+1)*padded_value_size;
            //vals_local = Vec((degree+1)*(degree+1)*(degree+1)*padded_value_size, 0.);
            //fmt::print("Memory usage of interpolant={:E} bytes \n", double(vals.size()*sizeof(double)));
            //fmt::print("{} function evaluations required\n", nnodes);

        }

        RegularGridInterpolant3D(InterpolationRule rule, int nx, int ny, int nz, int value_size) : 
            RegularGridInterpolant3D(rule, {0., 1., nx}, {0., 1., ny}, {0., 1., nz}, value_size, true) {

        }

        void interpolate(std::function<Vec(double, double, double)> &f);
        void interpolate_batch(std::function<Vec(Vec, Vec, Vec)> &f);
        void build_local_vals();

        inline int idx_dof(int i, int j, int k){
            int degree = rule.degree;
            return i*(ny*degree+1)*(nz*degree+1) + j*(nz*degree+1) + k;
        }

        inline int idx_cell(int i, int j, int k){
            return i*ny*nz + j*nz + k;
        }

        inline int idx_mesh(int i, int j, int k){
            return i*(ny+1)*(nz+1) + j*(nz+1) + k;
        }

        inline int idx_dof_local(int i, int j, int k){
            int degree = rule.degree;
            return i*(degree+1)*(degree+1) + j*(degree+1) + k;
        }

        int locate_unsafe(double x, double y, double z);
        void evaluate_batch_with_transform(Array& xyz, Array& fxyz);
        void evaluate_batch(Array& xyz, Array& fxyz);
        Vec evaluate(double x, double y, double z);
        void evaluate_inplace(double x, double y, double z, double* res);
        void evaluate_local(double x, double y, double z, int cell_idx, double* res);
        std::pair<double, double> estimate_error(std::function<Vec(Vec, Vec, Vec)> &f, int samples);
};
