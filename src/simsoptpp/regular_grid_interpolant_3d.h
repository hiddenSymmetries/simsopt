#pragma once

#include <iostream>
#include <vector>
#include <functional>

#include  <unordered_map>
#include <tuple>
#include <random>
#include <stdexcept>
#include <algorithm>
#include <fmt/core.h>
#include <stdint.h>
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
        Vec vals;
        Vec xdof, ydof, zdof;
        Vec xmesh, ymesh, zmesh;
        Vec xdoftensor_reduced, ydoftensor_reduced, zdoftensor_reduced;
        std::unordered_map<int, AlignedVec> all_local_vals_map;
        std::vector<bool> skip_cell;
        std::vector<uint32_t> reduced_to_full_map, full_to_reduced_map;

        uint32_t cells_to_skip, cells_to_keep, dofs_to_skip, dofs_to_keep;
        int local_vals_size;
        static const int simdcount = xsimd::simd_type<double>::size;
        //static const std::function<std::vector<bool>(Vec, Vec, Vec)> skipnothing = ;
        const InterpolationRule rule;
        Vec pkxs, pkys, pkzs;

    public:
        bool extrapolate;

        RegularGridInterpolant3D(InterpolationRule rule, RangeTriplet xrange, RangeTriplet yrange, RangeTriplet zrange, int value_size, bool extrapolate, std::function<std::vector<bool>(Vec, Vec, Vec)> skip) :
            rule(rule), 
            xmin(std::get<0>(xrange)), xmax(std::get<1>(xrange)), nx(std::get<2>(xrange)),
            ymin(std::get<0>(yrange)), ymax(std::get<1>(yrange)), ny(std::get<2>(yrange)),
            zmin(std::get<0>(zrange)), zmax(std::get<1>(zrange)), nz(std::get<2>(zrange)),
            value_size(value_size), extrapolate(extrapolate)
        {
    //std::function<std::vector<bool>(Vec, Vec, Vec)> skip = ;
            int degree = rule.degree;
            pkxs = Vec(degree+1, 0.);
            pkys = Vec(degree+1, 0.);
            pkzs = Vec(degree+1, 0.);
            hx = (xmax-xmin)/nx;
            hy = (ymax-ymin)/ny;
            hz = (zmax-zmin)/nz;

            // build a regular mesh on [xmin, xmax] x [ymin, ymax] x [zmin, zmax]
            xmesh = linspace(xmin, xmax, nx+1, true);
            ymesh = linspace(ymin, ymax, ny+1, true);
            zmesh = linspace(zmin, zmax, nz+1, true);

            int nmesh = (nx+1)*(ny+1)*(nz+1);
            Vec xmeshtensor(nmesh, 0.);
            Vec ymeshtensor(nmesh, 0.);
            Vec zmeshtensor(nmesh, 0.);

            for (int i = 0; i <= nx; ++i) {
                for (int j = 0; j <= ny; ++j) {
                    for (int k = 0; k <= nz; ++k) {
                        int offset = idx_mesh(i, j, k);
                        xmeshtensor[offset] = xmesh[i];
                        ymeshtensor[offset] = ymesh[j];
                        zmeshtensor[offset] = zmesh[k];
                    }
                }
            }
            // for each node in the mesh, check whether it is in the domain, or
            // outside of it (i.e. should be skipped)
            std::vector<bool> skip_mesh = skip(xmeshtensor, ymeshtensor, zmeshtensor);
            // cells are entirely ignored if *all* of its eight corners are
            // outside the domain
            skip_cell = std::vector<bool>(nx*ny*nz, false);
            cells_to_skip = 0;
            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    for (int k = 0; k < nz; ++k) {
                        bool skip_this_one = (
                                skip_mesh[idx_mesh(i  , j  , k)] && skip_mesh[idx_mesh(i  , j  , k+1)] &&
                                skip_mesh[idx_mesh(i  , j+1, k)] && skip_mesh[idx_mesh(i  , j+1, k+1)] &&
                                skip_mesh[idx_mesh(i+1, j  , k)] && skip_mesh[idx_mesh(i+1, j  , k+1)] &&
                                skip_mesh[idx_mesh(i+1, j+1, k)] && skip_mesh[idx_mesh(i+1, j+1, k+1)]
                                );
                        if(skip_this_one) {
                            skip_cell[idx_cell(i, j, k)] = true;
                            cells_to_skip++;
                        }
                    }
                }
            }
            cells_to_keep = nx*ny*nz - cells_to_skip;

            // now build the interpolation points in 1d.
            xdof = Vec(nx*degree+1, 0.);
            ydof = Vec(ny*degree+1, 0.);
            zdof = Vec(nz*degree+1, 0.);
            int i, j;
            for (i = 0; i < nx; ++i) {
                for (j = 0; j < degree+1; ++j) {
                    xdof[i*degree+j] = xmesh[i] + rule.nodes[j]*hx;
                }
            }
            for (i = 0; i < ny; ++i) {
                for (j = 0; j < degree+1; ++j) {
                    ydof[i*degree+j] = ymesh[i] + rule.nodes[j]*hy;
                }
            }
            for (i = 0; i < nz; ++i) {
                for (j = 0; j < degree+1; ++j) {
                    zdof[i*degree+j] = zmesh[i] + rule.nodes[j]*hz;
                }
            }
            uint32_t n =  (nx*degree+1)*(ny*degree+1)*(nz*degree+1);
            // turn these into a tensor product grid
            Vec xdoftensor(n, 0.);
            Vec ydoftensor(n, 0.);
            Vec zdoftensor(n, 0.);
            for (int i = 0; i <= nx*degree; ++i) {
                for (int j = 0; j <= ny*degree; ++j) {
                    for (int k = 0; k <= nz*degree; ++k) {
                        uint32_t offset = idx_dof(i, j, k);
                        xdoftensor[offset] = xdof[i];
                        ydoftensor[offset] = ydof[j];
                        zdoftensor[offset] = zdof[k];
                    }
                }
            }
            // now we need to figure out which of these dofs we keep, and which
            // to discard.  to do this, we loop over the cells, and mark all
            // dofs in cells that should not be skipped.

            std::vector<bool> skip_dof(n, true);
            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    for (int k = 0; k < nz; ++k) {
                        if(!skip_cell[idx_cell(i, j, k)]){
                            for (int ii = 0; ii <= degree; ++ii) {
                                for (int jj = 0; jj <= degree; ++jj) {
                                    for (int kk = 0; kk <= degree; ++kk) {
                                        skip_dof[idx_dof(i*degree + ii, j*degree + jj, k*degree + kk)] = false;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Count how many dofs are skipped in total, and how many to keep
            dofs_to_skip = 0;
            for (uint32_t i = 0; i < n; ++i) {
                dofs_to_skip += skip_dof[i];
            }
            dofs_to_keep = n - dofs_to_skip;
            // Build a map that maps indices from the reduced set of interpolation
            // points to the full set, and its inverse
            reduced_to_full_map = std::vector<uint32_t>(dofs_to_keep, 0);
            full_to_reduced_map = std::vector<uint32_t>(n, 0);
            uint32_t ctr = 0;
            for (uint32_t i = 0; i < n; ++i) {
                full_to_reduced_map[i] = i - ctr;
                if(skip_dof[i])
                    ctr++;
                else {
                    reduced_to_full_map[i-ctr] = i;
                }
            }

            // build the reduced list of interpolation points
            xdoftensor_reduced = Vec(dofs_to_keep, 0.);
            ydoftensor_reduced = Vec(dofs_to_keep, 0.);
            zdoftensor_reduced = Vec(dofs_to_keep, 0.);
            for (long i = 0; i < dofs_to_keep; ++i) {
                xdoftensor_reduced[i] = xdoftensor[reduced_to_full_map[i]];
                ydoftensor_reduced[i] = ydoftensor[reduced_to_full_map[i]];
                zdoftensor_reduced[i] = zdoftensor[reduced_to_full_map[i]];
            }
            vals = Vec(dofs_to_keep * value_size, 0.);

            padded_value_size = (value_size + simdcount) - (value_size % simdcount);
            //int nsimdblocks = padded_value_size/simdcount;
            int nnodes = (nx*degree+1)*(ny*degree+1)*(nz*degree+1);
            //vals = AlignedVec(nnodes*padded_value_size, 0.);
            local_vals_size = (degree+1)*(degree+1)*(degree+1)*padded_value_size;
        }
        RegularGridInterpolant3D(InterpolationRule rule, RangeTriplet xrange, RangeTriplet yrange, RangeTriplet zrange, int value_size, bool extrapolate) :
            RegularGridInterpolant3D(rule, xrange, yrange, zrange, value_size, extrapolate, [](Vec x, Vec y, Vec z){ return std::vector<bool>(x.size(), false); })
            {}
        //RegularGridInterpolant3D(InterpolationRule rule, int nx, int ny, int nz, int value_size) : 
        //    RegularGridInterpolant3D(rule, {0., 1., nx}, {0., 1., ny}, {0., 1., nz}, value_size, true) {

        //}

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
