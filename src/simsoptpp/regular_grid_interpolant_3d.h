#pragma once
#include "simdhelpers.h"
#include <unordered_map>
#include <algorithm>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <stdint.h>
#include <tuple>
#include <vector>

using Vec = std::vector<double>;
using RangeTriplet = std::tuple<double, double, int>;

Vec linspace(double min, double max, int n, bool endpoint);

class InterpolationRule {
    /* An InterpolationRule consists of a list of interpolation nodes and then
     * uses standard Lagrange interpolation, that is for each node x_i we get a
     * basis function p_i that satisfies p_i(x_i) = 1 and p_i(x_j)=0. This
     * function is given by p_i(x) = [Π_{i≠j} (x-x_i)/(x_j-x_i)]. To compute
     * this efficiently, we store the denominator in `scalings`, i.e.
     * scalings[j] = [Π_{i≠j} 1/(x_j-x_i)].
    */

    protected:
        void build_scalings(){
            for(int idx = 0; idx < degree+1; ++idx) {
                for(int i = 0; i < degree+1; ++i) {
                    if(i == idx) continue;
                    scalings[idx] *= 1./(nodes[idx]-nodes[i]);
                }
            }
        }
    public:
        Vec nodes;
        Vec scalings;
        const int degree;
        InterpolationRule(int degree) : degree(degree), nodes(degree+1, 0.), scalings(degree+1, 1.0) { }

        double basis_fun(int idx, double x) const {
            // evaluate the basisfunction p_idx at location x
            double res = scalings[idx];
            for(int i = 0; i < degree+1; ++i) {
                if(i == idx) continue;
                res *= (x-nodes[i]);
            }
            return res;
        }

        simd_t basis_fun(int idx, simd_t x) const {
            // evaluate the basisfunction p_idx at multiple locations stored in x
            simd_t res(scalings[idx]);
            for(int i = 0; i < degree+1; ++i) {
                if(i == idx) continue;
                res *= (x-simd_t(nodes[i]));
            }
            return res;
        }
};

template<class Array>
class RegularGridInterpolant3D {
    /* This class implements a vector-valued piecewise polynomial interpolant
     * on a regular grid in three dimensions.  There are many ways to
     * implemented interpolants, the implementation here is done to favour
     * speed over memory efficiency, and is designed for vector valued
     * functions for which SIMD vectorization pays off.
     *
     * The interpolant assumes a regular mesh on a rectangular cuboid. In order
     * to work on other geometries with reasonable efficiency, parts of the
     * mesh can be "skipped". This is done as follows: if a function is meant
     * to be interpolated on some irregular domain Ω, one first picks a
     * rectangular domain D so that Ω⊂D, and then defines a function `skip`
     * that returns true outside of Ω and false inside of Ω. Internally, we
     * then skip all cells for which the function returns true on all eight
     * corners. For some pictures that explain the idea behind the skip
     * function and its caveats, please check the discussion under
     * https://github.com/hiddenSymmetries/simsopt/pull/227
     */
    private:
        const int nx, ny, nz;  // number of cells in x, y, and z direction
        double hx, hy, hz; // gridsize in x, y, and z direction
        const double xmin, ymin, zmin; // lower bounds of the x, y, and z coordinates
        const double xmax, ymax, zmax; // lower bounds of the x, y, and z coordinates
        const int value_size; // number of output dimensions of the interpolant, i.e. space that is mapped into
        const InterpolationRule rule; // the interpolation rule to use on each cell in the grid
        const bool out_of_bounds_ok; // whether to do nothing or throw an error when the interpolant is queried at an out-of-bounds point

        // location of the mesh nodes in [xmin, xmax], [ymin, ymax], and [zmin, zmax]
        // has size nx+1, ny+1, nz+1 respectively
        Vec xmesh, ymesh, zmesh;

        // location of the mesh nodes in [xmin, xmax], [ymin, ymax], and [zmin, zmax]. superset of xmesh, ymesh, zmesh
        // has size nx*degree + 1, ny*degree + 1, and nz*degree + 1 respectively
        Vec xdof, ydof, zdof;
        // subset of the tensor product of the dof locations. if none of the dofs are skipped, these all have size
        // (nx*degree + 1) * (ny*degree + 1) * (nz*degree + 1), but now they have size dofs_to_keep
        Vec xdoftensor_reduced, ydoftensor_reduced, zdoftensor_reduced;

        Vec vals; // contains the values of the function to be interpolated at the dofs, of size dofs_to_keep * value_size
        std::unordered_map<int, AlignedPaddedVec> all_local_vals_map; // maps each cell to an array of size (degree+1)**3 * padded_value_size
        std::vector<bool> skip_cell; // whether to skip each cell or not
        // since we are skipping some dofs, we need mappings into the list of
        // reduced dofs, e.g. if we skip dofs 3, then reduced to full would
        // look like [0, 1, 2, 4, 5, ...]
        std::vector<uint32_t> reduced_to_full_map, full_to_reduced_map;

        uint32_t cells_to_skip, cells_to_keep, dofs_to_skip, dofs_to_keep; // which cells and dofs we skip and keep
        int local_vals_size;
        Vec pkxs, pkys, pkzs;

        static const int simdcount = xsimd::simd_type<double>::size; // vector width for simd instructions
        int padded_value_size; // smallest multiple of simdcount that is larger than value_size

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
        void evaluate_inplace(double x, double y, double z, double* res);
        void evaluate_local(double x, double y, double z, int cell_idx, double* res);

    public:

        RegularGridInterpolant3D(InterpolationRule rule, RangeTriplet xrange, RangeTriplet yrange, RangeTriplet zrange, int value_size, bool out_of_bounds_ok, std::function<std::vector<bool>(Vec, Vec, Vec)> skip) :
            rule(rule), 
            xmin(std::get<0>(xrange)), xmax(std::get<1>(xrange)), nx(std::get<2>(xrange)),
            ymin(std::get<0>(yrange)), ymax(std::get<1>(yrange)), ny(std::get<2>(yrange)),
            zmin(std::get<0>(zrange)), zmax(std::get<1>(zrange)), nz(std::get<2>(zrange)),
            value_size(value_size), out_of_bounds_ok(out_of_bounds_ok)
        {
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
            // Now we need to figure out which of these dofs we keep, and which
            // to discard.  To do this, we loop over the cells, and for each
            // cell that shouldn't be skipped, we mark all dofs in that cell.

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

            // round up value_size to nearest multiple of simdcount
            padded_value_size = (value_size + simdcount) - (value_size % simdcount);
            int nnodes = (nx*degree+1)*(ny*degree+1)*(nz*degree+1);
            local_vals_size = (degree+1)*(degree+1)*(degree+1)*padded_value_size;
        }
        RegularGridInterpolant3D(InterpolationRule rule, RangeTriplet xrange, RangeTriplet yrange, RangeTriplet zrange, int value_size, bool out_of_bounds_ok) :
            RegularGridInterpolant3D(rule, xrange, yrange, zrange, value_size, out_of_bounds_ok, [](Vec x, Vec y, Vec z){ return std::vector<bool>(x.size(), false); })
            {}

        void interpolate_batch(std::function<Vec(Vec, Vec, Vec)> &f); // build the interpolant

        Vec evaluate(double x, double y, double z); // evaluate the interpolant at one location
        void evaluate_batch(Array& xyz, Array& fxyz); // evluate the interpolant at multiple locations

        std::pair<double, double> estimate_error(std::function<Vec(Vec, Vec, Vec)> &f, int samples);
};


class UniformInterpolationRule : public InterpolationRule {
    protected:
        using InterpolationRule::build_scalings;
    public:
        using InterpolationRule::nodes;
        using InterpolationRule::scalings;
        using InterpolationRule::degree;
        UniformInterpolationRule(int degree) : InterpolationRule(degree) {
            double degreeinv = double(1.)/degree;
            for (int i = 0; i < degree+1; ++i) {
                nodes[i] = i*degreeinv;
            }
            build_scalings();
        }
};


class ChebyshevInterpolationRule : public InterpolationRule {
    protected:
        using InterpolationRule::build_scalings;
    public:
        using InterpolationRule::nodes;
        using InterpolationRule::scalings;
        using InterpolationRule::degree;
        ChebyshevInterpolationRule(int degree) : InterpolationRule(degree) {
            double degreeinv = double(1.)/degree;
            for (int i = 0; i < degree+1; ++i) {
                nodes[i] = (-0.5)*std::cos(i*M_PI*degreeinv) + 0.5;
            }
            build_scalings();
        }
};

