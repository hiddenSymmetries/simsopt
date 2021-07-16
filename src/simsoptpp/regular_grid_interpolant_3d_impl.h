#include "regular_grid_interpolant_3d.h"
#include <xtensor/xarray.hpp>
#include "xtensor/xlayout.hpp"
#define _USE_MATH_DEFINES
#include <math.h>

#define _EPS_ 1e-13

template<class Array>
const int RegularGridInterpolant3D<Array>::simdcount;

template<class Array>
void RegularGridInterpolant3D<Array>::interpolate(std::function<Vec(double, double, double)> &f) {
    int degree = rule.degree;
    Vec t;
    for (int i = 0; i <= nx*degree; ++i) {
        for (int j = 0; j <= ny*degree; ++j) {
            for (int k = 0; k <= nz*degree; ++k) {
                int offset = padded_value_size*idx_dof(i, j, k);
                t = f(xs[i], ys[j], zs[k]);
                for (int l = 0; l < value_size; ++l) {
                    vals[offset + l] = t[l];
                }
            }
        }
    }
    build_local_vals();
}

template<class Array>
void RegularGridInterpolant3D<Array>::interpolate_batch(std::function<Vec(Vec, Vec, Vec)> &f) {
    int degree = rule.degree;
    Vec xcoords((nx*degree+1)*(ny*degree+1)*(nz*degree+1), 0.);
    Vec ycoords((nx*degree+1)*(ny*degree+1)*(nz*degree+1), 0.);
    Vec zcoords((nx*degree+1)*(ny*degree+1)*(nz*degree+1), 0.);
    for (int i = 0; i <= nx*degree; ++i) {
        for (int j = 0; j <= ny*degree; ++j) {
            for (int k = 0; k <= nz*degree; ++k) {
                int offset = idx_dof(i, j, k);
                xcoords[offset] = xs[i];
                ycoords[offset] = ys[j];
                zcoords[offset] = zs[k];
            }
        }
    }
    Vec fxyz  = f(xcoords, ycoords, zcoords);
    for (int i = 0; i <= nx*degree; ++i) {
        for (int j = 0; j <= ny*degree; ++j) {
            for (int k = 0; k <= nz*degree; ++k) {
                int offset = value_size*idx_dof(i, j, k);
                int offset_padded = padded_value_size*idx_dof(i, j, k);
                for (int l = 0; l < value_size; ++l) {
                    vals[offset_padded + l] = fxyz[offset+l];
                }
            }
        }
    }
    build_local_vals();
}

template<class Array>
void RegularGridInterpolant3D<Array>::build_local_vals(){
    int degree = rule.degree;
    //all_local_vals = std::vector<AlignedVec>(
    //        nx*ny*nz,
    //        AlignedVec((degree+1)*(degree+1)*(degree+1)*padded_value_size, 0.)
    //        );
    all_local_vals = AlignedVec(nx*ny*nz*local_vals_size, 0.);
    for (int xidx = 0; xidx < nx; ++xidx) {
        for (int yidx = 0; yidx < ny; ++yidx) {
            for (int zidx = 0; zidx < nz; ++zidx) {
                int meshidx = idx_cell(xidx, yidx, zidx);
                for (int i = 0; i < degree+1; ++i) {
                    for (int j = 0; j < degree+1; ++j) {
                        int offset = padded_value_size*idx_dof(xidx*degree+i, yidx*degree+j, zidx*degree+0);
                        int offset_local = padded_value_size*idx_dof_local(i, j, 0);
                        memcpy(all_local_vals.data()+meshidx*local_vals_size + offset_local, vals.data()+offset, (degree+1)*padded_value_size*sizeof(double));
                    }
                }
            }
        }
    }
}

template<class Array>
void RegularGridInterpolant3D<Array>::evaluate_batch(Array& xyz, Array& fxyz){
    if(fxyz.layout() != xt::layout_type::row_major)
          throw std::runtime_error("fxyz needs to be in row-major storage order");
    int npoints = xyz.shape(0);
    for (int i = 0; i < npoints; ++i) {
        if(i < npoints-1){
            int idx = locate_unsafe(xyz(i+1, 0), xyz(i+1, 1), xyz(i+1, 2));
            double* ptr = all_local_vals.data()+idx*local_vals_size;
            __builtin_prefetch(ptr+(0*8), 0, 0);
            __builtin_prefetch(ptr+(1*8), 0, 0);
            __builtin_prefetch(ptr+(2*8), 0, 0);
            __builtin_prefetch(ptr+(3*8), 0, 0);
            __builtin_prefetch(ptr+(4*8), 0, 0);
            __builtin_prefetch(ptr+(5*8), 0, 0);
            __builtin_prefetch(ptr+(6*8), 0, 0);
            __builtin_prefetch(ptr+(7*8), 0, 0);
            __builtin_prefetch(ptr+(8*8), 0, 0);
            __builtin_prefetch(ptr+(9*8), 0, 0);
            __builtin_prefetch(ptr+(10*8), 0, 0);
        }
        evaluate_inplace(xyz(i, 0), xyz(i, 1), xyz(i, 2), fxyz.data() + value_size*i);
    }
}

template<class Array>
Vec RegularGridInterpolant3D<Array>::evaluate(double x, double y, double z){
    Vec fxyz(value_size, 0.);
    evaluate_inplace(x, y, z, fxyz.data());
    return fxyz;

}

template<class Array>
int RegularGridInterpolant3D<Array>::locate_unsafe(double x, double y, double z){
    int xidx = int(nx*(x-xmin)/(xmax-xmin)); // find idx so that xsmesh[xidx] <= x <= xs[xidx+1]
    int yidx = int(ny*(y-ymin)/(ymax-ymin));
    int zidx = int(nz*(z-zmin)/(zmax-zmin));
    return idx_cell(xidx, yidx, zidx);
}

template<class Array>
void RegularGridInterpolant3D<Array>::evaluate_inplace(double x, double y, double z, double* res){
    if(this->extrapolate){
        x = std::max(std::min(x, xmax-_EPS_), xmin+_EPS_);
        y = std::max(std::min(y, ymax-_EPS_), ymin+_EPS_);
        z = std::max(std::min(z, zmax-_EPS_), zmin+_EPS_);
    } else {
        if(x < xmin || x >= xmax)
            throw std::runtime_error(fmt::format("x={} not within [{}, {}]", x, xmin, xmax));
        if(y < ymin || y >= ymax)
            throw std::runtime_error(fmt::format("y={} not within [{}, {}]", y, ymin, ymax));
        if(z < zmin || z >= zmax)
            throw std::runtime_error(fmt::format("z={} not within [{}, {}]", z, zmin, zmax));
    }
    int xidx = int(nx*(x-xmin)/(xmax-xmin)); // find idx so that xsmesh[xidx] <= x <= xs[xidx+1]
    int yidx = int(ny*(y-ymin)/(ymax-ymin));
    int zidx = int(nz*(z-zmin)/(zmax-zmin));
    if(xidx < 0 || xidx >= nx)
        throw std::runtime_error(fmt::format("xidxs={} not within [0, {}]", xidx, nx-1));
    if(yidx < 0 || yidx >= ny)
        throw std::runtime_error(fmt::format("yidxs={} not within [0, {}]", yidx, ny-1));
    if(zidx < 0 || zidx >= nz)
        throw std::runtime_error(fmt::format("zidxs={} not within [0, {}]", zidx, nz-1));


    double xlocal = (x-xsmesh[xidx])/hx;
    double ylocal = (y-ysmesh[yidx])/hy;
    double zlocal = (z-zsmesh[zidx])/hz;
    if(xlocal < 0.-_EPS_ || xlocal > 1.+_EPS_)
        throw std::runtime_error(fmt::format("xlocal={} not within [0, 1]", xlocal));
    if(ylocal < 0.-_EPS_ || ylocal > 1.+_EPS_)
        throw std::runtime_error(fmt::format("ylocal={} not within [0, 1]", ylocal));
    if(zlocal < 0.-_EPS_ || zlocal > 1.+_EPS_)
        throw std::runtime_error(fmt::format("zlocal={} not within [0, 1]", zlocal));
    //std::cout << "local coordinates=(" << xlocal << ", " << ylocal << ", " << zlocal << ")" << std::endl;
    return evaluate_local(xlocal, ylocal, zlocal, idx_cell(xidx, yidx, zidx), res);
}

template<class Array>
void RegularGridInterpolant3D<Array>::evaluate_local(double x, double y, double z, int cell_idx, double* res)
{
    int degree = rule.degree;
    double* vals_local = all_local_vals.data()+cell_idx*local_vals_size;
    if(xsimd::simd_type<double>::size >= 3){
        simd_t xyz;
        xyz[0] = x;
        xyz[1] = y;
        xyz[2] = z;
        for (int k = 0; k < degree+1; ++k) {
            simd_t temp = this->rule.basis_fun(k, xyz);
            pkxs[k] = temp[0];
            pkys[k] = temp[1];
            pkzs[k] = temp[2];
        }
    } else {
        for (int k = 0; k < degree+1; ++k) {
            pkxs[k] = this->rule.basis_fun(k, x);
            pkys[k] = this->rule.basis_fun(k, y);
            pkzs[k] = this->rule.basis_fun(k, z);
        }
    }

    for(int l=0; l<padded_value_size; l += simdcount) {
        simd_t sumi(0.);
        int offset_local = l;
        double* val_ptr = &(vals_local[offset_local]);
        for (int i = 0; i < degree+1; ++i) {
            simd_t sumj(0.); 
            for (int j = 0; j < degree+1; ++j) {
                simd_t sumk(0.);
                for (int k = 0; k < degree+1; ++k) {
                    double pkz = pkzs[k];
                    sumk = xsimd::fma(xsimd::load_aligned(val_ptr), simd_t(pkz), sumk);
                    val_ptr += padded_value_size;
                }
                double pjy = pkys[j];
                sumj = xsimd::fma(sumk, simd_t(pjy), sumj);
            }
            double pix = pkxs[i];
            sumi = xsimd::fma(sumj, simd_t(pix), sumi);
        }
        for (int ll = 0; ll < std::min(simdcount, value_size-l); ++ll) {
            res[l+ll] = sumi[ll];
        }
    }
}

template<class Array>
std::pair<double, double> RegularGridInterpolant3D<Array>::estimate_error(std::function<Vec(Vec, Vec, Vec)> &f, int samples) {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, +1.0);
    double err = 0;
    double errsq = 0;
    Vec xs(samples, 0.);
    Vec ys(samples, 0.);
    Vec zs(samples, 0.);
    Array xyz = xt::zeros<double>({samples, 3});
    Array fhxyz = xt::zeros<double>({samples, value_size});
    for (int i = 0; i < samples; ++i) {
        xs[i] = xmin + distribution(generator)*(xmax-xmin);
        ys[i] = ymin + distribution(generator)*(ymax-ymin);
        zs[i] = zmin + distribution(generator)*(zmax-zmin);
        xyz(i, 0) = xs[i];
        xyz(i, 1) = ys[i];
        xyz(i, 2) = zs[i];
    }
    Vec fx = f(xs, ys, zs);
    this->evaluate_batch(xyz, fhxyz);
    for (int i = 0; i < samples; ++i) {
        double diff = 0.;
        for (int l = 0; l < value_size; ++l) {
            diff += std::pow(fx[value_size*i+l]-fhxyz(i, l), 2);
        }
        diff = std::sqrt(diff);
        err += diff;
        errsq += diff*diff;
    }
    double mean = err/samples;
    double std = std::sqrt((errsq - err*err/samples)/(samples-1)/samples);
    return std::make_pair(mean-std, mean+std);
}



Vec linspace(double min, double max, int n, bool endpoint) {
    Vec res(n, 0.);
    if(endpoint) {
        double h = (max-min)/(n-1);
        for (int i = 0; i < n; ++i)
            res[i] = min + i*h;
    } else {
        double h = (max-min)/n;
        for (int i = 0; i < n; ++i)
            res[i] = min + i*h;
    }
    return res;
}
