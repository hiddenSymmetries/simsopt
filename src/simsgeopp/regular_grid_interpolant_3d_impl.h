#include "regular_grid_interpolant_3d.h"
#include "xtensor/xlayout.hpp"
#define _USE_MATH_DEFINES
#include <math.h>

#define _EPS_ 1e-10

template<>
double basis_fun<1>(int idx, double x){
    if(idx == 0)
        return 1-x;
    else
        return x;
}

template<>
double basis_fun<2>(int idx, double x){
    if(idx == 0)
        return 2*(x-1)*(x-0.5);
    else if(idx == 1)
        return x*(x-1)*(-4);
    else
        return x*(x-0.5)*2;
}

template<>
double basis_fun<3>(int idx, double x){
    constexpr double onethird = 1./3;
    constexpr double twothirds = 2./3;
    if(idx == 0)
        return (x-onethird)*(x-twothirds)*(x-1)*(-9./2.);
    else if (idx == 1)
        return x*(x-twothirds)*(x-1)*(27./2.);
    else if (idx == 2)
        return x*(x-onethird)*(x-1)*(-27./2);
    else
        return x*(x-onethird)*(x-twothirds)*(9./2.);
}

template<>
double basis_fun<4>(int idx, double x){
    if(idx == 0)
        return (x-0.25)*(x-0.5)*(x-0.75)*(x-1.)/0.09375;
    else if (idx == 1)
        return x*(x-0.5)*(x-0.75)*(x-1.)/(-0.0234375);
    else if (idx == 2)
        return x*(x-0.25)*(x-0.75)*(x-1.)/(0.015625);
    else if (idx == 3)
        return x*(x-0.25)*(x-0.5)*(x-1.)/(-0.0234375);
    else
        return x*(x-0.25)*(x-0.5)*(x-0.75)/(0.09375);
}

template<class Array, int degree>
void RegularGridInterpolant3D<Array, degree>::interpolate(std::function<Vec(double, double, double)> &f) {
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

template<class Array, int degree>
void RegularGridInterpolant3D<Array, degree>::interpolate_batch(std::function<Vec(Vec, Vec, Vec)> &f) {
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

template<class Array, int degree>
void RegularGridInterpolant3D<Array, degree>::build_local_vals(){
    all_local_vals = std::vector<AlignedVec>(
            nx*ny*nz,
            AlignedVec((degree+1)*(degree+1)*(degree+1)*padded_value_size, 0.)
            );
    for (int xidx = 0; xidx < nx; ++xidx) {
        for (int yidx = 0; yidx < ny; ++yidx) {
            for (int zidx = 0; zidx < nz; ++zidx) {
                int meshidx = idx_cell(xidx, yidx, zidx);
                for (int i = 0; i < degree+1; ++i) {
                    for (int j = 0; j < degree+1; ++j) {
                        int offset = padded_value_size*idx_dof(xidx*degree+i, yidx*degree+j, zidx*degree+0);
                        int offset_local = padded_value_size*idx_dof_local(i, j, 0);
                        memcpy(all_local_vals[meshidx].data()+offset_local, vals.data()+offset, (degree+1)*padded_value_size*sizeof(double));
                    }
                }
            }
        }
    }
}


template<class Array, int degree>
void RegularGridInterpolant3D<Array, degree>::evaluate_batch_with_transform(Array& xyz, Array& fxyz){
    if(fxyz.layout() != xt::layout_type::row_major)
          throw std::runtime_error("fxyz needs to be in row-major storage order");
    int npoints = xyz.shape(0);
    for (int i = 0; i < npoints; ++i) {
        double r = std::sqrt(xyz(i, 0)*xyz(i, 0) + xyz(i, 1)*xyz(i, 1));
        double phi = std::atan2(xyz(i, 1), xyz(i, 0)) + M_PI;
        evaluate_inplace(r, phi, xyz(i, 2), &(fxyz(i, 0)));
    }
}

template<class Array, int degree>
void RegularGridInterpolant3D<Array, degree>::evaluate_batch(Array& xyz, Array& fxyz){
    if(fxyz.layout() != xt::layout_type::row_major)
          throw std::runtime_error("fxyz needs to be in row-major storage order");
    int npoints = xyz.shape(0);
    for (int i = 0; i < npoints; ++i) {
        evaluate_inplace(xyz(i, 0), xyz(i, 1), xyz(i, 2), &(fxyz(i, 0)));
    }
}

template<class Array, int degree>
Vec RegularGridInterpolant3D<Array, degree>::evaluate(double x, double y, double z){
    Vec fxyz(value_size, 0.);
    evaluate_inplace(x, y, z, fxyz.data());
    return fxyz;

}

template<class Array, int degree>
void RegularGridInterpolant3D<Array, degree>::evaluate_inplace(double x, double y, double z, double* res){
    if(x < xmin || x >= xmax)
        throw std::runtime_error(fmt::format("x={} not within [{}, {}]", x, xmin, xmax));
    if(y < ymin || y >= ymax)
        throw std::runtime_error(fmt::format("y={} not within [{}, {}]", y, ymin, ymax));
    if(z < zmin || z >= zmax)
        throw std::runtime_error(fmt::format("z={} not within [{}, {}]", z, zmin, zmax));
    int xidx = int(nx*(x-xmin)/(xmax-xmin)); // find idx so that xsmesh[xidx] <= x <= xs[xidx+1]
    int yidx = int(ny*(y-ymin)/(ymax-ymin));
    int zidx = int(nz*(z-zmin)/(zmax-zmin));
    if(xidx < 0 || xidx >= nx)
        throw std::runtime_error(fmt::format("xidxs={} not within [0, {}]", xidx, nx-1));
    if(yidx < 0 || yidx >= ny)
        throw std::runtime_error(fmt::format("yidxs={} not within [0, {}]", yidx, ny-1));
    if(zidx < 0 || zidx >= nz)
        throw std::runtime_error(fmt::format("zidxs={} not within [0, {}]", zidx, nz-1));

    //vals_local = all_local_vals[idx_cell(xidx, yidx, zidx)];
    //for (int i = 0; i < degree+1; ++i) {
    //    for (int j = 0; j < degree+1; ++j) {
    //        //for (int k = 0; k < degree+1; ++k) {
    //        //    int offset = padded_value_size*idx_dof(xidx*degree+i, yidx*degree+j, zidx*degree+k);
    //        //    int offset_local = padded_value_size*idx_dof_local(i, j, k);
    //        //    memcpy(vals_local.data()+offset_local, vals.data()+offset, padded_value_size*sizeof(double));
    //        //}
    //        int offset = padded_value_size*idx_dof(xidx*degree+i, yidx*degree+j, zidx*degree+0);
    //        int offset_local = padded_value_size*idx_dof_local(i, j, 0);
    //        memcpy(vals_local.data()+offset_local, vals.data()+offset, (degree+1)*padded_value_size*sizeof(double));
    //    }
    //}

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

template<class Array, int degree>
void RegularGridInterpolant3D<Array, degree>::evaluate_local(double x, double y, double z, int cell_idx, double* res)
{
    //Vec res(value_size, 0.);
    double* vals_local = all_local_vals[cell_idx].data();
    for(int l=0; l<padded_value_size; l += simdcount) {
        simd_t sumi(0.);
        int offset_local = l;
        for (int i = 0; i < degree+1; ++i) {
            simd_t sumj(0.); 
            for (int j = 0; j < degree+1; ++j) {
                simd_t sumk(0.);
                for (int k = 0; k < degree+1; ++k) {
                    double pkz = basis_fun<degree>(k, z);
                    //int offset_local = padded_value_size*idx_dof_local(i, j, k);
                    //sumk += xsimd::load_aligned(&(vals_local[offset_local]))*pkz;
                    sumk = xsimd::fma(xsimd::load_aligned(&(vals_local[offset_local])), simd_t(pkz), sumk);
                    offset_local += padded_value_size;
                }
                double pjy = basis_fun<degree>(j, y);
                //sumj += pjy * sumk;
                sumj = xsimd::fma(sumk, simd_t(pjy), sumj);
            }
            double pix = basis_fun<degree>(i, x);
            //sumi += pix * sumj;
            sumi = xsimd::fma(sumj, simd_t(pix), sumi);
        }
        for (int ll = 0; ll < std::min(simdcount, value_size-l); ++ll) {
            res[l+ll] = sumi[ll];
        }
        //xsimd::store_unaligned(&(res[l]), sumi);
    }
}

template<class Array, int degree>
std::pair<double, double> RegularGridInterpolant3D<Array, degree>::estimate_error(std::function<Vec(double, double, double)> &f, int samples) {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, +1.0);
    double err = 0;
    double errsq = 0;
    for (int i = 0; i < samples; ++i) {
        double x = xmin + distribution(generator)*(xmax-xmin);
        double y = ymin + distribution(generator)*(ymax-ymin);
        double z = zmin + distribution(generator)*(zmax-zmin);
        Vec fx = f(x, y, z);
        Vec fhx = this->evaluate(x, y, z);
        double diff = 0.;
        for (int l = 0; l < value_size; ++l) {
            diff += std::pow(fx[l]-fhx[l], 2);
        }
        diff = std::sqrt(diff);
        //fmt::print("x={}, y={}, z={}, diff={}\n", x, y, z, diff);
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
