#include "regular_grid_interpolant_3d.h"

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

template<int degree>
void RegularGridInterpolant3D<degree>::interpolate(std::function<Vec(double, double, double)> &f) {
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
}

template<int degree>
void RegularGridInterpolant3D<degree>::interpolate_batch(std::function<Vec(Vec, Vec, Vec)> &f) {
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
}


template<int degree>
Vec RegularGridInterpolant3D<degree>::evaluate(double x, double y, double z){
    int xidx = int(x*nx); // find idx so that xsmesh[xidx] <= x <= xs[xidx+1]
    int yidx = int(y*ny);
    int zidx = int(z*nz);
    if(xidx < 0 || xidx >= nx)
        throw std::runtime_error(fmt::format("xidxs={} not within [0, {}]", xidx, 0, nx-1));
    if(yidx < 0 || yidx >= ny)
        throw std::runtime_error(fmt::format("yidxs={} not within [0, {}]", yidx, 0, ny-1));
    if(zidx < 0 || zidx >= nz)
        throw std::runtime_error(fmt::format("zidxs={} not within [0, {}]", zidx, 0, nz-1));

    for (int i = 0; i < degree+1; ++i) {
        for (int j = 0; j < degree+1; ++j) {
            //for (int k = 0; k < degree+1; ++k) {
            //    int offset = padded_value_size*idx_dof(xidx*degree+i, yidx*degree+j, zidx*degree+k);
            //    int offset_local = padded_value_size*idx_dof_local(i, j, k);
            //    memcpy(vals_local.data()+offset_local, vals.data()+offset, padded_value_size*sizeof(double));
            //}
            int offset = padded_value_size*idx_dof(xidx*degree+i, yidx*degree+j, zidx*degree+0);
            int offset_local = padded_value_size*idx_dof_local(i, j, 0);
            memcpy(vals_local.data()+offset_local, vals.data()+offset, (degree+1)*padded_value_size*sizeof(double));
        }
    }

    double xlocal = nx*(x-xsmesh[xidx]);
    double ylocal = ny*(y-ysmesh[yidx]);
    double zlocal = nz*(z-zsmesh[zidx]);
    if(xlocal < 0. || xlocal > 1.)
        throw std::runtime_error(fmt::format("xlocal={} not within [0, 1]", xlocal));
    if(ylocal < 0. || ylocal > 1.)
        throw std::runtime_error(fmt::format("ylocal={} not within [0, 1]", ylocal));
    if(zlocal < 0. || zlocal > 1.)
        throw std::runtime_error(fmt::format("zlocal={} not within [0, 1]", zlocal));
    //std::cout << "local coordinates=(" << xlocal << ", " << ylocal << ", " << zlocal << ")" << std::endl;
    return evaluate_local(xlocal, ylocal, zlocal);
}

template<int degree>
Vec RegularGridInterpolant3D<degree>::evaluate_local(double x, double y, double z)
{
    Vec res(value_size, 0.);
    for (int l = 0; l < value_size; ++l) {
        double sumi = 0.;
        for (int i = 0; i < degree+1; ++i) {
            double sumj = 0.;
            for (int j = 0; j < degree+1; ++j) {
                double sumk = 0.;
                for (int k = 0; k < degree+1; ++k) {
                    double pkz = basis_fun<degree>(k, z);
                    int offset_local = padded_value_size*idx_dof_local(i, j, k);
                    sumk += vals_local[offset_local+l] * pkz;
                }
                double pjy = basis_fun<degree>(j, y);
                sumj += pjy * sumk;
            }
            double pix = basis_fun<degree>(i, x);
            sumi += pix * sumj;
        }
        res[l] = sumi;
    }
    return res;
}

template<int degree>
std::pair<double, double> RegularGridInterpolant3D<degree>::estimate_error(std::function<Vec(double, double, double)> &f, int samples) {
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(0.0, +1.0);
    double err = 0;
    double errsq = 0;
    for (int i = 0; i < samples; ++i) {
        double x = distribution(generator);
        double y = distribution(generator);
        double z = distribution(generator);
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


template class RegularGridInterpolant3D<1>;
template class RegularGridInterpolant3D<2>;
template class RegularGridInterpolant3D<3>;
template class RegularGridInterpolant3D<4>;



