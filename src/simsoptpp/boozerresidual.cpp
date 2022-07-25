#include "boozerresidual.h"

#include "xsimd/xsimd.hpp"
#include "simdhelpers.h"
#include "vec3dsimd.h"



double boozer_residual(double G, double iota, Array& xphi, Array& xtheta, Array& B){
    int nphi = B.shape(0);
    int ntheta = B.shape(1);
    Array res = xt::zeros<double>({nphi, ntheta, 3});
    int num_points = 3 * nphi * ntheta;
    for(int i=0; i<nphi; i++){
        for(int j=0; j<ntheta; j++){
            double B2 = B(i,j,0)*B(i,j,0) + B(i,j,1)*B(i,j,1) + B(i,j,2)*B(i,j,2);
            double rmodB = 1./sqrt(B2);
            for (int k = 0; k < 3; k++){
                res(i, j, k) = G* B(i, j, k)  - B2 * (xphi(i,j,k)+iota*xtheta(i,j,k));
                res(i, j, k) *= rmodB;
            }
        }
    }

    double val = 0.5*xt::sum(res*res)()/num_points;
    return val;
}








// G*dB_dc - 2*np.sum(B[..., None]*dB_dc, axis=2)[:, :, None, :] * tang[..., None] - B2[..., None, None] * (dxphi_dc + iota * dxtheta_dc)
// dB_dc = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_dc, optimize=True)

#include "xsimd/xsimd.hpp"
#include "simdhelpers.h"
#include "vec3dsimd.h"
std::tuple<double, Array> boozer_residual_ds(double G, double iota, Array& B, Array& dB_dx, Array& xphi, Array& xtheta, Array& dx_ds, Array& dxphi_ds, Array& dxtheta_ds){
    int nphi = xphi.shape(0);
    int ntheta = xtheta.shape(1);
    int ndofs = dx_ds.shape(3);
    int num_points = 3 * nphi * ntheta;
    double res = 0;
    Array dres = xt::zeros<double>({ndofs+2});

    constexpr int simd_size = xsimd::simd_type<double>::size;
    auto dx_ds_ij0 = vector_type(ndofs, 0);
    auto dx_ds_ij1 = vector_type(ndofs, 0);
    auto dx_ds_ij2 = vector_type(ndofs, 0);
    auto dxphi_ds_ij0 = vector_type(ndofs, 0);
    auto dxphi_ds_ij1 = vector_type(ndofs, 0);
    auto dxphi_ds_ij2 = vector_type(ndofs, 0);
    auto dxtheta_ds_ij0 = vector_type(ndofs, 0);
    auto dxtheta_ds_ij1 = vector_type(ndofs, 0);
    auto dxtheta_ds_ij2 = vector_type(ndofs, 0);
    
    simd_t it(iota); 
    simd_t GG(G); 
    for(int i=0; i<nphi; i++){
        for(int j=0; j<ntheta; j++){
            double B2ij = B(i,j,0)*B(i,j,0) + B(i,j,1)*B(i,j,1) + B(i,j,2)*B(i,j,2);
            double rB2ij = 1/B2ij;
            double wij = sqrt(rB2ij);

            double tang_ij0 = xphi(i,j,0)+iota*xtheta(i,j,0); 
            double tang_ij1 = xphi(i,j,1)+iota*xtheta(i,j,1); 
            double tang_ij2 = xphi(i,j,2)+iota*xtheta(i,j,2);  

            double resij0 = G*B(i, j, 0)  - B2ij * tang_ij0;
            double resij1 = G*B(i, j, 1)  - B2ij * tang_ij1;
            double resij2 = G*B(i, j, 2)  - B2ij * tang_ij2;
            
            double rtil_ij0, rtil_ij1, rtil_ij2;

            rtil_ij0 = resij0 * wij;
            rtil_ij1 = resij1 * wij;
            rtil_ij2 = resij2 * wij;

            res += 0.5*(rtil_ij0*rtil_ij0 + rtil_ij1*rtil_ij1 + rtil_ij2*rtil_ij2);
            
            simd_t Bij0(B(i, j, 0)); 
            simd_t Bij1(B(i, j, 1)); 
            simd_t Bij2(B(i, j, 2)); 

            simd_t dB_dxij00(dB_dx(i, j, 0, 0));
            simd_t dB_dxij10(dB_dx(i, j, 1, 0));
            simd_t dB_dxij20(dB_dx(i, j, 2, 0));
            simd_t dB_dxij01(dB_dx(i, j, 0, 1));
            simd_t dB_dxij11(dB_dx(i, j, 1, 1));
            simd_t dB_dxij21(dB_dx(i, j, 2, 1));
            simd_t dB_dxij02(dB_dx(i, j, 0, 2));
            simd_t dB_dxij12(dB_dx(i, j, 1, 2));
            simd_t dB_dxij22(dB_dx(i, j, 2, 2));
            
            simd_t bw_ij(wij); 
            simd_t btang_ij0(tang_ij0); 
            simd_t btang_ij1(tang_ij1); 
            simd_t btang_ij2(tang_ij2); 
            
            simd_t brtil_ij0(rtil_ij0); 
            simd_t brtil_ij1(rtil_ij1);
            simd_t brtil_ij2(rtil_ij2);

            for (int m = 0; m < ndofs; ++m) {
                dx_ds_ij0[m] = dx_ds(i,j,0,m);
                dx_ds_ij1[m] = dx_ds(i,j,1,m);
                dx_ds_ij2[m] = dx_ds(i,j,2,m);

                dxphi_ds_ij0[m] = dxphi_ds(i,j,0,m);
                dxphi_ds_ij1[m] = dxphi_ds(i,j,1,m);
                dxphi_ds_ij2[m] = dxphi_ds(i,j,2,m);

                dxtheta_ds_ij0[m] = dxtheta_ds(i,j,0,m);
                dxtheta_ds_ij1[m] = dxtheta_ds(i,j,1,m);
                dxtheta_ds_ij2[m] = dxtheta_ds(i,j,2,m);
             }
            
            for (int m = 0; m < ndofs; m+=simd_size) {
                simd_t dx_ds_ij0m = xs::load_aligned(&dx_ds_ij0[m]);
                simd_t dx_ds_ij1m = xs::load_aligned(&dx_ds_ij1[m]);
                simd_t dx_ds_ij2m = xs::load_aligned(&dx_ds_ij2[m]);

                simd_t dxphi_ds_ij0m = xs::load_aligned(&dxphi_ds_ij0[m]);
                simd_t dxphi_ds_ij1m = xs::load_aligned(&dxphi_ds_ij1[m]);
                simd_t dxphi_ds_ij2m = xs::load_aligned(&dxphi_ds_ij2[m]);

                simd_t dxtheta_ds_ij0m = xs::load_aligned(&dxtheta_ds_ij0[m]);
                simd_t dxtheta_ds_ij1m = xs::load_aligned(&dxtheta_ds_ij1[m]);
                simd_t dxtheta_ds_ij2m = xs::load_aligned(&dxtheta_ds_ij2[m]);

                auto dBij0m = xsimd::fma(dB_dxij00,dx_ds_ij0m,xsimd::fma(dB_dxij10,dx_ds_ij1m,dB_dxij20*dx_ds_ij2m)); 
                auto dBij1m = xsimd::fma(dB_dxij01,dx_ds_ij0m,xsimd::fma(dB_dxij11,dx_ds_ij1m,dB_dxij21*dx_ds_ij2m)); 
                auto dBij2m = xsimd::fma(dB_dxij02,dx_ds_ij0m,xsimd::fma(dB_dxij12,dx_ds_ij1m,dB_dxij22*dx_ds_ij2m));
                
                auto dB2_ijm = 2*(Bij0 * dBij0m + Bij1 * dBij1m + Bij2 * dBij2m);
                auto tang_ij0m = xsimd::fma(it, dxtheta_ds_ij0m , dxphi_ds_ij0m);
                auto tang_ij1m = xsimd::fma(it, dxtheta_ds_ij1m , dxphi_ds_ij1m);
                auto tang_ij2m = xsimd::fma(it, dxtheta_ds_ij2m , dxphi_ds_ij2m);
                    
                auto dresij0m = xsimd::fms(GG , dBij0m , xsimd::fma(dB2_ijm , btang_ij0 , B2ij * tang_ij0m));
                auto dresij1m = xsimd::fms(GG , dBij1m , xsimd::fma(dB2_ijm , btang_ij1 , B2ij * tang_ij1m));
                auto dresij2m = xsimd::fms(GG , dBij2m , xsimd::fma(dB2_ijm , btang_ij2 , B2ij * tang_ij2m));
               
                auto dmodB_ijm = 0.5 * dB2_ijm * wij;
                auto dw_ijm = -dmodB_ijm * rB2ij;
                auto drtil_ij0m = xsimd::fma(dresij0m , bw_ij , dw_ijm * resij0); 
                auto drtil_ij1m = xsimd::fma(dresij1m , bw_ij , dw_ijm * resij1); 
                auto drtil_ij2m = xsimd::fma(dresij2m , bw_ij , dw_ijm * resij2);
                
                auto dresm = xsimd::fma(brtil_ij0, drtil_ij0m, xsimd::fma(brtil_ij1, drtil_ij1m , brtil_ij2 * drtil_ij2m));
                
                int jjlimit = std::min(simd_size, ndofs-m);
                for(int jj = 0; jj < jjlimit; jj++){
                    dres(m+jj) += dresm[jj];
                }
            }
            
            double dres_ij0_iota = -B2ij * xtheta(i, j, 0);
            double dres_ij1_iota = -B2ij * xtheta(i, j, 1);
            double dres_ij2_iota = -B2ij * xtheta(i, j, 2);
             
            double drtil_ij0iota = dres_ij0_iota * wij; 
            double drtil_ij1iota = dres_ij1_iota * wij; 
            double drtil_ij2iota = dres_ij2_iota * wij;
            dres(ndofs + 0) += rtil_ij0 * drtil_ij0iota + rtil_ij1 * drtil_ij1iota + rtil_ij2 * drtil_ij2iota;
            
            double dres_ij0_dG = B(i, j, 0);
            double dres_ij1_dG = B(i, j, 1);
            double dres_ij2_dG = B(i, j, 2);

            double drtil_ij0_dG = wij * dres_ij0_dG;
            double drtil_ij1_dG = wij * dres_ij1_dG;
            double drtil_ij2_dG = wij * dres_ij2_dG;
            dres(ndofs + 1) += rtil_ij0 * drtil_ij0_dG + rtil_ij1 * drtil_ij1_dG + rtil_ij2 * drtil_ij2_dG;

        }
    }

    res/=num_points;
    dres/=num_points;

    auto tup = std::make_tuple(res, dres);
    return tup;
}


std::tuple<double, Array, Array> boozer_residual_ds2(double G, double iota, Array& B, Array& dB_dx, Array& d2B_dx2, Array& xphi, Array& xtheta, Array& dx_ds, Array& dxphi_ds, Array& dxtheta_ds){
    int nphi = xphi.shape(0);
    int ntheta = xtheta.shape(1);
    int ndofs = dx_ds.shape(3);
    int num_points = 3 * nphi * ntheta;
    double res = 0;
    Array dres = xt::zeros<double>({ndofs+2});
    Array d2res = xt::zeros<double>({ndofs+2, ndofs+2});

    constexpr int simd_size = xsimd::simd_type<double>::size;
    auto dx_ds_ij0 = vector_type(ndofs, 0);
    auto dx_ds_ij1 = vector_type(ndofs, 0);
    auto dx_ds_ij2 = vector_type(ndofs, 0);
    auto dxphi_ds_ij0 = vector_type(ndofs, 0);
    auto dxphi_ds_ij1 = vector_type(ndofs, 0);
    auto dxphi_ds_ij2 = vector_type(ndofs, 0);
    auto dxtheta_ds_ij0 = vector_type(ndofs, 0);
    auto dxtheta_ds_ij1 = vector_type(ndofs, 0);
    auto dxtheta_ds_ij2 = vector_type(ndofs, 0);

    auto drtilij0 = vector_type(ndofs+2, 0);
    auto drtilij1 = vector_type(ndofs+2, 0);
    auto drtilij2 = vector_type(ndofs+2, 0);
    
    auto dresij0 = vector_type(ndofs+2, 0);
    auto dresij1 = vector_type(ndofs+2, 0);
    auto dresij2 = vector_type(ndofs+2, 0);

    auto dw_ij = vector_type(ndofs+2, 0);

    auto dtang_ij0 = vector_type(ndofs+2, 0);
    auto dtang_ij1 = vector_type(ndofs+2, 0);
    auto dtang_ij2 = vector_type(ndofs+2, 0);

    auto dB2_ij = vector_type(ndofs+2, 0);
    auto dBij0 = vector_type(ndofs+2, 0);
    auto dBij1 = vector_type(ndofs+2, 0);
    auto dBij2 = vector_type(ndofs+2, 0);
    auto dmodB_ij = vector_type(ndofs+2, 0);
     
    simd_t it(iota); 
    simd_t GG(G); 
    for(int i=0; i<nphi; i++){
        for(int j=0; j<ntheta; j++){
            double B2ij = B(i,j,0)*B(i,j,0) + B(i,j,1)*B(i,j,1) + B(i,j,2)*B(i,j,2);
            double rB2ij = 1/B2ij;
            double wij = sqrt(rB2ij);
            double modB_ij = sqrt(B2ij);
            double powrmodBijthree = wij*wij*wij;

            double tang_ij0 = xphi(i,j,0)+iota*xtheta(i,j,0); 
            double tang_ij1 = xphi(i,j,1)+iota*xtheta(i,j,1); 
            double tang_ij2 = xphi(i,j,2)+iota*xtheta(i,j,2);  

            double resij0 = G*B(i, j, 0)  - B2ij * tang_ij0;
            double resij1 = G*B(i, j, 1)  - B2ij * tang_ij1;
            double resij2 = G*B(i, j, 2)  - B2ij * tang_ij2;
            
            double rtil_ij0, rtil_ij1, rtil_ij2;

            rtil_ij0 = resij0 * wij;
            rtil_ij1 = resij1 * wij;
            rtil_ij2 = resij2 * wij;

            res += 0.5*(rtil_ij0*rtil_ij0 + rtil_ij1*rtil_ij1 + rtil_ij2*rtil_ij2);
            
            simd_t Bij0(B(i, j, 0)); 
            simd_t Bij1(B(i, j, 1)); 
            simd_t Bij2(B(i, j, 2)); 

            simd_t dB_dxij00(dB_dx(i, j, 0, 0));
            simd_t dB_dxij10(dB_dx(i, j, 1, 0));
            simd_t dB_dxij20(dB_dx(i, j, 2, 0));
            simd_t dB_dxij01(dB_dx(i, j, 0, 1));
            simd_t dB_dxij11(dB_dx(i, j, 1, 1));
            simd_t dB_dxij21(dB_dx(i, j, 2, 1));
            simd_t dB_dxij02(dB_dx(i, j, 0, 2));
            simd_t dB_dxij12(dB_dx(i, j, 1, 2));
            simd_t dB_dxij22(dB_dx(i, j, 2, 2));
            
            simd_t bw_ij(wij); 
            simd_t btang_ij0(tang_ij0); 
            simd_t btang_ij1(tang_ij1); 
            simd_t btang_ij2(tang_ij2); 
            
            simd_t brtil_ij0(rtil_ij0); 
            simd_t brtil_ij1(rtil_ij1);
            simd_t brtil_ij2(rtil_ij2);

            for (int m = 0; m < ndofs; ++m) {
                dx_ds_ij0[m] = dx_ds(i,j,0,m);
                dx_ds_ij1[m] = dx_ds(i,j,1,m);
                dx_ds_ij2[m] = dx_ds(i,j,2,m);

                dxphi_ds_ij0[m] = dxphi_ds(i,j,0,m);
                dxphi_ds_ij1[m] = dxphi_ds(i,j,1,m);
                dxphi_ds_ij2[m] = dxphi_ds(i,j,2,m);

                dxtheta_ds_ij0[m] = dxtheta_ds(i,j,0,m);
                dxtheta_ds_ij1[m] = dxtheta_ds(i,j,1,m);
                dxtheta_ds_ij2[m] = dxtheta_ds(i,j,2,m);

                dmodB_ij[m] = dxtheta_ds(i,j,0,m);
                dmodB_ij[m] = dxtheta_ds(i,j,1,m);
                dmodB_ij[m] = dxtheta_ds(i,j,2,m);
            }
            
            for (int m = 0; m < ndofs; m+=simd_size) {
                simd_t dx_ds_ij0m = xs::load_aligned(&dx_ds_ij0[m]);
                simd_t dx_ds_ij1m = xs::load_aligned(&dx_ds_ij1[m]);
                simd_t dx_ds_ij2m = xs::load_aligned(&dx_ds_ij2[m]);

                simd_t dxphi_ds_ij0m = xs::load_aligned(&dxphi_ds_ij0[m]);
                simd_t dxphi_ds_ij1m = xs::load_aligned(&dxphi_ds_ij1[m]);
                simd_t dxphi_ds_ij2m = xs::load_aligned(&dxphi_ds_ij2[m]);

                simd_t dxtheta_ds_ij0m = xs::load_aligned(&dxtheta_ds_ij0[m]);
                simd_t dxtheta_ds_ij1m = xs::load_aligned(&dxtheta_ds_ij1[m]);
                simd_t dxtheta_ds_ij2m = xs::load_aligned(&dxtheta_ds_ij2[m]);

                auto dBij0m = xsimd::fma(dB_dxij00,dx_ds_ij0m,xsimd::fma(dB_dxij10,dx_ds_ij1m,dB_dxij20*dx_ds_ij2m)); 
                auto dBij1m = xsimd::fma(dB_dxij01,dx_ds_ij0m,xsimd::fma(dB_dxij11,dx_ds_ij1m,dB_dxij21*dx_ds_ij2m)); 
                auto dBij2m = xsimd::fma(dB_dxij02,dx_ds_ij0m,xsimd::fma(dB_dxij12,dx_ds_ij1m,dB_dxij22*dx_ds_ij2m));
                
                auto dB2_ijm = 2*(Bij0 * dBij0m + Bij1 * dBij1m + Bij2 * dBij2m);
                auto tang_ij0m = xsimd::fma(it, dxtheta_ds_ij0m , dxphi_ds_ij0m);
                auto tang_ij1m = xsimd::fma(it, dxtheta_ds_ij1m , dxphi_ds_ij1m);
                auto tang_ij2m = xsimd::fma(it, dxtheta_ds_ij2m , dxphi_ds_ij2m);
                    
                auto dresij0m = xsimd::fms(GG , dBij0m , xsimd::fma(dB2_ijm , btang_ij0 , B2ij * tang_ij0m));
                auto dresij1m = xsimd::fms(GG , dBij1m , xsimd::fma(dB2_ijm , btang_ij1 , B2ij * tang_ij1m));
                auto dresij2m = xsimd::fms(GG , dBij2m , xsimd::fma(dB2_ijm , btang_ij2 , B2ij * tang_ij2m));
               
                auto dmodB_ijm = 0.5 * dB2_ijm * wij;
                auto dw_ijm = -dmodB_ijm * rB2ij;
                auto drtil_ij0m = xsimd::fma(dresij0m , bw_ij , dw_ijm * resij0); 
                auto drtil_ij1m = xsimd::fma(dresij1m , bw_ij , dw_ijm * resij1); 
                auto drtil_ij2m = xsimd::fma(dresij2m , bw_ij , dw_ijm * resij2);
                
                auto dresm = xsimd::fma(brtil_ij0, drtil_ij0m, xsimd::fma(brtil_ij1, drtil_ij1m , brtil_ij2 * drtil_ij2m));
                
                int jjlimit = std::min(simd_size, ndofs-m);
                for(int jj = 0; jj < jjlimit; jj++){
                    dres(m+jj) += dresm[jj];

                    drtilij0[m+jj] = drtil_ij0m[jj];
                    drtilij1[m+jj] = drtil_ij1m[jj];
                    drtilij2[m+jj] = drtil_ij2m[jj];
                
                    dB2_ij[m+jj] = dB2_ijm[jj];
                    dtang_ij0[m+jj] = tang_ij0m[jj];
                    dtang_ij1[m+jj] = tang_ij1m[jj];
                    dtang_ij2[m+jj] = tang_ij2m[jj];
                    dresij0[m+jj] = dresij0m[jj];
                    dresij1[m+jj] = dresij1m[jj];
                    dresij2[m+jj] = dresij2m[jj];

                    dBij0[m+jj] = dBij0m[jj];
                    dBij1[m+jj] = dBij1m[jj];
                    dBij2[m+jj] = dBij2m[jj];

                    dw_ij[m+jj] = dw_ijm[jj];

                    dmodB_ij[m+jj] = dmodB_ijm[jj];
                }
            }
            
            double dres_ij0iota = -B2ij * xtheta(i, j, 0);
            double dres_ij1iota = -B2ij * xtheta(i, j, 1);
            double dres_ij2iota = -B2ij * xtheta(i, j, 2);
             
            double drtil_ij0iota = dres_ij0iota * wij; 
            double drtil_ij1iota = dres_ij1iota * wij; 
            double drtil_ij2iota = dres_ij2iota * wij;
            dres(ndofs + 0) += rtil_ij0 * drtil_ij0iota + rtil_ij1 * drtil_ij1iota + rtil_ij2 * drtil_ij2iota;
            
            drtilij0[ndofs + 0] = drtil_ij0iota; 
            drtilij1[ndofs + 0] = drtil_ij1iota; 
            drtilij2[ndofs + 0] = drtil_ij2iota; 


            double dres_ij0_dG = B(i, j, 0);
            double dres_ij1_dG = B(i, j, 1);
            double dres_ij2_dG = B(i, j, 2);

            double drtil_ij0_dG = wij * dres_ij0_dG;
            double drtil_ij1_dG = wij * dres_ij1_dG;
            double drtil_ij2_dG = wij * dres_ij2_dG;
            dres(ndofs + 1) += rtil_ij0 * drtil_ij0_dG + rtil_ij1 * drtil_ij1_dG + rtil_ij2 * drtil_ij2_dG;

            drtilij0[ndofs + 1] = drtil_ij0_dG; 
            drtilij1[ndofs + 1] = drtil_ij1_dG; 
            drtilij2[ndofs + 1] = drtil_ij2_dG; 
            
            // outer product d_rij0_dm (x) d_rij0_dm
            for(int m = 0; m < ndofs + 2; m++){
                simd_t drtilij0_dm(drtilij0[m]);
                simd_t drtilij1_dm(drtilij1[m]);
                simd_t drtilij2_dm(drtilij2[m]);
                for(int n = m; n < ndofs + 2; n+=simd_size){
                    simd_t drtilij0_dn = xs::load_aligned(&drtilij0[n]);
                    simd_t drtilij1_dn = xs::load_aligned(&drtilij1[n]);
                    simd_t drtilij2_dn = xs::load_aligned(&drtilij2[n]);
                    simd_t d2res_mn = drtilij0_dm * drtilij0_dn + drtilij1_dm * drtilij1_dn + drtilij2_dm * drtilij2_dn;
                
                    int jjlimit = std::min(simd_size, ndofs+2-n);
                    for(int jj = 0; jj < jjlimit; jj++){
                        d2res(m, n+jj) += d2res_mn[jj];
                    }
                }
            }

            // rij * d2rij_dmn
            for(int m = 0; m < ndofs; m+=simd_size){
                simd_t dx_ds_ij0m = xsimd::load_aligned(&dx_ds_ij0[m]);
                simd_t dx_ds_ij1m = xsimd::load_aligned(&dx_ds_ij1[m]);
                simd_t dx_ds_ij2m = xsimd::load_aligned(&dx_ds_ij2[m]);
                
                simd_t dxphi_ds_ij0m = xsimd::load_aligned(&dxphi_ds_ij0[m]);
                simd_t dxphi_ds_ij1m = xsimd::load_aligned(&dxphi_ds_ij1[m]);
                simd_t dxphi_ds_ij2m = xsimd::load_aligned(&dxphi_ds_ij2[m]);
                
                simd_t dxtheta_ds_ij0m = xsimd::load_aligned(&dxtheta_ds_ij0[m]);
                simd_t dxtheta_ds_ij1m = xsimd::load_aligned(&dxtheta_ds_ij1[m]);
                simd_t dxtheta_ds_ij2m = xsimd::load_aligned(&dxtheta_ds_ij2[m]);
                
                simd_t dBij0m = xsimd::load_aligned(&dBij0[m]);
                simd_t dBij1m = xsimd::load_aligned(&dBij1[m]);
                simd_t dBij2m = xsimd::load_aligned(&dBij2[m]);
                
                simd_t dB2_ijm = xsimd::load_aligned(&dB2_ij[m]);
                simd_t dmodB_ijm = xsimd::load_aligned(&dmodB_ij[m]);
                simd_t dw_ijm = xsimd::load_aligned(&dw_ij[m]);
                
                simd_t dresij0m = xsimd::load_aligned(&dresij0[m]);
                simd_t dresij1m = xsimd::load_aligned(&dresij1[m]);
                simd_t dresij2m = xsimd::load_aligned(&dresij2[m]);
                
                simd_t dtang_ij0m = xsimd::load_aligned(&dtang_ij0[m]);
                simd_t dtang_ij1m = xsimd::load_aligned(&dtang_ij1[m]);
                simd_t dtang_ij2m = xsimd::load_aligned(&dtang_ij2[m]);

                for(int n = m; n < ndofs; n++){
                    simd_t d2Bij0_mn(0.);
                    simd_t d2Bij1_mn(0.); 
                    simd_t d2Bij2_mn(0.); 
                    for(int l = 0; l < 3; l++){
                        simd_t dx_ds_ijln(dx_ds(i, j, l, n));

                        d2Bij0_mn = xsimd::fma(d2B_dx2(i, j, 0, l, 0) * dx_ds_ij0m , dx_ds_ijln, d2Bij0_mn);
                        d2Bij1_mn = xsimd::fma(d2B_dx2(i, j, 0, l, 1) * dx_ds_ij0m , dx_ds_ijln, d2Bij1_mn);
                        d2Bij2_mn = xsimd::fma(d2B_dx2(i, j, 0, l, 2) * dx_ds_ij0m , dx_ds_ijln, d2Bij2_mn);
                        d2Bij0_mn = xsimd::fma(d2B_dx2(i, j, 1, l, 0) * dx_ds_ij1m , dx_ds_ijln, d2Bij0_mn);
                        d2Bij1_mn = xsimd::fma(d2B_dx2(i, j, 1, l, 1) * dx_ds_ij1m , dx_ds_ijln, d2Bij1_mn);
                        d2Bij2_mn = xsimd::fma(d2B_dx2(i, j, 1, l, 2) * dx_ds_ij1m , dx_ds_ijln, d2Bij2_mn);
                        d2Bij0_mn = xsimd::fma(d2B_dx2(i, j, 2, l, 0) * dx_ds_ij2m , dx_ds_ijln, d2Bij0_mn);
                        d2Bij1_mn = xsimd::fma(d2B_dx2(i, j, 2, l, 1) * dx_ds_ij2m , dx_ds_ijln, d2Bij1_mn);
                        d2Bij2_mn = xsimd::fma(d2B_dx2(i, j, 2, l, 2) * dx_ds_ij2m , dx_ds_ijln, d2Bij2_mn);
                    }
                    
                    auto d2B2_ijmn = 2*(dBij0m*dBij0[n] + dBij1m*dBij1[n] + dBij2m*dBij2[n]+ B(i,j,0) * d2Bij0_mn + B(i,j,1) * d2Bij1_mn + B(i,j,2) * d2Bij2_mn);
                    auto term1_0 = -dtang_ij0[n] * dB2_ijm;
                    auto term1_1 = -dtang_ij1[n] * dB2_ijm;
                    auto term1_2 = -dtang_ij2[n] * dB2_ijm;

                    auto term2_0 = -dtang_ij0m * dB2_ij[n];
                    auto term2_1 = -dtang_ij1m * dB2_ij[n];
                    auto term2_2 = -dtang_ij2m * dB2_ij[n];

                    auto term3_0 = -tang_ij0 * d2B2_ijmn;
                    auto term3_1 = -tang_ij1 * d2B2_ijmn;
                    auto term3_2 = -tang_ij2 * d2B2_ijmn;

                    auto d2res_ij0mn = xsimd::fma(GG , d2Bij0_mn , term1_0) + term2_0 + term3_0;
                    auto d2res_ij1mn = xsimd::fma(GG , d2Bij1_mn , term1_1) + term2_1 + term3_1;
                    auto d2res_ij2mn = xsimd::fma(GG , d2Bij2_mn , term1_2) + term2_2 + term3_2;
                    
                    auto d2modB_ijmn = (2 * B2ij * d2B2_ijmn - dB2_ijm*dB2_ij[n]) * powrmodBijthree / 4. ;
                    auto d2wij_mn = (2. * dmodB_ijm * dmodB_ij[n] - modB_ij * d2modB_ijmn) * powrmodBijthree;
                   
                    auto d2rtil_0mn = dresij0m *dw_ij[n] + dresij0[n] * dw_ijm + d2res_ij0mn * wij + resij0 * d2wij_mn;
                    auto d2rtil_1mn = dresij1m *dw_ij[n] + dresij1[n] * dw_ijm + d2res_ij1mn * wij + resij1 * d2wij_mn;
                    auto d2rtil_2mn = dresij2m *dw_ij[n] + dresij2[n] * dw_ijm + d2res_ij2mn * wij + resij2 * d2wij_mn;
                    
                    auto d2res_mn = rtil_ij0 * d2rtil_0mn + rtil_ij1 * d2rtil_1mn +rtil_ij2 * d2rtil_2mn;

                    int jjlimit = std::min(simd_size, ndofs-m);
                    for(int jj = 0; jj < jjlimit; jj++){
                        d2res(m+jj, n) += d2res_mn[jj];
                    }
                }
                auto d2res_ij0miota = -(dB2_ijm * xtheta(i, j, 0) + B2ij * dxtheta_ds_ij0m); 
                auto d2res_ij1miota = -(dB2_ijm * xtheta(i, j, 1) + B2ij * dxtheta_ds_ij1m);
                auto d2res_ij2miota = -(dB2_ijm * xtheta(i, j, 2) + B2ij * dxtheta_ds_ij2m);
                

                auto d2rtil_ij0miota = d2res_ij0miota * wij + dres_ij0iota * dw_ijm ;
                auto d2rtil_ij1miota = d2res_ij1miota * wij + dres_ij1iota * dw_ijm ;
                auto d2rtil_ij2miota = d2res_ij2miota * wij + dres_ij2iota * dw_ijm ;
                auto d2res_miota = rtil_ij0*d2rtil_ij0miota+rtil_ij1*d2rtil_ij1miota+rtil_ij2*d2rtil_ij2miota;   

                int jjlimit = std::min(simd_size, ndofs-m);
                for(int jj = 0; jj < jjlimit; jj++){
                    d2res(m+jj, ndofs) += d2res_miota[jj];
                }

                auto d2res_ij0mG = dBij0m; 
                auto d2res_ij1mG = dBij1m;
                auto d2res_ij2mG = dBij2m;

                auto d2rtil_ij0mG = d2res_ij0mG * wij + dres_ij0_dG * dw_ijm;
                auto d2rtil_ij1mG = d2res_ij1mG * wij + dres_ij1_dG * dw_ijm;
                auto d2rtil_ij2mG = d2res_ij2mG * wij + dres_ij2_dG * dw_ijm;
                auto d2res_mG = rtil_ij0*d2rtil_ij0mG+rtil_ij1*d2rtil_ij1mG+rtil_ij2*d2rtil_ij2mG;

                for(int jj = 0; jj < jjlimit; jj++){
                    d2res(m+jj, ndofs+1) += d2res_mG[jj];
                }

            }
        }
    }

    res/=num_points;
    dres/=num_points;
    d2res/=num_points;
    
    // symmetrize the Hessian
    for(int m = 0; m < ndofs+2; m++){
        for(int n = m+1; n < ndofs+2; n++){
            d2res(n, m) = d2res(m, n);
        }
    }

    auto tup = std::make_tuple(res, dres, d2res);
    return tup;
}





//            for(int m = 0; m < ndofs; m++){
//                for(int n = m; n < ndofs; n+=simd_size){
//                    simd_t d2Bij0_mn(0.);
//                    simd_t d2Bij1_mn(0.); 
//                    simd_t d2Bij2_mn(0.); 
//                    
//                    simd_t dx_ds_ij0n = xsimd::load_aligned(&dx_ds_ij0[n]);
//                    simd_t dx_ds_ij1n = xsimd::load_aligned(&dx_ds_ij1[n]);
//                    simd_t dx_ds_ij2n = xsimd::load_aligned(&dx_ds_ij2[n]);
//
//                    simd_t dxphi_ds_ij0n = xsimd::load_aligned(&dxphi_ds_ij0[n]);
//                    simd_t dxphi_ds_ij1n = xsimd::load_aligned(&dxphi_ds_ij1[n]);
//                    simd_t dxphi_ds_ij2n = xsimd::load_aligned(&dxphi_ds_ij2[n]);
//
//                    simd_t dxtheta_ds_ij0n = xsimd::load_aligned(&dxtheta_ds_ij0[n]);
//                    simd_t dxtheta_ds_ij1n = xsimd::load_aligned(&dxtheta_ds_ij1[n]);
//                    simd_t dxtheta_ds_ij2n = xsimd::load_aligned(&dxtheta_ds_ij2[n]);
//
//                    simd_t dBij0n = xsimd::load_aligned(&dBij0[n]);
//                    simd_t dBij1n = xsimd::load_aligned(&dBij1[n]);
//                    simd_t dBij2n = xsimd::load_aligned(&dBij2[n]);
//                    
//                    simd_t dB2_ijn = xsimd::load_aligned(&dB2_ij[n]);
//                    simd_t dmodB_ijn = xsimd::load_aligned(&dmodB_ij[n]);
//                    simd_t dw_ijn = xsimd::load_aligned(&dw_ij[n]);
//                    
//                    simd_t dresij0n = xsimd::load_aligned(&dresij0[n]);
//                    simd_t dresij1n = xsimd::load_aligned(&dresij1[n]);
//                    simd_t dresij2n = xsimd::load_aligned(&dresij2[n]);
//                    
//                    for(int k = 0; k < 3; k++){
//                        auto d2Bij0_mn_0 = d2B_dx2(i, j, k, 0, 0) * dx_ds(i,j,k,m) * dx_ds_ij0n;
//                        auto d2Bij1_mn_0 = d2B_dx2(i, j, k, 0, 1) * dx_ds(i,j,k,m) * dx_ds_ij0n;
//                        auto d2Bij2_mn_0 = d2B_dx2(i, j, k, 0, 2) * dx_ds(i,j,k,m) * dx_ds_ij0n;
//                        
//                        auto d2Bij0_mn_1 = d2B_dx2(i, j, k, 1, 0) * dx_ds(i,j,k,m) * dx_ds_ij1n;
//                        auto d2Bij1_mn_1 = d2B_dx2(i, j, k, 1, 1) * dx_ds(i,j,k,m) * dx_ds_ij1n;
//                        auto d2Bij2_mn_1 = d2B_dx2(i, j, k, 1, 2) * dx_ds(i,j,k,m) * dx_ds_ij1n;
//
//                        auto d2Bij0_mn_2 = d2B_dx2(i, j, k, 2, 0) * dx_ds(i,j,k,m) * dx_ds_ij2n;
//                        auto d2Bij1_mn_2 = d2B_dx2(i, j, k, 2, 1) * dx_ds(i,j,k,m) * dx_ds_ij2n;
//                        auto d2Bij2_mn_2 = d2B_dx2(i, j, k, 2, 2) * dx_ds(i,j,k,m) * dx_ds_ij2n;
//                        
//                        d2Bij0_mn  += d2Bij0_mn_0 + d2Bij0_mn_1 + d2Bij0_mn_2;
//                        d2Bij1_mn  += d2Bij1_mn_0 + d2Bij1_mn_1 + d2Bij1_mn_2;
//                        d2Bij2_mn  += d2Bij2_mn_0 + d2Bij2_mn_1 + d2Bij2_mn_2;
//                    }
//
//                    
//                    auto d2B2_ijmn = 2*(dBij0[m]*dBij0n + dBij1[m]*dBij1n + dBij2[m]*dBij2n+ B(i,j,0) * d2Bij0_mn + B(i,j,1) * d2Bij1_mn + B(i,j,2) * d2Bij2_mn);
//                    auto term1_0 = -(dxphi_ds_ij0n + iota * dxtheta_ds_ij0n) * dB2_ij[m];
//                    auto term1_1 = -(dxphi_ds_ij1n + iota * dxtheta_ds_ij1n) * dB2_ij[m];
//                    auto term1_2 = -(dxphi_ds_ij2n + iota * dxtheta_ds_ij2n) * dB2_ij[m];
//
//                    auto term2_0 = -(dxphi_ds(i, j, 0, m) + iota * dxtheta_ds(i, j, 0, m)) * dB2_ijn;
//                    auto term2_1 = -(dxphi_ds(i, j, 1, m) + iota * dxtheta_ds(i, j, 1, m)) * dB2_ijn;
//                    auto term2_2 = -(dxphi_ds(i, j, 2, m) + iota * dxtheta_ds(i, j, 2, m)) * dB2_ijn;
//
//                    auto term3_0 = -(xphi(i,j,0) + iota * xtheta(i,j,0)) * d2B2_ijmn;
//                    auto term3_1 = -(xphi(i,j,1) + iota * xtheta(i,j,1)) * d2B2_ijmn;
//                    auto term3_2 = -(xphi(i,j,2) + iota * xtheta(i,j,2)) * d2B2_ijmn;
//
//                    auto d2res_ij0mn = G * d2Bij0_mn + term1_0 + term2_0 + term3_0;
//                    auto d2res_ij1mn = G * d2Bij1_mn + term1_1 + term2_1 + term3_1;
//                    auto d2res_ij2mn = G * d2Bij2_mn + term1_2 + term2_2 + term3_2;
//                    
//                    auto d2modB_ijmn = (2 * B2ij * d2B2_ijmn - dB2_ij[m]*dB2_ijn) / (4. * sqrt(B2ij*B2ij*B2ij) ) ;
//                    auto d2wij_mn = (2. * dmodB_ij[m] * dmodB_ijn - modB_ij * d2modB_ijmn)/ (modB_ij * modB_ij * modB_ij);
//                   
//                    auto d2rtil_0mn = dresij0[m] *dw_ijn + dresij0n * dw_ij[m] + d2res_ij0mn * wij + resij0 * d2wij_mn;
//                    auto d2rtil_1mn = dresij1[m] *dw_ijn + dresij1n * dw_ij[m] + d2res_ij1mn * wij + resij1 * d2wij_mn;
//                    auto d2rtil_2mn = dresij2[m] *dw_ijn + dresij2n * dw_ij[m] + d2res_ij2mn * wij + resij2 * d2wij_mn;
//                    
//                    auto d2res_mn = rtil_ij0 * d2rtil_0mn + rtil_ij1 * d2rtil_1mn +rtil_ij2 * d2rtil_2mn;
//                    
//                    int jjlimit = std::min(simd_size, ndofs-n);
//                    for(int jj = 0; jj < jjlimit; jj++){
//                        d2res(m, n+jj) += d2res_mn[jj];
//                    }
//                }

//std::tuple<double, Array, Array> boozer_residual_ds2(double G, double iota, Array& B, Array& dB_dx, Array& d2B_dx2, Array& xphi, Array& xtheta, Array& dx_ds, Array& dxphi_ds, Array& dxtheta_ds){
//    int nphi = xphi.shape(0);
//    int ntheta = xtheta.shape(1);
//    int ndofs = dx_ds.shape(3);
//    int num_points = 3 * nphi * ntheta;
//    double res = 0;
//    Array dres = xt::zeros<double>({ndofs+2});
//    Array d2res = xt::zeros<double>({ndofs+2, ndofs+2});
//
//    constexpr int simd_size = xsimd::simd_type<double>::size;
//    auto dx_ds_ij0 = vector_type(ndofs, 0);
//    auto dx_ds_ij1 = vector_type(ndofs, 0);
//    auto dx_ds_ij2 = vector_type(ndofs, 0);
//    auto dxphi_ds_ij0 = vector_type(ndofs, 0);
//    auto dxphi_ds_ij1 = vector_type(ndofs, 0);
//    auto dxphi_ds_ij2 = vector_type(ndofs, 0);
//    auto dxtheta_ds_ij0 = vector_type(ndofs, 0);
//    auto dxtheta_ds_ij1 = vector_type(ndofs, 0);
//    auto dxtheta_ds_ij2 = vector_type(ndofs, 0);
//
//    auto drtilij0 = vector_type(ndofs+2, 0);
//    auto drtilij1 = vector_type(ndofs+2, 0);
//    auto drtilij2 = vector_type(ndofs+2, 0);
//    
//    auto dresij0 = vector_type(ndofs+2, 0);
//    auto dresij1 = vector_type(ndofs+2, 0);
//    auto dresij2 = vector_type(ndofs+2, 0);
//
//    auto dw_ij = vector_type(ndofs+2, 0);
//
//    auto dtang_ij0 = vector_type(ndofs+2, 0);
//    auto dtang_ij1 = vector_type(ndofs+2, 0);
//    auto dtang_ij2 = vector_type(ndofs+2, 0);
//
//    auto dB2_ij = vector_type(ndofs+2, 0);
//    auto dBij0 = vector_type(ndofs+2, 0);
//    auto dBij1 = vector_type(ndofs+2, 0);
//    auto dBij2 = vector_type(ndofs+2, 0);
//
//    auto dmodB_ij = vector_type(ndofs+2, 0);
//
//     
//    simd_t it(iota); 
//    simd_t GG(G); 
//    for(int i=0; i<nphi; i++){
//        for(int j=0; j<ntheta; j++){
//            double B2ij = B(i,j,0)*B(i,j,0) + B(i,j,1)*B(i,j,1) + B(i,j,2)*B(i,j,2);
//            double rB2ij = 1/B2ij;
//            double wij = sqrt(rB2ij);
//            double modB_ij = sqrt(B2ij);
//
//            double tang_ij0 = xphi(i,j,0)+iota*xtheta(i,j,0); 
//            double tang_ij1 = xphi(i,j,1)+iota*xtheta(i,j,1); 
//            double tang_ij2 = xphi(i,j,2)+iota*xtheta(i,j,2);  
//
//            double resij0 = G*B(i, j, 0)  - B2ij * tang_ij0;
//            double resij1 = G*B(i, j, 1)  - B2ij * tang_ij1;
//            double resij2 = G*B(i, j, 2)  - B2ij * tang_ij2;
//            
//            double rtil_ij0, rtil_ij1, rtil_ij2;
//
//            rtil_ij0 = resij0 * wij;
//            rtil_ij1 = resij1 * wij;
//            rtil_ij2 = resij2 * wij;
//
//            res += 0.5*(rtil_ij0*rtil_ij0 + rtil_ij1*rtil_ij1 + rtil_ij2*rtil_ij2);
//            
//            simd_t Bij0(B(i, j, 0)); 
//            simd_t Bij1(B(i, j, 1)); 
//            simd_t Bij2(B(i, j, 2)); 
//
//            simd_t dB_dxij00(dB_dx(i, j, 0, 0));
//            simd_t dB_dxij10(dB_dx(i, j, 1, 0));
//            simd_t dB_dxij20(dB_dx(i, j, 2, 0));
//            simd_t dB_dxij01(dB_dx(i, j, 0, 1));
//            simd_t dB_dxij11(dB_dx(i, j, 1, 1));
//            simd_t dB_dxij21(dB_dx(i, j, 2, 1));
//            simd_t dB_dxij02(dB_dx(i, j, 0, 2));
//            simd_t dB_dxij12(dB_dx(i, j, 1, 2));
//            simd_t dB_dxij22(dB_dx(i, j, 2, 2));
//            
//            simd_t bw_ij(wij); 
//            simd_t btang_ij0(tang_ij0); 
//            simd_t btang_ij1(tang_ij1); 
//            simd_t btang_ij2(tang_ij2); 
//            
//            simd_t brtil_ij0(rtil_ij0); 
//            simd_t brtil_ij1(rtil_ij1);
//            simd_t brtil_ij2(rtil_ij2);
//
//            for (int m = 0; m < ndofs; ++m) {
//                dx_ds_ij0[m] = dx_ds(i,j,0,m);
//                dx_ds_ij1[m] = dx_ds(i,j,1,m);
//                dx_ds_ij2[m] = dx_ds(i,j,2,m);
//
//                dxphi_ds_ij0[m] = dxphi_ds(i,j,0,m);
//                dxphi_ds_ij1[m] = dxphi_ds(i,j,1,m);
//                dxphi_ds_ij2[m] = dxphi_ds(i,j,2,m);
//
//                dxtheta_ds_ij0[m] = dxtheta_ds(i,j,0,m);
//                dxtheta_ds_ij1[m] = dxtheta_ds(i,j,1,m);
//                dxtheta_ds_ij2[m] = dxtheta_ds(i,j,2,m);
//             }
//            
//            for (int m = 0; m < ndofs; m+=simd_size) {
//                simd_t dx_ds_ij0m = xs::load_aligned(&dx_ds_ij0[m]);
//                simd_t dx_ds_ij1m = xs::load_aligned(&dx_ds_ij1[m]);
//                simd_t dx_ds_ij2m = xs::load_aligned(&dx_ds_ij2[m]);
//
//                simd_t dxphi_ds_ij0m = xs::load_aligned(&dxphi_ds_ij0[m]);
//                simd_t dxphi_ds_ij1m = xs::load_aligned(&dxphi_ds_ij1[m]);
//                simd_t dxphi_ds_ij2m = xs::load_aligned(&dxphi_ds_ij2[m]);
//
//                simd_t dxtheta_ds_ij0m = xs::load_aligned(&dxtheta_ds_ij0[m]);
//                simd_t dxtheta_ds_ij1m = xs::load_aligned(&dxtheta_ds_ij1[m]);
//                simd_t dxtheta_ds_ij2m = xs::load_aligned(&dxtheta_ds_ij2[m]);
//
//                auto dBij0m = xsimd::fma(dB_dxij00,dx_ds_ij0m,xsimd::fma(dB_dxij10,dx_ds_ij1m,dB_dxij20*dx_ds_ij2m)); 
//                auto dBij1m = xsimd::fma(dB_dxij01,dx_ds_ij0m,xsimd::fma(dB_dxij11,dx_ds_ij1m,dB_dxij21*dx_ds_ij2m)); 
//                auto dBij2m = xsimd::fma(dB_dxij02,dx_ds_ij0m,xsimd::fma(dB_dxij12,dx_ds_ij1m,dB_dxij22*dx_ds_ij2m));
//                
//                auto dB2_ijm = 2*(Bij0 * dBij0m + Bij1 * dBij1m + Bij2 * dBij2m);
//                auto tang_ij0m = xsimd::fma(it, dxtheta_ds_ij0m , dxphi_ds_ij0m);
//                auto tang_ij1m = xsimd::fma(it, dxtheta_ds_ij1m , dxphi_ds_ij1m);
//                auto tang_ij2m = xsimd::fma(it, dxtheta_ds_ij2m , dxphi_ds_ij2m);
//                    
//                auto dresij0m = xsimd::fms(GG , dBij0m , xsimd::fma(dB2_ijm , btang_ij0 , B2ij * tang_ij0m));
//                auto dresij1m = xsimd::fms(GG , dBij1m , xsimd::fma(dB2_ijm , btang_ij1 , B2ij * tang_ij1m));
//                auto dresij2m = xsimd::fms(GG , dBij2m , xsimd::fma(dB2_ijm , btang_ij2 , B2ij * tang_ij2m));
//               
//                auto dmodB_ijm = 0.5 * dB2_ijm * wij;
//                auto dw_ijm = -dmodB_ijm * rB2ij;
//                auto drtil_ij0m = xsimd::fma(dresij0m , bw_ij , dw_ijm * resij0); 
//                auto drtil_ij1m = xsimd::fma(dresij1m , bw_ij , dw_ijm * resij1); 
//                auto drtil_ij2m = xsimd::fma(dresij2m , bw_ij , dw_ijm * resij2);
//                
//                auto dresm = xsimd::fma(brtil_ij0, drtil_ij0m, xsimd::fma(brtil_ij1, drtil_ij1m , brtil_ij2 * drtil_ij2m));
//                
//                int jjlimit = std::min(simd_size, ndofs-m);
//                for(int jj = 0; jj < jjlimit; jj++){
//                    dres(m+jj) += dresm[jj];
//
//                    drtilij0[m+jj] = drtil_ij0m[jj];
//                    drtilij1[m+jj] = drtil_ij1m[jj];
//                    drtilij2[m+jj] = drtil_ij2m[jj];
//                
//                    dB2_ij[m+jj] = dB2_ijm[jj];
//                    dtang_ij0[m+jj] = tang_ij0m[jj];
//                    dtang_ij1[m+jj] = tang_ij1m[jj];
//                    dtang_ij2[m+jj] = tang_ij2m[jj];
//                    dresij0[m+jj] = dresij0m[jj];
//                    dresij1[m+jj] = dresij1m[jj];
//                    dresij2[m+jj] = dresij2m[jj];
//
//                    dBij0[m+jj] = dBij0m[jj];
//                    dBij1[m+jj] = dBij1m[jj];
//                    dBij2[m+jj] = dBij2m[jj];
//
//                    dw_ij[m+jj] = dw_ijm[jj];
//
//                    dmodB_ij[m+jj] = dmodB_ijm[jj];
//                }
//            }
//            
//            double dres_ij0iota = -B2ij * xtheta(i, j, 0);
//            double dres_ij1iota = -B2ij * xtheta(i, j, 1);
//            double dres_ij2iota = -B2ij * xtheta(i, j, 2);
//             
//            double drtil_ij0iota = dres_ij0iota * wij; 
//            double drtil_ij1iota = dres_ij1iota * wij; 
//            double drtil_ij2iota = dres_ij2iota * wij;
//            dres(ndofs + 0) += rtil_ij0 * drtil_ij0iota + rtil_ij1 * drtil_ij1iota + rtil_ij2 * drtil_ij2iota;
//            
//            drtilij0[ndofs + 0] = drtil_ij0iota; 
//            drtilij1[ndofs + 0] = drtil_ij1iota; 
//            drtilij2[ndofs + 0] = drtil_ij2iota; 
//
//
//            double dres_ij0_dG = B(i, j, 0);
//            double dres_ij1_dG = B(i, j, 1);
//            double dres_ij2_dG = B(i, j, 2);
//
//            double drtil_ij0_dG = wij * dres_ij0_dG;
//            double drtil_ij1_dG = wij * dres_ij1_dG;
//            double drtil_ij2_dG = wij * dres_ij2_dG;
//            dres(ndofs + 1) += rtil_ij0 * drtil_ij0_dG + rtil_ij1 * drtil_ij1_dG + rtil_ij2 * drtil_ij2_dG;
//
//            drtilij0[ndofs + 1] = drtil_ij0_dG; 
//            drtilij1[ndofs + 1] = drtil_ij1_dG; 
//            drtilij2[ndofs + 1] = drtil_ij2_dG; 
//            
//            // outer product d_rij0_dm (x) d_rij0_dm
//            for(int m = 0; m < ndofs + 2; m++){
//                simd_t drtilij0_dm(drtilij0[m]);
//                simd_t drtilij1_dm(drtilij1[m]);
//                simd_t drtilij2_dm(drtilij2[m]);
//                for(int n = m; n < ndofs + 2; n+=simd_size){
//                    simd_t drtilij0_dn = xs::load_aligned(&drtilij0[n]);
//                    simd_t drtilij1_dn = xs::load_aligned(&drtilij1[n]);
//                    simd_t drtilij2_dn = xs::load_aligned(&drtilij2[n]);
//                    simd_t d2res_mn = drtilij0_dm * drtilij0_dn + drtilij1_dm * drtilij1_dn + drtilij2_dm * drtilij2_dn;
//                
//                    int jjlimit = std::min(simd_size, ndofs+2-n);
//                    for(int jj = 0; jj < jjlimit; jj++){
//                        d2res(m, n+jj) += d2res_mn[jj];
//                    }
//                }
//            }
//        
//            // rij * d2rij_dmn
//            for(int m = 0; m < ndofs; m++){
//                for(int n = m; n < ndofs; n++){
//                    double d2Bij0_mn = 0.;
//                    double d2Bij1_mn = 0.; 
//                    double d2Bij2_mn = 0.; 
//                    
//                    for(int k = 0; k < 3; k++){ 
//                        double d2Bij0_mn_0 = d2B_dx2(i, j, k, 0, 0) * dx_ds(i,j,k,m) * dx_ds(i, j, 0, n);
//                        double d2Bij1_mn_0 = d2B_dx2(i, j, k, 0, 1) * dx_ds(i,j,k,m) * dx_ds(i, j, 0, n);
//                        double d2Bij2_mn_0 = d2B_dx2(i, j, k, 0, 2) * dx_ds(i,j,k,m) * dx_ds(i, j, 0, n);
//                        
//                        double d2Bij0_mn_1 = d2B_dx2(i, j, k, 1, 0) * dx_ds(i,j,k,m) * dx_ds(i, j, 1, n);
//                        double d2Bij1_mn_1 = d2B_dx2(i, j, k, 1, 1) * dx_ds(i,j,k,m) * dx_ds(i, j, 1, n);
//                        double d2Bij2_mn_1 = d2B_dx2(i, j, k, 1, 2) * dx_ds(i,j,k,m) * dx_ds(i, j, 1, n);
//
//                        double d2Bij0_mn_2 = d2B_dx2(i, j, k, 2, 0) * dx_ds(i,j,k,m) * dx_ds(i, j, 2, n);
//                        double d2Bij1_mn_2 = d2B_dx2(i, j, k, 2, 1) * dx_ds(i,j,k,m) * dx_ds(i, j, 2, n);
//                        double d2Bij2_mn_2 = d2B_dx2(i, j, k, 2, 2) * dx_ds(i,j,k,m) * dx_ds(i, j, 2, n);
//                        
//                        d2Bij0_mn  += d2Bij0_mn_0 + d2Bij0_mn_1 + d2Bij0_mn_2;
//                        d2Bij1_mn  += d2Bij1_mn_0 + d2Bij1_mn_1 + d2Bij1_mn_2;
//                        d2Bij2_mn  += d2Bij2_mn_0 + d2Bij2_mn_1 + d2Bij2_mn_2;
//                    }
//
//                    
//                    double d2B2_ijmn = 2*(dBij0[m]*dBij0[n] + dBij1[m]*dBij1[n] + dBij2[m]*dBij2[n]+ B(i,j,0) * d2Bij0_mn + B(i,j,1) * d2Bij1_mn + B(i,j,2) * d2Bij2_mn);
//                    double term1_0 = -(dxphi_ds(i, j, 0, n) + iota * dxtheta_ds(i, j, 0, n)) * dB2_ij[m];
//                    double term1_1 = -(dxphi_ds(i, j, 1, n) + iota * dxtheta_ds(i, j, 1, n)) * dB2_ij[m];
//                    double term1_2 = -(dxphi_ds(i, j, 2, n) + iota * dxtheta_ds(i, j, 2, n)) * dB2_ij[m];
//
//                    double term2_0 = -(dxphi_ds(i, j, 0, m) + iota * dxtheta_ds(i, j, 0, m)) * dB2_ij[n];
//                    double term2_1 = -(dxphi_ds(i, j, 1, m) + iota * dxtheta_ds(i, j, 1, m)) * dB2_ij[n];
//                    double term2_2 = -(dxphi_ds(i, j, 2, m) + iota * dxtheta_ds(i, j, 2, m)) * dB2_ij[n];
//
//                    double term3_0 = -(xphi(i,j,0) + iota * xtheta(i,j,0)) * d2B2_ijmn;
//                    double term3_1 = -(xphi(i,j,1) + iota * xtheta(i,j,1)) * d2B2_ijmn;
//                    double term3_2 = -(xphi(i,j,2) + iota * xtheta(i,j,2)) * d2B2_ijmn;
//
//                    double d2res_ij0mn = G * d2Bij0_mn + term1_0 + term2_0 + term3_0;
//                    double d2res_ij1mn = G * d2Bij1_mn + term1_1 + term2_1 + term3_1;
//                    double d2res_ij2mn = G * d2Bij2_mn + term1_2 + term2_2 + term3_2;
//                    
//                    double d2modB_ijmn = (2 * B2ij * d2B2_ijmn - dB2_ij[m]*dB2_ij[n]) / (4. * sqrt(B2ij*B2ij*B2ij) ) ;
//                    double d2wij_mn = (2. * dmodB_ij[m] * dmodB_ij[n] - modB_ij * d2modB_ijmn)/ (modB_ij * modB_ij * modB_ij);
//                   
//                    double d2rtil_0mn = dresij0[m] *dw_ij[n] + dresij0[n] * dw_ij[m] + d2res_ij0mn * wij + resij0 * d2wij_mn;
//                    double d2rtil_1mn = dresij1[m] *dw_ij[n] + dresij1[n] * dw_ij[m] + d2res_ij1mn * wij + resij1 * d2wij_mn;
//                    double d2rtil_2mn = dresij2[m] *dw_ij[n] + dresij2[n] * dw_ij[m] + d2res_ij2mn * wij + resij2 * d2wij_mn;
//                    
//                    d2res(m, n) += rtil_ij0 * d2rtil_0mn + rtil_ij1 * d2rtil_1mn +rtil_ij2 * d2rtil_2mn;
//                }
//                double d2res_ij0miota = -(dB2_ij[m] * xtheta(i, j, 0) + B2ij * dxtheta_ds(i, j, 0, m)); 
//                double d2res_ij1miota = -(dB2_ij[m] * xtheta(i, j, 1) + B2ij * dxtheta_ds(i, j, 1, m));
//                double d2res_ij2miota = -(dB2_ij[m] * xtheta(i, j, 2) + B2ij * dxtheta_ds(i, j, 2, m));
//                
//
//                double d2rtil_ij0miota = d2res_ij0miota * wij + dres_ij0iota * dw_ij[m] ;
//                double d2rtil_ij1miota = d2res_ij1miota * wij + dres_ij1iota * dw_ij[m] ;
//                double d2rtil_ij2miota = d2res_ij2miota * wij + dres_ij2iota * dw_ij[m] ;
//                d2res(m, ndofs) += rtil_ij0*d2rtil_ij0miota+rtil_ij1*d2rtil_ij1miota+rtil_ij2*d2rtil_ij2miota;   
//
//                double d2res_ij0mG = dBij0[m]; 
//                double d2res_ij1mG = dBij1[m];
//                double d2res_ij2mG = dBij2[m];
//
//                double d2rtil_ij0mG = d2res_ij0mG * wij + dres_ij0_dG * dw_ij[m] ;
//                double d2rtil_ij1mG = d2res_ij1mG * wij + dres_ij1_dG * dw_ij[m] ;
//                double d2rtil_ij2mG = d2res_ij2mG * wij + dres_ij2_dG * dw_ij[m] ;
//                d2res(m, ndofs+1) += rtil_ij0*d2rtil_ij0mG+rtil_ij1*d2rtil_ij1mG+rtil_ij2*d2rtil_ij2mG;
//            }
//        }
//    }
//
//    res/=num_points;
//    dres/=num_points;
//    d2res/=num_points;
//    
//    // symmetrize the Hessian
//    for(int m = 0; m < ndofs+2; m++){
//        for(int n = m+1; n < ndofs+2; n++){
//            d2res(n, m) = d2res(m, n);
//        }
//    }
//
//    auto tup = std::make_tuple(res, dres, d2res);
//    return tup;
//}





//std::tuple<double, Array> boozer_residual_ds2(double G, double iota, Array& B, Array& dB_dx, Array& xphi, Array& xtheta, Array& dx_ds, Array& dxphi_ds, Array& dxtheta_ds){
//    int nphi = xphi.shape(0);
//    int ntheta = xtheta.shape(1);
//    int ndofs = dx_ds.shape(3);
//    int num_points = 3 * nphi * ntheta;
//    double res = 0;
//    Array dres = xt::zeros<double>({ndofs+2});
//    Array d2res = xt::zeros<double>({ndofs+2, ndofs+2});
//    
//    for(int i=0; i<nphi; i++){
//        for(int j=0; j<ntheta; j++){
//            double B2ij = B(i,j,0)*B(i,j,0) + B(i,j,1)*B(i,j,1) + B(i,j,2)*B(i,j,2);
//            double rB2ij = 1/B2ij;
//            double wij = sqrt(rB2ij);
//
//            double tang_ij0 = xphi(i,j,0)+iota*xtheta(i,j,0); 
//            double tang_ij1 = xphi(i,j,1)+iota*xtheta(i,j,1); 
//            double tang_ij2 = xphi(i,j,2)+iota*xtheta(i,j,2);  
//
//            double resij0 = G*B(i, j, 0)  - B2ij * tang_ij0;
//            double resij1 = G*B(i, j, 1)  - B2ij * tang_ij1;
//            double resij2 = G*B(i, j, 2)  - B2ij * tang_ij2;
//            
//            double rtil_ij0, rtil_ij1, rtil_ij2;
//
//            rtil_ij0 = resij0 * wij;
//            rtil_ij1 = resij1 * wij;
//            rtil_ij2 = resij2 * wij;
//
//            res += 0.5*(rtil_ij0*rtil_ij0 + rtil_ij1*rtil_ij1 + rtil_ij2*rtil_ij2);
//            
//
//
//            for (int m = 0; m < ndofs; ++m) {
//                double dBij0m = dB_dx(i, j, 0, 0)*dx_ds(i,j,0,m)+dB_dx(i, j, 1, 0)*dx_ds(i,j,1,m)+dB_dx(i, j, 2, 0)*dx_ds(i,j,2,m); 
//                double dBij1m = dB_dx(i, j, 0, 1)*dx_ds(i,j,0,m)+dB_dx(i, j, 1, 1)*dx_ds(i,j,1,m)+dB_dx(i, j, 2, 1)*dx_ds(i,j,2,m); 
//                double dBij2m = dB_dx(i, j, 0, 2)*dx_ds(i,j,0,m)+dB_dx(i, j, 1, 2)*dx_ds(i,j,1,m)+dB_dx(i, j, 2, 2)*dx_ds(i,j,2,m);
//                
//                double dB2_ijm = 2*(B(i, j, 0) * dBij0m + B(i, j, 1) * dBij1m + B(i, j, 2) * dBij2m);
//                double tang_ij0m = dxphi_ds(i, j, 0, m) + iota * dxtheta_ds(i, j, 0, m);
//                double tang_ij1m = dxphi_ds(i, j, 1, m) + iota * dxtheta_ds(i, j, 1, m);
//                double tang_ij2m = dxphi_ds(i, j, 2, m) + iota * dxtheta_ds(i, j, 2, m);
//                    
//                double dresij0m = G * dBij0m - dB2_ijm * tang_ij0 - B2ij * tang_ij0m;
//                double dresij1m = G * dBij1m - dB2_ijm * tang_ij1 - B2ij * tang_ij1m;
//                double dresij2m = G * dBij2m - dB2_ijm * tang_ij2 - B2ij * tang_ij2m;
//               
//                double dmodB_ijm = 0.5 * dB2_ijm * wij;
//                double dw_ijm = -dmodB_ijm * rB2ij;
//                double drtil_ij0m = dresij0m * wij + dw_ijm * resij0; 
//                double drtil_ij1m = dresij1m * wij + dw_ijm * resij1; 
//                double drtil_ij2m = dresij2m * wij + dw_ijm * resij2;
//                
//                dres(m) += rtil_ij0 * drtil_ij0m + rtil_ij1 * drtil_ij1m + rtil_ij2 * drtil_ij2m;
//            }
//            
//            double dres_ij0_iota = -B2ij * xtheta(i, j, 0);
//            double dres_ij1_iota = -B2ij * xtheta(i, j, 1);
//            double dres_ij2_iota = -B2ij * xtheta(i, j, 2);
//             
//            double drtil_ij0iota = dres_ij0_iota * wij; 
//            double drtil_ij1iota = dres_ij1_iota * wij; 
//            double drtil_ij2iota = dres_ij2_iota * wij;
//            dres(ndofs + 0) += rtil_ij0 * drtil_ij0iota + rtil_ij1 * drtil_ij1iota + rtil_ij2 * drtil_ij2iota;
//            
//            double dres_ij0_dG = B(i, j, 0);
//            double dres_ij1_dG = B(i, j, 1);
//            double dres_ij2_dG = B(i, j, 2);
//
//            double drtil_ij0_dG = wij * dres_ij0_dG;
//            double drtil_ij1_dG = wij * dres_ij1_dG;
//            double drtil_ij2_dG = wij * dres_ij2_dG;
//            dres(ndofs + 1) += rtil_ij0 * drtil_ij0_dG + rtil_ij1 * drtil_ij1_dG + rtil_ij2 * drtil_ij2_dG;
//
//        }
//    }
//
//    res/=num_points;
//    dres/=num_points;
//    auto tup = std::make_tuple(res, dres);
//    return tup;
//}















Array boozer_residual_dc(double G, Array& dB_dc, Array& B, Array& tang, Array& B2, Array& dxphi_dc, double iota, Array& dxtheta_dc){
    int nphi = dB_dc.shape(0);
    int ntheta = dB_dc.shape(1);
    int ndofs = dB_dc.shape(3);
    Array res = xt::zeros<double>({nphi, ntheta, 3, ndofs});
    double* B_dB_dc = new double[ndofs];
    for(int i=0; i<nphi; i++){
        for(int j=0; j<ntheta; j++){
            for (int m = 0; m < ndofs; ++m) {
                B_dB_dc[m] = B(i, j, 0)*dB_dc(i, j, 0, m) + B(i, j, 1)*dB_dc(i, j, 1, m) + B(i, j, 2)*dB_dc(i, j, 2, m);
            }
            double B2ij = B2(i, j);
            for (int d = 0; d < 3; ++d) {
                auto dB_dc_ptr = &(dB_dc(i, j, d, 0));
                auto res_ptr = &(res(i, j, d, 0));
                auto dxphi_dc_ptr = &(dxphi_dc(i, j, d, 0));
                auto dxtheta_dc_ptr = &(dxtheta_dc(i, j, d, 0));
                auto tangijd = tang(i, j, d);
                for (int m = 0; m < ndofs; ++m) {
                    res_ptr[m] = G*dB_dc_ptr[m]- 2*B_dB_dc[m]*tangijd- B2ij * (dxphi_dc_ptr[m] + iota*dxtheta_dc_ptr[m]);
                }
            }
        }
    }
    delete[] B_dB_dc;
    return res;
}
