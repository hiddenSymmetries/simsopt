#include "biot_savart_cuda.h"

#define BLOCK_SIZE 256

__global__ void biot_savart_cuda_device(int ntargets, int nsources, Vec3d *points, Vec3d *gamma, Vec3d *dgamma_by_dphi, Vec3d *B) {
    /*
       CUDA implementation of the Biot Savart kernel.
       This kernel expects to be executed with

       dim3 nBlocks2d;
       nBlocks2d.x = (ntargets + BLOCK_SIZE - 1) / BLOCK_SIZE;
       nBlocks2d.y = (nsources + BLOCK_SIZE - 1) / BLOCK_SIZE;
       biot_savart_cuda_device<<<nBlocks2d, BLOCK_SIZE>>>(ntargets, nsources, gpu_points, gpu_gamma, gpu_dgamma_by_dphi, gpu_B);

       where BLOCK_SIZE is a multiple of 32.

     */
    int tile = blockIdx.y;
    // shared memory
    __shared__ Vec3d share_gamma[BLOCK_SIZE];
    __shared__ Vec3d share_dgamma_by_dphi[BLOCK_SIZE];
    int j = tile*blockDim.x + threadIdx.x;
    if(j < nsources){
        share_gamma[threadIdx.x] = gamma[j];
        share_dgamma_by_dphi[threadIdx.x] = dgamma_by_dphi[j];
    }
    __syncthreads();

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < ntargets) {
        double B_x  = 0.;
        double B_y  = 0.;
        double B_z  = 0.;
        // In block, compute B-S
        for (int j = 0; j < BLOCK_SIZE; j++){
            if (tile*blockDim.x + j >= nsources)
                break;
            // compute the vector from target to source
            double diff_x = points[i].x - share_gamma[j].x;
            double diff_y = points[i].y - share_gamma[j].y;
            double diff_z = points[i].z - share_gamma[j].z;
            // compute distance between target and source
            double distSqr = diff_x*diff_x + diff_y*diff_y + diff_z*diff_z;
            double inv_norm_diff = rsqrt(distSqr);
            double invDist3 = inv_norm_diff*inv_norm_diff*inv_norm_diff;
            // compute cross product and reweight using distance
            B_x += invDist3 * (share_dgamma_by_dphi[j].y * diff_z - share_dgamma_by_dphi[j].z * diff_y);
            B_y += invDist3 * (share_dgamma_by_dphi[j].z * diff_x - share_dgamma_by_dphi[j].x * diff_z);
            B_z += invDist3 * (share_dgamma_by_dphi[j].x * diff_y - share_dgamma_by_dphi[j].y * diff_x);
        }
        atomicAdd(&(B[i].x), B_x);
        atomicAdd(&(B[i].y), B_y);
        atomicAdd(&(B[i].z), B_z);
    }
    __syncthreads();
}


void biot_savart_cuda(int ntargets, int nsources, double *points, double *gamma, double *dgamma_by_dphi, double *B) {
    long bytes_targets = 3 * ntargets * sizeof(double);
    long bytes_sources = 3 * nsources * sizeof(double);
    dim3 nBlocks2d;
    nBlocks2d.x = (ntargets + BLOCK_SIZE - 1) / BLOCK_SIZE;
    nBlocks2d.y = (nsources + BLOCK_SIZE - 1) / BLOCK_SIZE;

    Vec3d *gpu_points, *gpu_gamma, *gpu_dgamma_by_dphi, *gpu_B;
    cudaMalloc(&gpu_points, bytes_targets);
    cudaMalloc(&gpu_gamma, bytes_sources);
    cudaMalloc(&gpu_dgamma_by_dphi, bytes_sources);
    cudaMalloc(&gpu_B, bytes_targets);

    cudaDeviceSynchronize();
    cudaMemcpy(gpu_points, points, bytes_targets, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_gamma, gamma, bytes_sources, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_dgamma_by_dphi, dgamma_by_dphi, bytes_sources, cudaMemcpyHostToDevice);
    cudaMemset(gpu_B, 0, bytes_targets);
    biot_savart_cuda_device<<<nBlocks2d, BLOCK_SIZE>>>(ntargets, nsources, gpu_points, gpu_gamma, gpu_dgamma_by_dphi, gpu_B);
    cudaMemcpy(B, gpu_B, bytes_targets, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i=0; i<3*ntargets; i++)
        B[i] *= 1e-7/nsources;
}
