__global__ void GPU_rsqrt_biot_savart_B(int num_points, int num_quad_points, Vec3d *points, Vec3d *gamma, Vec3d *dgamma_by_dphi, Vec3d *B) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_points) {
        // initialize
        double B_x  = 0.;
        double B_y  = 0.;
        double B_z  = 0.;
        for (int tile = 0; tile < gridDim.x; tile++) {
            // shared memory
            __shared__ Vec3d share_gamma[BLOCK_SIZE];
            __shared__ Vec3d share_dgamma_by_dphi[BLOCK_SIZE];
            share_gamma[threadIdx.x] = gamma[tile*blockDim.x + threadIdx.x];
            share_dgamma_by_dphi[threadIdx.x] = dgamma_by_dphi[tile*blockDim.x + threadIdx.x];
            __syncthreads();

            // In block, compute B-S
            //#pragma unroll
            for (int j = 0; j < BLOCK_SIZE; j++){
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
            __syncthreads();
        }
        B[i].x = B_x; B[i].y = B_y; B[i].z = B_z;
    }
}
