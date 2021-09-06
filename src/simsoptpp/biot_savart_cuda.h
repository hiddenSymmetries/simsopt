typedef struct { double x, y, z;} Vec3d;
void biot_savart_cuda(int ntargets, int nsources, double *points, double *gamma, double *dgamma_by_dphi, double *B);
