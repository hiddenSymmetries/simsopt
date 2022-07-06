#include "winding_surface.h"
#include <math.h>

Array WindingSurfaceB(Array& points, Array& ws_points, Array& K)
{
    int num_points = points.shape(0);
    int num_ws_points = ws_points.shape(0);
    Array B        = xt::zeros<double>({points.shape(0), points.shape(1)});
    // compute BiotSavart from surface current K
    return B;
}

Array WindingSurfacedB(Array& points, Array& ws_points, Array& K)
{
    int num_points = points.shape(0);
    int num_ws_points = ws_points.shape(0);
    Array dB       = xt::zeros<double>({points.shape(0), points.shape(1), points.shape(1)});
    // compute BiotSavart dB from surface current K
    return dB;
}

Array WindingSurfaceA(Array& points, Array& ws_points, Array& K)
{
    int num_points = points.shape(0);
    int num_ws_points = ws_points.shape(0);
    Array A       = xt::zeros<double>({points.shape(0), points.shape(1)});
    // compute BiotSavart A from surface current K
    return A;
}

Array WindingSurfacedA(Array& points, Array& ws_points, Array& K)
{
    int num_points = points.shape(0);
    int num_ws_points = ws_points.shape(0);
    Array dA       = xt::zeros<double>({points.shape(0), points.shape(1), points.shape(1)});
    // compute BiotSavart dA from surface current K
    return dA;
}
