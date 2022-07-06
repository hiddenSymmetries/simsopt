#include "xtensor-python/pyarray.hpp"
typedef xt::pyarray<double> Array;

Array WindingSurfaceB(Array& points, Array& ws_points, Array& ws_normal, Array& K);
Array WindingSurfacedB(Array& points, Array& ws_points, Array& ws_normal, Array& K);
Array WindingSurfaceA(Array& points, Array& ws_points, Array& ws_normal, Array& K);
Array WindingSurfacedA(Array& points, Array& ws_points, Array& ws_normal, Array& K);
