#include "xtensor-python/pyarray.hpp"
typedef xt::pyarray<double> Array;

Array DommaschkB(Array& mArray, Array& nArray, Array& coeffs, Array& points);
Array DommaschkdB(Array& mArray, Array& nArray, Array& coeffs, Array& points);
