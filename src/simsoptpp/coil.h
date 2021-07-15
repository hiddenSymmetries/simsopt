#pragma once 
#include "curve.h"
#include "current.h"


using std::shared_ptr;

template<class Array>
class Coil {
    public:
        const shared_ptr<Curve<Array>> curve;
        const shared_ptr<CurrentBase<Array>> current;
        Coil(shared_ptr<Curve<Array>> curve, shared_ptr<CurrentBase<Array>> current) :
            curve(curve), current(current) { }
};
