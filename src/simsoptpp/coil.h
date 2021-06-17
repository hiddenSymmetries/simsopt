#pragma once 
#include "curve.h"
#include "current.h"


using std::shared_ptr;

template<class Array>
class Coil {
    public:
        const shared_ptr<Curve<Array>> curve;
        const shared_ptr<Current<Array>> current;
        Coil(shared_ptr<Curve<Array>> curve, shared_ptr<Current<Array>> current) :
            curve(curve), current(current) { }
};
