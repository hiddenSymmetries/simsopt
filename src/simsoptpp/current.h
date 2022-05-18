#pragma once 
#include <memory>

template<class Array>
class CurrentBase {
    public:
        virtual double get_value() = 0;
        virtual ~CurrentBase() {}//; = default;
};


template<class Array>
class Current : public CurrentBase<Array>{
    private:
        double value;
    public:
        Current(double value) : value(value) {}
        inline void set_dofs(Array& dofs) { value=dofs.data()[0]; };
        inline Array get_dofs() { return Array({value}); };
        double get_value() override { return value; }
};

