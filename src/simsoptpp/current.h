#pragma once 

template<class Array>
class Current {
    private:
        double value;
    public:
        Current(double value) : value(value) {}
        inline void set_dofs(Array& dofs) { value=dofs.data()[0]; };
        inline Array get_dofs() { return Array({value}); };
        inline double get_value() { return value; }
        inline void set_value(double val) { value = val; }
};
