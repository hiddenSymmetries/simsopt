#pragma once 
#include <memory>

template<class Array>
class CurrentBase {
    public:
        virtual std::complex<double> get_value() = 0;
        virtual ~CurrentBase() {}//; = default;
};


template<class Array>
class Current : public CurrentBase<Array>{
    private:
        std::complex<double> value;
    public:
        Current(std::complex<double> value) : value(value) {}
        inline void set_dofs(Array& dofs) { value=dofs.data()[0]; };
        inline Array get_dofs() { return Array({value}); };
        std::complex<double> get_value() override { return value; }
};

