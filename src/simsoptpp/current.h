#pragma once 
#include <memory>

template<class Array>
class CurrentBase {
    public:
        virtual double get_value() = 0;
        virtual ~CurrentBase() = default;
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

template<class Array>
class ScaledCurrent : public CurrentBase<Array> {
    private:
        const std::shared_ptr<CurrentBase<Array>> current;

    public:
        const double scale;
        ScaledCurrent(std::shared_ptr<CurrentBase<Array>> current, double scale) : current(current), scale(scale) {}
        double get_value() override { return scale * (current->get_value()); }
};

template<class Array>
class CurrentSum : public CurrentBase<Array> {
    private:
        const std::shared_ptr<CurrentBase<Array>> current_A;
        const std::shared_ptr<CurrentBase<Array>> current_B;
    public:
        CurrentSum(std::shared_ptr<CurrentBase<Array>> current_A, std::shared_ptr<CurrentBase<Array>> current_B) : current_A(current_A), current_B(current_B) {};
        double get_value() override { return  (current_A->get_value() + current_B->get_value()); }
};

