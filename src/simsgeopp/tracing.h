#pragma once
#include <memory>
#include <vector>
#include "magneticfield.h"

using std::shared_ptr;
using std::vector;
using std::tuple;

template<template<class, std::size_t, xt::layout_type> class T>
class GuidingCenterRHS {
    private:
        std::array<double, 3> BcrossGradAbsB = {0., 0., 0.};
        typename MagneticField<T>::Tensor2 rphiz = xt::zeros<double>({1, 3});
        shared_ptr<MagneticField<T>> field;
        double m, q, mu;
    public:
        static constexpr int Size = 4;
        using State = std::array<double, Size>;


        GuidingCenterRHS(shared_ptr<MagneticField<T>> field, double m, double q, double mu)
            : field(field), m(m), q(q), mu(mu) {

            }

        void operator()(const State &ys, array<double, 4> &dydt,
                const double t) {
            double x = ys[0];
            double y = ys[1];
            double z = ys[2];
            double vtang = ys[3];

            rphiz(0, 0) = std::sqrt(x*x+y*y);
            rphiz(0, 1) = std::atan2(y, x);
            if(rphiz(0, 1) < 0)
                rphiz(0, 1) += 2*M_PI;
            rphiz(0, 2) = z;

            field->set_points_cyl(rphiz);
            auto& GradAbsB = field->GradAbsB_ref();
            auto& B = field->B_ref();
            double AbsB = field->AbsB_ref()(0);
            BcrossGradAbsB[0] = (B(0, 1) * GradAbsB(0, 2)) - (B(0, 2) * GradAbsB(0, 1));
            BcrossGradAbsB[1] = (B(0, 2) * GradAbsB(0, 0)) - (B(0, 0) * GradAbsB(0, 2));
            BcrossGradAbsB[2] = (B(0, 0) * GradAbsB(0, 1)) - (B(0, 1) * GradAbsB(0, 0));
            double vperp2 = 2*mu*AbsB;
            double fak1 = (vtang/AbsB);
            double fak2 = (m/(q*pow(AbsB, 3)))*(0.5*vperp2 + vtang*vtang);
            dydt[0] = fak1*B(0, 0) + fak2*BcrossGradAbsB[0];
            dydt[1] = fak1*B(0, 1) + fak2*BcrossGradAbsB[1];
            dydt[2] = fak1*B(0, 2) + fak2*BcrossGradAbsB[2];
            dydt[3] = -mu*(B(0, 0)*GradAbsB(0, 0) + B(0, 1)*GradAbsB(0, 1) + B(0, 2)*GradAbsB(0, 2))/AbsB;
        }
};

template<template<class, std::size_t, xt::layout_type> class T>
class FieldlineRHS {
    private:
        typename MagneticField<T>::Tensor2 rphiz = xt::zeros<double>({1, 3});
        shared_ptr<MagneticField<T>> field;
    public:
        static constexpr int Size = 3;
        using State = std::array<double, Size>;

        FieldlineRHS(shared_ptr<MagneticField<T>> field)
            : field(field) {

            }
        void operator()(const array<double, 3> &ys, array<double, 3> &dydt,
                const double t) {
            double x = ys[0];
            double y = ys[1];
            double z = ys[2];
            rphiz(0, 0) = std::sqrt(x*x+y*y);
            rphiz(0, 1) = std::atan2(y, x);
            if(rphiz(0, 1) < 0)
                rphiz(0, 1) += 2*M_PI;
            rphiz(0, 2) = z;
            field->set_points_cyl(rphiz);
            auto& B = field->B_ref();
            dydt[0] = B(0, 0);
            dydt[1] = B(0, 1);
            dydt[2] = B(0, 2);
        }
};

class StoppingCriterion {
    public:
        // Should return true if the Criterion is satisfied.
        virtual bool operator()(int iter, double t, double x, double y, double z) = 0;
        virtual ~StoppingCriterion() {}
};

class IterationStoppingCriterion : public StoppingCriterion{
    private:
        int max_iter;
    public:
        IterationStoppingCriterion(int max_iter) : max_iter(max_iter) { };
        bool operator()(int iter, double t, double x, double y, double z) override {
            return iter>=max_iter;
        };
};

template<class Array>
class LevelsetStoppingCriterion : public StoppingCriterion{
    private:
        shared_ptr<RegularGridInterpolant3D<Array>> levelset;
    public:
        LevelsetStoppingCriterion(shared_ptr<RegularGridInterpolant3D<Array>> levelset) : levelset(levelset) { };
        bool operator()(int iter, double t, double x, double y, double z) override {
            double r = std::sqrt(x*x + y*y);
            double phi = std::atan2(y, x);
            if(phi < 0)
                phi += 2*M_PI;
            double f = levelset->evaluate(r, phi, z)[0];
            //fmt::print("Levelset at xyz=({}, {}, {}), rphiz=({}, {}, {}), f={}\n", x, y, z, r, phi, z, f);
            return f<0;
        };
};


template<template<class, std::size_t, xt::layout_type> class T>
tuple<vector<array<double, 5>>, vector<array<double, 6>>>
particle_guiding_center_tracing(
        shared_ptr<MagneticField<T>> field, double xinit, double yinit, double zinit,
        double m, double q, double vtotal, double vtang, double tmax, double tol, vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria);

template<template<class, std::size_t, xt::layout_type> class T>
tuple<vector<array<double, 4>>, vector<array<double, 5>>>
fieldline_tracing(
        shared_ptr<MagneticField<T>> field, double xinit, double yinit, double zinit,
        double tmax, double tol, vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria);


template<std::size_t m, std::size_t n>
std::array<double, m+n> join(const std::array<double, m>& a, const std::array<double, n>& b){
     std::array<double, m+n> res;
     for (int i = 0; i < m; ++i) {
         res[i] = a[i];
     }
     for (int i = 0; i < n; ++i) {
         res[i+m] = b[i];
     }
     return res;
}

