#pragma once
#include <memory>
#include <vector>
#include "magneticfield.h"

using std::shared_ptr;
using std::vector;
using std::tuple;

template<class Array>
class GuidingCenterRHS {
    private:
        vector<double> BcrossGradAbsB = {0., 0., 0.};
        Array xyz = xt::zeros<double>({1, 3});
    public:
        shared_ptr<MagneticField<Array>> field;
        double m, q, mu;

        GuidingCenterRHS(shared_ptr<MagneticField<Array>> field, double m, double q, double mu)
            : field(field), m(m), q(q), mu(mu) {

            }

        void operator()(const vector<double> &ys, vector<double> &dydt,
                const double t) {
            double x = ys[0];
            double y = ys[1];
            double z = ys[2];
            double vtang = ys[3];

            xyz(0) = x;
            xyz(1) = y;
            xyz(2) = z;

            field->set_points(xyz);
            Array& GradAbsB = field->GradAbsB_ref();
            Array& B = field->B_ref();
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

template<class Array>
class FieldlineRHS {
    private:
        Array xyz = xt::zeros<double>({1, 3});
    public:
        shared_ptr<MagneticField<Array>> field;

        FieldlineRHS(shared_ptr<MagneticField<Array>> field)
            : field(field) {

            }
        void operator()(const vector<double> &ys, vector<double> &dydt,
                const double t) {
            xyz(0) = ys[0];
            xyz(1) = ys[1];
            xyz(2) = ys[2];
            field->set_points(xyz);
            Array& B = field->B_ref();
            dydt[0] = B(0, 0);
            dydt[1] = B(0, 1);
            dydt[2] = B(0, 2);
        }
};

class StoppingCriterion {
    public:
        // Should return true if the Criterion is satisfied.
        virtual bool operator()(int iter, double t, const vector<double>& y) = 0;
};

class IterationStoppingCriterion : public StoppingCriterion{
    private:
        int max_iter;
    public:
        IterationStoppingCriterion(int max_iter) : max_iter(max_iter) { };
        bool operator()(int iter, double t, const vector<double>& y) override {
            return iter>=max_iter;
        };
};

template<class Array>
class LevelsetStoppingCriterion : public StoppingCriterion{
    private:
        shared_ptr<RegularGridInterpolant3D<Array>> levelset;
    public:
        LevelsetStoppingCriterion(shared_ptr<RegularGridInterpolant3D<Array>> levelset) : levelset(levelset) { };
        bool operator()(int iter, double t, const vector<double>& state) override {
            double x = state[0];
            double y = state[1];
            double z = state[2];
            double r = std::sqrt(x*x + y*y);
            double phi = std::atan2(y, x);
            if(phi < 0)
                phi += 2*M_PI;
            double f = levelset->evaluate(r, phi, z)[0];
            //fmt::print("Levelset at xyz=({}, {}, {}), rphiz=({}, {}, {}), f={}\n", x, y, z, r, phi, z, f);
            return f<0;
        };
};


template<class Array>
tuple<vector<double>, vector<vector<double>>> particle_guiding_center_tracing(
        shared_ptr<MagneticField<Array>> field, double xinit, double yinit, double zinit,
        double m, double q, double vtotal, double vtang, double tmax, double tol, vector<shared_ptr<StoppingCriterion>> stopping_criteria);

template<class Array>
tuple<vector<double>, vector<vector<double>>> particle_guiding_center_tracing(
        shared_ptr<MagneticField<Array>> field, double xinit, double yinit, double zinit,
        double m, double q, double vtotal, double vtang, double tmax, double tol){
    return particle_guiding_center_tracing(
            field, xinit, yinit, zinit, m, q, vtotal, vtang, tmax, tol, {});
}

template<class Array>
tuple<vector<double>, vector<vector<double>>, vector<vector<vector<double>>>> fieldline_tracing(
        shared_ptr<MagneticField<Array>> field, double xinit, double yinit, double zinit,
        double tmax, double tol, vector<double> phis, vector<shared_ptr<StoppingCriterion>> stopping_criteria);
