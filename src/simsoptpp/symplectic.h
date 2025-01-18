#pragma once
#include <memory>
#include <array>
#include <vector>
#include <tuple>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <gsl/gsl_vector.h>
#include "boozermagneticfield.h"
#include "tracing_helpers.h"

using std::vector;
using std::array;
using namespace std;
using namespace xt;
using Array2 = BoozerMagneticField::Array2;

class SymplField {
public:
        // Covaraint components of vector potential
        double Atheta, Azeta;
        // htheta = G/B, hzeta = I/B
        double htheta, hzeta;
        // field strength
        double modB;
        // poloidal momentum: m*htheta*vpar + q*Atheta;
        double ptheta;
        // H = vpar^2/2 + mu B
        double H;
        // vpar = (pzeta - q Azeta)/(m hzeta)
        double vpar;

        // Derivatives of above quantities wrt (s, theta, zeta)
        double dAtheta[3], dAzeta[3];
        double dhtheta[3], dhzeta[3];
        double dmodB[3];
    
        double dvpar[4], dH[4], dptheta[4];

        // mu = vperp^2/(2 B)
        // q = charge, m = mass
        double mu, q, m;

        shared_ptr<BoozerMagneticField> field;
        Array2 stz = xt::zeros<double>({1, 3});

        static constexpr int Size = 4;
        using State = array<double, Size>;
        static constexpr bool axis = false;

        SymplField(shared_ptr<BoozerMagneticField> field, double m, double q, double mu) :
            field(field), m(m), q(q), mu(mu) {
        }
        void eval_field(double x, double y, double z);
        double get_pzeta(double vpar);
        void get_val(double pzeta);
        void get_derivatives(double pzeta);
        double get_dsdt();
        double get_dthdt();
        double get_dzedt();
        double get_dvpardt();
};

tuple<vector<array<double, SymplField::Size+1>>, vector<array<double, SymplField::Size+2>>> solve_sympl(SymplField f, typename SymplField::State y, double tmax, double dt, double roottol, vector<double> zetas, vector<double> omegas, vector<shared_ptr<StoppingCriterion>> stopping_criteria, vector<double> vpars, bool zetas_stop=false, bool vpars_stop=false, bool forget_exact_path = false, bool predictor_step = true, double dt_save=1e-6);

class f_quasi_params{
public:
    double ptheta_old;
    double dt;
    array<double, 4> z;
    SymplField f;
};

class sympl_dense {
public:
    // for interpolation
    array<double, 2> bracket_s = {}; 
    array<double, 2> bracket_dsdt = {};
    array<double, 2> bracket_theta = {}; 
    array<double, 2> bracket_dthdt = {}; 
    array<double, 2> bracket_zeta = {}; 
    array<double, 2> bracket_dzedt = {};
    array<double, 2> bracket_vpar = {}; 
    array<double, 2> bracket_dvpardt = {}; 
    typedef typename SymplField::State State;

    double tlast = 0.0;
    double tcurrent = 0.0;
    void update(double t, double dt, array<double, 4>  y, SymplField f);
    void calc_state(double eval_t, State &temp);
};