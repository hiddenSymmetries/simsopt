#pragma once
#include <memory>
#include <vector>
#include "boozermagneticfield.h"
#include "shearalfvenwave.h"
#include "regular_grid_interpolant_3d.h"
#include "tracing_helpers.h"

using std::array;
using std::shared_ptr;
using std::vector;
using std::tuple;

tuple<vector<std::array<double, 6>>, vector<std::array<double, 7>>>
particle_guiding_center_boozer_perturbed_tracing(
        shared_ptr<ShearAlfvenWave> perturbed_field,
        std::array<double, 3> stz_init,
        double m,
        double q,
        double vtotal,
        double vtang,
        double mu,
        double tmax,
        double abstol,
        double reltol,
        bool vacuum,
        bool noK,
        vector<double> zetas,
        vector<double> omegas,
        vector<shared_ptr<StoppingCriterion>> stopping_criteria,
        double dt_save=1e-6,
        bool zetas_stop=false,
        bool vpars_stop=false,
        bool forget_exact_path=false,
        int axis=0,
        vector<double> vpars={});


tuple<vector<std::array<double, 5>>, vector<std::array<double, 6>>>
particle_guiding_center_boozer_tracing(
        shared_ptr<BoozerMagneticField> field, std::array<double, 3> stz_init,
        double m, double q, double vtotal, double vtang, double tmax, double dt, double abstol, double reltol, double roottol,
        bool vacuum, bool noK, bool solveSympl, vector<double> zetas={}, vector<double> omegas={},
        vector<shared_ptr<StoppingCriterion>> stopping_criteria={}, double dt_save=1e-6, vector<double> vpars={}, bool zetas_stop=false, bool vpars_stop=false, bool forget_exact_path=false, int axis=0, bool predictor_step=true);
