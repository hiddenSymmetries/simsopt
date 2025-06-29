#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>
#include "boozermagneticfield.h"
#include "symplectic.h"
#include "tracing_helpers.h"

using std::shared_ptr;
using std::vector;
using std::tuple;
using std::array;
using Array2 = BoozerMagneticField::Array2;

//
// Evaluates magnetic field in Boozer canonical coordinates (r, theta, zeta)
// and stores results in the SymplField object
//
void SymplField::eval_field(double s, double theta, double zeta)
{
    double Btheta, Bzeta, dBtheta, dBzeta, modB2;

    stz[0, 0] = s; stz[0, 1] = theta; stz[0, 2] = zeta;
    field->set_points(stz);
    // A = psi \nabla \theta - psip \nabla \zeta
    Atheta = s*field->psi0;
    Azeta =  -field->psip()(0);
    dAtheta[0] = field->psi0; // dAthetads
    dAzeta[0] = -field->iota()(0)*field->psi0; // dAzetads
    for (int i=1; i<3; i++)
    {
        dAtheta[i] = 0.0;
        dAzeta[i] = 0.0;
    }

    modB = field->modB()(0);
    dmodB[0] = field->modB_derivs_ref()(0);
    dmodB[1] = field->modB_derivs_ref()(1);
    dmodB[2] = field->modB_derivs_ref()(2);

    Btheta = field->I()(0);
    Bzeta = field->G()(0);
    dBtheta = field->dIds()(0);
    dBzeta = field->dGds()(0);

    modB2 = pow(modB, 2);

    htheta = Btheta/modB;
    hzeta = Bzeta/modB;
    dhtheta[0] = dBtheta/modB - Btheta*dmodB[0]/modB2;
    dhzeta[0] = dBzeta/modB - Bzeta*dmodB[0]/modB2;

    for (int i=1; i<3; i++)
    {
        dhtheta[i] = -Btheta*dmodB[i]/modB2;
        dhzeta[i] = -Bzeta*dmodB[i]/modB2;
    }

}

// compute pzeta for given vpar
double SymplField::get_pzeta(double vpar) {
    return vpar*hzeta*m + q*Azeta; // q*psi0
}

// computes values of H, ptheta and vpar at z=(s, theta, zeta, pzeta)
void SymplField::get_val(double pzeta) {
    vpar = (pzeta - q*Azeta)/(hzeta*m);
    H = m*pow(vpar,2)/2.0 + m*mu*modB;
    ptheta = m*htheta*vpar + q*Atheta;
}

// computes H, ptheta and vpar at z=(s, theta, zeta, pzeta) and their derivatives
void SymplField::get_derivatives(double pzeta) {
    get_val(pzeta);

    for (int i=0; i<3; i++)
        dvpar[i] = -q*dAzeta[i]/(hzeta*m) - (vpar/hzeta)*dhzeta[i];

    dvpar[3]   = 1.0/(hzeta*m); // dvpardpzeta

    for (int i=0; i<3; i++)
        dH[i] = m*vpar*dvpar[i] + m*mu*dmodB[i];
    dH[3]   = m*vpar*dvpar[3]; // dHdpzeta

    for (int i=0; i<3; i++)
        dptheta[i] = m*dvpar[i]*htheta + m*vpar*dhtheta[i] + q*dAtheta[i];

    dptheta[3] = m*htheta*dvpar[3]; // dpthetadpzeta
}

double SymplField::get_dsdt() {
    return (-dH[1] + dptheta[3]*dH[2] - dptheta[2]*dH[3])/dptheta[0];
}

double SymplField::get_dthdt() {
    return dH[0]/dptheta[0];
}

double SymplField::get_dzedt() {
    return (vpar - dH[0]/dptheta[0]*htheta)/hzeta;
}

double SymplField::get_dvpardt() {
    double dsdt = (-dH[1] + dptheta[3]*dH[2] - dptheta[2]*dH[3])/dptheta[0];
    double dthdt = dH[0]/dptheta[0];
    double dzdt = (vpar - dH[0]/dptheta[0]*htheta)/hzeta;
    double dpzdt = (-dH[2] + dH[0]*dptheta[2]/dptheta[0]);

    return dvpar[0] * dsdt + dvpar[1] * dthdt + dvpar[2] * dzdt + dvpar[3] * dpzdt;
}

int f_euler_quasi_func(const gsl_vector* x, void* p, gsl_vector* f)
{
    struct f_quasi_params * params = (struct f_quasi_params *)p;
    const double ptheta_old = (params->ptheta_old);
    const double dt = (params->dt);
    auto z = (params->z);
    SymplField field = (params->f);
    
    const double x0 = gsl_vector_get(x,0);
    const double x1 = gsl_vector_get(x,1);

    field.eval_field(x0, z[1], z[2]);
    field.get_derivatives(x1);

    const double f0 = (field.dptheta[0]*(field.ptheta - ptheta_old)
        + dt*(field.dH[1]*field.dptheta[0] - field.dH[0]*field.dptheta[1]))/pow(field.field->psi0 * field.q, 2); // corresponds with (2.6) in JPP 2020
    const double f1  = (field.dptheta[0]*(x1 - z[3])
        + dt*(field.dH[2]*field.dptheta[0] - field.dH[0]*field.dptheta[2]))/pow(field.field->psi0 * field.q, 2); // corresponds with (2.7) in JPP 2020

    gsl_vector_set(f, 0, f0);
    gsl_vector_set(f, 1, f1);

    return GSL_SUCCESS;
}

double cubic_hermite_interp(double t_last, double t_current, double y_last, double y_current, double dy_last, double dy_current, double t)
{
    double dt = t_current - t_last;
    return (3*dt*pow(t-t_last,2) - 2*pow(t-t_last,3))/pow(dt,3) * y_current 
            + (pow(dt,3)-3*dt*pow(t-t_last,2)+2*pow(t-t_last,3))/pow(dt,3) * y_last
            + pow(t-t_last,2)*(t-t_current)/pow(dt,2) * dy_current
            + (t-t_last)*pow(t-t_current,2)/pow(dt,2) * dy_last;
}

void sympl_dense::update(double t, double dt, array<double, 4>  y, SymplField f) {
    tlast = t;
    tcurrent = t+dt;

    bracket_s[0] = bracket_s[1];
    bracket_theta[0] = bracket_theta[1];
    bracket_zeta[0] = bracket_zeta[1];
    bracket_vpar[0] = bracket_vpar[1];

    bracket_dsdt[0] =  bracket_dsdt[1];
    bracket_dthdt[0] = bracket_dthdt[1];
    bracket_dzedt[0] = bracket_dzedt[1];
    bracket_dvpardt[0] = bracket_dvpardt[1];

    bracket_s[1] = y[0];
    bracket_theta[1] = y[1];
    bracket_zeta[1] = y[2];
    bracket_vpar[1] = y[3];

    bracket_dsdt[1] = f.get_dsdt();
    bracket_dthdt[1] = f.get_dthdt();
    bracket_dzedt[1] = f.get_dzedt();
    bracket_dvpardt[1] = f.get_dvpardt();
}

// Perform hermite interpolation between timesteps for computing stopping criteria
void sympl_dense::calc_state(double eval_t, State &temp) {
    assert (tlast <= eval_t && eval_t <= tcurrent);
    temp[0] = cubic_hermite_interp(tlast, tcurrent, bracket_s[0], bracket_s[1], bracket_dsdt[0], bracket_dsdt[1], eval_t);
    temp[1] = cubic_hermite_interp(tlast, tcurrent, bracket_theta[0], bracket_theta[1], bracket_dthdt[0], bracket_dthdt[1], eval_t);
    temp[2] = cubic_hermite_interp(tlast, tcurrent, bracket_zeta[0], bracket_zeta[1], bracket_dzedt[0], bracket_dzedt[1], eval_t);
    temp[3] = cubic_hermite_interp(tlast, tcurrent, bracket_vpar[0], bracket_vpar[1], bracket_dvpardt[0], bracket_dvpardt[1], eval_t);
}

// see https://github.com/itpplasma/SIMPLE/blob/master/SRC/
//         orbit_symplectic_quasi.f90:timestep_euler1_quasi
tuple<vector<array<double, SymplField::Size+1>>, vector<array<double, SymplField::Size+2>>> solve_sympl(SymplField f, typename SymplField::State y, double tmax, double dt, double roottol, vector<double> thetas, vector<double> zetas, vector<double> omega_thetas, vector<double> omega_zetas, vector<shared_ptr<StoppingCriterion>> stopping_criteria, vector<double> vpars, bool thetas_stop, bool zetas_stop, bool vpars_stop, bool forget_exact_path, bool predictor_step, double dt_save)
{
    double abstol = 0;
    if (zetas.size() > 0 && omega_zetas.size() == 0) {
        omega_zetas.insert(omega_zetas.end(), zetas.size(), 0.);
    } else if (zetas.size() !=  omega_zetas.size()) {
        throw std::invalid_argument("zetas and omega_zetas need to have matching length.");
    }
    if (thetas.size() > 0 && omega_thetas.size() == 0) {
        omega_thetas.insert(omega_thetas.end(), thetas.size(), 0.);
    } else if (thetas.size() !=  omega_thetas.size() and thetas.size() != 0) {
        throw std::invalid_argument("thetas and omega_thetas need to have matching length.");
    }

    typedef typename SymplField::State State;
    vector<array<double, SymplField::Size+1>> res = {};
    vector<array<double, SymplField::Size+2>> res_hits = {};
    double t = 0.0;
    bool stop = false;

    State z = {}; // s, theta, zeta, pzeta
    State temp = {};
    // y = [s, theta, zeta, vpar]
        
    // Translate y to z
    // y = [s, theta, zeta, vpar]
    // z = [s, theta, zeta, pzeta]
    // pzeta = m*vpar*hzeta + q*Azeta
    z[0] = y[0];
    z[1] = y[1];
    z[2] = y[2];
    f.eval_field(z[0], z[1], z[2]);
    z[3] = f.get_pzeta(y[3]);
    f.get_derivatives(z[3]);
    double ptheta_old = f.ptheta;

    double t_last = t;

    // for interpolation
    sympl_dense dense;
    dense.update(t, dt, y, f);

    // set up root solvers
    const gsl_multiroot_fsolver_type * Newt = gsl_multiroot_fsolver_hybrids;
    gsl_multiroot_fsolver *s_euler;
    s_euler = gsl_multiroot_fsolver_alloc(Newt, 2);

    struct f_quasi_params params = {ptheta_old, dt, z, f};
    gsl_multiroot_function F_euler_quasi = {&f_euler_quasi_func, 2, &params};
    gsl_vector* xvec_quasi = gsl_vector_alloc(2);

    int status;
    int iter = 0;
    double s_guess = z[0];
    double pzeta_guess = z[3];

    do {
        // Save initial point
        if (t==0){
            res.push_back(join<1,SymplField::Size>({t}, y));
        }

        params.ptheta_old = ptheta_old;
        params.z = z;
        params.dt = dt;
        gsl_vector_set(xvec_quasi, 0,s_guess);
        gsl_vector_set(xvec_quasi, 1, pzeta_guess);
        gsl_multiroot_fsolver_set(s_euler, &F_euler_quasi, xvec_quasi);

        int root_iter = 0;
        // Solve implicit part of time-step with some quasi-Newton
        // applied to f_euler1_quasi. This corresponds with (2.6)-(2.7) in JPP 2020,
        // which are solved for x = [s, pzeta].
        do
          {
            root_iter++;

            status = gsl_multiroot_fsolver_iterate(s_euler);
            //  printf("iter = %3u x = % .10e % .10e "
            //            "f(x) = % .10e % .10e\n",
            //            iter,
            //            gsl_vector_get (s_euler->x, 0),
            //            gsl_vector_get (s_euler->x, 1),
            //            gsl_vector_get (s_euler->f, 0),
            //            gsl_vector_get (s_euler->f, 1));

            if (status) {  /* check if solver is stuck */
                printf("iter = %3u x = % .10e % .10e "
                        "f(x) = % .10e % .10e\n",
                        root_iter,
                        gsl_vector_get (s_euler->x, 0),
                        gsl_vector_get (s_euler->x, 1),
                        gsl_vector_get (s_euler->f, 0),
                        gsl_vector_get (s_euler->f, 1));
              printf ("status = %s\n", gsl_strerror (status));
              break;
            }
            status = gsl_multiroot_test_residual(s_euler->f, roottol); //tolerance --> roottol ~ 1e-15
          }
        while (status == GSL_CONTINUE && root_iter < 20);
        iter++;

        z[0] = gsl_vector_get(s_euler->x, 0);  // s
        z[3] = gsl_vector_get(s_euler->x, 1);  // pzeta

        // We now evaluate the explicit part of the time-step at [s, pzeta]
        // given by the Euler step.
        f.eval_field(z[0], z[1], z[2]);
        f.get_derivatives(z[3]);

        // z[1] = theta
        // z[2] = zeta
        // dH[0] = dH/dr
        // dptheta[0] = dptheta/dr
        // htheta = G/B
        // hzeta = I/B
        z[1] = z[1] + dt*f.dH[0]/f.dptheta[0]; // (2.9) in JPP 2020
        z[2] = z[2] + dt*(f.vpar - f.dH[0]/f.dptheta[0]*f.htheta)/f.hzeta; // (2.10) in JPP 2020

        // Translate z back to y
        // y = [s, theta, zeta, vpar]
        // z = [s, theta, zeta, pzeta]
        // pzeta = m*vpar*hzeta + q*Azeta
        f.eval_field(z[0], z[1], z[2]);
        f.get_derivatives(z[3]);
        y[0] = z[0];
        y[1] = z[1];
        y[2] = z[2];
        y[3] = f.vpar;
        ptheta_old = f.ptheta;

        dense.update(t, dt, y, f); // tlast = t; tcurrent = t+dt;

        if (predictor_step) {
            s_guess = z[0] + dt*(-f.dH[1] + f.dptheta[3]*f.dH[2] - f.dptheta[2]*f.dH[3])/f.dptheta[0];
            pzeta_guess = z[3] + dt*(- f.dH[2] + f.dH[0]*f.dptheta[2]/f.dptheta[0]); // corresponds with (2.7s) in JPP 2020
        } else {
            s_guess = z[0];
            pzeta_guess = z[3];
        }

        t += dt;

        double t_current = t;

        stop = check_stopping_criteria<SymplField,sympl_dense>(f, iter, res_hits, dense, t_last, t_current, dt, abstol, thetas, zetas, 
            omega_thetas, omega_zetas, stopping_criteria, vpars, thetas_stop, zetas_stop, vpars_stop);

        // Save path if forget_exact_path = False
        if (forget_exact_path == 0) {
            double t_last; 
            // If we have hit a stopping criterion, we still want to save the trajectory up to that point
            if (stop) {
                t_last = res_hits.back()[0];
            } else {
                t_last = t_current;
            }
            // This will give the first save point after t_last
            double t_save_last = dt_save * std::ceil(t_last/dt_save);

            for (double t_save = t_save_last; t_save <= t_last; t_save += dt_save) {
                if (t_save != 0) { // t = 0 is already saved. 
                    dense.calc_state(t_save, temp);
                    res.push_back(join<1,SymplField::Size>({t_save}, {temp}));
                }
            }
        } 

        t_last = t_current;
    } while(t < tmax && !stop);
    // Save t = tmax
    if(!stop){
        t = tmax;
    } else {
        t = res_hits.back()[0];
    }
    dense.calc_state(t, y);
    res.push_back(join<1,SymplField::Size>({t}, {y}));

    gsl_multiroot_fsolver_free(s_euler);
    gsl_vector_free(xvec_quasi);

    return std::make_tuple(res, res_hits);
}