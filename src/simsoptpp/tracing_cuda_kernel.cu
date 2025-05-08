// #include "simdhelpers.h" // import above cuda_runtime to prevent collision for rsqrt
#include <cuda_runtime.h>
#include <iostream>
#include "tracing.h"
#include <math.h>
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;
#include "xtensor-python/pytensor.hpp"     // Numpy bindings
typedef xt::pytensor<double, 2, xt::layout_type::row_major> PyTensor;
using std::shared_ptr;
using std::vector;
namespace py = pybind11;

#define PARTICLES_PER_BLOCK 128

// Particle Data Structure
typedef struct particle_t {
    double state[4];
    double v_perp; // Velocity perpendicular
    double v_total;
    bool has_left;
    double dt;
    double dtmax;
    double t;
    double mu;
    double derivs[42] = {0.0};
    double x_temp[5], x_err[4];
    double s_shape[4], t_shape[4], z_shape[4];
    int i, j, k;
    double interpolation_loc[3];
    bool symmetry_exploited;
    int id;
    int step_attempt, step_accept;
} particle_t;



__host__ __device__ void shape(double x, double* shape){
    shape[0] = (1.0-x)*(2.0-x)*(3.0-x)/6.0;
    shape[1] = x*(2.0-x)*(3.0-x)/2.0;
    shape[2] = x*(x-1.0)*(3.0-x)/2.0;
    shape[3] = x*(x-1.0)*(x-2.0)/6.0;
    return;         
}

__host__  __device__ __forceinline__ void interpolate(particle_t& p, const double* __restrict__ data, double* out, const double* __restrict__ srange_arr, const double* __restrict__ trange_arr, const double* __restrict__ zrange_arr, int n){


    int ns = srange_arr[2];
    int nt = trange_arr[2];
    int nz = zrange_arr[2];

    // Need to interpolate modB, modB derivs, G, and iota

    /*
    From here it remains to perform the necessary interpolations
    As opposed to Cartesian coordinates, we don't need to monitor the surface dist via interpolation
    We also don't need to calculate the derivative of any of the interpolations
    This lets us interpolate everything in one set of nested loops 
    */

    // store interpolants in a common array, indexed the same as the columns of the quad info
    // modB, derivs of modB, G, iota

    // quad pts are indexed s t z (could be improved)
    for(int ii=0; ii<=3; ++ii){ // s grid
        if((p.i+ii) < ns){
            for(int jj=0; jj<=3; ++jj){ // theta grid           
                int wrap_j = (p.j+jj) % nt;
                for(int kk=0; kk<=3; ++kk){ // zeta grid
                    int wrap_k = (p.k+kk) % nz;
                    int row_idx = (p.i+ii)*nt*nz + wrap_j*nz + wrap_k;
                    
                    double shape_val = p.s_shape[ii]*p.t_shape[jj]*p.z_shape[kk];

                    for(int zz=0; zz<n; ++zz){
                        out[zz] += data[n*row_idx + zz]*shape_val;
                    }
                }
            }
        }

    }

}

__host__ __device__ __forceinline__ void calc_phi_alpha(double* __restrict__ phi_info, double* __restrict__ alpha_info, int i, double s, double theta, double z, double time, double psi0, const double* interpolants, double* saw_srange_arr, int* saw_m_arr, int* saw_n_arr, double* saw_phihats_arr, double saw_omega, int saw_nharmonics){
    // interpolants contains modB, dmodBdpsi, dmodBdtheta, dmodbdzeta, G, dGdpsi, I, dIdpsi, iota, diotadpsi
    double s_grid_size = (saw_srange_arr[1] - saw_srange_arr[0]) / (saw_srange_arr[2] - 1);
    int l_idx = (s - saw_srange_arr[0]) / s_grid_size;
    l_idx = l_idx * (l_idx >=0);
    int r_idx = min(l_idx+1, (int)saw_srange_arr[2]-1);
    double s_diff = s - (l_idx*s_grid_size + saw_srange_arr[0]);

    double iota_mn, slope, phi_i, phidot_i, sine, dphi_dpsi_i, dphi_dtheta_i, dphi_dzeta_i;

    // G + iota*I
    double G_iI = interpolants[4] + interpolants[8]*interpolants[6];
    
    // intermediate values
    int m = saw_m_arr[i];
    int n = saw_n_arr[i];
    iota_mn = interpolants[8]*m - n;

    int in_bounds = (s >= saw_srange_arr[0]) *(s <= saw_srange_arr[1]);
    slope = in_bounds*(saw_phihats_arr[r_idx*saw_nharmonics + i] -  saw_phihats_arr[l_idx*saw_nharmonics + i]) / (s_grid_size);
    phi_i = saw_phihats_arr[l_idx*saw_nharmonics + i] + in_bounds* slope * s_diff;
    phidot_i = phi_i * saw_omega*cos(m*theta - n*z + saw_omega*time);
    sine = sin(m*theta - n*z + saw_omega*time);
    dphi_dpsi_i = slope*sine/psi0;
    dphi_dtheta_i = phidot_i * m / saw_omega;
    dphi_dzeta_i = -phidot_i * n / saw_omega;
    
    // phi values: phi, phi_dot, dphi_dpsi, dphi_dtheta, dphi_dzeta
    phi_info[0] = phi_i * sine;
    phi_info[1] = phidot_i;
    phi_info[2] =  dphi_dpsi_i;
    phi_info[3] = dphi_dtheta_i;
    phi_info[4] = dphi_dzeta_i;

    // alpha values: alpha, alpha_dot, dalpha_dpsi, dalpha_dtheta, dalpha_dzeta
    double iota_mn_omega_G_iI = iota_mn / saw_omega * G_iI;
    alpha_info[0] = -phi_i*iota_mn_omega_G_iI;
    alpha_info[1] = -phidot_i * iota_mn_omega_G_iI;

    // sum = (dGds + diotads*I + iota*dIds) 
    // num = (diotads*m) / (G+ iota*I) - (iota*m -n)*sum
    double sum = (interpolants[5] + interpolants[9] * interpolants[6] + interpolants[8]*interpolants[7]);
    double num = interpolants[9]*m / G_iI - iota_mn*sum;
    double denom = (G_iI*G_iI);
    alpha_info[2] = -dphi_dpsi_i*iota_mn_omega_G_iI - (phi_i / saw_omega) * num / denom;
    alpha_info[3] = -dphi_dtheta_i * iota_mn_omega_G_iI;
    alpha_info[4] = -dphi_dzeta_i * iota_mn_omega_G_iI;
}

// out contains derivatives for x , y, z, v_par, and then norm of B and surface distance interpolation
__host__  __device__ void calc_derivs(particle_t& p, double* __restrict__ out, const double* __restrict__ srange_arr, const double* __restrict__ trange_arr, const double* __restrict__ zrange_arr, const double* __restrict__ quadpts_arr, double m, double q, double mu, double psi0,
                                     double* saw_srange_arr, int* saw_m_arr, int* saw_n_arr, double* saw_phihats_arr, double saw_omega, int saw_nharmonics){
    /*
    * Returns     
    out[0] = ds/dtime
    out[1] = dtheta/dtime
    out[2] = dzeta/dtime

    out[3] = dvpar/dtime;
    out[4] = modB;
    */
    // printf("calc_derivs particle %d\n", p.id);

        
    double interpolants[10] = {0.0};
    interpolate(p, quadpts_arr, interpolants, srange_arr, trange_arr, zrange_arr, 10);

    // interpolants now contains
    // modB, dmodBds, dmodBdtheta, dmodBdzeta, G, dGds, I, dIds, iota, diotads

    double s = sqrt(p.x_temp[0]*p.x_temp[0] + p.x_temp[1]*p.x_temp[1]);
    double theta = atan2(p.x_temp[1], p.x_temp[0]);
    double z = p.x_temp[2];
    double v_par = p.x_temp[3];
    double time = p.x_temp[4];
    if(p.symmetry_exploited){ // account for symmetry
        interpolants[2] *= -1.0;
        interpolants[3] *= -1.0;
    }

    // change derivatives wrt s to derivs wrt psi
    interpolants[1] /= psi0; // dmodBdpsi
    interpolants[5] /= psi0; // dGdpsi
    interpolants[7] /= psi0; // dIdpsi
    interpolants[9] /= psi0; // diotadpsi

    // double modB = interpolants[0];
    // double dmodBdpsi = interpolants[1]/psi0;
    // double dmodBdtheta = interpolants[2];
    // double dmodBdzeta = interpolants[3];
    // double G = interpolants[4];
    // double dGdpsi = interpolants[5] / psi0;
    // double I = interpolants[6];
    // double dIdpsi = interpolants[7]/psi0;
    // double iota = interpolants[8];
    // double diotadpsi = interpolants[9]/psi0;

    // contains phi, phidot, dphi_dpsi, dphi_dtheta, dphi_dzeta
    double phi_info_contrib[5];
    double phi = 0.0;
    double phidot = 0.0;
    double dphi_dpsi = 0.0;
    double dphi_dtheta = 0.0;
    double dphi_dzeta = 0.0;

    double alpha_info_contrib[5];
    double alpha = 0.0;
    double alphadot = 0.0;
    double dalpha_dpsi = 0.0;
    double dalpha_dtheta = 0.0;
    double dalpha_dzeta = 0.0;

    // #pragma unroll
    for(int i=0; i<saw_nharmonics; ++i){
        calc_phi_alpha(phi_info_contrib, alpha_info_contrib, i, s, theta, z, time, psi0, interpolants,
                        saw_srange_arr, saw_m_arr, saw_n_arr, saw_phihats_arr, saw_omega, saw_nharmonics);

        phi += phi_info_contrib[0];
        phidot += phi_info_contrib[1];
        dphi_dpsi += phi_info_contrib[2];
        dphi_dtheta += phi_info_contrib[3];
        dphi_dzeta += phi_info_contrib[4];

        alpha += alpha_info_contrib[0];
        alphadot += alpha_info_contrib[1];
        dalpha_dpsi += alpha_info_contrib[2];
        dalpha_dtheta += alpha_info_contrib[3];
        dalpha_dzeta += alpha_info_contrib[4];
    }

    // double phi = 0.0;
    // double phidot = 0.0;
    // double dphi_dpsi = 0.0;
    // double dphi_dtheta = 0.0;
    // double dphi_dzeta = 0.0;

    // contains alpha, alpha_dot, dalpha_dpsi, dalpha_dtheta

    // double alpha = 0.0;
    // double alphadot = 0.0;
    // double dalpha_dpsi = 0.0;
    // double dalpha_dtheta = 0.0;
    // double dalpha_dzeta = 0.0;


    // for(int i=0; i<saw_nharmonics; ++i){
    //     // intermediate values
    //     int m = saw_m_arr[i];
    //     int n = saw_n_arr[i];
    //     iota_mn = interpolants[8]*m - n;

    //     slope = (saw_phihats_arr[r_idx*saw_nharmonics + i] -  saw_phihats_arr[l_idx*saw_nharmonics + i]) / ((r_idx -l_idx)*s_grid_size);
    //     phi_i = saw_phihats_arr[l_idx*saw_nharmonics + i] + slope * s_diff;
    //     phidot_i = phi_i * cos(m*theta - n*z + saw_omega*time);
    //     sine = sin(m*theta - n*z + saw_omega*time);
    //     dphi_dpsi_i = slope*sine/psi0;
    //     dphi_dtheta_i = phidot_i * saw_m_arr[i] / saw_omega;
    //     dphi_dzeta_i = -phidot_i * saw_n_arr[i] / saw_omega;
        
    //     // phi values
    //     phi_info[0] += phi_i * sine;
    //     phi_info[1] += phidot_i;
    //     phi_info[2] +=  dphi_dpsi_i;
    //     phi_info[3] += dphi_dtheta_i;
    //     phi_info[4] += dphi_dzeta_i;

    //     // alpha values
    //     alpha_info[0] += -phi_i*iota_mn / G_iI;
    //     alpha_info[1] += -phidot_i * iota_mn / (saw_omega * G_iI);
    //     alpha_info[2] += -dphi_dpsi_i*iota_mn / (saw_omega * G_iI)
    //                 - (phi_i / saw_omega) * (interpolants[9]*m / G_iI - iota_mn*(interpolants[5] + interpolants[9] * interpolants[6] + interpolants[8]*interpolants[7])/(G_iI*G_iI));
    //     alpha_info[3] += -dphi_dtheta_i * iota_mn / (saw_omega*G_iI);
    //     alpha_info[4] += -dphi_dzeta_i * iota_mn / (saw_omega*G_iI);
    // }


    double fak1 = m*v_par*v_par/interpolants[0] + m*mu;
    double sdot = (-interpolants[2]*fak1/q + dalpha_dtheta*interpolants[0]*v_par - dphi_dtheta)/psi0;
    double tdot = interpolants[1]*fak1/q + (interpolants[8] - dalpha_dpsi*interpolants[4])*v_par*interpolants[0]/interpolants[4] + dphi_dpsi;

    out[0] = sdot*cos(theta) - s*sin(theta)*tdot;
    out[1] = sdot*sin(theta) + s*cos(theta)*tdot;
    out[2] = v_par*interpolants[0]/interpolants[4];
    out[3] = -interpolants[0]/(interpolants[4]*m) * (m*mu*(interpolants[3] + dalpha_dtheta*interpolants[1]*interpolants[4] 
                    + interpolants[2]*(interpolants[8] - dalpha_dpsi*interpolants[4])) + q*(alphadot*interpolants[4] 
                    + dalpha_dtheta*interpolants[4]*dphi_dpsi + (interpolants[8] - dalpha_dpsi*interpolants[4])*dphi_dtheta + dphi_dzeta)) 
                    + v_par/interpolants[0] * (interpolants[2]*dphi_dpsi - interpolants[1]*dphi_dtheta);

    out[4] = interpolants[0]; //modB
    out[5] = interpolants[4]; // G


}



__host__ __device__ void build_state(particle_t& p, int deriv_id, double* srange_arr, double* trange_arr, double* zrange_arr){
    // printf("build_state particle %d\n", p.id);


    const double b1 = 35.0 / 384.0, b3 = 500.0 / 1113.0, b4 = 125.0 / 192.0, b5 = -2187.0 / 6784.0, b6 = 11.0 / 84.0;
    double wgts[6] = {0.0}; 

    for (int i = 0; i < 4; i++) {
        p.x_temp[i] = p.state[i];
    }

    double deriv_time = p.t;

    switch(deriv_id){
        case 0:
            // wgts = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            break;
        case 1:
            // wgts = {1.0/5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
            deriv_time += 0.2*p.dt;
            wgts[0] = 1.0/5.0;
            break;
        case 2:
            // wgts = {3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0};
            deriv_time += 0.3*p.dt;
            wgts[0] = 3.0 / 40.0;
            wgts[1] = 9.0 / 40.0;
            break;
        case 3:
            // wgts = {44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0};
            deriv_time += 0.8*p.dt;
            wgts[0] = 44.0 / 45.0;
            wgts[1] = -56.0 / 15.0;
            wgts[2] = 32.0 / 9.0;
            break;
        case 4:
            // wgts = {19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0, 0.0, 0.0, 0.0};
            deriv_time += 8.0*p.dt/9.0;
            wgts[0] = 19372.0 / 6561.0;
            wgts[1] = -25360.0 / 2187.0;
            wgts[2] = 64448.0 / 6561.0;
            wgts[3] = -212.0 / 729.0;
            break;
        case 5:
            // wgts = {9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0,-5103.0 / 18656.0, 0.0, 0.0};
            deriv_time += p.dt;
            wgts[0] = 9017.0 / 3168.0;
            wgts[1] = -355.0 / 33.0;
            wgts[2] = 46732.0 / 5247.0;
            wgts[3] = 49.0 / 176.0;
            wgts[4] = -5103.0 / 18656.0;
            break;
        case 6:
            // wgts = {35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0};
            deriv_time += p.dt;
            wgts[0] = 35.0 / 384.0;
            wgts[2] = 500.0 / 1113.0;
            wgts[3] = 125.0 / 192.0; 
            wgts[4] = -2187.0 / 6784.0;
            wgts[5] = 11.0 / 84.0;
            break;
        default:
            break;
    }

    // create pt where deriv is to be computed
    for (int j=0; j<6; ++j){
        for(int i=0; i<4; ++i){
            p.x_temp[i] += p.dt * wgts[j] * p.derivs[6*j+i];
        }
    } 
    p.x_temp[4] = deriv_time;


    // transform to Boozer coordinates for B-field info
    double s = sqrt(p.x_temp[0]*p.x_temp[0] + p.x_temp[1]*p.x_temp[1]);
    double theta = atan2(p.x_temp[1], p.x_temp[0]);
    double z = p.x_temp[2];
    double v_par = p.x_temp[3];
    
    // we want to exploit periodicity in the B-field, but leave sine(theta) unchanged
    double t = fmod(theta, 2*M_PI);
    t += 2*M_PI*(t < 0);

    // we can modify z because it's only used to access the B-field location
    double period = zrange_arr[1];
    z = fmod(z, period);
    z += period*(z < 0);

    // exploit stellarator symmetry
    p.symmetry_exploited = t > M_PI;
    if(p.symmetry_exploited){
        z = period - z;
        t = 2*M_PI - t;
    }
    p.interpolation_loc[0] = s;
    p.interpolation_loc[1] = t;
    p.interpolation_loc[2] = z;

    /*
    * index into the grid and calculate weights
    */ 

    double s_grid_size = (srange_arr[1]-srange_arr[0]) / (srange_arr[2]-1);
    double theta_grid_size = (trange_arr[1]-trange_arr[0]) / (trange_arr[2]-1);
    double zeta_grid_size = (zrange_arr[1]-zrange_arr[0]) / (zrange_arr[2]-1);

    p.i = 3*((int) ((s - srange_arr[0]) / s_grid_size) / 3);
    p.j = 3*((int) ((t - trange_arr[0]) / theta_grid_size) / 3);
    p.k = 3*((int) ((z - zrange_arr[0]) / zeta_grid_size) / 3);

    // prevent out of bounds accesses
    p.i = min(p.i, (int)srange_arr[2]-4);
    p.j = min(p.j, (int)trange_arr[2]-4);
    p.k = min(p.k, (int)zrange_arr[2]-4);

    // normalized positions in local grid wrt e.g. r at index i
    // maps the position to [0,3] in the "meta grid"

    double s_rel = (s -  p.i*s_grid_size - srange_arr[0]) / s_grid_size;
    double theta_rel = (t -  p.j*theta_grid_size - trange_arr[0]) / theta_grid_size;
    double zeta_rel = (z - p.k*zeta_grid_size - zrange_arr[0]) / zeta_grid_size;
    
    shape(s_rel, p.s_shape);
    shape(theta_rel, p.t_shape);
    shape(zeta_rel, p.z_shape);

}


// set initial time step, calculate mu
__host__ __device__ void setup_particle(particle_t& p, double* srange_arr, double* trange_arr, double* zrange_arr, double* quadpts_arr,
                         double tmax, double m, double q, double psi0, double* saw_srange_arr, int* saw_m_arr, int* saw_n_arr, double* saw_phihats_arr, double saw_omega, int saw_nharmonics){
                             // double mu;
    // printf("setup particle %d\n", p.id);

    p.t = 0.0;
    p.dt = 0.0;
    build_state(p, 0, srange_arr, trange_arr, zrange_arr);

    // dummy call to get norm B
    calc_derivs(p, p.derivs, srange_arr, trange_arr, zrange_arr, quadpts_arr, m, q, -1, psi0, saw_srange_arr, saw_m_arr, saw_n_arr, saw_phihats_arr, saw_omega, saw_nharmonics);

    double v_perp2 = p.v_perp*p.v_perp;
    double denom = 1 / (2*p.derivs[4]);
    p.mu = v_perp2 * denom;

    p.dtmax = 0.5*M_PI*abs(p.derivs[5]) / (p.derivs[4]*p.v_total);
    p.dt = 1e-3*p.dtmax;

}

__host__ __device__ void adjust_time(particle_t& p, double tmax){
    // printf("adjust_time particle %d\n", p.id);

    if(p.has_left){
        return;
    }

    const double bhat1 = 71.0 / 57600.0, bhat3 = -71.0 / 16695.0, bhat4 = 71.0 / 1920.0, bhat5 = -17253.0 / 339200.0, bhat6 = 22.0 / 525.0, bhat7 = -1.0 / 40.0;

    // Compute  error
    // https://live.boost.org/doc/libs/1_82_0/libs/numeric/odeint/doc/html/boost_numeric_odeint/odeint_in_detail/steppers.html
    // resolve typo in boost docs: https://numerical.recipes/book.html
    double atol=1e-9;
    double rtol=1e-9;
    double err = 0.0;
    bool accept = true;
    for (int i = 0; i < 4; i++) {
        p.x_err[i] = p.dt*(bhat1 * p.derivs[i] + bhat3 * p.derivs[12+i] + bhat4 * p.derivs[18+i] + bhat5 * p.derivs[24+i] + bhat6 * p.derivs[30+i] + bhat7 * p.derivs[36+i]);
       
        if(i==3){ // account for scale of v_par in absolute tolerance
            atol *= 1e5;
        }
        p.x_err[i] = fabs(p.x_err[i]) / (atol + rtol*(fabs(p.state[i]) + p.dt*fabs(p.derivs[i])));      
        err = fmax(err, p.x_err[i]);
    }
    // Compute new step size
    double dt_new = p.dt*0.9*pow(err, -1.0/3.0);
    dt_new = fmax(dt_new, 0.2 * p.dt);  // Limit step size reduction
    dt_new = fmin(dt_new, 5.0 * p.dt);  // Limit step size increase
    dt_new = fmin(p.dtmax, dt_new);
    if ((0.5 < err) & (err < 1.0)){
        dt_new = p.dt;
    }
    p.step_attempt++;
    if (err <= 1.0) {
        // Accept the step
        p.t += p.dt;
        p.dt = fmin(dt_new, tmax - p.t);

        p.state[0] = p.x_temp[0];
        p.state[1] = p.x_temp[1];
        p.state[2] = p.x_temp[2];
        p.state[3] = p.x_temp[3];

        double s = sqrt(p.state[0]*p.state[0] + p.state[1]*p.state[1]);
        p.has_left = s >= 1;
        p.step_accept++;


    } else {
        // Reject the step and try again with smaller dt
        p.dt = dt_new;
    }

}
__host__ __device__    void trace_particle(particle_t& p, double* srange_arr, double* trange_arr, double* zrange_arr, double* quadpts_arr,
                         double tmax, double m, double q, double psi0, double* saw_srange_arr, int* saw_m_arr, int* saw_n_arr, double* saw_phihats_arr, double saw_omega, int saw_nharmonics){

    setup_particle(p, srange_arr, trange_arr, zrange_arr, quadpts_arr, tmax, m, q, psi0, saw_srange_arr, saw_m_arr, saw_n_arr, saw_phihats_arr, saw_omega, saw_nharmonics);

    int counter = 0;

    while(p.t < tmax){
        // if(counter % 1000){
        //     printf("particle %d position %.15e, %.15e, %.15e, %.15e, %.15e, dt=%.15e\n", p.id, p.t, p.state[0], p.state[1], p.state[2], p.state[3], p.dt);
        // }
        for(int k=0; k<7; ++k){
            build_state(p, k, srange_arr, trange_arr, zrange_arr);
            calc_derivs(p, p.derivs + 6*k, srange_arr, trange_arr, zrange_arr, quadpts_arr, m, q, p.mu, psi0, saw_srange_arr, saw_m_arr, saw_n_arr, saw_phihats_arr, saw_omega, saw_nharmonics);
        }
        adjust_time(p, tmax);
        
        double s = sqrt(p.state[0]*p.state[0] + p.state[1]*p.state[1]);
        if(s >= 1){
            printf("particle %d done s=%.15e\n", p.id, s);
            p.has_left = true;
            return;
        }

        counter++;

    }
    double s = sqrt(p.state[0]*p.state[0] + p.state[1]*p.state[1]);
    printf("particle %d done s=%.15e\n", p.id, s);
    return;
}

__global__ void particle_trace_kernel(particle_t* particles, double* srange_arr, double* trange_arr, double* zrange_arr, double* quadpts_arr,
                        double tmax, double m, double q, double psi0, int nparticles, double* saw_srange_arr, int* saw_m_arr, int* saw_n_arr, double* saw_phihats_arr, double saw_omega, int saw_nharmonics){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < nparticles){
        printf("tracing particle %d\n", idx);
        trace_particle(particles[idx], srange_arr, trange_arr, zrange_arr, quadpts_arr, tmax, m, q, psi0, saw_srange_arr, saw_m_arr, saw_n_arr, saw_phihats_arr, saw_omega, saw_nharmonics);
    }
}


// matches GuidingCenterVacuumBoozerPerturbedRHS
extern "C" vector<double> gpu_tracing_saw(py::array_t<double> quad_pts, py::array_t<double> srange,
        py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> stz_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        double tmax, double tol, double psi0, int nparticles, py::array_t<double> saw_srange, py::array_t<int> saw_m, py::array_t<int> saw_n, py::array_t<double> saw_phihats, double saw_omega, int saw_nharmonics){

    //  read data in from python
    auto ptr = stz_init.data();
    int size = stz_init.size();
    double stz_init_arr[size];
    std::memcpy(stz_init_arr, ptr, size * sizeof(double));
    
    py::buffer_info vtang_buf = vtang.request();
    double* vtang_arr = static_cast<double*>(vtang_buf.ptr);

    // contains b field
    py::buffer_info quadpts_buf = quad_pts.request();
    double* quadpts_arr = static_cast<double*>(quadpts_buf.ptr);

    py::buffer_info s_buf = srange.request();
    double* srange_arr = static_cast<double*>(s_buf.ptr);

    py::buffer_info t_buf = trange.request();
    double* trange_arr = static_cast<double*>(t_buf.ptr);

    py::buffer_info z_buf = zrange.request();
    double* zrange_arr = static_cast<double*>(z_buf.ptr);

    py::buffer_info saw_s_buf = saw_srange.request();
    double* saw_srange_arr = static_cast<double*>(saw_s_buf.ptr);

    py::buffer_info saw_m_buf = saw_m.request();
    int* saw_m_arr = static_cast<int*>(saw_m_buf.ptr);

    py::buffer_info saw_n_buf = saw_n.request();
    int* saw_n_arr = static_cast<int*>(saw_n_buf.ptr);

    py::buffer_info saw_phihats_buf = saw_phihats.request();
    double* saw_phihats_arr = static_cast<double*>(saw_phihats_buf.ptr);
    
    particle_t* particles =  new particle_t[nparticles];

    /*
    * y1 = s*cos(theta)
    * y2 = s*sin(theta)
    */

    // load initial conditions
    for(int i=0; i<nparticles; ++i){
        int start = 3*i;

        double s = stz_init_arr[start];
        double theta = stz_init_arr[start+1];
        
        // convert to alternative coordinates
        particles[i].state[0] = s*cos(theta);
        particles[i].state[1] = s*sin(theta);
        
        particles[i].state[2] = stz_init_arr[start+2];
        particles[i].state[3] = vtang_arr[i];
        particles[i].v_perp = sqrt(vtotal*vtotal -  vtang_arr[i]*vtang_arr[i]);
        particles[i].v_total = vtotal;
        particles[i].has_left = false;
        particles[i].t = 0;
        
        particles[i].step_accept = 0;
        particles[i].step_attempt = 0;
        particles[i].id = i;
        
    }
   
    
    particle_t* particles_d;
    cudaMalloc((void**)&particles_d, nparticles * sizeof(particle_t));
    cudaMemcpy(particles_d, particles, nparticles * sizeof(particle_t), cudaMemcpyHostToDevice);

    double* srange_d;
    cudaMalloc((void**)&srange_d, 3 * sizeof(double));
    cudaMemcpy(srange_d, srange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);

    double* zrange_d;
    cudaMalloc((void**)&zrange_d, 3 * sizeof(double));
    cudaMemcpy(zrange_d, zrange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);

    double* trange_d;
    cudaMalloc((void**)&trange_d, 3 * sizeof(double));
    cudaMemcpy(trange_d, trange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);


    double* quadpts_d;
    cudaMalloc((void**)&quadpts_d, quad_pts.size() * sizeof(double));
    cudaMemcpy(quadpts_d, quadpts_arr, quad_pts.size() * sizeof(double), cudaMemcpyHostToDevice);

    double* saw_srange_d;
    cudaMalloc((void**)&saw_srange_d, 3*sizeof(double));
    cudaMemcpy(saw_srange_d, saw_srange_arr, 3*sizeof(double), cudaMemcpyHostToDevice);
    
    int* saw_m_d;
    cudaMalloc((void**)&saw_m_d, saw_m.size()*sizeof(int));
    cudaMemcpy(saw_m_d, saw_m_arr, saw_m.size()*sizeof(int), cudaMemcpyHostToDevice);
    
    int* saw_n_d;
    cudaMalloc((void**)&saw_n_d, saw_n.size()*sizeof(int));
    cudaMemcpy(saw_n_d, saw_n_arr, saw_n.size()*sizeof(int), cudaMemcpyHostToDevice);

    double* saw_phihats_d;
    cudaMalloc((void**)&saw_phihats_d, saw_phihats.size()*sizeof(double));
    cudaMemcpy(saw_phihats_d, saw_phihats_arr, saw_phihats.size()*sizeof(double), cudaMemcpyHostToDevice);


    int nthreads = 1;
    int nblks = nparticles / nthreads + 1;
    std::cout << "starting particle tracing kernel\n";

       
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    particle_trace_kernel<<<nblks, nthreads>>>(particles_d, srange_d, trange_d, zrange_d, quadpts_d, tmax, m, q, psi0, nparticles, saw_srange_d, saw_m_d, saw_n_d, saw_phihats_d, saw_omega, saw_nharmonics);

    cudaMemcpy(particles, particles_d, nparticles * sizeof(particle_t), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "tracing kernels time (ms): " << milliseconds<< "\n";
    
    vector<double> particle_output(7*nparticles);
    for(int i=0; i<nparticles; ++i){
        double y1 = particles[i].state[0];
        double y2 = particles[i].state[1];
        double z = particles[i].state[2];
        double v_par = particles[i].state[3];

        // last location in Boozer coordinates
        particle_output[7*i] = sqrt(y1*y1 + y2*y2);
        particle_output[7*i + 1] = atan2(y2, y1);
        particle_output[7*i + 2] = z;
        particle_output[7*i + 3] = v_par;
        particle_output[7*i + 4] = particles[i].t;
        particle_output[7*i + 5] = particles[i].step_accept;
        particle_output[7*i + 6] = particles[i].step_attempt;
    }


    delete[] particles;

    return particle_output;
}

extern "C" py::array_t<double> test_interpolation(py::array_t<double> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> loc, int n){
    py::buffer_info quadpts_buf = quad_pts.request();
    double* quadpts_arr = static_cast<double*>(quadpts_buf.ptr);

    py::buffer_info s_buf = srange.request();
    double* srange_arr = static_cast<double*>(s_buf.ptr);

    py::buffer_info t_buf = trange.request();
    double* trange_arr = static_cast<double*>(t_buf.ptr);

    py::buffer_info z_buf = zrange.request();
    double* zrange_arr = static_cast<double*>(z_buf.ptr);

    py::buffer_info loc_buf = loc.request();
    double* loc_arr = static_cast<double*>(loc_buf.ptr);

    double out[n];

    // double s = loc_arr[0];
    double t = loc_arr[1];
    double z = loc_arr[2];
    // we want to exploit periodicity in the B-field, but leave sine(theta) unchanged
    t = fmod(t, 2*M_PI);
    t += 2*M_PI*(t < 0);

    // we can modify z because it's only used to access the B-field location
    double period = zrange_arr[1];
    z = fmod(z, period);
    z += period*(z < 0);

    
    // exploit stellarator symmetry
    bool symmetry_exploited = t > M_PI;
    if(symmetry_exploited){
        z = period - z;
        t = 2*M_PI - t;
    }
    loc_arr[1] = t;
    loc_arr[2] = z;

    if(symmetry_exploited){
        out[2] *= -1.0;
        out[3] *= -1.0;
    }

    auto result = py::array_t<double>(n, out);
    return result;

}

__global__ void test_gpu_interpolation_kernel(double* quad_pts, double* srange, double* trange, double* zrange, double* loc, double* out, int n, int n_points){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < n_points){
        double* loc_arr = loc + 3*idx;
        double* out_arr  =  out + idx*n;

        particle_t p;
        double s = loc_arr[0];
        double t = loc_arr[1];
        double z = loc_arr[2];

        p.state[0] = s*cos(t);
        p.state[1] = s*sin(t);
        p.state[2] = z;

        p.dt = 1e-3; //needed for build_state

        build_state(p, 0, srange, trange, zrange);
        
        interpolate(p, quad_pts, out_arr, srange, trange, zrange, n);

        if(p.symmetry_exploited){
            out_arr[2] *= -1.0;
            out_arr[3] *= -1.0;
        }
    }
}


extern "C" py::array_t<double> test_gpu_interpolation(py::array_t<double> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> loc, int n, int n_points){
    py::buffer_info quadpts_buf = quad_pts.request();
    double* quadpts_arr = static_cast<double*>(quadpts_buf.ptr);

    py::buffer_info s_buf = srange.request();
    double* srange_arr = static_cast<double*>(s_buf.ptr);

    py::buffer_info t_buf = trange.request();
    double* trange_arr = static_cast<double*>(t_buf.ptr);

    py::buffer_info z_buf = zrange.request();
    double* zrange_arr = static_cast<double*>(z_buf.ptr);

    py::buffer_info loc_buf = loc.request();
    double* loc_arr = static_cast<double*>(loc_buf.ptr);


    double* srange_d;
    cudaMalloc((void**)&srange_d, 3 * sizeof(double));
    cudaMemcpy(srange_d, srange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);

    double* zrange_d;
    cudaMalloc((void**)&zrange_d, 3 * sizeof(double));
    cudaMemcpy(zrange_d, zrange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);

    double* trange_d;
    cudaMalloc((void**)&trange_d, 3 * sizeof(double));
    cudaMemcpy(trange_d, trange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);

    double* quadpts_d;
    cudaMalloc((void**)&quadpts_d, quad_pts.size() * sizeof(double));
    cudaMemcpy(quadpts_d, quadpts_arr, quad_pts.size() * sizeof(double), cudaMemcpyHostToDevice);

    double* loc_d;
    cudaMalloc((void**)&loc_d, loc.size() * sizeof(double));
    cudaMemcpy(loc_d, loc_arr, loc.size() * sizeof(double), cudaMemcpyHostToDevice);


    double* out_d;
    cudaMalloc((void**)&out_d, n*n_points * sizeof(double));

    int nthreads = 256;
    int nblks = n_points / nthreads + 1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    test_gpu_interpolation_kernel<<<nblks, nthreads>>>(quadpts_d, srange_d, trange_d, zrange_d, loc_d, out_d, n, n_points);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();  // <-- Required for runtime issues
    if (err != cudaSuccess) {
        printf("Kernel runtime error: %s\n", cudaGetErrorString(err));
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "interpolation kernel time (ms): " << milliseconds<< "\n";
    
    double out[n*n_points];
    cudaMemcpy(&out, out_d, n*n_points * sizeof(double), cudaMemcpyDeviceToHost);
    auto result = py::array_t<double>(n*n_points, out);
    return result;

}


__global__ void test_gpu_derivs_kernel(double* quad_pts, double* srange, double* trange, double* zrange, double* loc, double* vpar, double vtotal, double* out, double m, double q, double psi0, int n_points){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < n_points){
        double* loc_arr = loc + 3*idx;
        double* out_arr  =  out + 4*idx;
        double vpar_val = vpar[idx];

        particle_t p;
        double s = loc_arr[0];
        double t = loc_arr[1];
        double z = loc_arr[2];

        p.state[0] = s*cos(t);
        p.state[1] = s*sin(t);
        p.state[2] = z;
        p.state[3] = vpar_val;
        p.v_total = vtotal;
        p.v_perp = sqrt(vtotal*vtotal -  vpar_val*vpar_val);

        // setup_particle(p, srange, trange, zrange, quad_pts, 1e-2, m, q, psi0);

        // calc_derivs(p, p.derivs, srange, trange, zrange, quad_pts, m, q, p.mu, psi0);

        out_arr[0] = p.derivs[0];
        out_arr[1] = p.derivs[1];
        out_arr[2] = p.derivs[2];
        out_arr[3] = p.derivs[3];

    }
}

extern "C" py::array_t<double> test_derivatives(py::array_t<double> quad_pts, py::array_t<double> srange, py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> loc, py::array_t<double> vpar, double v_total, double m, double q, double psi0, int n_points){
    py::buffer_info quadpts_buf = quad_pts.request();
    double* quadpts_arr = static_cast<double*>(quadpts_buf.ptr);

    py::buffer_info s_buf = srange.request();
    double* srange_arr = static_cast<double*>(s_buf.ptr);

    py::buffer_info t_buf = trange.request();
    double* trange_arr = static_cast<double*>(t_buf.ptr);

    py::buffer_info z_buf = zrange.request();
    double* zrange_arr = static_cast<double*>(z_buf.ptr);

    py::buffer_info loc_buf = loc.request();
    double* loc_arr = static_cast<double*>(loc_buf.ptr);

    py::buffer_info vpar_buf = vpar.request();
    double* vpar_arr = static_cast<double*>(vpar_buf.ptr);
    

    double* srange_d;
    cudaMalloc((void**)&srange_d, 3 * sizeof(double));
    cudaMemcpy(srange_d, srange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);

    double* zrange_d;
    cudaMalloc((void**)&zrange_d, 3 * sizeof(double));
    cudaMemcpy(zrange_d, zrange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);

    double* trange_d;
    cudaMalloc((void**)&trange_d, 3 * sizeof(double));
    cudaMemcpy(trange_d, trange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);

    double* quadpts_d;
    cudaMalloc((void**)&quadpts_d, quad_pts.size() * sizeof(double));
    cudaMemcpy(quadpts_d, quadpts_arr, quad_pts.size() * sizeof(double), cudaMemcpyHostToDevice);

    double* loc_d;
    cudaMalloc((void**)&loc_d, loc.size() * sizeof(double));
    cudaMemcpy(loc_d, loc_arr, loc.size() * sizeof(double), cudaMemcpyHostToDevice);

    double* vpar_d;
    cudaMalloc((void**)&vpar_d, vpar.size() * sizeof(double));
    cudaMemcpy(vpar_d, vpar_arr, vpar.size() * sizeof(double), cudaMemcpyHostToDevice);

    double* out_d;
    cudaMalloc((void**)&out_d, 4*n_points * sizeof(double));



    int nthreads = 256;
    int nblks = n_points / nthreads + 1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    test_gpu_derivs_kernel<<<nblks, nthreads>>>(quadpts_d, srange_d, trange_d, zrange_d, loc_d, vpar_d, v_total, out_d, m, q, psi0, n_points);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "interpolation kernel time (ms): " << milliseconds<< "\n";
    
    double out[4*n_points];
    cudaMemcpy(&out, out_d, 4*n_points * sizeof(double), cudaMemcpyDeviceToHost);
    auto result = py::array_t<double>(4*n_points, out);
    return result;
}

__global__ void test_gpu_timestep_kernel(particle_t* particles, double* srange_arr, double* trange_arr, double* zrange_arr, double* quadpts_arr,
                        double m, double q, double psi0, int nparticles){
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < nparticles){
        // setup_particle(particles[idx], srange_arr, trange_arr, zrange_arr, quadpts_arr, 1e-2, m, q, psi0);

        while(particles[idx].t == 0.0){
            for(int k=0; k<7; ++k){
                build_state(particles[idx], k, srange_arr, trange_arr, zrange_arr);
                // calc_derivs(particles[idx], particles[idx].derivs + 6*k, srange_arr, trange_arr, zrange_arr, quadpts_arr, m, q, particles[idx].mu, psi0);
            }
            adjust_time(particles[idx], 1e-2);
        }
    }
    return;
}



extern "C" vector<double> test_timestep(py::array_t<double> quad_pts, py::array_t<double> srange,
        py::array_t<double> trange, py::array_t<double> zrange, py::array_t<double> stz_init, double m, double q, double vtotal, py::array_t<double> vtang, 
        double tol, double psi0, int nparticles){

    //  read data in from python
    auto ptr = stz_init.data();
    int size = stz_init.size();
    double stz_init_arr[size];
    std::memcpy(stz_init_arr, ptr, size * sizeof(double));

    py::buffer_info vtang_buf = vtang.request();
    double* vtang_arr = static_cast<double*>(vtang_buf.ptr);

    // contains b field
    py::buffer_info quadpts_buf = quad_pts.request();
    double* quadpts_arr = static_cast<double*>(quadpts_buf.ptr);

    py::buffer_info s_buf = srange.request();
    double* srange_arr = static_cast<double*>(s_buf.ptr);

    py::buffer_info t_buf = trange.request();
    double* trange_arr = static_cast<double*>(t_buf.ptr);

    py::buffer_info z_buf = zrange.request();
    double* zrange_arr = static_cast<double*>(z_buf.ptr);


    particle_t* particles =  new particle_t[nparticles];

    // convert to alternative coordinates
    /*
    * y1 = s*cos(theta)
    * y2 = s*sin(theta)
    */

    // load initial conditions
    for(int i=0; i<nparticles; ++i){
        int start = 3*i;

        double s = stz_init_arr[start];
        double theta = stz_init_arr[start+1];
        
        // convert to alternative coordinates
        particles[i].state[0] = s*cos(theta);
        particles[i].state[1] = s*sin(theta);
        
        particles[i].state[2] = stz_init_arr[start+2];
        particles[i].state[3] = vtang_arr[i];
        particles[i].v_perp = sqrt(vtotal*vtotal -  vtang_arr[i]*vtang_arr[i]);
        particles[i].v_total = vtotal;
        particles[i].has_left = false;
        particles[i].t = 0;
        
        particles[i].step_accept = 0;
        particles[i].step_attempt = 0;
        particles[i].id = i;
    }
    
    particle_t* particles_d;
    cudaMalloc((void**)&particles_d, nparticles * sizeof(particle_t));
    cudaMemcpy(particles_d, particles, nparticles * sizeof(particle_t), cudaMemcpyHostToDevice);

    double* srange_d;
    cudaMalloc((void**)&srange_d, 3 * sizeof(double));
    cudaMemcpy(srange_d, srange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);

    double* zrange_d;
    cudaMalloc((void**)&zrange_d, 3 * sizeof(double));
    cudaMemcpy(zrange_d, zrange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);

    double* trange_d;
    cudaMalloc((void**)&trange_d, 3 * sizeof(double));
    cudaMemcpy(trange_d, trange_arr, 3 * sizeof(double), cudaMemcpyHostToDevice);


    double* quadpts_d;
    cudaMalloc((void**)&quadpts_d, quad_pts.size() * sizeof(double));
    cudaMemcpy(quadpts_d, quadpts_arr, quad_pts.size() * sizeof(double), cudaMemcpyHostToDevice);

    int nthreads = 256;
    int nblks = nparticles / nthreads + 1;
    std::cout << "starting particle tracing kernel\n";

       
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    test_gpu_timestep_kernel<<<nblks, nthreads>>>(particles_d, srange_d, trange_d, zrange_d, quadpts_d, m, q, psi0, nparticles);

    cudaMemcpy(particles, particles_d, nparticles * sizeof(particle_t), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "tracing kernels time (ms): " << milliseconds<< "\n";

    
    vector<double> particle_output(7*nparticles);
    for(int i=0; i<nparticles; ++i){
        double y1 = particles[i].state[0];
        double y2 = particles[i].state[1];
        double z = particles[i].state[2];
        double v_par = particles[i].state[3];

        double s = sqrt(y1*y1 + y2*y2);
        double theta = atan2(y2, y1);
        
        // last location in Boozer coordinates
        particle_output[7*i] = s;
        particle_output[7*i + 1] = theta;
        particle_output[7*i + 2] = z;
        particle_output[7*i + 3] = v_par;
        particle_output[7*i + 4] = particles[i].t;
        particle_output[7*i + 5] = particles[i].step_accept;
        particle_output[7*i + 6] = particles[i].step_attempt;
    }


    delete[] particles;

    return particle_output;
}