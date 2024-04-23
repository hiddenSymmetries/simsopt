#include "xtensor-python/pyarray.hpp"
typedef xt::pyarray<double> PyArray;
#include <math.h>

/** Compute quadratic flux and similar quantities.
 *
 * This function is used in simsopt.objectives.SquaredFlux.
 * See the documentation of that class for the definitions of the quantities
 * computed here.
 */
double integral_BdotN(PyArray& Bcoil, PyArray& Btarget, PyArray& n, std::string definition) {
    int nphi = Bcoil.shape(0);
    int ntheta = Bcoil.shape(1);
    double *Bcoil_ptr = Bcoil.data();
    double *Btarget_ptr = NULL;
    double *n_ptr = n.data();
    if(Bcoil.layout() != xt::layout_type::row_major)
            throw std::runtime_error("Bcoil needs to be in row-major storage order");
    if(Bcoil.shape(2) != 3)
        throw std::runtime_error("Bcoil has wrong shape.");
    if(Bcoil.size() != 3*nphi*ntheta)
        throw std::runtime_error("Bcoil has wrong size.");
    if(n.layout() != xt::layout_type::row_major)
            throw std::runtime_error("n needs to be in row-major storage order");
    if(n.shape(0) != nphi)
        throw std::runtime_error("n has wrong shape.");
    if(n.shape(1) != ntheta)
        throw std::runtime_error("n has wrong shape.");
    if(n.shape(2) != 3)
        throw std::runtime_error("n has wrong shape.");
    if(n.size() != 3*nphi*ntheta)
        throw std::runtime_error("n has wrong size.");
    if(Btarget.size() > 0){
        if(Btarget.layout() != xt::layout_type::row_major)
            throw std::runtime_error("Btarget needs to be in row-major storage order");
        if(Btarget.shape(0) != nphi)
            throw std::runtime_error("Btarget has wrong shape.");
        if(Btarget.shape(1) != ntheta)
            throw std::runtime_error("Btarget has wrong shape.");
        if(Btarget.size() != nphi*ntheta)
            throw std::runtime_error("Btarget has wrong size.");

        Btarget_ptr = Btarget.data();
    }
    // Convert "definition" from string to int for faster comparisons inside
    // the loop.
    enum definition_type {DEFINITION_QUADRATIC_FLUX, DEFINITION_NORMALIZED, DEFINITION_LOCAL};
    definition_type definition_int;
    if (definition == "quadratic flux") {
        definition_int = DEFINITION_QUADRATIC_FLUX;
    } else if (definition == "normalized") {
        definition_int = DEFINITION_NORMALIZED;
    } else if (definition == "local") {
        definition_int = DEFINITION_LOCAL;
    } else {
        throw std::runtime_error("Unrecognized value for 'definition'.");
    }
    double numerator_sum = 0.0;
    double denominator_sum = 0.0;
    double mod_B_squared;

    #pragma omp parallel for ordered reduction(+:numerator_sum, denominator_sum)
    for(int i=0; i<nphi*ntheta; i++){
        double normN = std::sqrt(
            n_ptr[3 * i + 0] * n_ptr[3 * i + 0] 
            + n_ptr[3 * i + 1] * n_ptr[3 * i + 1] 
            + n_ptr[3 * i + 2] * n_ptr[3 * i + 2]
        );
        double Nx = n_ptr[3 * i + 0] / normN;
        double Ny = n_ptr[3 * i + 1] / normN;
        double Nz = n_ptr[3 * i + 2] / normN;
        double BcoildotN = (
            Bcoil_ptr[3 * i + 0] * Nx 
            + Bcoil_ptr[3 * i + 1] * Ny 
            + Bcoil_ptr[3 * i + 2] * Nz
        );
        if(Btarget_ptr != NULL)
            BcoildotN -= Btarget_ptr[i];

        if (definition_int != DEFINITION_QUADRATIC_FLUX)
            mod_B_squared = 
                Bcoil_ptr[3 * i + 0] * Bcoil_ptr[3 * i + 0] 
                + Bcoil_ptr[3 * i + 1] * Bcoil_ptr[3 * i + 1] 
                + Bcoil_ptr[3 * i + 2] * Bcoil_ptr[3 * i + 2];
        
        if (definition_int == DEFINITION_QUADRATIC_FLUX) {
            numerator_sum += (BcoildotN * BcoildotN) * normN;
        } else if (definition_int == DEFINITION_NORMALIZED){
            numerator_sum += (BcoildotN * BcoildotN) * normN;
            denominator_sum += mod_B_squared * normN;
        } else if (definition_int == DEFINITION_LOCAL) {
            numerator_sum += (BcoildotN * BcoildotN) / mod_B_squared * normN;
        } else {
            throw std::runtime_error("Should never reach this point.");
        }
    }

    double result;
    if (definition_int == DEFINITION_NORMALIZED) {
        result = 0.5 * numerator_sum / denominator_sum;
    } else {
        result = 0.5 * numerator_sum / (nphi * ntheta);
    }

    return result;
}
