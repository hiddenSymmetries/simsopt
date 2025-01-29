#pragma once
#include <xtensor/xnoalias.hpp>
#include <stdexcept>
#include "xtensor-python/pytensor.hpp"
#include "boozermagneticfield.h"

using std::logic_error;
using std::shared_ptr;

#include <vector>       // For std::vector in Phihat
#include <algorithm>    // For std::sort in Phihat
#include <numeric>      // For std::iota in Phihat
#include <stdexcept>    // For std::invalid_argument Phihat
#include <set>          // For std::set in Phihat
#include <xtensor/xview.hpp> // To access parts of the xtensor 
                             // (for ShearAlfvenWave and ShearAlfvenHarmonic)

/**
* @brief Transverse Shear Alfvén Wave in Boozer coordinates
* 
* See Paul. et al,
* JPP (2023;89(5):905890515. doi:10.1017/S0022377823001095), and refs. therein.
**/
class ShearAlfvenWave {
public:
    using Array2 = xt::pytensor<double, 2, xt::layout_type::row_major>;

protected:
    virtual void _Phi_impl(Array2& Phi) {
        throw logic_error("_Phi_impl was not implemented");
    }

    virtual void _dPhidpsi_impl(Array2& dPhidpsi) {
        throw logic_error("_dPhidpsi_impl was not implemented");
    }

    virtual void _dPhidtheta_impl(Array2& dPhidtheta) {
        throw logic_error("_dPhidtheta_impl was not implemented");
    }

    virtual void _dPhidzeta_impl(Array2& dPhidzeta) {
        throw logic_error("_dPhidzeta_impl was not implemented");
    }

    virtual void _Phidot_impl(Array2& Phidot) {
        throw logic_error("_Phidot_impl was not implemented");
    }

    virtual void _alpha_impl(Array2& alpha) {
        throw logic_error("_alpha_impl was not implemented");
    }

    virtual void _dalphadpsi_impl(Array2& dalphadpsi) {
        throw logic_error("_dalphadpsi_impl was not implemented");
    }

    virtual void _dalphadtheta_impl(Array2& dalphadtheta) {
        throw logic_error("_dalphadtheta_impl was not implemented");
    }

    virtual void _dalphadzeta_impl(Array2& dalphadzeta) {
        throw logic_error("_dalphadzeta_impl was not implemented");
    }

    virtual void _alphadot_impl(Array2& alphadot) {
        throw logic_error("_alphadot_impl was not implemented");
    }
    shared_ptr<BoozerMagneticField> B0;
    Array2 points;
    Array2 data_Phi;
    Array2 data_dPhidpsi, data_dPhidtheta, data_dPhidzeta, data_Phidot;
    Array2 data_alpha;
    Array2 data_alphadot, data_dalphadpsi, data_dalphadtheta, data_dalphadzeta;
    long npoints;

public:
    ShearAlfvenWave(shared_ptr<BoozerMagneticField> B0field)
        : B0(B0field) {
        Array2 vals({{0., 0., 0., 0.}});
        this->set_points(vals);
    }

    virtual ~ShearAlfvenWave() {}

    virtual void set_points(Array2& p) {
        if (p.shape(1) != 4) {
            throw std::invalid_argument("Input tensor must have 4 columns: Boozer coordinates, and time (s, theta, zeta, time)");
        }
        npoints = p.shape(0);
        points.resize({npoints, 4});
        memcpy(points.data(), p.data(), 4 * npoints * sizeof(double));
        // Set points for B0 using the first three columns of p 
        // (s, theta, zeta):
        Array2 p_b0 = xt::view(p, xt::all(), xt::range(0, 3));
        B0->set_points(p_b0);
    }

    Array2 get_points() {
        return points;
    }

    Array2& Phi_ref() {
        data_Phi.resize({npoints, 1});
        _Phi_impl(data_Phi);
        return data_Phi;
    }

    Array2& dPhidpsi_ref() {
        data_dPhidpsi.resize({npoints, 1});
        _dPhidpsi_impl(data_dPhidpsi);
        return data_dPhidpsi;
    }

    Array2& Phidot_ref() {
        data_Phidot.resize({npoints, 1});
        _Phidot_impl(data_Phidot);
        return data_Phidot;
    }

    Array2& dPhidtheta_ref() {
        data_dPhidtheta.resize({npoints, 1});
        _dPhidtheta_impl(data_dPhidtheta);
        return data_dPhidtheta;
    }

    Array2& dPhidzeta_ref() {
        data_dPhidzeta.resize({npoints, 1});
        _dPhidzeta_impl(data_dPhidzeta);
        return data_dPhidzeta;
    }

    Array2& alpha_ref() {
        data_alpha.resize({npoints, 1});
        _alpha_impl(data_alpha);
        return data_alpha;
    }

    Array2& alphadot_ref() {
        data_alphadot.resize({npoints, 1});
        _alphadot_impl(data_alphadot);
        return data_alphadot;
    }

    Array2& dalphadtheta_ref() {
        data_dalphadtheta.resize({npoints, 1});
        _dalphadtheta_impl(data_dalphadtheta);
        return data_dalphadtheta;
    }

    Array2& dalphadpsi_ref() {
        data_dalphadpsi.resize({npoints, 1});
        _dalphadpsi_impl(data_dalphadpsi);
        return data_dalphadpsi;
    }

    Array2& dalphadzeta_ref() {
        data_dalphadzeta.resize({npoints, 1});
        _dalphadzeta_impl(data_dalphadzeta);
        return data_dalphadzeta;
    }
    
    std::shared_ptr<BoozerMagneticField> get_B0() const {
        return B0;
    }

    Array2 Phi() { return Phi_ref(); }
    Array2 dPhidpsi() { return dPhidpsi_ref(); }
    Array2 Phidot() { return Phidot_ref(); }
    Array2 dPhidtheta() { return dPhidtheta_ref(); }
    Array2 dPhidzeta() { return dPhidzeta_ref(); }
    Array2 alpha() { return alpha_ref(); }
    Array2 alphadot() { return alphadot_ref(); }
    Array2 dalphadtheta() { return dalphadtheta_ref(); }
    Array2 dalphadpsi() { return dalphadpsi_ref(); }
    Array2 dalphadzeta() { return dalphadzeta_ref(); }
};

/**
* @brief Class representing the profile of scalar potential
* with respect to normalized flux Boozer coordinate `s`.
*
* The `Phihat` class represents scalar potential profile (`Phihat`)
* as a function of the normalized toroidal Boozer coordinate `s`.
* It uses linear interpolation to compute the value of the
* scalar potential and its derivative at any given point within the domain.
*/
class Phihat {
private:
  std::vector<double> s_values;
  std::vector<double> Phihat_values;

  /**
  * @brief Validates the input vectors of normalized flux
  * Boozer coordinate `s` and corresponding scalar potential `Phi` values.
  *
  * Ensures that `s_values` and `Phihat_values` have the
  * same size and that all `s_values` are unique.
  *
  * @throws std::invalid_argument if the input vectors are not of
  * the same size or if `s_values` contains duplicates.
  */
  void validateInput() const {
    if (s_values.size() != Phihat_values.size()) {
      throw std::invalid_argument(
          "s_values and Phihat_values must have the same size.");
    }
    if (std::set<double>(s_values.begin(), s_values.end()).size() !=
        s_values.size()) {
      throw std::invalid_argument(
          "s_values contains duplicate entries; all s must be unique.");
    }
  }

  /**
  * @brief Sorts the input data based on `s_values`.
  *
  * Sorts both `s_values` and `Phihat_values` in ascending order
  * of `s_values` to ensure correct interpolation.
  */
  void sortData() {
    std::vector<size_t> indices(s_values.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [this](size_t i1, size_t i2) {
      return s_values[i1] < s_values[i2];
    });

    auto sorted_s_values = s_values;
    auto sorted_Phihat_values = Phihat_values;
    for (size_t i = 0; i < indices.size(); ++i) {
      s_values[i] = sorted_s_values[indices[i]];
      Phihat_values[i] = sorted_Phihat_values[indices[i]];
    }
  }

public:

  /**
  * @brief Constructs a Phihat object with the given `s` and `Phihat` values.
  *
  * Initializes the `Phihat` object with vectors of
  * `s` coordinates and their corresponding `Phihat` values.
  *
  * @param s_vals Vector of `s` coordinates.
  * @param Phihat_vals Vector of scalar potential values corresponding to `s`.
  * @throws std::invalid_argument if input vectors are not valid.
  */
  Phihat(const std::vector<double> &s_vals,
         const std::vector<double> &Phihat_vals)
      : s_values(s_vals), Phihat_values(Phihat_vals) {
    validateInput();
    sortData();
  }

  /**
  * @brief Interpolates the scalar potential `Phihat`
  * at a given `s` coordinate.
  *
  * Computes the value of the scalar potential `Phihat` using linear
  * interpolation between the nearest data points.
  * If `s` is outside the range of `s_values`,
  * returns the nearest boundary value.
  *
  * @param s The normalized toroidal Boozer coordinate.
  * @return Interpolated scalar potential value `Phihat` at the given `s`.
  */
  double operator()(double s) const {

    if (s < s_values.front()) {
      return Phihat_values.front();
    }
    if (s > s_values.back()) {
      return Phihat_values.back();
    }

    size_t i_left = 0;
    size_t i_right = s_values.size() - 1;
    for (int i = s_values.size() - 1; i >= 0; --i) {
      if (s_values[i] <= s && (i + 1 < s_values.size())) {
        i_left = i;
        i_right = i + 1;
        break;
      }
    }

    double slope = (Phihat_values[i_right] - Phihat_values[i_left]) /
                   (s_values[i_right] - s_values[i_left]);

    double Phi_at_s = Phihat_values[i_left] + slope * (s-s_values[i_left]);
    return Phi_at_s;
  }

  /**
  * @brief Computes the derivative of the scalar potential `Phihat`
  * at a given `s` coordinate.
  *
  * Computes the slope of `Phihat` at a given `s` using
  * linear interpolation between the nearest data points.
  * If `s` is outside the range of `s_values`, returns 0.0.
  *
  * @param s The normalized toroidal Boozer coordinate.
  * @return The derivative of `Phihat` at the given `s`.
  */
  double derivative(double s) const {
    if (s < s_values.front() || s > s_values.back()) {
      return 0.0;
    }

    size_t i_left = 0;
    size_t i_right = s_values.size() - 1;
    for (int i = s_values.size() - 1; i >= 0; --i) {
      if (s_values[i] <= s && (i + 1 < s_values.size())) {
        i_left = i;
        i_right = i + 1;
        break;
      }
    }

    return (Phihat_values[i_right] - Phihat_values[i_left]) /
           (s_values[i_right] - s_values[i_left]);
  }
  
  /**
  * @brief Returns the sorted s_values used for interpolation.
  *
  * @return A vector of sorted s_values.
  */
  const std::vector<double>& get_s_basis() const {
    return s_values;
  }
};


/**
* @brief Class representing a single harmonic Shear Alfvén Wave.
* See Paul. et al, JPP (2023;89(5):905890515. doi:10.1017/S0022377823001095)
*
* Initializes the Shear Alfvén Wave with the scalar potential of the form
* \f$ \Phi = \hat{\Phi}(s) \sin(m \theta - n \zeta + \omega t + \text{phase}) \f$
* and vector potential alpha determined by the ideal
* Ohm's law (i.e., zero electric field along the field line).
*
* @tparam T Template for tensor type. It should be a template class compatible with xtensor-like syntax.
*/
class ShearAlfvenHarmonic : public ShearAlfvenWave {
public:
    using Array2 = xt::pytensor<double, 2, xt::layout_type::row_major>;
    Phihat phihat;
    int Phim; // Poloidal mode number.
    int Phin; // Toroidal mode number.
    double omega; // Frequency of the wave.
    double phase; // Phase offset of the wave.

protected:
  void _Phi_impl(Array2& Phi) override {
    const Array2& points = this->get_points();
    const auto& ss = xt::view(points, xt::all(), 0);
    const auto& thetas = xt::view(points, xt::all(), 1);
    const auto& zetas = xt::view(points, xt::all(), 2);
    const auto& times = xt::view(points, xt::all(), 3);
    for (std::size_t i = 0; i < ss.size(); ++i) {
      xt::view(Phi, i, 0) = this->phihat(ss(i)) *
        sin(this->Phim * thetas(i) - this->Phin * zetas(i) +
        this->omega * times(i) + this->phase);
    }
  }

  void _dPhidpsi_impl(Array2& dPhidpsi) override {
    const Array2& points = this->get_points();
    const auto& ss = xt::view(points, xt::all(), 0);
    const auto& thetas = xt::view(points, xt::all(), 1);
    const auto& zetas = xt::view(points, xt::all(), 2);
    const auto& times = xt::view(points, xt::all(), 3);
    for (std::size_t i = 0; i < ss.size(); ++i) {
      xt::view(dPhidpsi, i, 0) =
        this->phihat.derivative(ss(i)) / (this->B0->psi0) *
        sin(this->Phim * thetas(i) - this->Phin * zetas(i) +
        this->omega * times(i) + this->phase);
     }
  }
  
  void _dPhidtheta_impl(Array2& dPhidtheta) override {
    xt::view(dPhidtheta, xt::all(), 0) =
      this->Phidot_ref()(0) * (this->Phim / this->omega);
  }

  void _dPhidzeta_impl(Array2& dPhidzeta) override {
    xt::view(dPhidzeta, xt::all(), 0) =
      -this->Phidot_ref()(0) * (this->Phin / this->omega);
  }
  
  void _Phidot_impl(Array2& Phidot) override {
    const Array2& points = this->get_points();
    const auto& ss = xt::view(points, xt::all(), 0);
    const auto& thetas = xt::view(points, xt::all(), 1);
    const auto& zetas = xt::view(points, xt::all(), 2);
    const auto& times = xt::view(points, xt::all(), 3);
    for (std::size_t i = 0; i < ss.size(); ++i) {
      xt::view(Phidot, i, 0) =
        this->phihat(ss(i)) * this->omega *
        cos(this->Phim * thetas(i) - this->Phin * zetas(i) +
        this->omega * times(i) + this->phase);
    }
  }

  void _alpha_impl(Array2& alpha) override {
    xt::view(alpha, xt::all(), 0) =
      -this->Phi_ref()(0) *
      ((this->B0->iota_ref()(0) * this->Phim - this->Phin) /
      (this->omega * (this->B0->G_ref()(0) +
      this->B0->iota_ref()(0) * this->B0->I_ref()(0))));
  }
      
  void _alphadot_impl(Array2& alphadot) override {
    xt::view(alphadot, xt::all(), 0) =
      -this->Phidot_ref()(0) *
      ((this->B0->iota_ref()(0) * this->Phim - this->Phin) /
      (this->omega * (this->B0->G_ref()(0) +
      this->B0->iota_ref()(0) * this->B0->I_ref()(0))));
  }
    
  void _dalphadpsi_impl(Array2& dalphadpsi) override {
    const auto& diotadpsi_values = this->B0->diotads_ref()(0) / this->B0->psi0;
    const auto& dGdpsi_values = this->B0->dGds_ref()(0) / this->B0->psi0;
    const auto& dIdpsi_values = this->B0->dIds_ref()(0) / this->B0->psi0;
    const auto& iota_values = this->B0->iota_ref()(0);
    const auto& G_values = this->B0->G_ref()(0);
    const auto& I_values = this->B0->I_ref()(0);
    const auto& dPhidpsi_values = this->dPhidpsi_ref()(0);
    const auto& Phi_values = this->Phi_ref()(0);
  
    xt::view(dalphadpsi, xt::all(), 0) =
      -dPhidpsi_values * (iota_values * this->Phim - this->Phin) /
      (this->omega * (G_values + iota_values * I_values)) -
      (Phi_values / this->omega) *
        (diotadpsi_values * this->Phim /
          (G_values + iota_values * I_values) -
         (iota_values * this->Phim - this->Phin) /
          ((G_values + iota_values * I_values) *
           (G_values + iota_values * I_values)) *
          (dGdpsi_values + diotadpsi_values * I_values +
            iota_values * dIdpsi_values));
  }
  
  void _dalphadtheta_impl(Array2& dalphadtheta) override {
    const auto& iota_values = this->B0->iota_ref()(0);
    const auto& G_values = this->B0->G_ref()(0);
    const auto& I_values = this->B0->I_ref()(0);
    const auto& dPhidtheta_values = this->dPhidtheta_ref()(0);
    xt::view(dalphadtheta, xt::all(), 0) =
      -dPhidtheta_values * (iota_values * this->Phim - this->Phin) /
      (this->omega * (G_values + iota_values * I_values));
  }
  
  void _dalphadzeta_impl(Array2& dalphadzeta) override {
    const auto& iota_values = this->B0->iota_ref()(0);
    const auto& G_values = this->B0->G_ref()(0);
    const auto& I_values = this->B0->I_ref()(0);
    const auto& dPhidzeta_values = this->dPhidzeta_ref()(0);
    xt::view(dalphadzeta, xt::all(), 0) =
      -dPhidzeta_values * (iota_values * this->Phim - this->Phin) /
      (this->omega * (G_values + iota_values * I_values));
  }
  
  public:
      /**
      * @brief Constructor for the ShearAlfvenHarmonic class.
      *
      * Initializes the Shear Alfvén Harmonic with a given profile `phihat`, mode numbers `m` and `n`,
      * wave frequency `omega`, phase `phase`, and equilibrium magnetic field `B0`.
      *
      * @param phihat_in Profile of the scalar potential.
      * @param Phim Poloidal mode number.
      * @param Phin Toroidal mode number.
      * @param omega Frequency of the wave.
      * @param phase Phase offset of the wave.
      * @param B0field Shared pointer to the equilibrium Boozer magnetic field.
      */
      ShearAlfvenHarmonic(
          const Phihat& phihat_in,
          int Phim,
          int Phin,
          double omega,
          double phase,
          shared_ptr<BoozerMagneticField> B0field
      ) : 
      ShearAlfvenWave(B0field),
      phihat(phihat_in),
      Phim(Phim),
      Phin(Phin),
      omega(omega),
      phase(phase) {}
      
      /**
      * @brief Returns radial amplitude Phihat of the ShearAlfvenHarmonic
      */
      const Phihat& get_phihat() const {
        return phihat;
      }
};


/**
* @brief Class representing a superposition of multiple Shear Alfvén waves.
*
* This class models the superposition of multiple Shear Alfvén waves, combining their scalar
* potential `Phi`, vector potential `alpha`, and their respective derivatives.
*/
class ShearAlfvenWavesSuperposition : public ShearAlfvenWave {
public:
  using Array2 = xt::pytensor<double, 2, xt::layout_type::row_major>;
  //List of waves in superposition:
  std::vector<std::shared_ptr<ShearAlfvenWave>> waves; 

  /**
  * @brief Adds a new wave to the superposition.
  *
  *  Adds a new wave to the superposition after verifying 
  *  that it has the same equilibrium magnetic field `B0`.
  *
  * @param wave Shared pointer to a ShearAlfvenWave object to be added.
  * @throws std::invalid_argument if the wave's `B0` field does not 
  * match the superposition's `B0`.
  */
  void add_wave(const std::shared_ptr<ShearAlfvenWave>& wave) {
    if (wave->get_B0() != this->B0) {
      throw std::invalid_argument(
        "The wave's B0 field does not match the superposition's B0 field."
      );
    }
    waves.push_back(wave);
  }
  
  /**
  * @brief Constructor for ShearAlfvenWavesSuperposition.
  *
  * Initializes the superposition with a base wave, 
  * setting its `B0` field as the reference field
  * for all subsequent waves added to the superposition.
  *
  * @param base_wave Shared pointer to the initial ShearAlfvenWave object.
  * @throws std::invalid_argument if the base wave is not provided.
  */
  ShearAlfvenWavesSuperposition(std::shared_ptr<ShearAlfvenWave> base_wave)
    : ShearAlfvenWave(base_wave->get_B0()) {
    if (!base_wave) {
      throw std::invalid_argument(
        "Base wave must be provided to initialize the superposition."
      );
    }
    add_wave(base_wave);
  }
  
  /**
  * @brief Sets the points (s, theta, zeta, time)
  *
  * Sets the points for the superposition and propagates them to all waves.
  *
  * @param p A tensor representing the points in Boozer coordinates 
  *          and time (s, theta, zeta, time).
  */
  void set_points(Array2& p) override {
    ShearAlfvenWave::set_points(p);
    for (const auto& wave : waves) {
      wave->set_points(p);  // Propagate points to each wave
    }
  }
  
protected:
  void _Phi_impl(Array2& Phi) override {
    Phi.fill(0.0);
    for (const auto& wave : waves) {
      Phi += wave->Phi();
    }
  }
  
  void _dPhidpsi_impl(Array2& dPhidpsi) override {
    dPhidpsi.fill(0.0);
    for (const auto& wave : waves) {
      dPhidpsi += wave->dPhidpsi();
    }
  }
  
  void _dPhidtheta_impl(Array2& dPhidtheta) override {
    dPhidtheta.fill(0.0);
    for (const auto& wave : waves) {
      dPhidtheta += wave->dPhidtheta();
    }
  }
  
  void _dPhidzeta_impl(Array2& dPhidzeta) override {
    dPhidzeta.fill(0.0);
    for (const auto& wave : waves) {
      dPhidzeta += wave->dPhidzeta();
    }
  }
  
  void _Phidot_impl(Array2& Phidot) override {
    Phidot.fill(0.0);
      for (const auto& wave : waves) {
      Phidot += wave->Phidot();
    }
  }
  
  void _alpha_impl(Array2& alpha) override {
    alpha.fill(0.0);
    for (const auto& wave : waves) {
      alpha += wave->alpha();
    }
  }
  
  void _dalphadpsi_impl(Array2& dalphadpsi) override {
    dalphadpsi.fill(0.0);
    for (const auto& wave : waves) {
      dalphadpsi += wave->dalphadpsi();
    }
  }
  
  void _dalphadtheta_impl(Array2& dalphadtheta) override {
    dalphadtheta.fill(0.0);
    for (const auto& wave : waves) {
      dalphadtheta += wave->dalphadtheta();
    }
  }
  
  void _dalphadzeta_impl(Array2& dalphadzeta) override {
    dalphadzeta.fill(0.0);
    for (const auto& wave : waves) {
      dalphadzeta += wave->dalphadzeta();
    }
  }
  
  void _alphadot_impl(Array2& alphadot) override {
    alphadot.fill(0.0);
    for (const auto& wave : waves) {
      alphadot += wave->alphadot();
    }
  }
};
