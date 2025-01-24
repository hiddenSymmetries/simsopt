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

    void set_points(Array2& p) {
        if (p.shape(1) != 4) {
            throw std::invalid_argument("Input tensor must have 4 columns: Boozer coordinates, and time (s, theta, zeta, time)");
        }
        npoints = p.shape(0);
        points.resize({npoints, 4});
        memcpy(points.data(), p.data(), 4 * npoints * sizeof(double));
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
};
