#pragma once
#include <xtensor/xarray.hpp>
#include <xtensor/xnoalias.hpp>
#include <stdexcept>
#include "cachedarray.h"
#include "cache.h"
#include "cachedtensor.h"
#include <vector>       // For std::vector
                        // (for Phihat and ShearAlfvenWavesSuperposition)
#include <algorithm>    // For std::sort (for Phihat)
#include <numeric>      // For std::iota (for Phihat)
#include <stdexcept>    // For std::invalid_argument (for Phihat)
#include <set>          // For std::set (for Phihat)
#include <xtensor/xview.hpp> // To access parts of the xtensor
                             // (for ShearAlfvenWave and ShearAlfvenHarmonic)

using std::logic_error;
using std::vector;
using std::shared_ptr;
using std::make_shared;

template<template<class, std::size_t, xt::layout_type> class T>
class BoozerMagneticField {
    public:
      using Tensor1 = T<double, 1, xt::layout_type::row_major>;
      using Tensor2 = T<double, 2, xt::layout_type::row_major>;
      using Tensor3 = T<double, 3, xt::layout_type::row_major>;
      using Tensor4 = T<double, 4, xt::layout_type::row_major>;
      double psi0;

    protected:
        virtual void _set_points_cb() { }
        virtual void _K_impl(Tensor2& K) { throw logic_error("_K_impl was not implemented"); }
        virtual void _dKdtheta_impl(Tensor2& dKdtheta) { throw logic_error("_dKdtheta_impl was not implemented"); }
        virtual void _dKdzeta_impl(Tensor2& dKdzeta) { throw logic_error("_dKdzeta_impl was not implemented"); }
        virtual void _K_derivs_impl(Tensor2& K_derivs) { throw logic_error("_K_derivs_impl was not implemented"); }
        virtual void _nu_impl(Tensor2& nu) { throw logic_error("_nu_impl was not implemented"); }
        virtual void _dnudtheta_impl(Tensor2& dnudtheta) { throw logic_error("_dnudtheta_impl was not implemented"); }
        virtual void _dnudzeta_impl(Tensor2& dnudzeta) { throw logic_error("_dnudzeta_impl was not implemented"); }
        virtual void _dnuds_impl(Tensor2& dnuds) { throw logic_error("_dnuds_impl was not implemented"); }
        virtual void _nu_derivs_impl(Tensor2& nu_derivs) { throw logic_error("_nu_derivs_impl was not implemented"); }
        virtual void _R_impl(Tensor2& R) { throw logic_error("_R_impl was not implemented"); }
        virtual void _Z_impl(Tensor2& Z) { throw logic_error("_Z_impl was not implemented"); }
        virtual void _dRdtheta_impl(Tensor2& dRdtheta) { throw logic_error("_dRdtheta_impl was not implemented"); }
        virtual void _dZdtheta_impl(Tensor2& dZdtheta) { throw logic_error("_dZdtheta_impl was not implemented"); }
        virtual void _dRdzeta_impl(Tensor2& dRdzeta) { throw logic_error("_dRdzeta_impl was not implemented"); }
        virtual void _dZdzeta_impl(Tensor2& dZdzeta) { throw logic_error("_dZdzeta_impl was not implemented"); }
        virtual void _dRds_impl(Tensor2& dRds) { throw logic_error("_dRds_impl was not implemented"); }
        virtual void _dZds_impl(Tensor2& dZds) { throw logic_error("_dZds_impl was not implemented"); }
        virtual void _R_derivs_impl(Tensor2& R_derivs) { throw logic_error("_R_derivs_impl was not implemented"); }
        virtual void _Z_derivs_impl(Tensor2& Z_derivs) { throw logic_error("_Z_derivs_impl was not implemented"); }
        virtual void _modB_impl(Tensor2& modB) { throw logic_error("_modB_impl was not implemented"); }
        virtual void _dmodBdtheta_impl(Tensor2& dmodBdtheta) { throw logic_error("_dmodBdtheta_impl was not implemented"); }
        virtual void _dmodBdzeta_impl(Tensor2& dmodBdzeta) { throw logic_error("_dmodBdzeta_impl was not implemented"); }
        virtual void _dmodBds_impl(Tensor2& dmodBds) { throw logic_error("_dmodBds_impl was not implemented"); }
        virtual void _modB_derivs_impl(Tensor2& modB_derivs) { throw logic_error("_modB_derivs_impl was not implemented"); }
        virtual void _I_impl(Tensor2& I) { throw logic_error("_I_impl was not implemented"); }
        virtual void _G_impl(Tensor2& G) { throw logic_error("_G_impl was not implemented"); }
        virtual void _dIds_impl(Tensor2& dIds) { throw logic_error("_dIds_impl was not implemented"); }
        virtual void _dGds_impl(Tensor2& dGds) { throw logic_error("_dGds_impl was not implemented"); }
        virtual void _psip_impl(Tensor2& psip) { throw logic_error("_psip_impl was not implemented"); }
        virtual void _iota_impl(Tensor2& iota) { throw logic_error("_iota_impl was not implemented"); }
        virtual void _diotads_impl(Tensor2& diotads) { throw logic_error("_diotads_impl was not implemented"); }
        virtual void _set_points() { }

        CachedTensor<T, 2> points;
        CachedTensor<T, 2> data_modB, data_dmodBdtheta, data_dmodBdzeta, data_dmodBds,\
          data_modB_derivs, data_G, data_iota, data_dGds, data_diotads, data_psip, \
          data_I, data_dIds, data_R, data_Z, data_nu, data_K, data_dRdtheta, data_dRdzeta, \
          data_dRds, data_R_derivs, data_dZdtheta, data_dZdzeta, data_dZds, data_Z_derivs, \
          data_dnudtheta, data_dnudzeta, data_dnuds, data_nu_derivs, data_dKdtheta, \
          data_dKdzeta, data_K_derivs;
        int npoints;

    public:
        BoozerMagneticField(double psi0) : psi0(psi0) {
            Tensor2 vals({{0., 0., 0.}});
            this->set_points(vals);
        }

        virtual void invalidate_cache() {
            data_modB.invalidate_cache();
            data_K.invalidate_cache();
            data_K_derivs.invalidate_cache();
            data_dKdtheta.invalidate_cache();
            data_dKdzeta.invalidate_cache();
            data_nu.invalidate_cache();
            data_dnudtheta.invalidate_cache();
            data_dnudzeta.invalidate_cache();
            data_dnuds.invalidate_cache();
            data_nu_derivs.invalidate_cache();
            data_R.invalidate_cache();
            data_dRdtheta.invalidate_cache();
            data_dRdzeta.invalidate_cache();
            data_dRds.invalidate_cache();
            data_R_derivs.invalidate_cache();
            data_Z.invalidate_cache();
            data_dZdtheta.invalidate_cache();
            data_dZdzeta.invalidate_cache();
            data_dZds.invalidate_cache();
            data_Z_derivs.invalidate_cache();
            data_dmodBdtheta.invalidate_cache();
            data_dmodBdzeta.invalidate_cache();
            data_dmodBds.invalidate_cache();
            data_modB_derivs.invalidate_cache();
            data_I.invalidate_cache();
            data_G.invalidate_cache();
            data_psip.invalidate_cache();
            data_iota.invalidate_cache();
            data_dGds.invalidate_cache();
            data_dIds.invalidate_cache();
            data_diotads.invalidate_cache();
        }

        BoozerMagneticField& set_points(Tensor2& p) {
            this->invalidate_cache();
            this->points.invalidate_cache();
            npoints = p.shape(0);
            Tensor2& _points = points.get_or_create({npoints, 3});
            memcpy(_points.data(), p.data(), 3*npoints*sizeof(double));
            this->_set_points_cb();
            return *this;
        }

        Tensor2 get_points() {
            return get_points_ref();
        }

        Tensor2& get_points_ref() {
            return points.get_or_create({npoints, 3});
        }

        Tensor2& K_ref() {
            return data_K.get_or_create_and_fill({npoints, 1}, [this](Tensor2& K) { return _K_impl(K);});
        }

        Tensor2& K_derivs_ref() {
            return data_K_derivs.get_or_create_and_fill({npoints, 2}, [this](Tensor2& K_derivs) { return _K_derivs_impl(K_derivs);});
        }

        Tensor2& dKdtheta_ref() {
            return data_dKdtheta.get_or_create_and_fill({npoints, 1}, [this](Tensor2& dKdtheta) { return _dKdtheta_impl(dKdtheta);});
        }

        Tensor2& dKdzeta_ref() {
            return data_dKdzeta.get_or_create_and_fill({npoints, 1}, [this](Tensor2& dKdzeta) { return _dKdzeta_impl(dKdzeta);});
        }

        Tensor2& nu_ref() {
            return data_nu.get_or_create_and_fill({npoints, 1}, [this](Tensor2& nu) { return _nu_impl(nu);});
        }

        Tensor2& dnudtheta_ref() {
            return data_dnudtheta.get_or_create_and_fill({npoints, 1}, [this](Tensor2& dnudtheta) { return _dnudtheta_impl(dnudtheta);});
        }

        Tensor2& dnudzeta_ref() {
            return data_dnudzeta.get_or_create_and_fill({npoints, 1}, [this](Tensor2& dnudzeta) { return _dnudzeta_impl(dnudzeta);});
        }

        Tensor2& dnuds_ref() {
            return data_dnuds.get_or_create_and_fill({npoints, 1}, [this](Tensor2& dnuds) { return _dnuds_impl(dnuds);});
        }

        Tensor2& nu_derivs_ref() {
            return data_nu_derivs.get_or_create_and_fill({npoints, 3}, [this](Tensor2& nu_derivs) { return _nu_derivs_impl(nu_derivs);});
        }

        Tensor2& R_ref() {
            return data_R.get_or_create_and_fill({npoints, 1}, [this](Tensor2& R) { return _R_impl(R);});
        }

        Tensor2& dRdtheta_ref() {
            return data_dRdtheta.get_or_create_and_fill({npoints, 1}, [this](Tensor2& dRdtheta) { return _dRdtheta_impl(dRdtheta);});
        }

        Tensor2& dRdzeta_ref() {
            return data_dRdzeta.get_or_create_and_fill({npoints, 1}, [this](Tensor2& dRdzeta) { return _dRdzeta_impl(dRdzeta);});
        }

        Tensor2& dRds_ref() {
            return data_dRds.get_or_create_and_fill({npoints, 1}, [this](Tensor2& dRds) { return _dRds_impl(dRds);});
        }

        Tensor2& R_derivs_ref() {
            return data_R_derivs.get_or_create_and_fill({npoints, 3}, [this](Tensor2& R_derivs) { return _R_derivs_impl(R_derivs);});
        }

        Tensor2& Z_ref() {
            return data_Z.get_or_create_and_fill({npoints, 1}, [this](Tensor2& Z) { return _Z_impl(Z);});
        }

        Tensor2& dZdtheta_ref() {
            return data_dZdtheta.get_or_create_and_fill({npoints, 1}, [this](Tensor2& dZdtheta) { return _dZdtheta_impl(dZdtheta);});
        }

        Tensor2& dZdzeta_ref() {
            return data_dZdzeta.get_or_create_and_fill({npoints, 1}, [this](Tensor2& dZdzeta) { return _dZdzeta_impl(dZdzeta);});
        }

        Tensor2& dZds_ref() {
            return data_dZds.get_or_create_and_fill({npoints, 1}, [this](Tensor2& dZds) { return _dZds_impl(dZds);});
        }

        Tensor2& Z_derivs_ref() {
            return data_Z_derivs.get_or_create_and_fill({npoints, 3}, [this](Tensor2& Z_derivs) { return _Z_derivs_impl(Z_derivs);});
        }

        Tensor2& modB_ref() {
            return data_modB.get_or_create_and_fill({npoints, 1}, [this](Tensor2& modB) { return _modB_impl(modB);});
        }

        Tensor2& dmodBdtheta_ref() {
            return data_dmodBdtheta.get_or_create_and_fill({npoints, 1}, [this](Tensor2& dmodBdtheta) { return _dmodBdtheta_impl(dmodBdtheta);});
        }

        Tensor2& dmodBdzeta_ref() {
            return data_dmodBdzeta.get_or_create_and_fill({npoints, 1}, [this](Tensor2& dmodBdzeta) { return _dmodBdzeta_impl(dmodBdzeta);});
        }

        Tensor2& dmodBds_ref() {
            return data_dmodBds.get_or_create_and_fill({npoints, 1}, [this](Tensor2& dmodBds) { return _dmodBds_impl(dmodBds);});
        }

        Tensor2& modB_derivs_ref() {
            return data_modB_derivs.get_or_create_and_fill({npoints, 3}, [this](Tensor2& modB_derivs) { return _modB_derivs_impl(modB_derivs);});
        }

        Tensor2& I_ref() {
            return data_I.get_or_create_and_fill({npoints, 1}, [this](Tensor2& I) { return _I_impl(I);});
        }

        Tensor2& G_ref() {
            return data_G.get_or_create_and_fill({npoints, 1}, [this](Tensor2& G) { return _G_impl(G);});
        }

        Tensor2& psip_ref() {
            return data_psip.get_or_create_and_fill({npoints, 1}, [this](Tensor2& psip) { return _psip_impl(psip);});
        }

        Tensor2& iota_ref() {
            return data_iota.get_or_create_and_fill({npoints, 1}, [this](Tensor2& iota) { return _iota_impl(iota);});
        }

        Tensor2& dGds_ref() {
            return data_dGds.get_or_create_and_fill({npoints, 1}, [this](Tensor2& dGds) { return _dGds_impl(dGds);});
        }

        Tensor2& dIds_ref() {
            return data_dIds.get_or_create_and_fill({npoints, 1}, [this](Tensor2& dIds) { return _dIds_impl(dIds);});
        }

        Tensor2& diotads_ref() {
            return data_diotads.get_or_create_and_fill({npoints, 1}, [this](Tensor2& diotads) { return _diotads_impl(diotads);});
        }

        Tensor2 K() { return K_ref(); }
        Tensor2 dKdtheta() { return dKdtheta_ref(); }
        Tensor2 dKdzeta() { return dKdzeta_ref(); }
        Tensor2 K_derivs() { return K_derivs_ref(); }
        Tensor2 nu() { return nu_ref(); }
        Tensor2 dnudtheta() { return dnudtheta_ref(); }
        Tensor2 dnudzeta() { return dnudzeta_ref(); }
        Tensor2 dnuds() { return dnuds_ref(); }
        Tensor2 nu_derivs() { return nu_derivs_ref(); }
        Tensor2 R() { return R_ref(); }
        Tensor2 Z() { return Z_ref(); }
        Tensor2 dRdtheta() { return dRdtheta_ref(); }
        Tensor2 dZdtheta() { return dZdtheta_ref(); }
        Tensor2 dRdzeta() { return dRdzeta_ref(); }
        Tensor2 dZdzeta() { return dZdzeta_ref(); }
        Tensor2 dRds() { return dRds_ref(); }
        Tensor2 dZds() { return dZds_ref(); }
        Tensor2 R_derivs() {return R_derivs_ref();}
        Tensor2 Z_derivs() {return Z_derivs_ref();}
        Tensor2 modB() { return modB_ref(); }
        Tensor2 dmodBdtheta() { return dmodBdtheta_ref(); }
        Tensor2 dmodBdzeta() { return dmodBdzeta_ref(); }
        Tensor2 dmodBds() { return dmodBds_ref(); }
        Tensor2 modB_derivs() { return modB_derivs_ref(); }
        Tensor2 G() { return G_ref(); }
        Tensor2 I() { return I_ref(); }
        Tensor2 psip() { return psip_ref(); }
        Tensor2 iota() { return iota_ref(); }
        Tensor2 dGds() { return dGds_ref(); }
        Tensor2 dIds() { return dIds_ref(); }
        Tensor2 diotads() { return diotads_ref(); }

};


/**
* @brief Class representing a generic Shear Alfvén Wave.
* See Paul. et al,
* JPP (2023;89(5):905890515. doi:10.1017/S0022377823001095),
* and refs. therein.
*
* The Shear Alfvén Wave (SAW) propagating in equilibrium field B0 is
* represented by scalar potential `Phi` and vector potential parameter alpha, * s.t. the SAW magnetic field is represented by the curl of (alpha*B0).
*
* This class provides a framework for representing SAW in Boozer coordinates, * with attributes for computing scalar & vector potential and their
* derivatives: `Phi`, `dPhidpsi`, `Phidot`, etc.
*
* It is designed to be a base class,
* to be extended to implement specific behaviors.
*
* @tparam T Template for tensor type.
* It should be a template class compatible with xtensor-like syntax.
*/
template <template <class, std::size_t, xt::layout_type> class T>
class ShearAlfvenWave {
public:
  using Tensor2 = T<double, 2, xt::layout_type::row_major>;

protected:
  std::shared_ptr<BoozerMagneticField<T>> B0;
  CachedTensor<T, 2> points;
  CachedTensor<T, 2> data_Phi, data_dPhidpsi, data_Phidot, data_dPhidtheta,
    data_dPhidzeta, data_alpha, data_alphadot, data_dalphadtheta,
    data_dalphadpsi, data_dalphadzeta;
  int npoints;

  /**
  * @brief Virtual method to compute the scalar potential `Phi`.
  *
  * This method should be overridden by derived classes to provide
  * the specific implementation for computing the Phi tensor.
  *
  * @param Phi Reference to the Phi tensor to be filled.
  */
  virtual void _Phi_impl(Tensor2 &Phi) {
    throw logic_error("_Phi_impl was not implemented");
  }

  /**
  * @brief Virtual method to compute the partial derivative of
  * scalar potential `Phi` w.r.t. toroidal flux coordinate `psi` .
  *
  * This method should be overridden by derived classes to provide
  * the specific implementation for computing the dPhidpsi tensor.
  *
  * @param dPhidpsi Reference to the dPhidpsi tensor to be filled.
  */
  virtual void _dPhidpsi_impl(Tensor2 &dPhidpsi) {
    throw logic_error("_dPhidpsi_impl was not implemented");
  }

  /**
  * @brief Virtual method to compute the partial derivative of
  * scalar potential `Phi` w.r.t. poloidal Boozer angle `theta`.
  *
  * This method should be overridden by derived classes to provide
  * specific implementation for computing the dPhidtheta tensor.
  *
  * @param dPhidtheta Reference to the dPhidtheta tensor to be filled.
  */
  virtual void _dPhidtheta_impl(Tensor2 &dPhidtheta) {
    throw logic_error("_dPhidtheta_impl was not implemented");
  }

  /**
  * @brief Virtual method to compute the partial derivative of
  * scalar potential `Phi` w.r.t. toroidal Boozer angle `zeta`.
  *
  * This method should be overridden by derived classes to provide the
  * specific implementation for computing the dPhidzeta tensor.
  *
  * @param dPhidzeta Reference to the dPhidzeta tensor to be filled.
  */
  virtual void _dPhidzeta_impl(Tensor2 &dPhidzeta) {
    throw logic_error("_dPhidzeta_impl was not implemented");
  }

  /**
  * @brief Virtual method to compute the partial derivative of
  * scalar potential `Phi` w.r.t. time.
  *
  * This method should be overridden by derived classes to provide the
  * specific implementation for computing the Phidot tensor.
  *
  * @param Phidot Reference to the Phidot tensor to be filled.
  */
  virtual void _Phidot_impl(Tensor2 &Phidot) {
    throw logic_error("_Phidot_impl was not implemented");
  }

  /**
  * @brief Virtual method to compute the vector potential parameter `alpha`.
  *
  * This method should be overridden by derived classes to provide the
  * specific implementation for computing the alpha tensor.
  *
  * @param alpha Reference to the alpha tensor to be filled.
  */
  virtual void _alpha_impl(Tensor2 &alpha) {
    throw logic_error("_alpha_impl was not implemented");
  }

  /**
  * @brief Virtual method to compute the partial derivative of the
  * vector potential parameter `alpha` w.r.t. toroidal flux `psi`
  *
  * This method should be overridden by derived classes to provide the
  * specific implementation for computing the dalphadpsi tensor.
  *
  * @param dalphadpsi Reference to the dalphadpsi tensor to be filled.
  */
  virtual void _dalphadpsi_impl(Tensor2 &dalphadpsi) {
    throw logic_error("_dalphadpsi_impl was not implemented");
  }

  /**
  * @brief Virtual method to compute the partial derivative of the
  * vector potential parameter `alpha` w.r.t. poloidal Boozer angle `theta`
  *
  * This method should be overridden by derived classes to provide the
  * specific implementation for computing the alphadot tensor.
  *
  * @param alphadot Reference to the alphadot tensor to be filled.
  */
  virtual void _dalphadtheta_impl(Tensor2 &dalphadtheta) {
    throw logic_error("_dalphadtheta_impl was not implemented");
  }

  /**
  * @brief Virtual method to compute the partial derivative of the
  * vector potential parameter `alpha` w.r.t. toroidal Boozer angle `zeta`
  *
  * This method should be overridden by derived classes to provide the
  * specific implementation for computing the dalphadzeta tensor.
  *
  * @param dalphadzeta Reference to the dalphadzeta tensor to be filled.
  */
  virtual void _dalphadzeta_impl(Tensor2 &dalphadzeta) {
    throw logic_error("_dalphadzeta_impl was not implemented");
  }

  /**
  * @brief Virtual method to compute the partial derivative of the
  * vector potential parameter `alpha` w.r.t. time
  *
  * This method should be overridden by derived classes to provide the
  * specific implementation for computing the alphadot tensor.
  *
  * @param alphadot Reference to the alphadot tensor to be filled.
  */
  virtual void _alphadot_impl(Tensor2 &alphadot) {
    throw logic_error("_alphadot_impl was not implemented");
  }

  virtual void _set_points() {}
  virtual void _set_points_cb() {}

public:

  /**
  * @brief Constructor for ShearAlfvenWave with an equilibrium magnetic field.
  *
  * @param B0field Shared pointer to the equilibrium Boozer magnetic field.
  */
  ShearAlfvenWave(std::shared_ptr<BoozerMagneticField<T>> B0field)
        : B0(B0field) {
      Tensor2 vals({{0., 0., 0., 0.}});
      this->set_points(vals);
  }

  virtual ~ShearAlfvenWave() {}

  virtual void invalidate_cache() {
    data_Phi.invalidate_cache();
    data_Phidot.invalidate_cache();
    data_dPhidpsi.invalidate_cache();
    data_dPhidtheta.invalidate_cache();
    data_dPhidzeta.invalidate_cache();
    data_alpha.invalidate_cache();
    data_alphadot.invalidate_cache();
    data_dalphadtheta.invalidate_cache();
    data_dalphadpsi.invalidate_cache();
    data_dalphadzeta.invalidate_cache();
  }

  /**
  * @brief Sets the points for the perturbation and the equilibrium field.
  *
  * @param p A tensor representing the points
  * in Boozer coordinates and time (s, theta, zeta, time).
  *
  * @return Reference to the current instance
  * @throws std::invalid_argument if the input tensor does not have 4 columns.
  */
  virtual ShearAlfvenWave &set_points(Tensor2 &p) {
    this->invalidate_cache();
    this->points.invalidate_cache();
    if (p.shape(1) != 4) {
        throw std::invalid_argument(
        "Input tensor must have 4 columns: Boozer coordinates,"
        "and time (s, theta, zeta, time)"
        );
    }
    // Set points for ShearAlfvenWave (4 columns: s, theta, zeta, time)
    npoints = p.shape(0);
    Tensor2 &_points = points.get_or_create({npoints, 4});
    memcpy(_points.data(), p.data(), 4 * npoints * sizeof(double));
    this->_set_points_cb();

    // Set points for B0 using the first three columns of p (s, theta, zeta)
    Tensor2 p_b0 = xt::view(p, xt::all(), xt::range(0, 3));
    B0->set_points(p_b0);
    return *this;
  }

  Tensor2 get_points() { return get_points_ref(); }

  Tensor2 &get_points_ref() { return points.get_or_create({npoints, 4}); }

  Tensor2 &Phi_ref() {
    return data_Phi.get_or_create_and_fill(
        {npoints, 1}, [this](Tensor2 &Phi) { return _Phi_impl(Phi); });
  }

  Tensor2 &dPhidpsi_ref() {
    return data_dPhidpsi.get_or_create_and_fill(
        {npoints, 1},
        [this](Tensor2 &dPhidpsi) { return _dPhidpsi_impl(dPhidpsi); });
  }

  Tensor2 &Phidot_ref() {
    return data_Phidot.get_or_create_and_fill(
        {npoints, 1}, [this](Tensor2 &Phidot) { return _Phidot_impl(Phidot); });
  }

  Tensor2 &dPhidtheta_ref() {
    return data_dPhidtheta.get_or_create_and_fill(
        {npoints, 1},
        [this](Tensor2 &dPhidtheta) { return _dPhidtheta_impl(dPhidtheta); });
  }

  Tensor2 &dPhidzeta_ref() {
    return data_dPhidzeta.get_or_create_and_fill(
        {npoints, 1},
        [this](Tensor2 &dPhidzeta) { return _dPhidzeta_impl(dPhidzeta); });
  }

  Tensor2 &alpha_ref() {
    return data_alpha.get_or_create_and_fill(
        {npoints, 1}, [this](Tensor2 &alpha) { return _alpha_impl(alpha); });
  }

  Tensor2 &alphadot_ref() {
    return data_alphadot.get_or_create_and_fill(
        {npoints, 1},
        [this](Tensor2 &alphadot) { return _alphadot_impl(alphadot); });
  }

  Tensor2 &dalphadtheta_ref() {
    return data_dalphadtheta.get_or_create_and_fill(
        {npoints, 1}, [this](Tensor2 &dalphadtheta) {
          return _dalphadtheta_impl(dalphadtheta);
        });
  }

  Tensor2 &dalphadpsi_ref() {
    return data_dalphadpsi.get_or_create_and_fill(
        {npoints, 1},
        [this](Tensor2 &dalphadpsi) { return _dalphadpsi_impl(dalphadpsi); });
  }

  Tensor2 &dalphadzeta_ref() {
    return data_dalphadzeta.get_or_create_and_fill(
        {npoints, 1}, [this](Tensor2 &dalphadzeta) {
          return _dalphadzeta_impl(dalphadzeta);
        });
  }

  Tensor2 Phi() { return Phi_ref(); }
  Tensor2 dPhidpsi() { return dPhidpsi_ref(); }
  Tensor2 Phidot() { return Phidot_ref(); }
  Tensor2 dPhidtheta() { return dPhidtheta_ref(); }
  Tensor2 dPhidzeta() { return dPhidzeta_ref(); }
  Tensor2 alpha() { return alpha_ref(); }
  Tensor2 alphadot() { return alphadot_ref(); }
  Tensor2 dalphadtheta() { return dalphadtheta_ref(); }
  Tensor2 dalphadpsi() { return dalphadpsi_ref(); }
  Tensor2 dalphadzeta() { return dalphadzeta_ref(); }

  std::shared_ptr<BoozerMagneticField<T>> get_B0() const {
            return B0;
        }
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
* \f$ \Phi=\hat{\Phi}(s)\sin(m\theta - n\zeta + \omega t + \text{phase}) \f$
* and vector potential alpha determined by the ideal
* Ohm's law (i.e., zero electric field along the field line).
*
* @tparam T Template for tensor type. It should be a template class
* compatible with xtensor-like syntax.
*/
template <template <class, std::size_t, xt::layout_type> class T>
class ShearAlfvenHarmonic : public ShearAlfvenWave<T> {
public:
  using Tensor2 = T<double, 2, xt::layout_type::row_major>;
  Phihat phihat;
  int Phim;       // Poloidal mode number.
  int Phin;       // Toroidal mode number.
  double omega;   // Frequency of the wave.
  double phase;   // Phase offset of the wave.

protected:

  /**
  * @brief Computes te scalar potential `Phi` at each point.
  *
  * This method calculates the scalar potential \f$ \Phi \f$
  * based on the given harmonic parameters and profile \f$ \hat{\Phi}(s) \f$.
  *
  * @param Phi Tensor to be filled with the computed scalar potential values.
  */
  void _Phi_impl(Tensor2 &Phi) override {
    const Tensor2 &points = this->get_points_ref();
    const auto &ss = xt::view(points, xt::all(), 0);
    const auto &thetas = xt::view(points, xt::all(), 1);
    const auto &zetas = xt::view(points, xt::all(), 2);
    const auto &times = xt::view(points, xt::all(), 3);
    for (std::size_t i = 0; i < ss.size(); ++i) {
      xt::view(Phi, i, 0) = this->phihat(ss(i))
                          * sin(this->Phim * thetas(i) - this->Phin * zetas(i) + this->omega * times(i) + this->phase);
    }
  }

  /**
  * @brief Computes the partial derivative of `Phi` with respect
  * to toroidal flux coordinate `psi`.
  *
  * This method computes the partial derivative
  * \f$ \frac{\partial \Phi}{\partial \psi} \f$ using the derivative
  * of \f$ \hat{\Phi}(s) \f$ and the harmonic parameters.
  *
  * @param dPhidpsi Tensor to be filled with the computed values.
  */
  void _dPhidpsi_impl(Tensor2 &dPhidpsi) override {
    const Tensor2 &points = this->get_points_ref();
    const auto &ss = xt::view(points, xt::all(), 0);
    const auto &thetas = xt::view(points, xt::all(), 1);
    const auto &zetas = xt::view(points, xt::all(), 2);
    const auto &times = xt::view(points, xt::all(), 3);
    for (std::size_t i = 0; i < ss.size(); ++i) {
      xt::view(dPhidpsi, i, 0) =
        this->phihat.derivative(ss(i)) / (this->B0->psi0)
          * sin(this->Phim * thetas(i) - this->Phin * zetas(i)
            + this->omega * times(i) + this->phase);
    }
  }

  /**
  * @brief Computes the partial derivative of `Phi`
  * with respect to poloidal angle `theta`.
  *
  * @param dPhidtheta Tensor to be filled with the computed values.
  */
  void _dPhidtheta_impl(Tensor2 &dPhidtheta) override {
    const auto Phidot_values = xt::squeeze(this->Phidot_ref(), 1);

    xt::view(dPhidtheta, xt::all(), 0) =
      Phidot_values * (this->Phim / this->omega);
  }

  /**
  * @brief Computes the partial derivative of `Phi`
  * with respect to toroidal angle `zeta`.
  *
  * @param dPhidzeta Tensor to be filled with the computed values.
  */
  void _dPhidzeta_impl(Tensor2 &dPhidzeta) override {
    const auto Phidot_values = xt::squeeze(this->Phidot_ref(), 1);

    xt::view(dPhidzeta, xt::all(), 0) =
      -Phidot_values * (this->Phin / this->omega);
  }

  /**
  * @brief Computes the time derivative of `Phi`.
  *
  * This method computes the time derivative
  * \f$ \frac{\partial \Phi}{\partial t} \f$.
  *
  * @param Phidot Tensor to be filled with the computed time derivative values.
  */
  void _Phidot_impl(Tensor2 &Phidot) override {
    const Tensor2 &points = this->get_points_ref();
    const auto &ss = xt::view(points, xt::all(), 0);
    const auto thetas = xt::view(points, xt::all(), 1);
    const auto zetas = xt::view(points, xt::all(), 2);
    const auto times = xt::view(points, xt::all(), 3);

    for (std::size_t i = 0; i < ss.size(); ++i) {
      xt::view(Phidot, i, 0) = this->phihat(ss(i)) * this->omega
        * cos(this->Phim * thetas(i) - this->Phin * zetas(i)
          + this->omega * times(i) + this->phase);
    }
  }

  /**
  * @brief Computes the vector potential parameter `alpha`.
  *
  * @param alpha Tensor to be filled with the computed
  * vector potential parameter values.
  */
  void _alpha_impl(Tensor2 &alpha) override {
    const auto phi = xt::squeeze(this->Phi_ref(), 1);
    const auto iota_ref = xt::squeeze(this->B0->iota_ref(), 1);
    const auto G_ref = xt::squeeze(this->B0->G_ref(), 1);
    const auto I_ref = xt::squeeze(this->B0->I_ref(), 1);

    xt::view(alpha, xt::all(), 0) =
        - phi * ((iota_ref * this->Phim - this->Phin)
                  / (this->omega * (G_ref + iota_ref * I_ref)));
  }

  /**
  * @brief Computes the time derivative of `alpha`.
  *
  * This method computes the time derivative
  * \f$ \frac{\partial \alpha}{\partial t} \f$.
  *
  * @param alphadot Tensor to be filled with the computed
  * time derivative values.
  */
  void _alphadot_impl(Tensor2 &alphadot) override {
    const auto phidot = xt::squeeze(this->Phidot_ref(), 1);
    const auto iota_ref = xt::squeeze(this->B0->iota_ref(), 1);
    const auto G_ref = xt::squeeze(this->B0->G_ref(), 1);
    const auto I_ref = xt::squeeze(this->B0->I_ref(), 1);

    xt::view(alphadot, xt::all(), 0) =
      -phidot * ((iota_ref * this->Phim - this->Phin)
        / (this->omega * (G_ref + iota_ref * I_ref)));
  }

  /**
  * @brief Computes the partial derivative of `alpha`
  * with respect to toroidal flux coordinate `psi`.
  *
  * This method computes the partial derivative
  * \f$ \frac{\partial \alpha}{\partial \psi} \f$ using the derivative
  * of \f$ \hat{\Phi}(s) \f$ and the harmonic parameters.
  *
  * @param dalphadpsi Tensor to be filled with the computed values.
  */
  void _dalphadpsi_impl(Tensor2 &dalphadpsi) override {
    const auto diotadpsi_values = xt::squeeze(this->B0->diotads_ref(), 1)
                            / this->B0->psi0;
    const auto dGdpsi_values = xt::squeeze(this->B0->dGds_ref(), 1)
                         / this->B0->psi0;
    const auto dIdpsi_values = xt::squeeze(this->B0->dIds_ref(), 1)
                         / this->B0->psi0;
    const auto iota_values = xt::squeeze(this->B0->iota_ref(), 1);
    const auto G_values = xt::squeeze(this->B0->G_ref(), 1);
    const auto I_values = xt::squeeze(this->B0->I_ref(), 1);
    const auto dPhidpsi_values = xt::squeeze(this->dPhidpsi_ref(), 1);
    const auto Phi_values = xt::squeeze(this->Phi_ref(), 1);

    xt::view(dalphadpsi, xt::all(), 0) =
      - dPhidpsi_values * (iota_values * this->Phim - this->Phin)
                        / (this->omega * (G_values + iota_values * I_values))
      - (Phi_values / this->omega)
      * ( diotadpsi_values * this->Phim / (G_values + iota_values * I_values)
          - (iota_values * this->Phim - this->Phin)
            / (
                (G_values + iota_values * I_values)
                * (G_values + iota_values * I_values)
              )
            * (
                dGdpsi_values + diotadpsi_values * I_values
                + iota_values * dIdpsi_values
              )
        );
  }

  /**
  * @brief Computes the partial derivative of `alpha`
  * with respect to poloidal angle `theta`.
  *
  * @param dalphadtheta Tensor to be filled with the computed values.
  */
  void _dalphadtheta_impl(Tensor2 &dalphadtheta) override {
    const auto iota_values = xt::squeeze(this->B0->iota_ref(), 1);
    const auto G_values = xt::squeeze(this->B0->G_ref(), 1);
    const auto I_values = xt::squeeze(this->B0->I_ref(), 1);
    const auto dPhidtheta_values = xt::squeeze(this->dPhidtheta_ref(), 1);

    xt::view(dalphadtheta, xt::all(), 0) =
      - dPhidtheta_values * (iota_values * this->Phim - this->Phin)
        / (this->omega * (G_values + iota_values * I_values));
  }

  /**
  * @brief Computes the partial derivative of `alpha`
  * with respect to toroidal angle `zeta`.
  *
  * @param dalphadzeta Tensor to be filled with the computed values.
  */
  void _dalphadzeta_impl(Tensor2 &dalphadzeta) override {
    const auto iota_values = xt::squeeze(this->B0->iota_ref(), 1);
    const auto G_values = xt::squeeze(this->B0->G_ref(), 1);
    const auto I_values = xt::squeeze(this->B0->I_ref(), 1);
    const auto dPhidzeta_values = xt::squeeze(this->dPhidzeta_ref(), 1);

    xt::view(dalphadzeta, xt::all(), 0) =
        - dPhidzeta_values * (iota_values * this->Phim - this->Phin)
          / (this->omega * (G_values + iota_values * I_values));
  }

public:

  /**
  * @brief Constructor for the ShearAlfvenHarmonic class.
  *
  * Initializes the Shear Alfvén Harmonic with a given profile `phihat`,
  * mode numbers `m` and `n`,
  * wave frequency `omega`, phase `phase`, and equilibrium magnetic field `B0`.
  *
  * @param phihat_in Profile of the scalar potential.
  * @param Phim Poloidal mode number.
  * @param Phin Toroidal mode number.
  * @param omega Frequency of the wave.
  * @param phase Phase offset of the wave.
  * @param B0field Shared pointer to the equilibrium Boozer magnetic field.
  */
  ShearAlfvenHarmonic(const Phihat &phihat_in, int Phim, int Phin, double omega,
                      double phase,
                      shared_ptr<BoozerMagneticField<T>> B0field)
      : ShearAlfvenWave<T>(B0field),
        phihat(phihat_in), Phim(Phim), Phin(Phin), omega(omega), phase(phase) {}
        
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
* This class models the superposition of multiple Shear Alfvén waves,
* combining their scalar potential `Phi`, vector potential `alpha`,
* and their respective derivatives.
*
* @tparam T Template for tensor type.
* It should be a template class compatible with xtensor-like syntax.
*/
template <template <class, std::size_t, xt::layout_type> class T>
class ShearAlfvenWavesSuperposition : public ShearAlfvenWave<T> {
public:
  using Tensor2 = T<double, 2, xt::layout_type::row_major>;
  std::vector<std::shared_ptr<ShearAlfvenWave<T>>> waves;

  /**
  * @brief Adds a new wave to the superposition after verifying
  * that it has the same equilibrium magnetic fieldn`B0`.
  *
  * @param wave Shared pointer to a ShearAlfvenWave object to be added.
  * @throws std::invalid_argument if the wave's
  * `B0` field does not match the superposition's `B0`.
  */
  void add_wave(const std::shared_ptr<ShearAlfvenWave<T>> &wave) {
    if (wave->get_B0() != this->B0) {
        throw std::invalid_argument("The wave's B0 field does not match the superposition's B0 field.");
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
  ShearAlfvenWavesSuperposition(std::shared_ptr<ShearAlfvenWave<T>> base_wave)
        : ShearAlfvenWave<T>(base_wave->get_B0()) {
      if (!base_wave) {
        throw std::invalid_argument("Base wave must be provided to initialize the superposition.");
      }
      add_wave(base_wave);
  }

  /**
  * @brief Sets the points for the superposition and propagates them to all waves.
  *
  * This method sets the points for the superposition
  * and ensures that all waves in the superposition
  * are updated with the same points.
  *
  * @param p A tensor representing the points in
  * Boozer coordinates and time (s, theta, zeta, time).
  * @return Reference to the current instance for chaining.
  */
  ShearAlfvenWave<T> &set_points(Tensor2 &p) override {
    ShearAlfvenWave<T>::set_points(p);
    for (const auto &wave : waves) {
      wave->set_points(this->get_points_ref());
    }
    return *this;
  }

protected:

  /**
  * @brief Computes the scalar potential `Phi` as
  * the sum of `Phi` of all waves.
  *
  * @param Phi Tensor to be filled with the computed scalar potential values.
  */
  void _Phi_impl(Tensor2 &Phi) override {
    Phi.fill(0.0);
    for (const auto &wave : waves) {
      Phi += wave->Phi();
    }
  }

  /**
  * @brief Computes the partial derivative of `Phi` with respect to toroidal flux coordinate `psi`.
  *
  * @param dPhidpsi Tensor to be filled with the computed derivative values.
  */
  void _dPhidpsi_impl(Tensor2 &dPhidpsi) override {
    dPhidpsi.fill(0.0);
    for (const auto &wave : waves) {
      dPhidpsi += wave->dPhidpsi();
    }
  }

  /**
  * @brief Computes the partial derivative of `Phi` with respect to poloidal angle `theta`.
  *
  * @param dPhidtheta Tensor to be filled with the computed derivative values.
  */
  void _dPhidtheta_impl(Tensor2 &dPhidtheta) override {
    dPhidtheta.fill(0.0);
    for (const auto &wave : waves) {
      dPhidtheta += wave->dPhidtheta();
    }
  }

  /**
  * @brief Computes the partial derivative of `Phi`
  * with respect to toroidal angle `zeta`.
  *
  * @param dPhidzeta Tensor to be filled with the computed derivative values.
  */
  void _dPhidzeta_impl(Tensor2 &dPhidzeta) override {
    dPhidzeta.fill(0.0);
    for (const auto &wave : waves) {
      dPhidzeta += wave->dPhidzeta();
    }
  }

  /**
  * @brief Computes the partial time derivative of `Phi`.
  *
  * @param Phidot Tensor to be filled with the computed time derivative values.
  */
  void _Phidot_impl(Tensor2 &Phidot) override {
    Phidot.fill(0.0);
    for (const auto &wave : waves) {
      Phidot += wave->Phidot();
    }
  }

  /**
  * @brief Computes the vector potential parameter `alpha`
  * as the sum of `alpha` of all waves.
  *
  * @param alpha Tensor to be filled with the computed vector potential parameter values.
  */
  void _alpha_impl(Tensor2 &alpha) override {
    alpha.fill(0.0);
    for (const auto &wave : waves) {
      alpha += wave->alpha();
    }
  }

  /**
  * @brief Computes the partial derivative of `alpha`
  * with respect to toroidal flux `psi`.
  *
  * @param dalphadpsi Tensor to be filled with the computed derivative values.
  */
  void _dalphadpsi_impl(Tensor2 &dalphadpsi) override {
    dalphadpsi.fill(0.0);
    for (const auto &wave : waves) {
      dalphadpsi += wave->dalphadpsi();
    }
  }

  /**
  * @brief Computes the partial derivative of `alpha`
  * with respect to poloidal angle `theta`.
  *
  * @param dalphadtheta Tensor to be filled with the computed derivative values.
  */
  void _dalphadtheta_impl(Tensor2 &dalphadtheta) override {
    dalphadtheta.fill(0.0);
    for (const auto &wave : waves) {
      dalphadtheta += wave->dalphadtheta();
    }
  }

  /**
  * @brief Computes the partial derivative of `alpha` with respect to toroidal angle `zeta`.
  *
  * @param dalphadzeta Tensor to be filled with the computed derivative values.
  */
  void _dalphadzeta_impl(Tensor2 &dalphadzeta) override {
    dalphadzeta.fill(0.0);
    for (const auto &wave : waves) {
      dalphadzeta += wave->dalphadzeta();
    }
  }

  /**
  * @brief Computes the partial time derivative of the vector potential parameter `alpha`.
  *
  * @param alphadot Tensor to be filled with the computed time derivative values.
  */
  void _alphadot_impl(Tensor2 &alphadot) override {
    alphadot.fill(0.0);
    for (const auto &wave : waves) {
      alphadot += wave->alphadot();
    }
  }
};
