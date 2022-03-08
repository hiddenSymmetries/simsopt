#pragma once
#include <xtensor/xarray.hpp>
#include <xtensor/xnoalias.hpp>
#include <stdexcept>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>


#include "cachedarray.h"
#include "cache.h"
#include "cachedtensor.h"

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
