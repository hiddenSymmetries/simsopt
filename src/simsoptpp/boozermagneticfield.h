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
        virtual void _modB_impl(Tensor2& modB) { throw logic_error("_modB_impl was not implemented"); }
        virtual void _dmodBdtheta_impl(Tensor2& dmodBdtheta) { throw logic_error("_dmodBdtheta_impl was not implemented"); }
        virtual void _dmodBdzeta_impl(Tensor2& dmodBdzeta) { throw logic_error("_dmodBdzeta_impl was not implemented"); }
        virtual void _dmodBds_impl(Tensor2& dmodBds) { throw logic_error("_dmodBds_impl was not implemented"); }
        virtual void _d2modBdtheta2_impl(Tensor2& d2modBdtheta2) { throw logic_error("_d2modBdtheta2_impl was not implemented"); }
        virtual void _d2modBdzeta2_impl(Tensor2& d2modBdzeta2) { throw logic_error("_d2modBdzeta2_impl was not implemented"); }
        virtual void _d2modBds2_impl(Tensor2& d2modBds2) { throw logic_error("_d2modBds2_impl was not implemented"); }
        virtual void _d2modBdthetadzeta_impl(Tensor2& d2modBdthetadzeta) { throw logic_error("_d2modBdthetadzeta_impl was not implemented"); }
        virtual void _d2modBdsdzeta_impl(Tensor2& d2modBdsdzeta) { throw logic_error("_d2modBdsdzeta_impl was not implemented"); }
        virtual void _d2modBdsdtheta_impl(Tensor2& d2modBdsdtheta) { throw logic_error("_d2modBdsdtheta_impl was not implemented"); }
        virtual void _G_impl(Tensor2& G) { throw logic_error("_G_impl was not implemented"); }
        virtual void _psip_impl(Tensor2& psip) { throw logic_error("_psip_impl was not implemented"); }
        virtual void _iota_impl(Tensor2& iota) { throw logic_error("_iota_impl was not implemented"); }
        virtual void _dGds_impl(Tensor2& dGds) { throw logic_error("_dGds_impl was not implemented"); }
        virtual void _diotads_impl(Tensor2& diotads) { throw logic_error("_diotads_impl was not implemented"); }
        virtual void _set_points() { }

        CachedTensor<T, 2> points;
        CachedTensor<T, 2> data_modB, data_dmodBdtheta, data_dmodBdzeta, data_dmodBds,\
          data_d2modBdtheta2, data_d2modBdzeta2, data_d2modBds2, data_d2modBdthetadzeta,\
          data_d2modBdsdzeta, data_d2modBdsdtheta, data_G, data_iota, data_dGds, data_diotads, data_psip;
        int npoints;

    public:
        BoozerMagneticField(double psi0) : psi0(psi0) {
            Tensor2 vals({{0., 0., 0.}});
            this->set_points(vals);
        }

        virtual void invalidate_cache() {
            data_modB.invalidate_cache();
            data_dmodBdtheta.invalidate_cache();
            data_dmodBdzeta.invalidate_cache();
            data_dmodBds.invalidate_cache();
            data_d2modBdtheta2.invalidate_cache();
            data_d2modBdzeta2.invalidate_cache();
            data_d2modBds2.invalidate_cache();
            data_d2modBdthetadzeta.invalidate_cache();
            data_d2modBdsdzeta.invalidate_cache();
            data_d2modBdsdtheta.invalidate_cache();
            data_G.invalidate_cache();
            data_psip.invalidate_cache();
            data_iota.invalidate_cache();
            data_dGds.invalidate_cache();
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

        Tensor2& d2modBdtheta2_ref() {
            return data_d2modBdtheta2.get_or_create_and_fill({npoints, 1}, [this](Tensor2& d2modBdtheta2) { return _d2modBdtheta2_impl(d2modBdtheta2);});
        }

        Tensor2& d2modBdzeta2_ref() {
            return data_d2modBdzeta2.get_or_create_and_fill({npoints, 1}, [this](Tensor2& d2modBdzeta2) { return _d2modBdzeta2_impl(d2modBdzeta2);});
        }

        Tensor2& d2modBds2_ref() {
            return data_d2modBds2.get_or_create_and_fill({npoints, 1}, [this](Tensor2& d2modBds2) { return _d2modBds2_impl(d2modBds2);});
        }

        Tensor2& d2modBdthetadzeta_ref() {
            return data_d2modBdthetadzeta.get_or_create_and_fill({npoints, 1}, [this](Tensor2& d2modBdthetadzeta) { return _d2modBdthetadzeta_impl(d2modBdthetadzeta);});
        }

        Tensor2& d2modBdsdzeta_ref() {
            return data_d2modBdsdzeta.get_or_create_and_fill({npoints, 1}, [this](Tensor2& d2modBdsdzeta) { return _d2modBdsdzeta_impl(d2modBdsdzeta);});
        }

        Tensor2& d2modBdsdtheta_ref() {
            return data_d2modBdsdtheta.get_or_create_and_fill({npoints, 1}, [this](Tensor2& d2modBdsdtheta) { return _d2modBdsdtheta_impl(d2modBdsdtheta);});
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

        Tensor2& diotads_ref() {
            return data_diotads.get_or_create_and_fill({npoints, 1}, [this](Tensor2& diotads) { return _diotads_impl(diotads);});
        }

        Tensor2 modB() { return modB_ref(); }
        Tensor2 dmodBdtheta() { return dmodBdtheta_ref(); }
        Tensor2 dmodBdzeta() { return dmodBdzeta_ref(); }
        Tensor2 dmodBds() { return dmodBds_ref(); }
        Tensor2 d2modBdtheta2() { return d2modBdtheta2_ref(); }
        Tensor2 d2modBdzeta2() { return d2modBdzeta2_ref(); }
        Tensor2 d2modBds2() { return d2modBds2_ref(); }
        Tensor2 d2modBdthetadzeta() { return d2modBdthetadzeta_ref(); }
        Tensor2 d2modBdsdzeta() { return d2modBdsdzeta_ref(); }
        Tensor2 d2modBdsdtheta() { return d2modBdsdtheta_ref(); }
        Tensor2 G() { return G_ref(); }
        Tensor2 psip() { return psip_ref(); }
        Tensor2 iota() { return iota_ref(); }
        Tensor2 dGds() { return dGds_ref(); }
        Tensor2 diotads() { return diotads_ref(); }

};
