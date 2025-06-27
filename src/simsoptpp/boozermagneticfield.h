#pragma once
#include <xtensor/xnoalias.hpp>
#include <stdexcept>
#include "xtensor-python/pytensor.hpp"
#include <string>

using std::logic_error;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::string;

class BoozerMagneticField {
    public:
        using Array2 = xt::pytensor<double, 2, xt::layout_type::row_major>;
        double psi0;
        string field_type; // "vac" for vacuum, "nok" for no-K, "" for full Boozer field.

    protected:
        virtual void _set_points_cb() {}
        virtual void _K_impl(Array2& K) {
            throw logic_error("_K_impl was not implemented");
        }
        virtual void _dKdtheta_impl(Array2& dKdtheta) {
            throw logic_error("_dKdtheta_impl was not implemented");
        }
        virtual void _dKdzeta_impl(Array2& dKdzeta) {
            throw logic_error("_dKdzeta_impl was not implemented");
        }
        virtual void _K_derivs_impl(Array2& K_derivs) {
            throw logic_error("_K_derivs_impl was not implemented");
        }
        virtual void _nu_impl(Array2& nu) {
            throw logic_error("_nu_impl was not implemented");
        }
        virtual void _dnudtheta_impl(Array2& dnudtheta) {
            throw logic_error("_dnudtheta_impl was not implemented");
        }
        virtual void _dnudzeta_impl(Array2& dnudzeta) {
            throw logic_error("_dnudzeta_impl was not implemented");
        }
        virtual void _dnuds_impl(Array2& dnuds) {
            throw logic_error("_dnuds_impl was not implemented"); 
        }
        virtual void _nu_derivs_impl(Array2& nu_derivs) {
            throw logic_error("_nu_derivs_impl was not implemented");
        }
        virtual void _R_impl(Array2& R) {
            throw logic_error("_R_impl was not implemented");
        }
        virtual void _Z_impl(Array2& Z) {
            throw logic_error("_Z_impl was not implemented");
        }
        virtual void _dRdtheta_impl(Array2& dRdtheta) {
            throw logic_error("_dRdtheta_impl was not implemented");
        }
        virtual void _dZdtheta_impl(Array2& dZdtheta) {
            throw logic_error("_dZdtheta_impl was not implemented");
        }
        virtual void _dRdzeta_impl(Array2& dRdzeta) {
            throw logic_error("_dRdzeta_impl was not implemented");
        }
        virtual void _dZdzeta_impl(Array2& dZdzeta) {
            throw logic_error("_dZdzeta_impl was not implemented");
        }
        virtual void _dRds_impl(Array2& dRds) {
            throw logic_error("_dRds_impl was not implemented");
        }
        virtual void _dZds_impl(Array2& dZds) {
            throw logic_error("_dZds_impl was not implemented");
        }
        virtual void _R_derivs_impl(Array2& R_derivs) {
            throw logic_error("_R_derivs_impl was not implemented");
        }
        virtual void _Z_derivs_impl(Array2& Z_derivs) {
            throw logic_error("_Z_derivs_impl was not implemented");
        }
        virtual void _modB_impl(Array2& modB) {
            throw logic_error("_modB_impl was not implemented");
        }
        virtual void _dmodBdtheta_impl(Array2& dmodBdtheta) {
            throw logic_error("_dmodBdtheta_impl was not implemented");
        }
        virtual void _dmodBdzeta_impl(Array2& dmodBdzeta) {
            throw logic_error("_dmodBdzeta_impl was not implemented");
        }
        virtual void _dmodBds_impl(Array2& dmodBds) {
            throw logic_error("_dmodBds_impl was not implemented");
        }
        virtual void _modB_derivs_impl(Array2& modB_derivs) {
            throw logic_error("_modB_derivs_impl was not implemented");
        }
        virtual void _I_impl(Array2& I) {
            throw logic_error("_I_impl was not implemented");
        }
        virtual void _G_impl(Array2& G) {
            throw logic_error("_G_impl was not implemented");
        }
        virtual void _dIds_impl(Array2& dIds) {
            throw logic_error("_dIds_impl was not implemented");
        }
        virtual void _dGds_impl(Array2& dGds) {
            throw logic_error("_dGds_impl was not implemented");
        }
        virtual void _psip_impl(Array2& psip) {
            throw logic_error("_psip_impl was not implemented");
        }
        virtual void _iota_impl(Array2& iota) {
            throw logic_error("_iota_impl was not implemented");
        }
        virtual void _diotads_impl(Array2& diotads) {
            throw logic_error("_diotads_impl was not implemented");
        }
        virtual void _set_points() {}
        Array2 points, points_sym;
        Array2 data_modB, data_modB_derivs;
        Array2 data_dmodBds, data_dmodBdtheta, data_dmodBdzeta;
        Array2 data_G, data_dGds;
        Array2 data_iota, data_diotads;
        Array2 data_psip;
        Array2 data_I, data_dIds;
        Array2 data_nu, data_nu_derivs;
        Array2 data_dnuds, data_dnudtheta, data_dnudzeta;
        Array2 data_K, data_K_derivs;
        Array2 data_dKdtheta, data_dKdzeta;
        Array2 data_R, data_R_derivs;
        Array2 data_dRds, data_dRdtheta, data_dRdzeta;
        Array2 data_Z, data_Z_derivs;
        Array2 data_dZds, data_dZdtheta, data_dZdzeta;
        long npoints;
        bool status_modB=false, status_modB_derivs=false;
        bool status_dmodBds=false, status_dmodBdtheta=false, status_dmodBdzeta=false;
        bool status_G=false, status_dGds=false;
        bool status_iota=false, status_diotads=false; 
        bool status_psip=false;
        bool status_I=false, status_dIds=false;  
        bool status_nu=false, status_nu_derivs=false;
        bool status_dnuds=false, status_dnudtheta=false, status_dnudzeta=false;
        bool status_K=false, status_K_derivs=false;
        bool status_dKdtheta=false, status_dKdzeta=false;
        bool status_R=false, status_R_derivs=false;
        bool status_dRds=false, status_dRdtheta=false, status_dRdzeta=false;
        bool status_Z=false, status_Z_derivs=false;
        bool status_dZds=false, status_dZdtheta=false, status_dZdzeta=false;

    public:
        BoozerMagneticField(double psi0, string field_type) : psi0(psi0), field_type(field_type) {
            Array2 vals({{0., 0., 0.}});
            this->set_points(vals);
        }

        void set_points(Array2& p) {
            npoints = p.shape(0);
            points.resize({npoints, 3});
            Array2& _points = points;
            memcpy(_points.data(), p.data(), 3*npoints*sizeof(double));
            this->_set_points_cb();
            status_modB = false; 
            status_modB_derivs = false;
            status_dmodBds = false;
            status_dmodBdtheta = false;
            status_dmodBdzeta = false;
            status_G = false;
            status_dGds = false;
            status_iota = false;
            status_diotads = false;
            status_psip = false;
            status_I = false;
            status_dIds = false;
            status_nu = false;
            status_nu_derivs = false;
            status_dnuds = false;
            status_dnudtheta = false;
            status_dnudzeta = false;
            status_K = false;
            status_K_derivs = false;
            status_dKdtheta = false;
            status_dKdzeta = false;
            status_R = false;
            status_R_derivs = false;
            status_dRds = false;
            status_dRdtheta = false;
            status_dRdzeta = false;
            status_Z = false;
            status_Z_derivs = false;
            status_dZds = false;
            status_dZdtheta = false;
            status_dZdzeta = false;
        }

        Array2 get_points() {
            return get_points_ref();
        }

        Array2& get_points_ref() {
            return points;
        }

        Array2& get_sym_points_ref() {
            return points_sym;
        }

        Array2& K_ref() {
            if (!status_K) {
                data_K.resize({npoints, 1});
                _K_impl(data_K);
                status_K = true;
            }
            return data_K;
        }

        Array2& K_derivs_ref() {
            if (!status_K_derivs) {
                data_K_derivs.resize({npoints, 2});
                _K_derivs_impl(data_K_derivs);
                status_K_derivs = true;
            }
            return data_K_derivs;
        }

        Array2& dKdtheta_ref() {
            if (!status_dKdtheta) {
                data_dKdtheta.resize({npoints, 1});
                _dKdtheta_impl(data_dKdtheta);
                status_dKdtheta = true;
            }
            return data_dKdtheta;
        }

        Array2& dKdzeta_ref() {
            if (!status_dKdzeta) {
                data_dKdzeta.resize({npoints, 1});
                _dKdzeta_impl(data_dKdzeta);
                status_dKdzeta = true;
            }
            return data_dKdzeta;
        }

        Array2& nu_ref() {
            if (!status_nu) {
                data_nu.resize({npoints, 1});
                _nu_impl(data_nu);
                status_nu = true;
            }
            return data_nu;
        }

        Array2& dnudtheta_ref() {
            if (!status_dnudtheta) {
                data_dnudtheta.resize({npoints, 1});
                _dnudtheta_impl(data_dnudtheta);
                status_dnudtheta = true;
            }
            return data_dnudtheta;
        }

        Array2& dnudzeta_ref() {
            if (!status_dnudzeta) {
                data_dnudzeta.resize({npoints, 1});
                _dnudzeta_impl(data_dnudzeta);
                status_dnudzeta = true;
            }
            return data_dnudzeta;
        }

        Array2& dnuds_ref() {
            if (!status_dnuds) {
                data_dnuds.resize({npoints, 1});
                _dnuds_impl(data_dnuds);
                status_dnuds = true;
            }
            return data_dnuds;
        }

        Array2& nu_derivs_ref() {
            if (!status_nu_derivs) {
                data_nu_derivs.resize({npoints, 3});
                _nu_derivs_impl(data_nu_derivs);
                status_nu_derivs = true;
            }
            return data_nu_derivs;
        }

        Array2& R_ref() {
            if (!status_R) {
                data_R.resize({npoints, 1});
                _R_impl(data_R);
                status_R = true;
            }
            return data_R;
        }

        Array2& dRdtheta_ref() {
            if (!status_dRdtheta) {
                data_dRdtheta.resize({npoints, 1});
                _dRdtheta_impl(data_dRdtheta);
                status_dRdtheta = true;
            }
            return data_dRdtheta;
        }

        Array2& dRdzeta_ref() {
            if (!status_dRdzeta) {
                data_dRdzeta.resize({npoints, 1});
                _dRdzeta_impl(data_dRdzeta);
                status_dRdzeta = true;
            }
            return data_dRdzeta;
        }

        Array2& dRds_ref() {
            if (!status_dRds) {
                data_dRds.resize({npoints, 1});
                _dRds_impl(data_dRds);
                status_dRds = true;
            }
            return data_dRds;
        }

        Array2& R_derivs_ref() {
            if (!status_R_derivs) {
                data_R_derivs.resize({npoints, 3});
                _R_derivs_impl(data_R_derivs);
                status_R_derivs = true;
            }
            return data_R_derivs;
        }

        Array2& Z_ref() {
            if (!status_Z) {
                data_Z.resize({npoints, 1});
                _Z_impl(data_Z);
                status_Z = true;
            }
            return data_Z;
        }

        Array2& dZdtheta_ref() {
            if (!status_dZdtheta) {
                data_dZdtheta.resize({npoints, 1});
                _dZdtheta_impl(data_dZdtheta);
                status_dZdtheta = true;
            }
            return data_dZdtheta;
        }

        Array2& dZdzeta_ref() {
            if (!status_dZdzeta) {
                data_dZdzeta.resize({npoints, 1});
                _dZdzeta_impl(data_dZdzeta);
                status_dZdzeta = true;
            }
            return data_dZdzeta;
        }

        Array2& dZds_ref() {
            if (!status_dZds) {
                data_dZds.resize({npoints, 1});
                _dZds_impl(data_dZds);
                status_dZds = true;
            }
            return data_dZds;
        }

        Array2& Z_derivs_ref() {
            if (!status_Z_derivs) {
                data_Z_derivs.resize({npoints, 3});
                _Z_derivs_impl(data_Z_derivs);
                status_Z_derivs = true;
            }
            return data_Z_derivs;
        }

        Array2& modB_ref() {
            if (!status_modB) {
                data_modB.resize({npoints, 1});
                _modB_impl(data_modB);
                status_modB = true;
            }
            return data_modB;
        }

        Array2& dmodBdtheta_ref() {
            if (!status_dmodBdtheta) {
                data_dmodBdtheta.resize({npoints, 1});
                _dmodBdtheta_impl(data_dmodBdtheta);
                status_dmodBdtheta = true;
            }
            return data_dmodBdtheta;
        }

        Array2& dmodBdzeta_ref() {
            if (!status_dmodBdzeta) {
                data_dmodBdzeta.resize({npoints, 1});
                _dmodBdzeta_impl(data_dmodBdzeta);
                status_dmodBdzeta = true;
            }
            return data_dmodBdzeta;
        }

        Array2& dmodBds_ref() {
            if (!status_dmodBds) {
                data_dmodBds.resize({npoints, 1});
                _dmodBds_impl(data_dmodBds);
                status_dmodBds = true;
            }
            return data_dmodBds;
        }

        Array2& modB_derivs_ref() {
            if (!status_modB_derivs) {
                data_modB_derivs.resize({npoints, 3});
                _modB_derivs_impl(data_modB_derivs);
                status_modB_derivs = true;
            }
            return data_modB_derivs;
        }

        Array2& I_ref() {
            if (!status_I) {
                data_I.resize({npoints, 1});
                _I_impl(data_I);
                status_I = true;
            }
            return data_I;
        }

        Array2& G_ref() {
            if (!status_G) {
                data_G.resize({npoints, 1});
                _G_impl(data_G);
                status_G = true;
            }
            return data_G;
        }

        Array2& psip_ref() {
            if (!status_psip) {
                data_psip.resize({npoints, 1});
                _psip_impl(data_psip);
                status_psip = true;
            }
            return data_psip;
        }

        Array2& iota_ref() {
            if (!status_iota) {
                data_iota.resize({npoints, 1});
                _iota_impl(data_iota);
                status_iota = true;
            }
            return data_iota;
        }

        Array2& dGds_ref() {
            if (!status_dGds) {
                data_dGds.resize({npoints, 1});
                _dGds_impl(data_dGds);
                status_dGds = true;
            }
            return data_dGds;
        }

        Array2& dIds_ref() {
            if (!status_dIds) {
                data_dIds.resize({npoints, 1});
                _dIds_impl(data_dIds);
                status_dIds = true;
            }
            return data_dIds;
        }

        Array2& diotads_ref() {
            if (!status_diotads) {
                data_diotads.resize({npoints, 1});
                _diotads_impl(data_diotads);
                status_diotads = true;
            }
            return data_diotads;
        }

        Array2 K() { return K_ref(); }
        Array2 dKdtheta() { return dKdtheta_ref(); }
        Array2 dKdzeta() { return dKdzeta_ref(); }
        Array2 K_derivs() { return K_derivs_ref(); }
        Array2 nu() { return nu_ref(); }
        Array2 dnudtheta() { return dnudtheta_ref(); }
        Array2 dnudzeta() { return dnudzeta_ref(); }
        Array2 dnuds() { return dnuds_ref(); }
        Array2 nu_derivs() { return nu_derivs_ref(); }
        Array2 R() { return R_ref(); }
        Array2 Z() { return Z_ref(); }
        Array2 dRdtheta() { return dRdtheta_ref(); }
        Array2 dZdtheta() { return dZdtheta_ref(); }
        Array2 dRdzeta() { return dRdzeta_ref(); }
        Array2 dZdzeta() { return dZdzeta_ref(); }
        Array2 dRds() { return dRds_ref(); }
        Array2 dZds() { return dZds_ref(); }
        Array2 R_derivs() {return R_derivs_ref();}
        Array2 Z_derivs() {return Z_derivs_ref();}
        Array2 modB() { return modB_ref(); }
        Array2 dmodBdtheta() { return dmodBdtheta_ref(); }
        Array2 dmodBdzeta() { return dmodBdzeta_ref(); }
        Array2 dmodBds() { return dmodBds_ref(); }
        Array2 modB_derivs() { return modB_derivs_ref(); }
        Array2 G() { return G_ref(); }
        Array2 I() { return I_ref(); }
        Array2 psip() { return psip_ref(); }
        Array2 iota() { return iota_ref(); }
        Array2 dGds() { return dGds_ref(); }
        Array2 dIds() { return dIds_ref(); }
        Array2 diotads() { return diotads_ref(); }

};
