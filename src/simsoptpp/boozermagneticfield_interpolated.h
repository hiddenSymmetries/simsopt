#pragma once

#include "boozermagneticfield.h"
#include "xtensor/xlayout.hpp"
#include "regular_grid_interpolant_3d.h"

template<template<class, std::size_t, xt::layout_type> class T>
class InterpolatedBoozerField : public BoozerMagneticField<T> {
    public:
        using typename BoozerMagneticField<T>::Tensor2;
    private:

        CachedTensor<T, 2> points_cyl_sym;
        shared_ptr<RegularGridInterpolant3D<Tensor2>> interp_modB, interp_dmodBdtheta, \
          interp_dmodBdzeta, interp_dmodBds, interp_d2modBdtheta2, interp_d2modBdzeta2, \
          interp_d2modBds2, interp_d2modBdsdtheta, interp_d2modBdsdzeta, \
          interp_d2modBdthetadzeta, interp_G, interp_iota, interp_dGds, \
          interp_diotads, interp_psip;
        bool status_modB = false, status_dmodBdtheta = false, status_dmodBdzeta = false, \
          status_dmodBds = false, status_d2modBdtheta2 = false, status_d2modBdzeta2 = false, \
          status_d2modBds2 = false, status_d2modBdsdtheta = false, status_d2modBdsdzeta = false, \
          status_d2modBdthetadzeta = false, status_G = false, status_iota = false,
          status_dGds = false, status_diotads = false, status_psip = false;
        const bool extrapolate;
        const bool stellsym = false;
        const int nfp = 1;
        vector<bool> symmetries = vector<bool>(1, false);

    protected:
      void _psip_impl(Tensor2& psip) override {
          if(!interp_psip)
              interp_psip = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, s_range, angle0_range, angle0_range, 1, extrapolate);
          if(!status_psip) {
              Tensor2 old_points = this->field->get_points();
              string which_scalar = "psip";
              std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                return fbatch_scalar(s,theta,zeta,which_scalar);
              };
              interp_psip->interpolate_batch(fbatch);
              this->field->set_points(old_points);
              status_psip = true;
          }
          Tensor2& stz = this->get_points_ref();
          Tensor2& stz0 = points_cyl_sym.get_or_create({npoints, 3});
          exploit_fluxfunction_points(stz, stz0);
          interp_psip->evaluate_batch(stz0, psip);
      }

        void _G_impl(Tensor2& G) override {
            if(!interp_G)
                interp_G = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, s_range, angle0_range, angle0_range, 1, extrapolate);
            if(!status_G) {
                Tensor2 old_points = this->field->get_points();
                string which_scalar = "G";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_G->interpolate_batch(fbatch);
                this->field->set_points(old_points);
                status_G = true;
            }
            Tensor2& stz = this->get_points_ref();
            Tensor2& stz0 = points_cyl_sym.get_or_create({npoints, 3});
            exploit_fluxfunction_points(stz, stz0);
            interp_G->evaluate_batch(stz0, G);
        }

        void _iota_impl(Tensor2& iota) override {
            if(!interp_iota)
                interp_iota = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, s_range, angle0_range, angle0_range, 1, extrapolate);
            if(!status_iota) {
                Tensor2 old_points = this->field->get_points();
                string which_scalar = "iota";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_iota->interpolate_batch(fbatch);
                this->field->set_points(old_points);
                status_iota = true;
            }
            Tensor2& stz = this->get_points_ref();
            Tensor2& stz0 = points_cyl_sym.get_or_create({npoints, 3});
            exploit_fluxfunction_points(stz, stz0);
            interp_iota->evaluate_batch(stz0, iota);
        }

        void _dGds_impl(Tensor2& dGds) override {
            if(!interp_dGds)
                interp_dGds = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, s_range, angle0_range, angle0_range, 1, extrapolate);
            if(!status_dGds) {
                Tensor2 old_points = this->field->get_points();
                string which_scalar = "dGds";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_dGds->interpolate_batch(fbatch);
                this->field->set_points(old_points);
                status_dGds = true;
            }
            Tensor2& stz = this->get_points_ref();
            Tensor2& stz0 = points_cyl_sym.get_or_create({npoints, 3});
            exploit_fluxfunction_points(stz, stz0);
            interp_dGds->evaluate_batch(stz0, dGds);
        }

        void _diotads_impl(Tensor2& diotads) override {
            if(!interp_diotads)
                interp_diotads = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, s_range, angle0_range, angle0_range, 1, extrapolate);
            if(!status_diotads) {
                Tensor2 old_points = this->field->get_points();
                string which_scalar = "diotads";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_diotads->interpolate_batch(fbatch);
                this->field->set_points(old_points);
                status_diotads = true;
            }
            Tensor2& stz = this->get_points_ref();
            Tensor2& stz0 = points_cyl_sym.get_or_create({npoints, 3});
            exploit_fluxfunction_points(stz, stz0);
            interp_diotads->evaluate_batch(stz0, diotads);
        }

        void _modB_impl(Tensor2& modB) override {
            if(!interp_modB)
                interp_modB = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_modB) {
                Tensor2 old_points = this->field->get_points();
                string which_scalar = "modB";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_modB->interpolate_batch(fbatch);
                this->field->set_points(old_points);
                status_modB = true;
            }
            if(nfp > 1 || stellsym){
                Tensor2& stz = this->get_points_ref();
                Tensor2& stz_sym = points_cyl_sym.get_or_create({npoints, 3});
                exploit_symmetries_points(stz, stz_sym);
                interp_modB->evaluate_batch(stz_sym, modB);
            } else {
                interp_modB->evaluate_batch(this->get_points_ref(), modB);
            }
        }

        void _dmodBdtheta_impl(Tensor2& dmodBdtheta) override {
            if(!interp_dmodBdtheta)
                interp_dmodBdtheta = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_dmodBdtheta) {
                Tensor2 old_points = this->field->get_points();
                string which_scalar = "dmodBdtheta";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_dmodBdtheta->interpolate_batch(fbatch);
                this->field->set_points(old_points);
                status_dmodBdtheta = true;
            }
            if(nfp > 1 || stellsym){
                Tensor2& stz = this->get_points_ref();
                Tensor2& stz_sym = points_cyl_sym.get_or_create({npoints, 3});
                exploit_symmetries_points(stz, stz_sym);
                interp_dmodBdtheta->evaluate_batch(stz_sym, dmodBdtheta);
                apply_symmetries_to_modB_derivatives(dmodBdtheta);
            } else {
                interp_dmodBdtheta->evaluate_batch(this->get_points_ref(), dmodBdtheta);
            }
        }

        void _dmodBdzeta_impl(Tensor2& dmodBdzeta) override {
            if(!interp_dmodBdzeta)
                interp_dmodBdzeta = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_dmodBdzeta) {
                Tensor2 old_points = this->field->get_points();
                string which_scalar = "dmodBdzeta";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_dmodBdzeta->interpolate_batch(fbatch);
                this->field->set_points(old_points);
                status_dmodBdzeta = true;
            }
            if(nfp > 1 || stellsym){
                Tensor2& stz = this->get_points_ref();
                Tensor2& stz_sym = points_cyl_sym.get_or_create({npoints, 3});
                exploit_symmetries_points(stz, stz_sym);
                interp_dmodBdzeta->evaluate_batch(stz_sym, dmodBdzeta);
                apply_symmetries_to_modB_derivatives(dmodBdzeta);
            } else {
                interp_dmodBdzeta->evaluate_batch(this->get_points_ref(), dmodBdzeta);
            }
        }

        void _dmodBds_impl(Tensor2& dmodBds) override {
            if(!interp_dmodBds)
                interp_dmodBds = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_dmodBds) {
                Tensor2 old_points = this->field->get_points();
                string which_scalar = "dmodBds";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_dmodBds->interpolate_batch(fbatch);
                this->field->set_points(old_points);
                status_dmodBds = true;
            }
            if(nfp > 1 || stellsym){
                Tensor2& stz = this->get_points_ref();
                Tensor2& stz_sym = points_cyl_sym.get_or_create({npoints, 3});
                exploit_symmetries_points(stz, stz_sym);
                interp_dmodBds->evaluate_batch(stz_sym, dmodBds);
            } else {
                interp_dmodBds->evaluate_batch(this->get_points_ref(), dmodBds);
            }
        }

        void _d2modBds2_impl(Tensor2& d2modBds2) override {
            if(!interp_d2modBds2)
                interp_d2modBds2 = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_d2modBds2) {
                Tensor2 old_points = this->field->get_points();
                string which_scalar = "d2modBds2";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_d2modBds2->interpolate_batch(fbatch);
                this->field->set_points(old_points);
                status_d2modBds2 = true;
            }
            if(nfp > 1 || stellsym){
                Tensor2& stz = this->get_points_ref();
                Tensor2& stz_sym = points_cyl_sym.get_or_create({npoints, 3});
                exploit_symmetries_points(stz, stz_sym);
                interp_d2modBds2->evaluate_batch(stz_sym, d2modBds2);
            } else {
                interp_d2modBds2->evaluate_batch(this->get_points_ref(), d2modBds2);
            }
        }

        void _d2modBdtheta2_impl(Tensor2& d2modBdtheta2) override {
            if(!interp_d2modBdtheta2)
                interp_d2modBdtheta2 = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_d2modBdtheta2) {
                Tensor2 old_points = this->field->get_points();
                string which_scalar = "d2modBdtheta2";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_d2modBdtheta2->interpolate_batch(fbatch);
                this->field->set_points(old_points);
                status_d2modBdtheta2 = true;
            }
            if(nfp > 1 || stellsym){
                Tensor2& stz = this->get_points_ref();
                Tensor2& stz_sym = points_cyl_sym.get_or_create({npoints, 3});
                exploit_symmetries_points(stz, stz_sym);
                interp_d2modBdtheta2->evaluate_batch(stz_sym, d2modBdtheta2);
            } else {
                interp_d2modBdtheta2->evaluate_batch(this->get_points_ref(), d2modBdtheta2);
            }
        }

        void _d2modBdzeta2_impl(Tensor2& d2modBdzeta2) override {
            if(!interp_d2modBdzeta2)
                interp_d2modBdzeta2 = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_d2modBdzeta2) {
                Tensor2 old_points = this->field->get_points();
                string which_scalar = "d2modBdzeta2";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_d2modBdzeta2->interpolate_batch(fbatch);
                this->field->set_points(old_points);
                status_d2modBdzeta2 = true;
            }
            if(nfp > 1 || stellsym){
                Tensor2& stz = this->get_points_ref();
                Tensor2& stz_sym = points_cyl_sym.get_or_create({npoints, 3});
                exploit_symmetries_points(stz, stz_sym);
                interp_d2modBdzeta2->evaluate_batch(stz_sym, d2modBdzeta2);
            } else {
                interp_d2modBdzeta2->evaluate_batch(this->get_points_ref(), d2modBdzeta2);
            }
        }

        void _d2modBdsdtheta_impl(Tensor2& d2modBdsdtheta) override {
            if(!interp_d2modBdsdtheta)
                interp_d2modBdsdtheta = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_d2modBdsdtheta) {
                Tensor2 old_points = this->field->get_points();
                string which_scalar = "d2modBdsdtheta";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_d2modBdsdtheta->interpolate_batch(fbatch);
                this->field->set_points(old_points);
                status_d2modBdsdtheta = true;
            }
            if(nfp > 1 || stellsym){
                Tensor2& stz = this->get_points_ref();
                Tensor2& stz_sym = points_cyl_sym.get_or_create({npoints, 3});
                exploit_symmetries_points(stz, stz_sym);
                interp_d2modBdsdtheta->evaluate_batch(stz_sym, d2modBdsdtheta);
                apply_symmetries_to_modB_derivatives(d2modBdsdtheta);
            } else {
                interp_d2modBdsdtheta->evaluate_batch(this->get_points_ref(), d2modBdsdtheta);
            }
        }

        void _d2modBdsdzeta_impl(Tensor2& d2modBdsdzeta) override {
            if(!interp_d2modBdsdzeta)
                interp_d2modBdsdzeta = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_d2modBdsdzeta) {
                Tensor2 old_points = this->field->get_points();
                string which_scalar = "d2modBdsdzeta";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_d2modBdsdzeta->interpolate_batch(fbatch);
                this->field->set_points(old_points);
                status_d2modBdsdzeta = true;
            }
            if(nfp > 1 || stellsym){
                Tensor2& stz = this->get_points_ref();
                Tensor2& stz_sym = points_cyl_sym.get_or_create({npoints, 3});
                exploit_symmetries_points(stz, stz_sym);
                interp_d2modBdsdzeta->evaluate_batch(stz_sym, d2modBdsdzeta);
                apply_symmetries_to_modB_derivatives(d2modBdsdzeta);
            } else {
                interp_d2modBdsdzeta->evaluate_batch(this->get_points_ref(), d2modBdsdzeta);
            }
        }

        void _d2modBdthetadzeta_impl(Tensor2& d2modBdthetadzeta) override {
            if(!interp_d2modBdthetadzeta)
                interp_d2modBdthetadzeta = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
            if(!status_d2modBdthetadzeta) {
                Tensor2 old_points = this->field->get_points();
                string which_scalar = "d2modBdthetadzeta";
                std::function<Vec(Vec, Vec, Vec)> fbatch = [this,which_scalar](Vec s, Vec theta, Vec zeta) {
                  return fbatch_scalar(s,theta,zeta,which_scalar);
                };
                interp_d2modBdthetadzeta->interpolate_batch(fbatch);
                this->field->set_points(old_points);
                status_d2modBdthetadzeta = true;
            }
            if(nfp > 1 || stellsym){
                Tensor2& stz = this->get_points_ref();
                Tensor2& stz_sym = points_cyl_sym.get_or_create({npoints, 3});
                exploit_symmetries_points(stz, stz_sym);
                interp_d2modBdthetadzeta->evaluate_batch(stz_sym, d2modBdthetadzeta);
            } else {
                interp_d2modBdthetadzeta->evaluate_batch(this->get_points_ref(), d2modBdthetadzeta);
            }
        }

        void exploit_fluxfunction_points(Tensor2& stz, Tensor2& stz0){
            int npoints = stz.shape(0);
            double* dataptr = &(stz(0, 0));
            double* datasymptr = &(stz0(0, 0));
            for (int i = 0; i < npoints; ++i) {
                double s = dataptr[3*i+0];
                datasymptr[3*i+0] = s;
                datasymptr[3*i+1] = 0.;
                datasymptr[3*i+2] = 0.;
            }
        }

        void exploit_symmetries_points(Tensor2& stz, Tensor2& stz_sym){
            int npoints = stz.shape(0);
            if(symmetries.size() != npoints)
                symmetries = vector<bool>(npoints, false);
            double period = (2*M_PI)/nfp;
            double* dataptr = &(stz(0, 0));
            double* datasymptr = &(stz_sym(0, 0));
            for (int i = 0; i < npoints; ++i) {
                double s = dataptr[3*i+0];
                double theta = dataptr[3*i+1];
                double zeta = dataptr[3*i+2];
                // Restrict theta to [0,2 pi]
                int theta_mult = int(theta/(2*M_PI));
                theta = theta - theta_mult * 2*M_PI;
                // Restrict zeta to [0,2 pi/nfp]
                int zeta_mult = int(zeta/period);
                zeta = zeta - zeta_mult * period;
                if(theta > M_PI && stellsym) {
                    zeta = period-zeta;
                    theta = 2*M_PI-theta;
                    symmetries[i] = true;
                } else{
                    symmetries[i] = false;
                }
                datasymptr[3*i+0] = s;
                datasymptr[3*i+1] = theta;
                datasymptr[3*i+2] = zeta;
            }
        }

        void apply_symmetries_to_modB_derivatives(Tensor2& field){
            int npoints = field.shape(0);
            for (int i = 0; i < npoints; ++i) {
                if(symmetries[i])
                    field(i, 0) = -field(i, 0);
            }
        }

        Vec fbatch_scalar(Vec s, Vec theta, Vec zeta, string which_scalar) {
            int npoints = s.size();
            Tensor2 points = xt::zeros<double>({npoints, 3});
            for(int i=0; i<npoints; i++) {
                points(i, 0) = s[i];
                if ((which_scalar != "G") &&  (which_scalar != "iota") && (which_scalar != "dGds") && (which_scalar != "diotads")) {
                  points(i, 1) = theta[i];
                  points(i, 2) = zeta[i];
                }
            }
            this->field->set_points(points);
            Tensor2 scalar;
            if (which_scalar == "modB") {
              scalar = this->field->modB();
            } else if (which_scalar == "dmodBdtheta") {
              scalar = this->field->dmodBdtheta();
            } else if (which_scalar == "dmodBdzeta") {
              scalar = this->field->dmodBdzeta();
            } else if (which_scalar == "dmodBds") {
              scalar = this->field->dmodBds();
            } else if (which_scalar == "d2modBdtheta2") {
              scalar = this->field->d2modBdtheta2();
            } else if (which_scalar == "d2modBdzeta2") {
              scalar = this->field->d2modBdzeta2();
            } else if (which_scalar == "d2modBds2") {
              scalar = this->field->d2modBds2();
            } else if (which_scalar == "d2modBdsdtheta") {
              scalar = this->field->d2modBdsdtheta();
            } else if (which_scalar == "d2modBdsdzeta") {
              scalar = this->field->d2modBdsdzeta();
            } else if (which_scalar == "d2modBdthetadzeta") {
              scalar = this->field->d2modBdthetadzeta();
            } else if (which_scalar == "G") {
              scalar = this->field->G();
            } else if (which_scalar == "psip") {
              scalar = this->field->psip();
            } else if (which_scalar == "iota") {
              scalar = this->field->iota();
            } else if (which_scalar == "dGds") {
              scalar = this->field->dGds();
            } else if (which_scalar == "diotads") {
              scalar = this->field->diotads();
            } else {
              throw std::runtime_error("Incorrect value for which_scalar.");
            }
            return Vec(scalar.data(), scalar.data()+npoints);
        }

    public:
        const shared_ptr<BoozerMagneticField<T>> field;
        const RangeTriplet s_range, theta_range, zeta_range, angle0_range = {0., M_PI, 1};
        using BoozerMagneticField<T>::npoints;
        const InterpolationRule rule;

        InterpolatedBoozerField(
                shared_ptr<BoozerMagneticField<T>> field, InterpolationRule rule,
                RangeTriplet s_range, RangeTriplet theta_range, RangeTriplet zeta_range,
                bool extrapolate, int nfp, bool stellsym) :
            BoozerMagneticField<T>(field->psi0), field(field), rule(rule), s_range(s_range), theta_range(theta_range), zeta_range(zeta_range), extrapolate(extrapolate), nfp(nfp), stellsym(stellsym)
        {}

        InterpolatedBoozerField(
                shared_ptr<BoozerMagneticField<T>> field, int degree,
                RangeTriplet s_range, RangeTriplet theta_range, RangeTriplet zeta_range,
                bool extrapolate, int nfp, bool stellsym) : InterpolatedBoozerField(field, UniformInterpolationRule(degree), s_range, theta_range, zeta_range, extrapolate, nfp, stellsym) {}

                std::pair<double, double> estimate_error_modB(int samples) {
                    if(!interp_modB) {
                      interp_modB = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, s_range, theta_range, zeta_range, 1, extrapolate);
                    }
                    std::function<Vec(Vec, Vec, Vec)> fbatch = [this](Vec s, Vec theta, Vec zeta) {
                      return fbatch_scalar(s,theta,zeta,"modB");
                    };
                    if(!status_modB) {
                        Tensor2 old_points = this->field->get_points();
                        interp_modB->interpolate_batch(fbatch);
                        this->field->set_points(old_points);
                        status_modB = true;
                    }
                    return interp_modB->estimate_error(fbatch, samples);
                }

                std::pair<double, double> estimate_error_G(int samples) {
                    if(!interp_G) {
                      interp_G = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, s_range, angle0_range, angle0_range, 1, extrapolate);
                    }
                    std::function<Vec(Vec, Vec, Vec)> fbatch = [this](Vec s, Vec theta, Vec zeta) {
                      return fbatch_scalar(s,theta,zeta,"G");
                    };
                    if(!status_G) {
                        Tensor2 old_points = this->field->get_points();
                        interp_G->interpolate_batch(fbatch);
                        this->field->set_points(old_points);
                        status_G = true;
                    }
                    return interp_G->estimate_error(fbatch, samples);
                }

                std::pair<double, double> estimate_error_iota(int samples) {
                    if(!interp_iota) {
                      interp_iota = std::make_shared<RegularGridInterpolant3D<Tensor2>>(rule, s_range, angle0_range, angle0_range, 1, extrapolate);
                    }
                    std::function<Vec(Vec, Vec, Vec)> fbatch = [this](Vec s, Vec theta, Vec zeta) {
                      return fbatch_scalar(s,theta,zeta,"iota");
                    };
                    if(!status_iota) {
                        Tensor2 old_points = this->field->get_points();
                        interp_iota->interpolate_batch(fbatch);
                        this->field->set_points(old_points);
                        status_iota = true;
                    }
                    return interp_iota->estimate_error(fbatch, samples);
                }
};
