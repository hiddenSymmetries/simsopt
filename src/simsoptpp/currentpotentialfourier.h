#pragma once

#pragma once
#include <vector>
using std::vector;

#include <string>
using std::string;

#include <map>
using std::map;
#include <stdexcept>
using std::logic_error;

#include "xtensor/xarray.hpp"
#include "cachedarray.h"
#include <Eigen/Dense>

#include "currentpotential.h"

template<class Array>
class CurrentPotentialFourier : public CurrentPotential<Array> {

    public:
        const shared_ptr<Surface<Array>> winding_surfae;
        using CurrentPotential<Array>::quadpoints_phi;
        using CurrentPotential<Array>::quadpoints_theta;
        using CurrentPotential<Array>::numquadpoints_phi;
        using CurrentPotential<Array>::numquadpoints_theta;
        Array phic;
        Array phis;
        int nfp;
        int mpol;
        int ntor;
        bool stellsym;

        CurrentPotentialFourier(shared_ptr<Surface<Array>> _winding_surface, int _mpol, int _ntor, int _nfp, bool _stellsym, vector<double> _quadpoints_phi, vector<double> _quadpoints_theta)
            : CurrentPotential<Array>(_winding_surface, _quadpoints_phi, _quadpoints_theta), mpol(_mpol), ntor(_ntor), nfp(_nfp), stellsym(_stellsym) {
                this->allocate();
            }

        void allocate() {
            phic = xt::zeros<double>({mpol+1, 2*ntor+1});
            phis = xt::zeros<double>({mpol+1, 2*ntor+1});
        }

        int num_dofs() override {
            if(stellsym)
                return (mpol+1)*(2*ntor+1) - (ntor + 1);
            else
                return 2*(mpol+1)*(2*ntor+1) - ntor - (ntor + 1);
        }

        void set_dofs_impl(const vector<double>& dofs) override {
            int shift = (mpol+1)*(2*ntor+1);
            int counter = 0;
            if(stellsym) {
                for (int i = ntor+1; i < shift; ++i)
                    phis.data()[i] = dofs[counter++];
            } else {
                for (int i = ntor+1; i < shift; ++i)
                    phis.data()[i] = dofs[counter++];
                for (int i = ntor; i < shift; ++i)
                    phic.data()[i] = dofs[counter++];
            }
        }

        vector<double> get_dofs() override {
            auto res = vector<double>(num_dofs(), 0.);
            int shift = (mpol+1)*(2*ntor+1);
            int counter = 0;
            if(stellsym) {
                for (int i = ntor+1; i < shift; ++i)
                    res[counter++] = phis.data()[i];
            } else {
                for (int i = ntor+1; i < shift; ++i)
                    res[counter++] = phis.data()[i];
                for (int i = ntor; i < shift; ++i)
                    res[counter++] = phic.data()[i];
            }
            return res;
        }

        virtual void Phi_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override;
        virtual void Phidash1_impl(Array& data) override;
        virtual void Phidash2_impl(Array& data) override;
        virtual void Phidash1dash1_impl(Array& data) override;
        virtual void Phidash2dash2_impl(Array& data) override;
        virtual void Phidash1dash2_impl(Array& data) override;

        virtual void dPhi_by_dcoeff_impl(Array& data) override;
        virtual void dPhidash1_by_dcoeff_impl(Array& data) override;
        virtual void dPhidash2_by_dcoeff_impl(Array& data) override;
        virtual void dPhidash1dash1_by_dcoeff_impl(Array& data) override;
        virtual void dPhidash1dash2_by_dcoeff_impl(Array& data) override;
        virtual void dPhidash2dash2_by_dcoeff_impl(Array& data) override;
};
