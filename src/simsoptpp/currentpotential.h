#pragma once
#include <vector>
using std::vector;

#include <string>
using std::string;

#include <map>
using std::map;
#include <stdexcept>
using std::logic_error;

#include "xtensor/xlayout.hpp"

#include "xtensor/xarray.hpp"
#include "cachedarray.h"
#include "surface.h"
#include <Eigen/Dense>
using std::shared_ptr;

template<class Array>
class CurrentPotential {
    private:
        map<string, CachedArray<Array>> cache;
        map<string, CachedArray<Array>> cache_persistent;

        Array& check_the_cache(string key, vector<int> dims, std::function<void(Array&)> impl){
            auto loc = cache.find(key);
            if(loc == cache.end()){ // Key not found --> allocate array
                loc = cache.insert(std::make_pair(key, CachedArray<Array>(xt::zeros<double>(dims)))).first;
            }
            if(!((loc->second).status)){ // needs recomputing
                impl((loc->second).data);
                (loc->second).status = true;
            }
            return (loc->second).data;
        }

        Array& check_the_persistent_cache(string key, vector<int> dims, std::function<void(Array&)> impl){
            auto loc = cache_persistent.find(key);
            if(loc == cache_persistent.end()){ // Key not found --> allocate array
                loc = cache_persistent.insert(std::make_pair(key, CachedArray<Array>(xt::zeros<double>(dims)))).first;
            }
            if(!((loc->second).status)){ // needs recomputing
                impl((loc->second).data);
                (loc->second).status = true;
            }
            return (loc->second).data;
        }

    public:
        int numquadpoints_phi;
        int numquadpoints_theta;
        Array quadpoints_phi;
        Array quadpoints_theta;
        const shared_ptr<Surface<Array>> winding_surface;

    public:

        CurrentPotential(shared_ptr<Surface<Array>> winding_surface, vector<double> _quadpoints_phi, vector<double> _quadpoints_theta) {
            numquadpoints_phi = _quadpoints_phi.size();
            numquadpoints_theta = _quadpoints_theta.size();

            quadpoints_phi = xt::zeros<double>({numquadpoints_phi});
            for (int i = 0; i < numquadpoints_phi; ++i) {
                quadpoints_phi[i] = _quadpoints_phi[i];
            }
            quadpoints_theta = xt::zeros<double>({numquadpoints_theta});
            for (int i = 0; i < numquadpoints_theta; ++i) {
                quadpoints_theta[i] = _quadpoints_theta[i];
            }
        }

        void invalidate_cache() {

            for (auto it = cache.begin(); it != cache.end(); ++it) {
                (it->second).status = false;
            }
        }

        virtual void set_dofs(const vector<double>& _dofs) {
            this->set_dofs_impl(_dofs);
            this->invalidate_cache();
        }

        void K_impl(Array& data);

        virtual int num_dofs() = 0;
        virtual void set_dofs_impl(const vector<double>& _dofs) = 0;
        virtual vector<double> get_dofs() = 0;

        virtual void Phi_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) = 0;
        virtual void Phidash1_impl(Array& data)  { throw logic_error("Phidash1_impl was not implemented"); };
        virtual void Phidash2_impl(Array& data)  { throw logic_error("Phidash2_impl was not implemented"); };
        virtual void Phidash1dash1_impl(Array& data)  { throw logic_error("Phidash1Phidash1_impl was not implemented"); };
        virtual void Phidash1dash2_impl(Array& data)  { throw logic_error("Phidash1Phidash2_impl was not implemented"); };
        virtual void Phidash2dash2_impl(Array& data)  { throw logic_error("Phidash2Phidash2_impl was not implemented"); };

        virtual void dPhi_by_dcoeff_impl(Array& data) { throw logic_error("dPhi_by_dcoeff_impl was not implemented"); };
        virtual void dPhidash1_by_dcoeff_impl(Array& data) { throw logic_error("dPhidash1_by_dcoeff_impl was not implemented"); };
        virtual void dPhidash2_by_dcoeff_impl(Array& data) { throw logic_error("dPhidash2_by_dcoeff_impl was not implemented"); };
        virtual void dPhidash2dash2_by_dcoeff_impl(Array& data) { throw logic_error("dPhidash2dash2_by_dcoeff_impl was not implemented"); };
        virtual void dPhidash1dash2_by_dcoeff_impl(Array& data) { throw logic_error("dPhidash1dash2_by_dcoeff_impl was not implemented"); };
        virtual void dPhidash1dash1_by_dcoeff_impl(Array& data) { throw logic_error("dPhidash1dash1_by_dcoeff_impl was not implemented"); };

        Array& Phi() {
            return check_the_cache("Phi", {numquadpoints_phi, numquadpoints_theta,3}, [this](Array& A) { return Phi_impl(A, this->quadpoints_phi, this->quadpoints_theta);});
        }
        Array& Phidash1() {
            return check_the_cache("Phidash1", {numquadpoints_phi, numquadpoints_theta,3}, [this](Array& A) { return Phidash1_impl(A);});
        }
        Array& Phidash2() {
            return check_the_cache("Phidash2", {numquadpoints_phi, numquadpoints_theta,3}, [this](Array& A) { return Phidash2_impl(A);});
        }
        Array& Phidash1dash1() {
            return check_the_cache("Phidash1dash1", {numquadpoints_phi, numquadpoints_theta,3}, [this](Array& A) { return Phidash1dash1_impl(A);});
        }
        Array& Phidash1dash2() {
            return check_the_cache("Phidash1dash2", {numquadpoints_phi, numquadpoints_theta,3}, [this](Array& A) { return Phidash1dash2_impl(A);});
        }
        Array& Phidash2dash2() {
            return check_the_cache("Phidash2dash2", {numquadpoints_phi, numquadpoints_theta,3}, [this](Array& A) { return Phidash2dash2_impl(A);});
        }
        Array& dPhidash1dash1_by_dcoeff() {
            return check_the_cache("dPhidash1dash1_by_dcoeff", {numquadpoints_phi, numquadpoints_theta,3,num_dofs()}, [this](Array& A) { return dPhidash1dash1_by_dcoeff_impl(A);});
        }
        Array& dPhidash1dash2_by_dcoeff() {
            return check_the_cache("dPhidash1dash2_by_dcoeff", {numquadpoints_phi, numquadpoints_theta,3,num_dofs()}, [this](Array& A) { return dPhidash1dash2_by_dcoeff_impl(A);});
        }
        Array& dPhidash2dash2_by_dcoeff() {
            return check_the_cache("dPhidash2dash2_by_dcoeff", {numquadpoints_phi, numquadpoints_theta,3,num_dofs()}, [this](Array& A) { return dPhidash2dash2_by_dcoeff_impl(A);});
        }
        Array& dPhi_by_dcoeff() {
            return check_the_persistent_cache("dPhi_by_dcoeff", {numquadpoints_phi, numquadpoints_theta,3,num_dofs()}, [this](Array& A) { return dPhi_by_dcoeff_impl(A);});
        }
        Array& dPhidash1_by_dcoeff() {
            return check_the_persistent_cache("dPhidash1_by_dcoeff", {numquadpoints_phi, numquadpoints_theta,3,num_dofs()}, [this](Array& A) { return dPhidash1_by_dcoeff_impl(A);});
        }
        Array& dPhidash2_by_dcoeff() {
            return check_the_persistent_cache("dPhidash2_by_dcoeff", {numquadpoints_phi, numquadpoints_theta,3,num_dofs()}, [this](Array& A) { return dPhidash2_by_dcoeff_impl(A);});
        }

        virtual ~CurrentPotential() = default;
};
