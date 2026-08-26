#pragma once
#include <vector>
using std::vector;

#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> Array;

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
        double net_poloidal_current_amperes;
        double net_toroidal_current_amperes;

    public:

        CurrentPotential(
                vector<double> _quadpoints_phi, vector<double> _quadpoints_theta,
                double _net_poloidal_current_amperes, double _net_toroidal_current_amperes)
            {
            numquadpoints_phi = _quadpoints_phi.size();
            numquadpoints_theta = _quadpoints_theta.size();
            net_poloidal_current_amperes = _net_poloidal_current_amperes;
            net_toroidal_current_amperes = _net_toroidal_current_amperes;

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

        void K_impl_helper(Array& data, Array& dg1, Array& dg2, Array& normal);
        void K_GI_impl_helper(Array& data, Array& dg1, Array& dg2, Array& normal);
        void K_by_dcoeff_impl_helper(Array& data, Array& dg1, Array& dg2, Array& normal);
        void K_matrix_impl_helper(Array& data, Array& dg1, Array& dg2, Array& normal);
        void K_rhs_impl_helper(Array& data, Array& dg1, Array& dg2, Array& normal);

        virtual int num_dofs() { throw logic_error("num_dofs was not implemented"); };
        virtual void set_dofs_impl(const vector<double>& _dofs) { throw logic_error("set_dofs_impl was not implemented"); };
        virtual vector<double> get_dofs() { throw logic_error("get_dofs was not implemented"); };

        virtual void Phi_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) { throw logic_error("Phi_impl was not implemented"); };
        virtual void Phidash1_impl(Array& data)  { throw logic_error("Phidash1_impl was not implemented"); };
        virtual void Phidash2_impl(Array& data)  { throw logic_error("Phidash2_impl was not implemented"); };
        virtual void dPhidash1_by_dcoeff_impl(Array& data)  { throw logic_error("dPhidash1_by_dcoeff_impl was not implemented"); };
        virtual void dPhidash2_by_dcoeff_impl(Array& data)  { throw logic_error("dPhidash2_by_dcoeff_impl was not implemented"); };

        Array& Phi() {
            return check_the_cache("Phi", {numquadpoints_phi, numquadpoints_theta}, [this](Array& A) { return Phi_impl(A, this->quadpoints_phi, this->quadpoints_theta);});
        }
        Array& Phidash1() {
            return check_the_cache("Phidash1", {numquadpoints_phi, numquadpoints_theta}, [this](Array& A) { return Phidash1_impl(A);});
        }
        Array& Phidash2() {
            return check_the_cache("Phidash2", {numquadpoints_phi, numquadpoints_theta}, [this](Array& A) { return Phidash2_impl(A);});
        }
        Array& dPhidash1_by_dcoeff() {
            return check_the_persistent_cache("dPhidash1_by_dcoeff", {numquadpoints_phi, numquadpoints_theta,num_dofs()}, [this](Array& A) { return dPhidash1_by_dcoeff_impl(A);});
        }
        Array& dPhidash2_by_dcoeff() {
            return check_the_persistent_cache("dPhidash2_by_dcoeff", {numquadpoints_phi, numquadpoints_theta,num_dofs()}, [this](Array& A) { return dPhidash2_by_dcoeff_impl(A);});
        }

        virtual ~CurrentPotential() = default;
};
