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

template<class Array>
struct CachedArray {
    Array data;
    bool status;
    CachedArray(Array _data) : data(_data), status(false) {}
};

template<class Array>
Array curve_vjp_contraction(const Array& mat, const Array& v){
    int numquadpoints = mat.shape(0);
    int numdofs = mat.shape(2);
    Array res = xt::zeros<double>({numdofs});
    for (int i = 0; i < numdofs; ++i) {
        for (int j = 0; j < numquadpoints; ++j) {
            for (int k = 0; k < 3; ++k) {
                res(i) += mat(j, k, i) * v(j, k);
            }
        }
    }
    return res;
}

template<class Array>
class Curve {
    private:
        map<string, CachedArray<Array>> cache;

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

    // We'd really like these to be protected, but I'm not sure that plays well
    // with accessing them from python child classes. 
    public://protected:
        int numquadpoints;
        vector<double> quadpoints;

    public:

        Curve(vector<double> _quadpoints) : quadpoints(_quadpoints) {
            numquadpoints = quadpoints.size();
        }

        void invalidate_cache() {
            for (auto it = cache.begin(); it != cache.end(); ++it) {
                (it->second).status = false;
            }
        }

        void set_dofs(const vector<double>& _dofs) {
            this->set_dofs_impl(_dofs);
            this->invalidate_cache();
        }

        virtual int num_dofs() = 0;
        virtual void set_dofs_impl(const vector<double>& _dofs) = 0;
        virtual vector<double> get_dofs() = 0;

        virtual void gamma_impl(Array& data) = 0;
        virtual void gammadash_impl(Array& data) { throw logic_error("gammadash_impl was not implemented"); };
        virtual void gammadashdash_impl(Array& data) { throw logic_error("gammadashdash_impl was not implemented"); };
        virtual void gammadashdashdash_impl(Array& data) { throw logic_error("gammadashdashdash_impl was not implemented"); };

        virtual void dgamma_by_dcoeff_impl(Array& data) { throw logic_error("dgamma_by_dcoeff_impl was not implemented"); };
        virtual void dgammadash_by_dcoeff_impl(Array& data) { throw logic_error("dgammadash_by_dcoeff_impl was not implemented"); };
        virtual void dgammadashdash_by_dcoeff_impl(Array& data) { throw logic_error("dgammadashdash_by_dcoeff_impl was not implemented"); };
        virtual void dgammadashdashdash_by_dcoeff_impl(Array& data) { throw logic_error("dgammadashdashdash_by_dcoeff_impl was not implemented"); };

        virtual void kappa_impl(Array& data) { throw logic_error("kappa_impl was not implemented"); };
        virtual void dkappa_by_dcoeff_impl(Array& data) { throw logic_error("dkappa_by_dcoeff_impl was not implemented"); };

        virtual void torsion_impl(Array& data) { throw logic_error("torsion_impl was not implemented"); };
        virtual void dtorsion_by_dcoeff_impl(Array& data) { throw logic_error("dtorsion_by_dcoeff_impl was not implemented"); };

        void incremental_arclength_impl(Array& data) { 
            auto dg = this->gammadash();
            for (int i = 0; i < numquadpoints; ++i) {
                data(i) = std::sqrt(dg(i, 0)*dg(i, 0)+dg(i, 1)*dg(i, 1)+dg(i, 2)*dg(i, 2));
            }
        };
        void dincremental_arclength_by_dcoeff_impl(Array& data) {
            auto dg = this->gammadash();
            auto dgdc = this->dgammadash_by_dcoeff();
            auto l = this->incremental_arclength();
            for (int i = 0; i < numquadpoints; ++i) {
                for (int c = 0; c < num_dofs(); ++c) {
                    data(i, c) = (1./l(i)) * (dg(i, 0) * dgdc(i, 0, c) + dg(i, 1) * dgdc(i, 1, c) + dg(i, 2) * dgdc(i, 2, c));
                }
            }
        };

        Array& gamma() {
            return check_the_cache("gamma", {numquadpoints, 3}, [this](Array& A) { return gamma_impl(A);});
        }
        Array& gammadash() {
            return check_the_cache("gammadash", {numquadpoints, 3}, [this](Array& A) { return gammadash_impl(A);});
        }
        Array& gammadashdash() {
            return check_the_cache("gammadashdash", {numquadpoints, 3}, [this](Array& A) { return gammadashdash_impl(A);});
        }
        Array& gammadashdashdash() {
            return check_the_cache("gammadashdashdash", {numquadpoints, 3}, [this](Array& A) { return gammadashdashdash_impl(A);});
        }

        Array& dgamma_by_dcoeff() {
            return check_the_cache("dgamma_by_dcoeff", {numquadpoints, 3, num_dofs()}, [this](Array& A) { return dgamma_by_dcoeff_impl(A);});
        }
        Array& dgammadash_by_dcoeff() {
            return check_the_cache("dgammadash_by_dcoeff", {numquadpoints, 3, num_dofs()}, [this](Array& A) { return dgammadash_by_dcoeff_impl(A);});
        }
        Array& dgammadashdash_by_dcoeff() {
            return check_the_cache("dgammadashdash_by_dcoeff", {numquadpoints, 3, num_dofs()}, [this](Array& A) { return dgammadashdash_by_dcoeff_impl(A);});
        }
        Array& dgammadashdashdash_by_dcoeff() {
            return check_the_cache("dgammadashdashdash_by_dcoeff", {numquadpoints, 3, num_dofs()}, [this](Array& A) { return dgammadashdashdash_by_dcoeff_impl(A);});
        }

        virtual Array dgamma_by_dcoeff_vjp(Array& v) {
            return curve_vjp_contraction<Array>(dgamma_by_dcoeff(), v);
        };

        virtual Array dgammadash_by_dcoeff_vjp(Array& v) {
            return curve_vjp_contraction<Array>(dgammadash_by_dcoeff(), v);
        };

        virtual Array dgammadashdash_by_dcoeff_vjp(Array& v) {
            return curve_vjp_contraction<Array>(dgammadashdash_by_dcoeff(), v);
        };

        virtual Array dgammadashdashdash_by_dcoeff_vjp(Array& v) {
            return curve_vjp_contraction<Array>(dgammadashdashdash_by_dcoeff(), v);
        };

        Array& kappa() {
            return check_the_cache("kappa", {numquadpoints}, [this](Array& A) { return kappa_impl(A);});
        }

        Array& dkappa_by_dcoeff() {
            return check_the_cache("dkappa_by_dcoeff", {numquadpoints, num_dofs()}, [this](Array& A) { return dkappa_by_dcoeff_impl(A);});
        }

        Array& torsion() {
            return check_the_cache("torsion", {numquadpoints}, [this](Array& A) { return torsion_impl(A);});
        }

        Array& dtorsion_by_dcoeff() {
            return check_the_cache("dtorsion_by_dcoeff", {numquadpoints, num_dofs()}, [this](Array& A) { return dtorsion_by_dcoeff_impl(A);});
        }

        Array& incremental_arclength() {
            return check_the_cache("incremental_arclength", {numquadpoints}, [this](Array& A) { return incremental_arclength_impl(A);});
        }

        Array& dincremental_arclength_by_dcoeff() {
            return check_the_cache("dincremental_arclength_by_dcoeff", {numquadpoints, num_dofs()}, [this](Array& A) { return dincremental_arclength_by_dcoeff_impl(A);});
        }

        virtual ~Curve() = default;
};
