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
#include "curve.h"
#include <Eigen/Dense>

template<class Array>
Array surface_vjp_contraction(const Array& mat, const Array& v);

template<class Array>
class Surface {
    private:
        /* The cache object contains data that has to be recomputed everytime
         * the dofs change.  However, some data does not have to be recomputed,
         * e.g. dgamma_by_dcoeff, since we assume a representation that is
         * linear in the dofs.  For that data we use the cache_persistent
         * object */
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

        std::unique_ptr<Eigen::FullPivHouseholderQR<Eigen::MatrixXd>> qr; //QR factorisation of dgamma_by_dcoeff, for least squares fitting.

    // We'd really like these to be protected, but I'm not sure that plays well
    // with accessing them from python child classes.
    public://protected:
        int numquadpoints_phi;
        int numquadpoints_theta;
        Array quadpoints_phi;
        Array quadpoints_theta;

    public:

        Surface(vector<double> _quadpoints_phi, vector<double> _quadpoints_theta) {
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

        void least_squares_fit(Array& target_values);
        void fit_to_curve(Curve<Array>& curve, double radius, bool flip_theta);
        void scale(double scale);
        void extend_via_normal(double scale);

        void invalidate_cache() {

            for (auto it = cache.begin(); it != cache.end(); ++it) {
                (it->second).status = false;
            }
        }

        virtual void set_dofs(const vector<double>& _dofs) {
            this->set_dofs_impl(_dofs);
            this->invalidate_cache();
        }

        virtual int num_dofs() = 0;
        virtual void set_dofs_impl(const vector<double>& _dofs) = 0;
        virtual vector<double> get_dofs() = 0;

        virtual void gamma_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) = 0;
        virtual void gamma_lin(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) = 0;
        virtual void gammadash1_impl(Array& data)  { throw logic_error("gammadash1_impl was not implemented"); };
        virtual void gammadash2_impl(Array& data)  { throw logic_error("gammadash2_impl was not implemented"); };
        virtual void gammadash1dash1_impl(Array& data)  { throw logic_error("gammadash1gammadash1_impl was not implemented"); };
        virtual void gammadash1dash2_impl(Array& data)  { throw logic_error("gammadash1gammadash2_impl was not implemented"); };
        virtual void gammadash2dash2_impl(Array& data)  { throw logic_error("gammadash2gammadash2_impl was not implemented"); };

        virtual void dgamma_by_dcoeff_impl(Array& data) { throw logic_error("dgamma_by_dcoeff_impl was not implemented"); };
        virtual void dgammadash1_by_dcoeff_impl(Array& data) { throw logic_error("dgammadash1_by_dcoeff_impl was not implemented"); };
        virtual void dgammadash2_by_dcoeff_impl(Array& data) { throw logic_error("dgammadash2_by_dcoeff_impl was not implemented"); };
        virtual void dgammadash2dash2_by_dcoeff_impl(Array& data) { throw logic_error("dgammadash2dash2_by_dcoeff_impl was not implemented"); };
        virtual void dgammadash1dash2_by_dcoeff_impl(Array& data) { throw logic_error("dgammadash1dash2_by_dcoeff_impl was not implemented"); };
        virtual void dgammadash1dash1_by_dcoeff_impl(Array& data) { throw logic_error("dgammadash1dash1_by_dcoeff_impl was not implemented"); };

        virtual Array dgamma_by_dcoeff_vjp(Array& v) {
            return surface_vjp_contraction<Array>(dgamma_by_dcoeff(), v);
        };

        virtual Array dgammadash1_by_dcoeff_vjp(Array& v) {
            return surface_vjp_contraction<Array>(dgammadash1_by_dcoeff(), v);
        };

        virtual Array dgammadash2_by_dcoeff_vjp(Array& v) {
            return surface_vjp_contraction<Array>(dgammadash2_by_dcoeff(), v);
        };

        void surface_curvatures_impl(Array& data);
        void dsurface_curvatures_by_dcoeff_impl(Array& data);
        void first_fund_form_impl(Array& data);
        void dfirst_fund_form_by_dcoeff_impl(Array& data);
        void second_fund_form_impl(Array& data);
        void dsecond_fund_form_by_dcoeff_impl(Array& data);
        void normal_impl(Array& data);
        void dnormal_by_dcoeff_impl(Array& data);
        void d2normal_by_dcoeffdcoeff_impl(Array& data);
        Array dnormal_by_dcoeff_vjp(Array& v);

        void unitnormal_impl(Array& data);
        void dunitnormal_by_dcoeff_impl(Array& data);

        double area();
        void darea_by_dcoeff_impl(Array& data);
        void d2area_by_dcoeffdcoeff_impl(Array& data);

        double volume();
        void dvolume_by_dcoeff_impl(Array& data);
        void d2volume_by_dcoeffdcoeff_impl(Array& data);

        Array& gamma() {
            return check_the_cache("gamma", {numquadpoints_phi, numquadpoints_theta,3}, [this](Array& A) { return gamma_impl(A, this->quadpoints_phi, this->quadpoints_theta);});
        }
        Array& gammadash1() {
            return check_the_cache("gammadash1", {numquadpoints_phi, numquadpoints_theta,3}, [this](Array& A) { return gammadash1_impl(A);});
        }
        Array& gammadash2() {
            return check_the_cache("gammadash2", {numquadpoints_phi, numquadpoints_theta,3}, [this](Array& A) { return gammadash2_impl(A);});
        }
        Array& gammadash1dash1() {
            return check_the_cache("gammadash1dash1", {numquadpoints_phi, numquadpoints_theta,3}, [this](Array& A) { return gammadash1dash1_impl(A);});
        }
        Array& gammadash1dash2() {
            return check_the_cache("gammadash1dash2", {numquadpoints_phi, numquadpoints_theta,3}, [this](Array& A) { return gammadash1dash2_impl(A);});
        }
        Array& gammadash2dash2() {
            return check_the_cache("gammadash2dash2", {numquadpoints_phi, numquadpoints_theta,3}, [this](Array& A) { return gammadash2dash2_impl(A);});
        }
        Array& dgammadash1dash1_by_dcoeff() {
            return check_the_cache("dgammadash1dash1_by_dcoeff", {numquadpoints_phi, numquadpoints_theta,3,num_dofs()}, [this](Array& A) { return dgammadash1dash1_by_dcoeff_impl(A);});
        }
        Array& dgammadash1dash2_by_dcoeff() {
            return check_the_cache("dgammadash1dash2_by_dcoeff", {numquadpoints_phi, numquadpoints_theta,3,num_dofs()}, [this](Array& A) { return dgammadash1dash2_by_dcoeff_impl(A);});
        }
        Array& dgammadash2dash2_by_dcoeff() {
            return check_the_cache("dgammadash2dash2_by_dcoeff", {numquadpoints_phi, numquadpoints_theta,3,num_dofs()}, [this](Array& A) { return dgammadash2dash2_by_dcoeff_impl(A);});
        }
        Array& surface_curvatures() {
            return check_the_cache("surface_curvatures", {numquadpoints_phi, numquadpoints_theta,4}, [this](Array& A) { return surface_curvatures_impl(A);});
        }
        Array& dsurface_curvatures_by_dcoeff() {
            return check_the_cache("dsurface_curvatures_by_dcoeff", {numquadpoints_phi, numquadpoints_theta,4,num_dofs()}, [this](Array& A) { return dsurface_curvatures_by_dcoeff_impl(A);});
        }
        Array& first_fund_form() {
            return check_the_cache("first_fund_form", {numquadpoints_phi, numquadpoints_theta,3}, [this](Array& A) { return first_fund_form_impl(A);});
        }
        Array& dfirst_fund_form_by_dcoeff() {
            return check_the_cache("dfirst_fund_form_by_dcoeff", {numquadpoints_phi, numquadpoints_theta,3,num_dofs()}, [this](Array& A) { return dfirst_fund_form_by_dcoeff_impl(A);});
        }
        Array& second_fund_form() {
            return check_the_cache("second_fund_form", {numquadpoints_phi, numquadpoints_theta,3}, [this](Array& A) { return second_fund_form_impl(A);});
        }
        Array& dsecond_fund_form_by_dcoeff() {
            return check_the_cache("dsecond_fund_form_by_dcoeff", {numquadpoints_phi, numquadpoints_theta,3,num_dofs()}, [this](Array& A) { return dsecond_fund_form_by_dcoeff_impl(A);});
        }
        Array& dgamma_by_dcoeff() {
            return check_the_persistent_cache("dgamma_by_dcoeff", {numquadpoints_phi, numquadpoints_theta,3,num_dofs()}, [this](Array& A) { return dgamma_by_dcoeff_impl(A);});
        }
        Array& dgammadash1_by_dcoeff() {
            return check_the_persistent_cache("dgammadash1_by_dcoeff", {numquadpoints_phi, numquadpoints_theta,3,num_dofs()}, [this](Array& A) { return dgammadash1_by_dcoeff_impl(A);});
        }
        Array& dgammadash2_by_dcoeff() {
            return check_the_persistent_cache("dgammadash2_by_dcoeff", {numquadpoints_phi, numquadpoints_theta,3,num_dofs()}, [this](Array& A) { return dgammadash2_by_dcoeff_impl(A);});
        }
        Array& normal() {
            return check_the_cache("normal", {numquadpoints_phi, numquadpoints_theta,3}, [this](Array& A) { return normal_impl(A);});
        }
        Array& dnormal_by_dcoeff() {
            return check_the_cache("dnormal_by_dcoeff", {numquadpoints_phi, numquadpoints_theta,3, num_dofs()}, [this](Array& A) { return dnormal_by_dcoeff_impl(A);});
        }
        Array& d2normal_by_dcoeffdcoeff() {
            return check_the_cache("d2normal_by_dcoeffdcoeff", {numquadpoints_phi, numquadpoints_theta,3, num_dofs(), num_dofs() }, [this](Array& A) { return d2normal_by_dcoeffdcoeff_impl(A);});
        }
        Array& unitnormal() {
            return check_the_cache("unitnormal", {numquadpoints_phi, numquadpoints_theta,3}, [this](Array& A) { return unitnormal_impl(A);});
        }
        Array& dunitnormal_by_dcoeff() {
            return check_the_cache("dunitnormal_by_dcoeff", {numquadpoints_phi, numquadpoints_theta, 3, num_dofs()}, [this](Array& A) { return dunitnormal_by_dcoeff_impl(A);});
        }
        Array& darea_by_dcoeff() {
            return check_the_cache("darea_by_dcoeff", {num_dofs()}, [this](Array& A) { return darea_by_dcoeff_impl(A);});
        }
        Array& d2area_by_dcoeffdcoeff() {
            return check_the_cache("d2area_by_dcoeffdcoeff", {num_dofs(), num_dofs()}, [this](Array& A) { return d2area_by_dcoeffdcoeff_impl(A);});
        }
        Array& dvolume_by_dcoeff() {
            return check_the_cache("dvolume_by_dcoeff", {num_dofs()}, [this](Array& A) { return dvolume_by_dcoeff_impl(A);});
        }
        Array& d2volume_by_dcoeffdcoeff() {
            return check_the_cache("d2volume_by_dcoeffdcoeff", {num_dofs(), num_dofs()}, [this](Array& A) { return d2volume_by_dcoeffdcoeff_impl(A);});
        }


        virtual ~Surface() = default;
};
