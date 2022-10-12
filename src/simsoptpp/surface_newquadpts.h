#pragma once

#include "surface.h"

template<class Array>
class SurfaceNewQuadPoints : public Surface<Array> {
    /*
       SurfaceNewQuadPoints is a surface wrapper class that uses the supplied
       set of quadrature points when compared to that of parent surface object

       */

    public:
        using Surface<Array>::quadpoints_phi;
        using Surface<Array>::quadpoints_theta;
        using Surface<Array>::numquadpoints_phi;
        using Surface<Array>::numquadpoints_theta;

        Surface<Array>& parent_surface;

        SurfaceNewQuadPoints(Surface<Array>& _surface, vector<double> _quadpoints_phi, vector<double> _quadpoints_theta)
            : Surface<Array>(_quadpoints_phi, _quadpoints_theta), parent_surface(_surface) {
            }

        int num_dofs() override {
            return 0;
        }

        void set_dofs_impl(const vector<double>& dofs) override {
            if (dofs.size()){
                throw logic_error("SurfaceNewQuadPoints has zero DOFs.\nYou should not call set_dofs_impl for SurfaceNewQuadPoints class with finite sized DOFs vector");
            }
        }

        vector<double> get_dofs() override {
            return {}
        }

        void gamma_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override;
        void gamma_lin(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override;
        void gammadash1_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override;
        void gammadash2_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override;
        void gammadash1dash1_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override;
        void gammadash2dash2_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override;
        void gammadash1dash2_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override;

        void dgamma_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override;
        void dgammadash1_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override;
        void dgammadash2_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override;
        void dgammadash1dash1_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override;
        void dgammadash1dash2_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override;
        void dgammadash2dash2_by_dcoeff_impl(Array& data, Array& quadpoints_phi, Array& quadpoints_theta) override;
};
