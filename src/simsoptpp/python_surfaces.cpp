#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;
using std::shared_ptr;
using std::vector;

namespace py = pybind11;
#include "pycurve.h"
#include "surface.h"
#include "pysurface.h"
#include "surfacerzfourier.h"
typedef SurfaceRZFourier<PyArray> PySurfaceRZFourier;
#include "surfacexyzfourier.h"
typedef SurfaceXYZFourier<PyArray> PySurfaceXYZFourier;
#include "surfacexyztensorfourier.h"
typedef SurfaceXYZTensorFourier<PyArray> PySurfaceXYZTensorFourier;

template <class PySurfaceRZFourierBase = PySurfaceRZFourier> class PySurfaceRZFourierTrampoline : public PySurfaceTrampoline<PySurfaceRZFourierBase> {
    public:
        using PySurfaceTrampoline<PySurfaceRZFourierBase>::PySurfaceTrampoline;
        using PySurfaceRZFourierBase::mpol;
        using PySurfaceRZFourierBase::ntor;
        using PySurfaceRZFourierBase::nfp;
        using PySurfaceRZFourierBase::stellsym;

        int num_dofs() override {
            return PySurfaceRZFourierBase::num_dofs();
        }

        void set_dofs_impl(const vector<double>& _dofs) override {
            PySurfaceRZFourierBase::set_dofs_impl(_dofs);
        }

        vector<double> get_dofs() override {
            return PySurfaceRZFourierBase::get_dofs();
        }

        void gamma_impl(PyArray& data, PyArray& quadpoints_phi, PyArray& quadpoints_theta) override {
            PySurfaceRZFourierBase::gamma_impl(data, quadpoints_phi, quadpoints_theta);
        }

        void gamma_lin(PyArray& data, PyArray& quadpoints_phi, PyArray& quadpoints_theta) override {
            PySurfaceRZFourierBase::gamma_lin(data, quadpoints_phi, quadpoints_theta);
        }


        void fit_to_curve(PyCurve& curve, double radius) {
            PySurfaceRZFourierBase::fit_to_curve(curve, radius);
        }
};

template <class PySurfaceXYZFourierBase = PySurfaceXYZFourier> class PySurfaceXYZFourierTrampoline : public PySurfaceTrampoline<PySurfaceXYZFourierBase> {
    public:
        using PySurfaceTrampoline<PySurfaceXYZFourierBase>::PySurfaceTrampoline;

        int num_dofs() override {
            return PySurfaceXYZFourierBase::num_dofs();
        }

        void set_dofs_impl(const vector<double>& _dofs) override {
            PySurfaceXYZFourierBase::set_dofs_impl(_dofs);
        }

        vector<double> get_dofs() override {
            return PySurfaceXYZFourierBase::get_dofs();
        }

        void gamma_impl(PyArray& data, PyArray& quadpoints_phi, PyArray& quadpoints_theta) override {
            PySurfaceXYZFourierBase::gamma_impl(data, quadpoints_phi, quadpoints_theta);
        }

        void gamma_lin(PyArray& data, PyArray& quadpoints_phi, PyArray& quadpoints_theta) override {
            PySurfaceXYZFourierBase::gamma_lin(data, quadpoints_phi, quadpoints_theta);
        }

        void fit_to_curve(PyCurve& curve, double radius) {
            PySurfaceXYZFourierBase::fit_to_curve(curve, radius);
        }
};

template <class PySurfaceXYZTensorFourierBase = PySurfaceXYZTensorFourier> class PySurfaceXYZTensorFourierTrampoline : public PySurfaceTrampoline<PySurfaceXYZTensorFourierBase> {
    public:
        using PySurfaceTrampoline<PySurfaceXYZTensorFourierBase>::PySurfaceTrampoline;

        int num_dofs() override {
            return PySurfaceXYZTensorFourierBase::num_dofs();
        }

        void set_dofs_impl(const vector<double>& _dofs) override {
            PySurfaceXYZTensorFourierBase::set_dofs_impl(_dofs);
        }

        vector<double> get_dofs() override {
            return PySurfaceXYZTensorFourierBase::get_dofs();
        }

        void gamma_impl(PyArray& data, PyArray& quadpoints_phi, PyArray& quadpoints_theta) override {
            PySurfaceXYZTensorFourierBase::gamma_impl(data, quadpoints_phi, quadpoints_theta);
        }

        void gamma_lin(PyArray& data, PyArray& quadpoints_phi, PyArray& quadpoints_theta) override {
            PySurfaceXYZTensorFourierBase::gamma_lin(data, quadpoints_phi, quadpoints_theta);
        }


        void fit_to_curve(PyCurve& curve, double radius) {
            PySurfaceXYZTensorFourierBase::fit_to_curve(curve, radius);
        }
};

template <typename T, typename S> void register_common_surface_methods(S &s) {
    s.def("gamma", &T::gamma)
     .def("gamma_impl", &T::gamma_impl)
     .def("gamma_lin", &T::gamma_lin)
     .def("dgamma_by_dcoeff", &T::dgamma_by_dcoeff)
     .def("dgamma_by_dcoeff_vjp", &T::dgamma_by_dcoeff_vjp)
     .def("gammadash1", &T::gammadash1)
     .def("dgammadash1_by_dcoeff", &T::dgammadash1_by_dcoeff)
     .def("dgammadash1_by_dcoeff_vjp", &T::dgammadash1_by_dcoeff_vjp)
     .def("gammadash2", &T::gammadash2)
     .def("dgammadash2_by_dcoeff", &T::dgammadash2_by_dcoeff)
     .def("gammadash1dash1", &T::gammadash1dash1, "Returns a `(n_phi, n_theta, 3)` array containing partial^2_{phi,phi} Gamma(phi_i, theta_j) for i in {1, ..., n_phi}, j in{1, ..., n_theta}")
     .def("gammadash1dash2", &T::gammadash1dash2, "Returns a `(n_phi, n_theta, 3)` array containing partial^2_{phi,theta} Gamma(phi_i, theta_j) for i in {1, ..., n_phi}, j in{1, ..., n_theta}")
     .def("gammadash2dash2", &T::gammadash2dash2, "Returns a `(n_phi, n_theta, 3)` array containing partial^2_{theta,theta} Gamma(phi_i, theta_j) for i in {1, ..., n_phi}, j in{1, ..., n_theta}")
     .def("dgammadash1dash1_by_dcoeff", &T::dgammadash1dash1_by_dcoeff, "Returns a `(n_phi, n_theta, 3)` array containing derivatives of `gammadash1dash1` wrt surface coefficients.")
     .def("dgammadash1dash2_by_dcoeff", &T::dgammadash1dash2_by_dcoeff, "Returns a `(n_phi, n_theta, 3)` array containing derivatives of `gammadash1dash2` wrt surface coefficients.")
     .def("dgammadash2dash2_by_dcoeff", &T::dgammadash2dash2_by_dcoeff, "Returns a `(n_phi, n_theta, 3)` array containing derivatives of `gammadash2dash2` wrt surface coefficients.")
     .def("surface_curvatures", &T::surface_curvatures, "Returns a `(n_phi, n_theta, 4)` array containing [G(phi_i, theta_j),K(phi_i, theta_j),kappa_1(phi_i, theta_j),kappa_2(phi_i, theta_j)] for i in {1, ..., n_phi}, j in {1, ..., n_theta} where H is the mean curvature, K is the Gaussian curvature, and kappa_{1,2} are the principal curvatures with kappa_1>kappa_2.")
     .def("dsurface_curvatures_by_dcoeff", &T::dsurface_curvatures_by_dcoeff, "Returns a `(n_phi, n_theta, 4, ndofs)` array containing the derivatives of `surface_curvatures` wrt the surface coefficients.")
     .def("first_fund_form", &T::first_fund_form, "Returns a `(n_phi, n_theta, 3)` array containing [partial_{phi} Gamma(phi_i, theta_j) cdot partial_{phi} Gamma(phi_i, theta_j), partial_{phi} Gamma(phi_i, theta_j) cdot partial_{theta} Gamma(phi_i, theta_j), partial_{theta} Gamma(phi_i, theta_j) cdot partial_{theta} Gamma(phi_i, theta_j)] for i in {1, ..., n_phi}, j in {1, ..., n_theta}.")
     .def("dfirst_fund_form_by_dcoeff", &T::dfirst_fund_form_by_dcoeff, "Returns a `(n_phi, n_theta, 3, ndofs)` array containing the derivatives of `first_fund_form` wrt the surface coefficients.")
     .def("second_fund_form", &T::second_fund_form, "Returns a `(n_phi, n_theta, 3)` array containing [n(phi_i, theta_j) cdot partial^2_{phi,phi} Gamma(phi_i, theta_j), n(phi_i, theta_j) cdot partial^2_{phi,theta} Gamma(phi_i, theta_j), n(phi_i, theta_j) cdot partial^2_{theta,theta} Gamma(phi_i, theta_j)] for i in {1, ..., n_phi}, j in {1, ..., n_theta} where n is the unit normal.")
     .def("dsecond_fund_form_by_dcoeff", &T::dsecond_fund_form_by_dcoeff, "Returns a `(n_phi, n_theta, 3, ndofs)` array containing the derivatives of `second_fund_form` wrt the surface coefficients.")
     .def("dgammadash2_by_dcoeff_vjp", &T::dgammadash2_by_dcoeff_vjp)
     .def("normal", &T::normal)
     .def("dnormal_by_dcoeff", &T::dnormal_by_dcoeff)
     .def("dnormal_by_dcoeff_vjp", &T::dnormal_by_dcoeff_vjp)
     .def("d2normal_by_dcoeffdcoeff", &T::d2normal_by_dcoeffdcoeff)
     .def("unitnormal", &T::unitnormal)
     .def("dunitnormal_by_dcoeff", &T::dunitnormal_by_dcoeff)
     .def("area", &T::area)
     .def("darea_by_dcoeff", &T::darea_by_dcoeff)
     .def("darea", &T::darea_by_dcoeff) // shorthand
     .def("d2area_by_dcoeffdcoeff", &T::d2area_by_dcoeffdcoeff)
     .def("volume", &T::volume)
     .def("dvolume_by_dcoeff", &T::dvolume_by_dcoeff)
     .def("dvolume", &T::dvolume_by_dcoeff) // shorthand
     .def("d2volume_by_dcoeffdcoeff", &T::d2volume_by_dcoeffdcoeff)
     .def("fit_to_curve", &T::fit_to_curve, py::arg("curve"), py::arg("radius"), py::arg("flip_theta") = false)
     .def("scale", &T::scale)
     .def("extend_via_normal", &T::extend_via_normal)
     .def("least_squares_fit", &T::least_squares_fit)
     .def("invalidate_cache", &T::invalidate_cache)
     .def("set_dofs", &T::set_dofs)
     .def("set_dofs_impl", &T::set_dofs_impl)
     .def("get_dofs", &T::get_dofs)
     .def_readonly("quadpoints_phi", &T::quadpoints_phi)
     .def_readonly("quadpoints_theta", &T::quadpoints_theta);
}

void init_surfaces(py::module_ &m){
    auto pysurface = py::class_<PySurface, shared_ptr<PySurface>, PySurfaceTrampoline<PySurface>>(m, "Surface")
        .def(py::init<vector<double>,vector<double>>());
    register_common_surface_methods<PySurface>(pysurface);

    auto pysurfacerzfourier = py::class_<PySurfaceRZFourier, shared_ptr<PySurfaceRZFourier>, PySurfaceRZFourierTrampoline<PySurfaceRZFourier>>(m, "SurfaceRZFourier")
        .def(py::init<int, int, int, bool, vector<double>, vector<double>>())
        .def_readwrite("rc", &PySurfaceRZFourier::rc)
        .def_readwrite("rs", &PySurfaceRZFourier::rs)
        .def_readwrite("zc", &PySurfaceRZFourier::zc)
        .def_readwrite("zs", &PySurfaceRZFourier::zs)
        .def_readwrite("mpol", &PySurfaceRZFourier::mpol)
        .def_readwrite("ntor", &PySurfaceRZFourier::ntor)
        .def_readwrite("nfp", &PySurfaceRZFourier::nfp)
        .def_readwrite("stellsym", &PySurfaceRZFourier::stellsym)
        .def("allocate", &PySurfaceRZFourier::allocate);
    register_common_surface_methods<PySurfaceRZFourier>(pysurfacerzfourier);

    auto pysurfacexyzfourier = py::class_<PySurfaceXYZFourier, shared_ptr<PySurfaceXYZFourier>, PySurfaceXYZFourierTrampoline<PySurfaceXYZFourier>>(m, "SurfaceXYZFourier")
        .def(py::init<int, int, int, bool, vector<double>, vector<double>>())
        .def_readwrite("xc", &PySurfaceXYZFourier::xc)
        .def_readwrite("xs", &PySurfaceXYZFourier::xs)
        .def_readwrite("yc", &PySurfaceXYZFourier::yc)
        .def_readwrite("ys", &PySurfaceXYZFourier::ys)
        .def_readwrite("zc", &PySurfaceXYZFourier::zc)
        .def_readwrite("zs", &PySurfaceXYZFourier::zs)
        .def_readwrite("mpol",&PySurfaceXYZFourier::mpol)
        .def_readwrite("ntor",&PySurfaceXYZFourier::ntor)
        .def_readwrite("nfp", &PySurfaceXYZFourier::nfp)
        .def_readwrite("stellsym", &PySurfaceXYZFourier::stellsym);
    register_common_surface_methods<PySurfaceXYZFourier>(pysurfacexyzfourier);

    auto pysurfacexyztensorfourier = py::class_<PySurfaceXYZTensorFourier, shared_ptr<PySurfaceXYZTensorFourier>, PySurfaceXYZTensorFourierTrampoline<PySurfaceXYZTensorFourier>>(m, "SurfaceXYZTensorFourier")
        .def(py::init<int, int, int, bool, vector<bool>, vector<double>, vector<double>>())
        .def_readwrite("xcs", &PySurfaceXYZTensorFourier::x)
        .def_readwrite("ycs", &PySurfaceXYZTensorFourier::y)
        .def_readwrite("zcs", &PySurfaceXYZTensorFourier::z)
        .def_readwrite("nfp", &PySurfaceXYZTensorFourier::nfp)
        .def_readwrite("ntor", &PySurfaceXYZTensorFourier::ntor)
        .def_readwrite("mpol", &PySurfaceXYZTensorFourier::mpol)
        .def_readwrite("nfp", &PySurfaceXYZTensorFourier::nfp)
        .def_readwrite("stellsym", &PySurfaceXYZTensorFourier::stellsym)
        .def_readwrite("clamped_dims", &PySurfaceXYZTensorFourier::clamped_dims);
    register_common_surface_methods<PySurfaceXYZTensorFourier>(pysurfacexyztensorfourier);
}
