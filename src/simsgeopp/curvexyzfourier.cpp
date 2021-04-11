#pragma once

#include "curve.cpp"

template<class Array>
class CurveXYZFourier : public Curve<Array> {
    /*
       CurveXYZFourier is a curve that is represented in cartesian
       coordinates using the following Fourier series: 

           x(phi) = \sum_{m=0}^{order} x_{c,m}cos(m*phi) + \sum_{m=1}^order x_{s,m}sin(m*phi)
           y(phi) = \sum_{m=0}^{order} y_{c,m}cos(m*phi) + \sum_{m=1}^order y_{s,m}sin(m*phi)
           z(phi) = \sum_{m=0}^{order} z_{c,m}cos(m*phi) + \sum_{m=1}^order z_{s,m}sin(m*phi)

       The dofs are stored in the order 

           [x_{c,0},...,x_{c,order},x_{s,1},...,x_{s,order},y_{c,0},....]

       */
    private:
        int order;
    public:
        using Curve<Array>::quadpoints;
        using Curve<Array>::numquadpoints;
        vector<vector<double>> dofs;

        CurveXYZFourier(int _numquadpoints, int _order) : Curve<Array>(_numquadpoints), order(_order) {
            dofs = vector<vector<double>> {
                vector<double>(2*order+1, 0.), 
                vector<double>(2*order+1, 0.), 
                vector<double>(2*order+1, 0.)
            };
        }

        CurveXYZFourier(vector<double> _quadpoints, int _order) : Curve<Array>(_quadpoints), order(_order) {
            dofs = vector<vector<double>> {
                vector<double>(2*order+1, 0.), 
                vector<double>(2*order+1, 0.), 
                vector<double>(2*order+1, 0.)
            };
        }

        CurveXYZFourier(Array _quadpoints, int _order) : Curve<Array>(_quadpoints), order(_order) {
            dofs = vector<vector<double>> {
                vector<double>(2*order+1, 0.), 
                vector<double>(2*order+1, 0.), 
                vector<double>(2*order+1, 0.)
            };
        }

        inline int num_dofs() override {
            return 3*(2*order+1);
        }

        void set_dofs_impl(const vector<double>& _dofs) override {
            int counter = 0;
            for (int i = 0; i < 3; ++i) {
                dofs[i][0] = _dofs[counter++];
                for (int j = 1; j < order+1; ++j) {
                    dofs[i][2*j-1] = _dofs[counter++];
                    dofs[i][2*j] = _dofs[counter++];
                }
            }
        }

        vector<double> get_dofs() override {
            auto _dofs = vector<double>(num_dofs(), 0.);
            int counter = 0;
            for (int i = 0; i < 3; ++i) {
                _dofs[counter++] = dofs[i][0];
                for (int j = 1; j < order+1; ++j) {
                    _dofs[counter++] = dofs[i][2*j-1];
                    _dofs[counter++] = dofs[i][2*j];
                }
            }
            return _dofs;
        }

        void gamma_impl(Array& data, Array& quadpoints) override {
            int numquadpoints = quadpoints.size();
            data *= 0;
            for (int k = 0; k < numquadpoints; ++k) {
                for (int i = 0; i < 3; ++i) {
                    data(k, i) += dofs[i][0];
                    for (int j = 1; j < order+1; ++j) {
                        data(k, i) += dofs[i][2*j-1]*sin(2*M_PI*j*quadpoints[k]);
                        data(k, i) += dofs[i][2*j]*cos(2*M_PI*j*quadpoints[k]);
                    }
                }
            }
        }

        void gammadash_impl(Array& data) override {
            data *= 0;
            for (int k = 0; k < numquadpoints; ++k) {
                for (int i = 0; i < 3; ++i) {
                    for (int j = 1; j < order+1; ++j) {
                        data(k, i) += +dofs[i][2*j-1]*2*M_PI*j*cos(2*M_PI*j*quadpoints[k]);
                        data(k, i) += -dofs[i][2*j]*2*M_PI*j*sin(2*M_PI*j*quadpoints[k]);
                    }
                }
            }
        }

        void gammadashdash_impl(Array& data) override {
            data *= 0;
            for (int k = 0; k < numquadpoints; ++k) {
                for (int i = 0; i < 3; ++i) {
                    for (int j = 1; j < order+1; ++j) {
                        data(k, i) += -dofs[i][2*j-1] * (2*M_PI*j)*(2*M_PI*j)*sin(2*M_PI*j*quadpoints[k]);
                        data(k, i) += -dofs[i][2*j]   * (2*M_PI*j)*(2*M_PI*j)*cos(2*M_PI*j*quadpoints[k]);
                    }
                }
            }
        }

        void gammadashdashdash_impl(Array& data) override {
            data *= 0;
            for (int k = 0; k < numquadpoints; ++k) {
                for (int i = 0; i < 3; ++i) {
                    for (int j = 1; j < order+1; ++j) {
                        data(k, i) += -dofs[i][2*j-1] * (2*M_PI*j)*(2*M_PI*j)*(2*M_PI*j)*cos(2*M_PI*j*quadpoints[k]);
                        data(k, i) += +dofs[i][2*j]   * (2*M_PI*j)*(2*M_PI*j)*(2*M_PI*j)*sin(2*M_PI*j*quadpoints[k]);
                    }
                }
            }
        }

        void dgamma_by_dcoeff_impl(Array& data) override {
            for (int k = 0; k < numquadpoints; ++k) {
                for (int i = 0; i < 3; ++i) {
                    data(k, i, i*(2*order+1)) = 1.;
                    for (int j = 1; j < order+1; ++j) {
                        data(k, i, i*(2*order+1) + 2*j-1) = sin(2*M_PI*j*quadpoints[k]);
                        data(k, i, i*(2*order+1) + 2*j  ) = cos(2*M_PI*j*quadpoints[k]);
                    }
                }
            }
        }

        void dgammadash_by_dcoeff_impl(Array& data) override {
            for (int k = 0; k < numquadpoints; ++k) {
                for (int i = 0; i < 3; ++i) {
                    for (int j = 1; j < order+1; ++j) {
                        data(k, i, i*(2*order+1) + 2*j-1) = +2*M_PI*j*cos(2*M_PI*j*quadpoints[k]);
                        data(k, i, i*(2*order+1) + 2*j  ) = -2*M_PI*j*sin(2*M_PI*j*quadpoints[k]);
                    }
                }
            }
        }

        void dgammadashdash_by_dcoeff_impl(Array& data) override {
            for (int k = 0; k < numquadpoints; ++k) {
                for (int i = 0; i < 3; ++i) {
                    for (int j = 1; j < order+1; ++j) {
                        data(k, i, i*(2*order+1) + 2*j-1) = -(2*M_PI*j)*(2*M_PI*j)*sin(2*M_PI*j*quadpoints[k]);
                        data(k, i, i*(2*order+1) + 2*j  ) = -(2*M_PI*j)*(2*M_PI*j)*cos(2*M_PI*j*quadpoints[k]);
                    }
                }
            }
        }

        void dgammadashdashdash_by_dcoeff_impl(Array& data) override {
            for (int k = 0; k < numquadpoints; ++k) {
                for (int i = 0; i < 3; ++i) {
                    for (int j = 1; j < order+1; ++j) {
                        data(k, i, i*(2*order+1) + 2*j-1) = -(2*M_PI*j)*(2*M_PI*j)*(2*M_PI*j)*cos(2*M_PI*j*quadpoints[k]);
                        data(k, i, i*(2*order+1) + 2*j  ) = +(2*M_PI*j)*(2*M_PI*j)*(2*M_PI*j)*sin(2*M_PI*j*quadpoints[k]);
                    }
                }
            }
        }

        Array dgamma_by_dcoeff_vjp(Array& v) override {
            Array res = xt::zeros<double>({num_dofs()});
            for (int k = 0; k < numquadpoints; ++k) {
                for (int i = 0; i < 3; ++i) {
                    res(i*(2*order+1)) += v(k, i);
                    for (int j = 1; j < order+1; ++j) {
                        res(i*(2*order+1) + 2*j-1) += sin(2*M_PI*j*quadpoints[k]) * v(k, i);
                        res(i*(2*order+1) + 2*j) += cos(2*M_PI*j*quadpoints[k]) * v(k, i);
                    }
                }
            }
            return res;
        }

        Array dgammadash_by_dcoeff_vjp(Array& v) override {
            Array res = xt::zeros<double>({num_dofs()});
            for (int k = 0; k < numquadpoints; ++k) {
                for (int i = 0; i < 3; ++i) {
                    for (int j = 1; j < order+1; ++j) {
                        res(i*(2*order+1) + 2*j-1) += +2*M_PI*j*cos(2*M_PI*j*quadpoints[k]) * v(k, i);
                        res(i*(2*order+1) + 2*j) += -2*M_PI*j*sin(2*M_PI*j*quadpoints[k]) * v(k, i);
                    }
                }
            }
            return res;
        }
};
