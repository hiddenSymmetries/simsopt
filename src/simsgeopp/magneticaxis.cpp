#pragma once

#include "curve.cpp"

template<class Array>
class StelleratorSymmetricCylindricalFourierCurve : public Curve<Array> {
    private:
        int order;
        int nfp;
    public:
        using Curve<Array>::quadpoints;
        using Curve<Array>::numquadpoints;
        vector<vector<double>> dofs;

        StelleratorSymmetricCylindricalFourierCurve(int _numquadpoints, int _order, int _nfp) : Curve<Array>(std::vector<double>(_numquadpoints, 0.)), order(_order), nfp(_nfp) {
            for (int i = 0; i < numquadpoints; ++i) {
                this->quadpoints[i] = ((double)i)/(nfp*numquadpoints);
            }
            dofs = vector<vector<double>> {
                vector<double>(order+1, 0.), 
                vector<double>(order, 0.), 
            };
        }

        StelleratorSymmetricCylindricalFourierCurve(vector<double> _quadpoints, int _order, int _nfp) : Curve<Array>(_quadpoints), order(_order), nfp(_nfp) {
            dofs = vector<vector<double>> {
                vector<double>(order+1, 0.), 
                vector<double>(order, 0.), 
            };
        }

        inline int get_nfp() {
            return nfp;
        }

        inline int num_dofs() override {
            return 2*order+1;
        }

        void set_dofs_impl(const vector<double>& _dofs) override {
            for (int i = 0; i < order+1; ++i)
                dofs[0][i] = _dofs[i];
            for (int i = 0; i < order; ++i)
                dofs[1][i] = _dofs[order + 1 + i];
        }

        vector<double> get_dofs() override {
            auto _dofs = vector<double>(num_dofs(), 0.);
            int counter = 0;
            for (int i = 0; i < order+1; ++i)
                _dofs[counter++] = dofs[0][i];
            for (int i = 0; i < order; ++i)
                _dofs[counter++] = dofs[1][i];
            return _dofs;
        }

        void gamma_impl(Array& data) override {
            data *= 0;
            for (int k = 0; k < numquadpoints; ++k) {
                for (int i = 0; i < order+1; ++i) {
                    data(k, 0) += dofs[0][i] * cos(nfp * 2 * M_PI * i * quadpoints[k]) * cos(2 * M_PI * quadpoints[k]);
                    data(k, 1) += dofs[0][i] * cos(nfp * 2 * M_PI * i * quadpoints[k]) * sin(2 * M_PI * quadpoints[k]);
                }
                for (int i = 1; i < order+1; ++i) {
                    data(k, 2) += dofs[1][i-1] * sin(nfp * 2 * M_PI * i * quadpoints[k]);
                }
            }
        }

        void gammadash_impl(Array& data) override {
            data *= 0;
            for (int k = 0; k < numquadpoints; ++k) {
                for (int i = 0; i < order+1; ++i) {
                    data(k, 0) += dofs[0][i] * (
                            -(nfp * 2 * M_PI * i) * sin(nfp * 2 * M_PI * i * quadpoints[k]) * cos(2 * M_PI * quadpoints[k])
                            -(2 * M_PI) *           cos(nfp * 2 * M_PI * i * quadpoints[k]) * sin(2 * M_PI * quadpoints[k])
                            );
                    data(k, 1) += dofs[0][i] * (
                            -(nfp * 2 * M_PI * i) * sin(nfp * 2 * M_PI * i * quadpoints[k]) * sin(2 * M_PI * quadpoints[k])
                            +(2 * M_PI) *           cos(nfp * 2 * M_PI * i * quadpoints[k]) * cos(2 * M_PI * quadpoints[k])
                            );
                }
                for (int i = 1; i < order+1; ++i) {
                    data(k, 2) += dofs[1][i-1] * (nfp * 2 * M_PI * i) * cos(nfp * 2 * M_PI * i * quadpoints[k]);
                }
            }
        }

        void gammadashdash_impl(Array& data) override {
            data *= 0;
            for (int k = 0; k < numquadpoints; ++k) {
                for (int i = 0; i < order+1; ++i) {
                    data(k, 0) += dofs[0][i] * (
                            +2*(nfp * 2 * M_PI * i)*(2 * M_PI) *       sin(nfp * 2 * M_PI * i * quadpoints[k]) * sin(2 * M_PI * quadpoints[k])
                            -(pow(nfp * 2 * M_PI * i, 2) + pow(2*M_PI, 2)) * cos(nfp * 2 * M_PI * i * quadpoints[k]) * cos(2 * M_PI * quadpoints[k])
                            );
                    data(k, 1) += dofs[0][i] * (
                            -2*(nfp * 2 * M_PI * i) * (2 * M_PI) *     sin(nfp * 2 * M_PI * i * quadpoints[k]) * cos(2 * M_PI * quadpoints[k])
                            -(pow(nfp * 2 * M_PI * i, 2) + pow(2*M_PI, 2)) * cos(nfp * 2 * M_PI * i * quadpoints[k]) * sin(2 * M_PI * quadpoints[k])
                            );
                }
                for (int i = 1; i < order+1; ++i) {
                    data(k, 2) -= dofs[1][i-1] * pow(nfp * 2 * M_PI * i, 2) * sin(nfp * 2 * M_PI * i * quadpoints[k]);
                }
            }
        }

        void gammadashdashdash_impl(Array& data) override {
            data *= 0;
            for (int k = 0; k < numquadpoints; ++k) {
                for (int i = 0; i < order+1; ++i) {
                    data(k, 0) += dofs[0][i] * (
                            +2*pow(nfp * 2 * M_PI * i, 2)*(2 * M_PI) *       cos(nfp * 2 * M_PI * i * quadpoints[k]) * sin(2 * M_PI * quadpoints[k])
                            +2*(nfp * 2 * M_PI * i)*pow(2 * M_PI, 2) *       sin(nfp * 2 * M_PI * i * quadpoints[k]) * cos(2 * M_PI * quadpoints[k])
                            +(pow(nfp * 2 * M_PI * i, 2) + pow(2*M_PI, 2))*(nfp * 2 * M_PI * i)* sin(nfp * 2 * M_PI * i * quadpoints[k]) * cos(2 * M_PI * quadpoints[k])
                            +(pow(nfp * 2 * M_PI * i, 2) + pow(2*M_PI, 2))*(2 * M_PI)*           cos(nfp * 2 * M_PI * i * quadpoints[k]) * sin(2 * M_PI * quadpoints[k])
                            );
                    data(k, 1) += dofs[0][i] * (
                            -2*pow(nfp * 2 * M_PI * i, 2) * (2 * M_PI) *     cos(nfp * 2 * M_PI * i * quadpoints[k]) * cos(2 * M_PI * quadpoints[k])
                            +2*(nfp * 2 * M_PI * i) * pow(2 * M_PI, 2) *     sin(nfp * 2 * M_PI * i * quadpoints[k]) * sin(2 * M_PI * quadpoints[k])
                            +(pow(nfp * 2 * M_PI * i, 2) + pow(2*M_PI, 2)) * (nfp * 2 * M_PI * i) * sin(nfp * 2 * M_PI * i * quadpoints[k]) * sin(2 * M_PI * quadpoints[k])
                            -(pow(nfp * 2 * M_PI * i, 2) + pow(2*M_PI, 2)) * (2 * M_PI) *           cos(nfp * 2 * M_PI * i * quadpoints[k]) * cos(2 * M_PI * quadpoints[k])
                            );
                }
                for (int i = 1; i < order+1; ++i) {
                    data(k, 2) -= dofs[1][i-1] * pow(nfp * 2 * M_PI * i, 3) * cos(nfp * 2 * M_PI * i * quadpoints[k]);
                }
            }
        }

        void dgamma_by_dcoeff_impl(Array& data) override {
            for (int k = 0; k < numquadpoints; ++k) {
                for (int i = 0; i < order+1; ++i) {
                    data(k, 0, i) = cos(nfp * 2 * M_PI * i * quadpoints[k]) * cos(2 * M_PI * quadpoints[k]);
                    data(k, 1, i) = cos(nfp * 2 * M_PI * i * quadpoints[k]) * sin(2 * M_PI * quadpoints[k]);
                }
                for (int i = 1; i < order+1; ++i) {
                    data(k, 2, order + i) = sin(nfp * 2 * M_PI * i * quadpoints[k]);
                }
            }
        }

        void dgammadash_by_dcoeff_impl(Array& data) override {
            for (int k = 0; k < numquadpoints; ++k) {
                for (int i = 0; i < order+1; ++i) {
                    data(k, 0, i) = (
                            -(nfp * 2 * M_PI * i) * sin(nfp * 2 * M_PI * i * quadpoints[k]) * cos(2 * M_PI * quadpoints[k])
                            -(2 * M_PI) *           cos(nfp * 2 * M_PI * i * quadpoints[k]) * sin(2 * M_PI * quadpoints[k])
                            );
                    data(k, 1, i) = (
                            -(nfp * 2 * M_PI * i) * sin(nfp * 2 * M_PI * i * quadpoints[k]) * sin(2 * M_PI * quadpoints[k])
                            +(2 * M_PI) *           cos(nfp * 2 * M_PI * i * quadpoints[k]) * cos(2 * M_PI * quadpoints[k])
                            );
                }
                for (int i = 1; i < order+1; ++i) {
                    data(k, 2, order + i) = (
                            (nfp * 2 * M_PI * i) * cos(nfp * 2 * M_PI * i * quadpoints[k])
                            );
                }
            }
        }

        void dgammadashdash_by_dcoeff_impl(Array& data) override {
            for (int k = 0; k < numquadpoints; ++k) {
                for (int i = 0; i < order+1; ++i) {
                    data(k, 0, i) = (
                            +2*(nfp * 2 * M_PI * i)*(2 * M_PI) *       sin(nfp * 2 * M_PI * i * quadpoints[k]) * sin(2 * M_PI * quadpoints[k])
                            -(pow(nfp * 2 * M_PI * i, 2) + pow(2*M_PI, 2)) * cos(nfp * 2 * M_PI * i * quadpoints[k]) * cos(2 * M_PI * quadpoints[k])
                            );
                    data(k, 1, i) = (
                            -2*(nfp * 2 * M_PI * i) * (2 * M_PI) *     sin(nfp * 2 * M_PI * i * quadpoints[k]) * cos(2 * M_PI * quadpoints[k])
                            -(pow(nfp * 2 * M_PI * i, 2) + pow(2*M_PI, 2)) * cos(nfp * 2 * M_PI * i * quadpoints[k]) * sin(2 * M_PI * quadpoints[k])
                            );
                }
                for (int i = 1; i < order+1; ++i) {
                    data(k, 2, order + i) = -1. * (
                            pow(nfp * 2 * M_PI * i, 2) * sin(nfp * 2 * M_PI * i * quadpoints[k])
                            );
                }
            }
        }

        void dgammadashdashdash_by_dcoeff_impl(Array& data) override {
            for (int k = 0; k < numquadpoints; ++k) {
                for (int i = 0; i < order+1; ++i) {
                    data(k, 0, i) = (
                            +2*pow(nfp * 2 * M_PI * i, 2)*(2 * M_PI) *       cos(nfp * 2 * M_PI * i * quadpoints[k]) * sin(2 * M_PI * quadpoints[k])
                            +2*(nfp * 2 * M_PI * i)*pow(2 * M_PI, 2) *       sin(nfp * 2 * M_PI * i * quadpoints[k]) * cos(2 * M_PI * quadpoints[k])
                            +(pow(nfp * 2 * M_PI * i, 2) + pow(2*M_PI, 2))*(nfp * 2 * M_PI * i)* sin(nfp * 2 * M_PI * i * quadpoints[k]) * cos(2 * M_PI * quadpoints[k])
                            +(pow(nfp * 2 * M_PI * i, 2) + pow(2*M_PI, 2))*(2 * M_PI)*           cos(nfp * 2 * M_PI * i * quadpoints[k]) * sin(2 * M_PI * quadpoints[k])
                            );
                    data(k, 1, i) = (
                            -2*pow(nfp * 2 * M_PI * i, 2) * (2 * M_PI) *     cos(nfp * 2 * M_PI * i * quadpoints[k]) * cos(2 * M_PI * quadpoints[k])
                            +2*(nfp * 2 * M_PI * i) * pow(2 * M_PI, 2) *     sin(nfp * 2 * M_PI * i * quadpoints[k]) * sin(2 * M_PI * quadpoints[k])
                            +(pow(nfp * 2 * M_PI * i, 2) + pow(2*M_PI, 2)) * (nfp * 2 * M_PI * i) * sin(nfp * 2 * M_PI * i * quadpoints[k]) * sin(2 * M_PI * quadpoints[k])
                            -(pow(nfp * 2 * M_PI * i, 2) + pow(2*M_PI, 2)) * (2 * M_PI) *           cos(nfp * 2 * M_PI * i * quadpoints[k]) * cos(2 * M_PI * quadpoints[k])
                            );
                }
                for (int i = 1; i < order+1; ++i) {
                    data(k, 2, order + i) = -1. * (
                            pow(nfp * 2 * M_PI * i, 3) * cos(nfp * 2 * M_PI * i * quadpoints[k])
                            );
                }
            }
        }

};
