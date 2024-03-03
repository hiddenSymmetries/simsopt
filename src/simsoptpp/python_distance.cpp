#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
namespace py = pybind11;
#include "xtensor-python/pyarray.hpp"     // Numpy bindings
typedef xt::pyarray<double> PyArray;
using std::vector;
using std::tuple;
using std::set;
using std::floor;


template<class T>
bool empty_intersection(const set<T>& x, const set<T>& y)
{
    auto i = x.begin();
    auto j = y.begin();
    while (i != x.end() && j != y.end())
    {
        if (*i == *j)
            return false;
        else if (*i < *j)
            ++i;
        else
            ++j;
    }
    return true;
}

bool two_points_too_close_exist(PyArray& XA, PyArray& XB, double threshold_squared){
    for (int k = 0; k < XA.shape(0); ++k) {
        for (int l = 0; l < XB.shape(0); ++l) {
            double dist = std::pow(XA(k, 0) - XB(l, 0), 2) + std::pow(XA(k, 1) - XB(l, 1), 2) + std::pow(XA(k, 2) - XB(l, 2), 2);
            if(dist < threshold_squared)
                return true;
        }
    }
    return false;
}

vector<tuple<int, int>> get_close_candidates_pdist(vector<PyArray>& pointClouds, double threshold, int num_base_curves) {
    /*
       Returns all pairings of the given pointClouds that have two points that
       are less than `threshold` away. The estimate is approximate (for
       speed), so this function may return too many (but not too few!)
       pairings.

       The basic idea of this function is the following:
       - Assume we want to compare pointcloud A and B.
       - We create a uniform grid of cell size threshold.
       - Loop over points in cloud A, mark all cells that have a point in it (via the `set` variables below).
       - Loop over points in cloud B, mark all cells that have a point in it and also all cells in the 8 neighbouring cells around it.
       - Check whether the intersection between the two sets is non-empty.
       */
    vector<set<tuple<int, int, int>>> sets(pointClouds.size());
    vector<set<tuple<int, int, int>>> sets_extended(pointClouds.size());
#pragma omp parallel for
    for (int p = 0; p < pointClouds.size(); ++p) {
        PyArray& points = pointClouds[p];
        for (int l = 0; l < points.shape(0); ++l) {
            int i = floor(points(l, 0)/threshold);
            int j = floor(points(l, 1)/threshold);
            int k = floor(points(l, 2)/threshold);
            sets[p].insert({i, j, k});
            for (int ii = -1; ii <= 1; ++ii) {
                for (int jj = -1; jj <= 1; ++jj) {
                    for (int kk = -1; kk <= 1; ++kk) {
                        sets_extended[p].insert({i + ii, j + jj, k + kk});
                    }
                }
            }
        }
    }


    vector<tuple<int, int>> candidates_1;
    for (int i = 0; i < pointClouds.size(); ++i) {
        for (int j = 0; j < i; ++j) {
            if(j < num_base_curves)
                candidates_1.push_back({i, j});
        }
    }
    vector<tuple<int, int>> candidates_2;
#pragma omp parallel for
    for (int k = 0; k < candidates_1.size(); ++k) {
        int i = std::get<0>(candidates_1[k]);
        int j = std::get<1>(candidates_1[k]);
        bool check = empty_intersection(sets_extended[i], sets[j]);
#pragma omp critical
        if(!check)
            candidates_2.push_back({i, j});
    }

    double t2 = threshold*threshold;
    vector<tuple<int, int>> candidates_3;
#pragma omp parallel for
    for (int k = 0; k < candidates_2.size(); ++k) {
        int i = std::get<0>(candidates_2[k]);
        int j = std::get<1>(candidates_2[k]);
        bool check = two_points_too_close_exist(pointClouds[i], pointClouds[j], t2);
#pragma omp critical
        if(check)
            candidates_3.push_back({i, j});
    }
    return candidates_3;
}

vector<tuple<int, int>> get_close_candidates_cdist(vector<PyArray>& pointCloudsA, vector<PyArray>& pointCloudsB, double threshold) {
    /*
       */
    vector<set<tuple<int, int, int>>> sets_A(pointCloudsA.size());
    vector<set<tuple<int, int, int>>> sets_B_extended(pointCloudsB.size());
#pragma omp parallel for
    for (int p = 0; p < pointCloudsA.size(); ++p) {
        PyArray& points = pointCloudsA[p];
        for (int l = 0; l < points.shape(0); ++l) {
            int i = floor(points(l, 0)/threshold);
            int j = floor(points(l, 1)/threshold);
            int k = floor(points(l, 2)/threshold);
            sets_A[p].insert({i, j, k});
        }
    }
    for (int p = 0; p < pointCloudsB.size(); ++p) {
        set<tuple<int, int, int>> s;
        set<tuple<int, int, int>> s_extended;
        PyArray& points = pointCloudsB[p];
        for (int l = 0; l < points.shape(0); ++l) {
            int i = floor(points(l, 0)/threshold);
            int j = floor(points(l, 1)/threshold);
            int k = floor(points(l, 2)/threshold);
            for (int ii = -1; ii <= 1; ++ii) {
                for (int jj = -1; jj <= 1; ++jj) {
                    for (int kk = -1; kk <= 1; ++kk) {
                        sets_B_extended[p].insert({i + ii, j + jj, k + kk});
                    }
                }
            }
        }
    }


    vector<tuple<int, int>> candidates_1;
    for (int i = 0; i < pointCloudsA.size(); ++i) {
        for (int j = 0; j < pointCloudsB.size(); ++j) {
            candidates_1.push_back({i, j});
        }
    }
    vector<tuple<int, int>> candidates_2;
#pragma omp parallel for
    for (int k = 0; k < candidates_1.size(); ++k) {
        int i = std::get<0>(candidates_1[k]);
        int j = std::get<1>(candidates_1[k]);
        bool check = empty_intersection(sets_A[i], sets_B_extended[j]);
#pragma omp critical
        if(!check)
            candidates_2.push_back({i, j});
    }

    double t2 = threshold*threshold;
    vector<tuple<int, int>> candidates_3;
#pragma omp parallel for
    for (int k = 0; k < candidates_2.size(); ++k) {
        int i = std::get<0>(candidates_2[k]);
        int j = std::get<1>(candidates_2[k]);
        bool check = two_points_too_close_exist(pointCloudsA[i], pointCloudsB[j], t2);
#pragma omp critical
        if(check)
            candidates_3.push_back({i, j});
    }
    return candidates_3;
}

void init_distance(py::module_ &m){

    m.def("get_pointclouds_closer_than_threshold_within_collection", &get_close_candidates_pdist, "In a list of point clouds, get all pairings that are closer than threshold to each other.", py::arg("pointClouds"), py::arg("threshold"), py::arg("num_base_curves"));
    m.def("get_pointclouds_closer_than_threshold_between_two_collections", &get_close_candidates_cdist, "Between two lists of pointclouds, get all pairings that are closer than threshold to each other.", py::arg("pointCloudsA"), py::arg("pointCloudsB"), py::arg("threshold"));
    m.def("compute_linking_number", [](const vector<PyArray>& gammas, const vector<PyArray>& gammadashs, const PyArray& dphis, const double downsample) {
        int ncurves = gammas.size();
        // assert(dphis.size() == ncurves);
        // assert(gammadashs.size() == ncurves);

        int linking_number = 0;
        #pragma omp parallel for reduction(+:linking_number)
        for (int p = 1; p < ncurves; p++) {
            int linknphi1 = gammas[p].shape(0);
            const double *curve1_ptr = gammas[p].data();
            const double *curve1dash_ptr = gammadashs[p].data();
            // assert(gammas[p].size() == gammadashs[p].size());
            for (int q = 0; q < p; q++) {
                // assert(gammas[q].size() == gammadashs[q].size());
                int linknphi2 = gammas[q].shape(0);
                const double *curve2_ptr = gammas[q].data();
                const double *curve2dash_ptr = gammadashs[q].data();
                double difference[3] = { 0 };
                double total = 0;
                double dr, det;
                for (int i=0; i < linknphi1; i += downsample){
                    for (int j=0; j < linknphi2; j += downsample){
                        difference[0] = (curve1_ptr[3*i+0] - curve2_ptr[3*j+0]);
                        difference[1] = (curve1_ptr[3*i+1] - curve2_ptr[3*j+1]);
                        difference[2] = (curve1_ptr[3*i+2] - curve2_ptr[3*j+2]);
                        dr = std::sqrt(difference[0]*difference[0] + difference[1]*difference[1] + difference[2]*difference[2]);
                        det = curve1dash_ptr[3*i+0]*(curve2dash_ptr[3*j+1]*difference[2] 
                            - curve2dash_ptr[3*j+2]*difference[1]) 
                            - curve1dash_ptr[3*i+1]*(curve2dash_ptr[3*j+0]*difference[2] 
                            - curve2dash_ptr[3*j+2]*difference[0]) 
                            + curve1dash_ptr[3*i+2]*(curve2dash_ptr[3*j+0]*difference[1] 
                            - curve2dash_ptr[3*j+1]*difference[0]);
                        total += det / (dr * dr * dr);
                    }
                }
                linking_number += std::round(std::abs(total * dphis[p] * dphis[q]) / (4 * M_PI));
            }
        }
        return linking_number;
    });

}
