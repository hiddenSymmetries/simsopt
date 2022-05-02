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

}
