#pragma once

#include <string>
#include <vector>
#include <xtensor/xarray.hpp>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include "cachedarray.h"


using std::string;
using std::vector;

template<class Array>
class Cache {
    private:
        std::map<string, CachedArray<Array>> cache;
    public:
        bool get_status(string key) const {
            auto loc = cache.find(key);
            if(loc == cache.end()){ // Key not found
                return false;
            }
            if(!(loc->second.status)){ // needs recomputing
                return false;
            }
            return true;
        }
        Array& get_or_create(string key, vector<int> dims){
            auto loc = cache.find(key);
            if(loc == cache.end()){ // Key not found --> allocate array
                loc = cache.insert(std::make_pair(key, CachedArray<Array>(xt::zeros<double>(dims)))).first; 
                //fmt::print("Create a new array for key {} of size [{}] at {}\n", key, fmt::join(dims, ", "), fmt::ptr(loc->second.data.data()));
            } else if(loc->second.data.shape(0) != dims[0]) { // key found but not the right number of points
                loc->second = CachedArray<Array>(xt::zeros<double>(dims));
                //fmt::print("Create a new array for key {} of size [{}] at {}\n", key, fmt::join(dims, ", "), fmt::ptr(loc->second.data.data()));
            } else {
                //fmt::print("Existing array found for key {} of size [{}] at {}\n", key, fmt::join(dims, ", "), fmt::ptr(loc->second.data.data()));
            }
            loc->second.status = true;
            return loc->second.data;
        }

        Array& get_or_create_and_fill(string key, vector<int> dims, std::function<void(Array&)> impl) {
            auto loc = cache.find(key);
            if(loc == cache.end()){ // Key not found --> allocate array
                loc = cache.insert(std::make_pair(key, CachedArray<Array>(xt::zeros<double>(dims)))).first; 
                //fmt::print("Create a new array for key {} of size [{}] at {}\n", key, fmt::join(dims, ", "), fmt::ptr(loc->second.data.data()));
            } else if(loc->second.data.shape(0) != dims[0]) { // key found but not the right number of points
                loc->second = CachedArray<Array>(xt::zeros<double>(dims));
                //fmt::print("Create a new array for key {} of size [{}] at {}\n", key, fmt::join(dims, ", "), fmt::ptr(loc->second.data.data()));
            }
            if(!(loc->second.status)){ // needs recomputing
                //fmt::print("Fill array for key {} of size [{}] at {}\n", key, fmt::join(dims, ", "), fmt::ptr(loc->second.data.data()));
                impl(loc->second.data);
                loc->second.status = true;
            }
            return loc->second.data;
        }

        void invalidate_cache(){
            for (auto it = cache.begin(); it != cache.end(); ++it) {
                it->second.status = false;
            }
        }
};
