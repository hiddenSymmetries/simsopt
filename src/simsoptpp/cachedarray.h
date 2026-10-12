#pragma once

#include <utility>

template<class Array>
struct CachedArray {
    Array data;
    bool status;
    CachedArray(Array _data) : data(std::move(_data)), status(false) {}
};


