#pragma once

template<class Array>
struct CachedArray {
    Array data;
    bool status;
    CachedArray(Array _data) : data(_data), status(false) {}
};


