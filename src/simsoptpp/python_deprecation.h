#pragma once

#include <string>
#include <utility>
#include "pybind11/pybind11.h"

namespace py = pybind11;

// Message for a deprecated accessor that returns a live view of a cached array.
inline std::string cache_ref_deprecation_message(const std::string& name, const std::string& replacement) {
    return name + "() is deprecated and will be removed at a future date; please use " + replacement
        + "() instead, which returns an independent copy. The array returned by " + name
        + "() is a live view of the field cache that later evaluations overwrite in place.";
}

// Message for a deprecated internal field-cache hook, which has no public replacement.
inline std::string fieldcache_deprecation_message(const std::string& name, const std::string& accessors) {
    return name + "() is deprecated and will be removed at a future date. It is an internal field-cache hook"
        " with no public replacement; please use the public accessors, such as " + accessors
        + ", which return independent copies.";
}

// Returns a callable that emits DeprecationWarning and then forwards to `method`,
// returning exactly what `method` returns (references stay references).
// PyErr_WarnEx stack level 1 attributes the warning to the calling Python frame.
template <typename C, typename R, typename... Args>
auto deprecated_alias(R (C::*method)(Args...), std::string message) {
    return [method, message = std::move(message)](C& self, Args... args) -> R {
        if (PyErr_WarnEx(PyExc_DeprecationWarning, message.c_str(), 1) < 0)
            throw py::error_already_set();
        return (self.*method)(std::forward<Args>(args)...);
    };
}

// Docstring of the private accessor `_<quantity>_ref`.
inline std::string cache_ref_doc(const std::string& quantity) {
    return "Private. As `" + quantity + "`, but returns a reference to the cached array (this array should be"
        " read only). The array is overwritten in place by later evaluations; copy it before holding on to it.";
}

// Docstring of the deprecated public alias `<quantity>_ref`.
inline std::string deprecated_cache_ref_doc(const std::string& quantity) {
    return "Deprecated alias of `_" + quantity + "_ref`; use `" + quantity + "` for an independent copy."
        " Returns a live view of the cached array that later evaluations overwrite in place.";
}

// Registers the private accessor `_<quantity>_ref` and its deprecated public alias `<quantity>_ref`.
// pybind11 copies the name and docstring strings, so passing temporaries is safe.
template <typename S, typename M>
void def_cache_ref(S& c, const std::string& quantity, M method) {
    const std::string ref = quantity + "_ref";
    c.def(("_" + ref).c_str(), method, cache_ref_doc(quantity).c_str());
    c.def(ref.c_str(), deprecated_alias(method, cache_ref_deprecation_message(ref, quantity)),
          deprecated_cache_ref_doc(quantity).c_str());
}
