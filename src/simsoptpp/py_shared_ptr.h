#pragma once

// see https://github.com/pybind/pybind11/issues/1389

#include <pybind11/pybind11.h>
#include <memory>
namespace py = pybind11;

template <typename T> class py_shared_ptr {
    private:
        std::shared_ptr<T> _impl;

    public:
        using element_type = T;

        py_shared_ptr(T *ptr) {
            py::object pyobj = py::cast(ptr);
            PyObject* pyptr = pyobj.ptr();
            Py_INCREF(pyptr);
            std::shared_ptr<PyObject> vec_py_ptr(
                    pyptr, [](PyObject *ob) { Py_DECREF(ob); });
            _impl = std::shared_ptr<T>(vec_py_ptr, ptr);
        }

        py_shared_ptr(std::shared_ptr<T> r): _impl(r){}

        operator std::shared_ptr<T>() { return _impl; }

        T* get() const {return _impl.get();}
};
