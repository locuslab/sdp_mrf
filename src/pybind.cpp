#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdio.h>

extern "C" {
#include "solvers.h"
}

namespace py = pybind11;

using arr = py::array;
float *data(arr& a) { return (float*) a.mutable_data(); }

float py_M4(arr A, arr h, arr V, float eps, int max_iter)
{
    int n = A.shape(0);
    int d = V.shape(1);

    return M4(n, d, data(A), data(h), data(V), eps, max_iter);
}

float py_M4_plus(arr A, arr h, arr Z, int k, float eps, int max_iter)
{
    int n = A.shape(0);
    int d = Z.shape(1);

    return M4_plus(n, d, k, data(A), data(h), data(Z), eps, max_iter);
}

PYBIND11_MODULE(EXTENSION_NAME, m) {
    m.doc() = "";
    m.def("M4", &py_M4, "M4");
    m.def("M4_plus", &py_M4_plus, "M4_plus");
}
