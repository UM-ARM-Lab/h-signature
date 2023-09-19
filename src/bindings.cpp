#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include <h_signature/h_signature.h>

namespace py = pybind11;

PYBIND11_MODULE(pyh_signature, m) {
    m.doc() = "H-signature";
    m.def("get_h_signature", &get_h_signature, "Get H-signature");
}