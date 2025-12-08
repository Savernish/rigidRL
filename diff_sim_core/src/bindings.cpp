#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <iostream>
#include "core.h"
#include "activations.h"
#include "optimizers.h"

namespace py = pybind11;

// A simple C++ function to test the bridge
float add_cpp(float a, float b) {
    std::cout << "[C++] Adding " << a << " + " << b << std::endl;
    return a + b;
}

PYBIND11_MODULE(forgeNN_cpp, m) {
    m.doc() = "forgeNN++: C++ Core (Eigen Backend)";

    m.def("add", &add_cpp, "A test function that adds two numbers");

    py::class_<Tensor>(m, "Tensor")
        // 1. Constructors
        .def(py::init<int, int, bool>(), py::arg("rows"), py::arg("cols"), py::arg("requires_grad")=false)
        .def(py::init<int, bool>(), py::arg("size"), py::arg("requires_grad")=false)
        .def(py::init<std::vector<float>, bool>(), py::arg("data"), py::arg("requires_grad")=false)
        
        // 2. Direct methods
        .def("set", &Tensor::set)
        .def("get", &Tensor::get)
        .def("rows", &Tensor::rows)
        .def("cols", &Tensor::cols)
        .def("backward", &Tensor::backward)
        .def("zero_grad", &Tensor::zero_grad)
        
        // 3. Properties
        .def_property_readonly("shape", [](const Tensor& t) {
            return std::make_pair(t.rows(), t.cols());
        })
        .def_property("requires_grad", &Tensor::get_requires_grad, &Tensor::set_requires_grad)
        .def_property("data", &Tensor::get_data, &Tensor::set_data)
        .def_property("grad", &Tensor::get_grad, &Tensor::set_grad)


        .def("sin", &Tensor::sin, py::keep_alive<0, 1>())
        .def("cos", &Tensor::cos, py::keep_alive<0, 1>())
        .def("select", &Tensor::select, py::keep_alive<0, 1>())
        .def("__getitem__", &Tensor::select, py::keep_alive<0, 1>()) // Enable t[i] syntax
        .def_static("stack", &Tensor::stack) // Static method

        // 4. Operators (with keep_alive to manage graph lifetime)
        // Keep 'this' (1) and 'other' (2) alive as long as 'result' (0) is alive
        .def("__add__", &Tensor::operator+, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("__sub__", &Tensor::operator-, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("__mul__", (Tensor (Tensor::*)(const Tensor&)) &Tensor::operator*, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("__mul__", (Tensor (Tensor::*)(float)) &Tensor::operator*, py::keep_alive<0, 1>())
        .def("__rmul__", (Tensor (Tensor::*)(float)) &Tensor::operator*, py::keep_alive<0, 1>())
        .def("__truediv__", &Tensor::operator/, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())


        // Reductions
        .def("sum", (Tensor (Tensor::*)()) &Tensor::sum)
        .def("sum", (Tensor (Tensor::*)(int)) &Tensor::sum)
        .def("mean", (Tensor (Tensor::*)()) &Tensor::mean)
        .def("mean", (Tensor (Tensor::*)(int)) &Tensor::mean)
        .def("min", &Tensor::min)
        .def("max", &Tensor::max)

        // Math
        .def("exp", &Tensor::exp)
        .def("log", &Tensor::log)
        .def("sqrt", &Tensor::sqrt)
        .def("abs", &Tensor::abs)
        .def("clamp", &Tensor::clamp)
        
        .def("transpose", &Tensor::transpose)
        .def("matmul", &Tensor::matmul, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("__matmul__", &Tensor::matmul, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        
        // New Features
        .def("pow", &Tensor::pow, py::keep_alive<0, 1>())
        .def("__pow__", &Tensor::pow, py::keep_alive<0, 1>())
        .def("reshape", &Tensor::reshape, py::keep_alive<0, 1>())
        .def_static("cat", &Tensor::cat);

    // Module-level Activations
    m.def("relu", &relu, py::keep_alive<0, 1>());
    m.def("tanh", (Tensor (*)(const Tensor&)) &tanh, py::keep_alive<0, 1>());

    py::class_<SGD>(m, "SGD")
        .def(py::init<std::vector<Tensor*>, float>(), py::arg("params"), py::arg("lr"))
        .def("step", &SGD::step)
        .def("zero_grad", &SGD::zero_grad);

    py::class_<Adam>(m, "Adam")
        .def(py::init<std::vector<Tensor*>, float, float, float, float>(), 
             py::arg("params"), py::arg("lr")=0.001, py::arg("beta1")=0.9, py::arg("beta2")=0.999, py::arg("epsilon")=1e-8)
        .def("step", &Adam::step)
        .def("zero_grad", &Adam::zero_grad);

    py::class_<AdamW>(m, "AdamW")
        .def(py::init<std::vector<Tensor*>, float, float, float, float, float>(), 
             py::arg("params"), py::arg("lr")=0.001, py::arg("beta1")=0.9, py::arg("beta2")=0.999, py::arg("epsilon")=1e-8, py::arg("weight_decay")=0.0)
        .def("step", &AdamW::step)
        .def("zero_grad", &AdamW::zero_grad);
}