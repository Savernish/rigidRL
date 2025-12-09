#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <iostream>
#include "engine/tensor.h"
#include "engine/activations.h"
#include "engine/optimizers.h"
#include "engine/body.h"
#include "renderer/sdl_renderer.h"
#include "engine/engine.h"

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
        .def_static("stack", &Tensor::stack, py::keep_alive<0, 1>()) // Static method

        // 4. Operators (with keep_alive to manage graph lifetime)
        // Keep 'this' (1) and 'other' (2) alive as long as 'result' (0) is alive
        .def("__add__", &Tensor::operator+, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("__sub__", &Tensor::operator-, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("__mul__", (Tensor (Tensor::*)(const Tensor&) const) &Tensor::operator*, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("__mul__", (Tensor (Tensor::*)(float) const) &Tensor::operator*, py::keep_alive<0, 1>())
        .def("__rmul__", (Tensor (Tensor::*)(float) const) &Tensor::operator*, py::keep_alive<0, 1>())
        .def("__truediv__", &Tensor::operator/, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())


        // Reductions
        .def("sum", (Tensor (Tensor::*)()) &Tensor::sum, py::keep_alive<0, 1>())
        .def("sum", (Tensor (Tensor::*)(int)) &Tensor::sum, py::keep_alive<0, 1>())
        .def("mean", (Tensor (Tensor::*)()) &Tensor::mean, py::keep_alive<0, 1>())
        .def("mean", (Tensor (Tensor::*)(int)) &Tensor::mean, py::keep_alive<0, 1>())
        .def("min", &Tensor::min, py::keep_alive<0, 1>())
        .def("max", &Tensor::max, py::keep_alive<0, 1>())

        // Math
        .def("exp", &Tensor::exp, py::keep_alive<0, 1>())
        .def("log", &Tensor::log, py::keep_alive<0, 1>())
        .def("sqrt", &Tensor::sqrt, py::keep_alive<0, 1>())
        .def("abs", &Tensor::abs, py::keep_alive<0, 1>())
        .def("clamp", &Tensor::clamp, py::keep_alive<0, 1>())
        
        .def("transpose", &Tensor::transpose, py::keep_alive<0, 1>())
        .def("matmul", &Tensor::matmul, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("__matmul__", &Tensor::matmul, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        
        // New Features
        .def("pow", &Tensor::pow, py::keep_alive<0, 1>())
        .def("__pow__", &Tensor::pow, py::keep_alive<0, 1>())
        .def("reshape", &Tensor::reshape, py::keep_alive<0, 1>())
        .def_static("cat", &Tensor::cat, py::keep_alive<0, 1>());

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

    py::class_<Body>(m, "Body")
        .def(py::init<float, float, float, float, float>(), 
             py::arg("x"), py::arg("y"), py::arg("mass"), py::arg("width"), py::arg("height"))
        .def("step", py::overload_cast<const Tensor&, const Tensor&, float>(&Body::step), py::arg("forces"), py::arg("torque"), py::arg("dt"))
        .def("step", py::overload_cast<float>(&Body::step), py::arg("dt"))
        .def("apply_force", &Body::apply_force)
        .def("apply_force_at_point", &Body::apply_force_at_point, py::arg("force"), py::arg("point"))
        .def("apply_torque", &Body::apply_torque)
        .def("reset_forces", &Body::reset_forces)
        .def_property_readonly("pos", [](Body& b) -> Tensor& { return b.pos; }, py::return_value_policy::reference_internal)
        .def_property_readonly("vel", [](Body& b) -> Tensor& { return b.vel; }, py::return_value_policy::reference_internal)
        .def_property_readonly("rotation", [](Body& b) -> Tensor& { return b.rotation; }, py::return_value_policy::reference_internal)
        .def_property_readonly("ang_vel", [](Body& b) -> Tensor& { return b.ang_vel; }, py::return_value_policy::reference_internal)
        .def("get_x", &Body::get_x)
        .def("get_y", &Body::get_y)
        .def("get_rotation", &Body::get_rotation)
        .def_readwrite("is_static", &Body::is_static)
        .def_readwrite("friction", &Body::friction)
        .def_readwrite("restitution", &Body::restitution);

    py::class_<Renderer>(m, "Renderer")
        .def("get_width", &Renderer::get_width, "Get the window width in pixels.")
        .def("get_height", &Renderer::get_height, "Get the window height in pixels.")
        .def("get_scale", &Renderer::get_scale, "Get the pixels-per-meter scale factor.")
        .def("clear", &Renderer::clear, "Clear the screen with the background color.")
        .def("present", &Renderer::present, "Swap buffers and present the rendered frame.")
        .def("process_events", &Renderer::process_events, "Handle window events (close, resize). Returns False if quit.")
        .def("draw_box", &Renderer::draw_box, py::arg("x"), py::arg("y"), py::arg("w"), py::arg("h"), py::arg("rotation"),
             py::arg("r")=1.0f, py::arg("g")=1.0f, py::arg("b")=1.0f,
             "Draw a rectangle defined by center (x,y), width, height, and rotation.")
        .def("draw_line", &Renderer::draw_line, py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"), 
             py::arg("r")=1.0f, py::arg("g")=1.0f, py::arg("b")=1.0f,
             "Draw a line between (x1,y1) and (x2,y2) with color (r,g,b).");

    py::class_<SDLRenderer, Renderer>(m, "SDLRenderer")
        .def(py::init<int, int, float>(), py::arg("width")=800, py::arg("height")=600, py::arg("scale")=50.0f);

    py::class_<Engine>(m, "Engine")
        .def(py::init<int, int, float, float, int>(), py::arg("width")=800, py::arg("height")=600, py::arg("scale")=50.0f, py::arg("dt")=0.016f, py::arg("substeps")=10)
        .def("add_body", &Engine::add_body)
        .def("set_gravity", &Engine::set_gravity)
        .def("step", &Engine::step, "Run one simulation step. Returns False if Quit event received.")
        .def("update", &Engine::update, "Run one physics step (forces, collision, integration).")
        .def("render_bodies", &Engine::render_bodies, "Render all bodies + colliders.")
        .def("add_collider", &Engine::add_collider, py::arg("x"), py::arg("y"), py::arg("width"), py::arg("height"), py::arg("rotation")=0.0f,
             py::return_value_policy::reference, "Add a static box collider (ground, wall, platform).")
        .def("clear_colliders", &Engine::clear_colliders, "Remove all static colliders.")
        .def("get_renderer", &Engine::get_renderer, py::return_value_policy::reference);
}