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

PYBIND11_MODULE(rigidRL, m) {
    m.doc() = "rigidRL: C++ Core (Eigen Backend)";

    m.def("add", &add_cpp, "A test function that adds two numbers");

    py::class_<Tensor>(m, "Tensor")
        // 1. Constructors
        .def(py::init<int, int, bool>(), py::arg("rows"), py::arg("cols"), py::arg("requires_grad")=false)
        .def(py::init<int, bool>(), py::arg("size"), py::arg("requires_grad")=false)
        .def(py::init<std::vector<float>, bool>(), py::arg("data"), py::arg("requires_grad")=false)
        
        // 2. Direct methods
        .def("set", &Tensor::Set)
        .def("get", &Tensor::Get)
        .def("rows", &Tensor::Rows)
        .def("cols", &Tensor::Cols)
        .def("backward", &Tensor::Backward)
        .def("zero_grad", &Tensor::ZeroGrad)
        
        // 3. Properties
        .def_property_readonly("shape", [](const Tensor& t) {
            return std::make_pair(t.Rows(), t.Cols());
        })
        .def_property("requires_grad", &Tensor::GetRequiresGrad, &Tensor::SetRequiresGrad)
        .def_property("data", &Tensor::GetData, &Tensor::SetData)
        .def_property("grad", &Tensor::GetGrad, &Tensor::SetGrad)



        .def("sin", &Tensor::Sin, py::keep_alive<0, 1>())
        .def("cos", &Tensor::Cos, py::keep_alive<0, 1>())
        .def("exp", &Tensor::Exp, py::keep_alive<0, 1>())
        .def("log", &Tensor::Log, py::keep_alive<0, 1>())
        .def("select", &Tensor::Select, py::keep_alive<0, 1>())
        .def("__getitem__", &Tensor::Select, py::keep_alive<0, 1>()) // Enable t[i] syntax
        .def_static("stack", &Tensor::Stack, py::keep_alive<0, 1>()) // Static method

        // 4. Operators (with keep_alive to manage graph lifetime)
        // Keep 'this' (1) and 'other' (2) alive as long as 'result' (0) is alive
        .def("__add__", &Tensor::operator+, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("__sub__", &Tensor::operator-, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("__mul__", (Tensor (Tensor::*)(const Tensor&) const) &Tensor::operator*, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("__mul__", (Tensor (Tensor::*)(float) const) &Tensor::operator*, py::keep_alive<0, 1>())
        .def("__rmul__", (Tensor (Tensor::*)(float) const) &Tensor::operator*, py::keep_alive<0, 1>())
        .def("__truediv__", &Tensor::operator/, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())


        // Reductions
        .def("sum", (Tensor (Tensor::*)()) &Tensor::Sum, py::keep_alive<0, 1>())
        .def("sum", (Tensor (Tensor::*)(int)) &Tensor::Sum, py::keep_alive<0, 1>())
        .def("mean", (Tensor (Tensor::*)()) &Tensor::Mean, py::keep_alive<0, 1>())
        .def("mean", (Tensor (Tensor::*)(int)) &Tensor::Mean, py::keep_alive<0, 1>())
        .def("min", &Tensor::Min, py::keep_alive<0, 1>())
        .def("max", &Tensor::Max, py::keep_alive<0, 1>())

        // Math
        .def("exp", &Tensor::Exp, py::keep_alive<0, 1>())
        .def("log", &Tensor::Log, py::keep_alive<0, 1>())
        .def("sqrt", &Tensor::Sqrt, py::keep_alive<0, 1>())
        .def("abs", &Tensor::Abs, py::keep_alive<0, 1>())
        .def("clamp", &Tensor::Clamp, py::keep_alive<0, 1>())
        
        .def("transpose", &Tensor::Transpose, py::keep_alive<0, 1>())
        .def("matmul", &Tensor::Matmul, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        .def("__matmul__", &Tensor::Matmul, py::keep_alive<0, 1>(), py::keep_alive<0, 2>())
        
        // New Features
        .def("pow", &Tensor::Pow, py::keep_alive<0, 1>())
        .def("__pow__", &Tensor::Pow, py::keep_alive<0, 1>())
        .def("reshape", &Tensor::Reshape, py::keep_alive<0, 1>())
        .def_static("cat", &Tensor::Cat, py::keep_alive<0, 1>())
        .def_static("gaussian_log_prob", &Tensor::GaussianLogProb, 
            py::arg("action"), py::arg("mean"), py::arg("log_std"),
            py::keep_alive<0, 1>(), py::keep_alive<0, 2>(), py::keep_alive<0, 3>());
    
    // Module-level function for convenience
    m.def("gaussian_log_prob", &Tensor::GaussianLogProb, 
        py::arg("action"), py::arg("mean"), py::arg("log_std"),
        py::keep_alive<0, 1>(), py::keep_alive<0, 2>(), py::keep_alive<0, 3>());

    // Module-level Activations
    m.def("relu", &relu, py::keep_alive<0, 1>());
    m.def("tanh", (Tensor (*)(const Tensor&)) &tanh, py::keep_alive<0, 1>());

    py::class_<SGD>(m, "SGD")
        .def(py::init<std::vector<Tensor*>, float>(), py::arg("params"), py::arg("lr"))
        .def("step", &SGD::Step)
        .def("zero_grad", &SGD::ZeroGrad);

    py::class_<Adam>(m, "Adam")
        .def(py::init<std::vector<Tensor*>, float, float, float, float>(), 
             py::arg("params"), py::arg("lr")=0.001, py::arg("beta1")=0.9, py::arg("beta2")=0.999, py::arg("epsilon")=1e-8)
        .def("step", &Adam::Step)
        .def("zero_grad", &Adam::ZeroGrad);

    py::class_<AdamW>(m, "AdamW")
        .def(py::init<std::vector<Tensor*>, float, float, float, float, float>(), 
             py::arg("params"), py::arg("lr")=0.001, py::arg("beta1")=0.9, py::arg("beta2")=0.999, py::arg("epsilon")=1e-8, py::arg("weight_decay")=0.0)
        .def("step", &AdamW::Step)
        .def("zero_grad", &AdamW::ZeroGrad);

    py::class_<Body>(m, "Body")
        .def(py::init<float, float, float, float, float>(), 
             py::arg("x"), py::arg("y"), py::arg("mass"), py::arg("width"), py::arg("height"))
        .def("step", py::overload_cast<const Tensor&, const Tensor&, float>(&Body::Step), py::arg("forces"), py::arg("torque"), py::arg("dt"))
        .def("step", py::overload_cast<float>(&Body::Step), py::arg("dt"))
        .def("apply_force", &Body::ApplyForce)
        .def("apply_force_at_point", &Body::ApplyForceAtPoint, py::arg("force"), py::arg("point"))
        .def("apply_torque", &Body::ApplyTorque)
        .def("reset_forces", &Body::ResetForces)
        .def_property_readonly("pos", [](Body& b) -> Tensor& { return b.pos; }, py::return_value_policy::reference_internal)
        .def_property_readonly("vel", [](Body& b) -> Tensor& { return b.vel; }, py::return_value_policy::reference_internal)
        .def_property_readonly("rotation", [](Body& b) -> Tensor& { return b.rotation; }, py::return_value_policy::reference_internal)
        .def_property_readonly("ang_vel", [](Body& b) -> Tensor& { return b.ang_vel; }, py::return_value_policy::reference_internal)
        .def("get_x", &Body::GetX)
        .def("get_y", &Body::GetY)
        .def("get_rotation", &Body::GetRotation)
        .def("set_rotation", [](Body& b, float angle) {
            b.rotation = Tensor(std::vector<float>{angle}, true);
        }, py::arg("angle"))
        .def_readwrite("is_static", &Body::is_static)
        .def_readwrite("friction", &Body::friction)
        .def_readwrite("restitution", &Body::restitution)
        .def("add_motor", &Body::AddMotor, py::arg("motor"), py::keep_alive<1, 2>())
        .def("add_box_shape", &Body::AddBoxShape, py::arg("w"), py::arg("h"), py::arg("off_x")=0.0f, py::arg("off_y")=0.0f,
             "Add a box shape to the body.")
        .def("add_circle_shape", &Body::AddCircleShape, py::arg("radius"), py::arg("off_x")=0.0f, py::arg("off_y")=0.0f,
             "Add a circle shape to the body.")
        .def("add_triangle_shape", &Body::AddTriangleShape, 
             py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"), py::arg("x3"), py::arg("y3"),
             "Add a triangle shape defined by 3 local vertices.")
        .def("clear_shapes", &Body::ClearShapes, "Remove all shapes from the body.")
        .def_static("Circle", &Body::Circle, 
             py::arg("x"), py::arg("y"), py::arg("mass"), py::arg("radius"),
             py::arg("friction")=0.3f, py::arg("restitution")=0.2f,
             py::return_value_policy::take_ownership,
             "Create a circle body.")
        .def_static("Rect", &Body::Rect,
             py::arg("x"), py::arg("y"), py::arg("mass"), py::arg("width"), py::arg("height"),
             py::arg("friction")=0.3f, py::arg("restitution")=0.2f,
             py::return_value_policy::take_ownership,
             "Create a rectangle body.")
        .def_static("Triangle", &Body::Triangle,
             py::arg("x"), py::arg("y"), py::arg("mass"),
             py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"), py::arg("x3"), py::arg("y3"),
             py::arg("friction")=0.3f, py::arg("restitution")=0.2f,
             py::return_value_policy::take_ownership,
             "Create a triangle body. Vertices are relative to body center.")
        .def_property_readonly("motors", [](Body& b) { return b.motors; }, py::return_value_policy::reference_internal);

    py::class_<Motor>(m, "Motor")
        .def(py::init<>())
        .def(py::init<float, float>(), py::arg("local_x"), py::arg("local_y"))
        .def(py::init<float, float, float, float, float, float>(), 
             py::arg("local_x"), py::arg("local_y"), py::arg("width"), py::arg("height"), py::arg("mass"), py::arg("max_thrust"))
        .def_readwrite("local_x", &Motor::local_x)
        .def_readwrite("local_y", &Motor::local_y)
        .def_readwrite("width", &Motor::width)
        .def_readwrite("height", &Motor::height)
        .def_readwrite("mass", &Motor::mass)
        .def_readwrite("thrust", &Motor::thrust)
        .def_readwrite("max_thrust", &Motor::max_thrust)
        .def_readwrite("angle", &Motor::angle)
        .def("set_thrust", &Motor::SetThrust, py::arg("thrust"));

    py::class_<Renderer>(m, "Renderer")
        .def("get_width", &Renderer::GetWidth, "Get the window width in pixels.")
        .def("get_height", &Renderer::GetHeight, "Get the window height in pixels.")
        .def("get_scale", &Renderer::GetScale, "Get the pixels-per-meter scale factor.")
        .def("clear", &Renderer::Clear, "Clear the screen with the background color.")
        .def("present", &Renderer::Present, "Swap buffers and present the rendered frame.")
        .def("process_events", &Renderer::ProcessEvents, "Handle window events (close, resize). Returns False if quit.")
        .def("draw_box", &Renderer::DrawBox, py::arg("x"), py::arg("y"), py::arg("w"), py::arg("h"), py::arg("rotation"),
             py::arg("r")=1.0f, py::arg("g")=1.0f, py::arg("b")=1.0f,
             "Draw a rectangle defined by center (x,y), width, height, and rotation.")
        .def("draw_line", &Renderer::DrawLine, py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"), 
             py::arg("r")=1.0f, py::arg("g")=1.0f, py::arg("b")=1.0f,
             "Draw a line between (x1,y1) and (x2,y2) with color (r,g,b).")
        .def("draw_circle", &Renderer::DrawCircle, py::arg("centerX"), py::arg("centerY"), py::arg("radius"), py::arg("r")=1.0f, py::arg("g")=1.0f, py::arg("b")=1.0f,
             "Draw a circle outline defined by center (x,y), radius, and color (r,g,b).")
        .def("draw_circle_filled", &Renderer::DrawCircleFilled, py::arg("centerX"), py::arg("centerY"), py::arg("radius"), py::arg("r")=1.0f, py::arg("g")=1.0f, py::arg("b")=1.0f,
             "Draw a filled circle defined by center (x,y), radius, and color (r,g,b).")
        .def("draw_triangle", &Renderer::DrawTriangle, py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"), py::arg("x3"), py::arg("y3"),
             py::arg("r")=1.0f, py::arg("g")=1.0f, py::arg("b")=1.0f,
             "Draw a triangle outline defined by 3 vertices.")
        .def("draw_triangle_filled", &Renderer::DrawTriangleFilled, py::arg("x1"), py::arg("y1"), py::arg("x2"), py::arg("y2"), py::arg("x3"), py::arg("y3"),
             py::arg("r")=1.0f, py::arg("g")=1.0f, py::arg("b")=1.0f,
             "Draw a filled triangle defined by 3 vertices.")
        .def("draw_box_filled", &Renderer::DrawBoxFilled, py::arg("x"), py::arg("y"), py::arg("w"), py::arg("h"), py::arg("rotation"),
             py::arg("r")=1.0f, py::arg("g")=1.0f, py::arg("b")=1.0f,
             "Draw a filled rectangle defined by center (x,y), width, height, and rotation.");

    py::class_<SDLRenderer, Renderer>(m, "SDLRenderer")
        .def(py::init<int, int, float>(), py::arg("width")=800, py::arg("height")=600, py::arg("scale")=50.0f);

    py::class_<Engine>(m, "Engine")
        .def(py::init<int, int, float, float, int, bool>(), 
             py::arg("width")=800, py::arg("height")=600, py::arg("scale")=50.0f, 
             py::arg("dt")=0.016f, py::arg("substeps")=10, py::arg("headless")=false)
        .def("add_body", &Engine::AddBody, py::keep_alive<1, 2>())
        .def("set_gravity", &Engine::SetGravity)
        .def("step", &Engine::Step, "Run one simulation step. Returns False if Quit event received.")
        .def("update", &Engine::Update, "Run one physics step (forces, collision, integration).")
        .def("render_bodies", &Engine::RenderBodies, "Render all bodies + colliders.")
        .def("add_collider", &Engine::AddCollider, py::arg("x"), py::arg("y"), py::arg("width"), py::arg("height"), py::arg("rotation")=0.0f, py::arg("friction")=0.5f,
             py::return_value_policy::reference, "Add a static box collider with optional friction (0-1).")
        .def("clear_colliders", &Engine::ClearColliders, "Remove all static colliders.")
        .def("clear_bodies", &Engine::ClearBodies, "Remove all dynamic bodies (for episode reset).")
        .def("get_renderer", &Engine::GetRenderer, py::return_value_policy::reference)
        .def("is_headless", &Engine::IsHeadless, "Check if engine is running in headless mode.");
}