#include "activations.h"
#include <iostream>

Tensor relu(const Tensor& input) {
    Tensor result(input.get_data().rows(), input.get_data().cols(), false);
    result.set_data(input.get_data().cwiseMax(0.0f));

    if (input.get_requires_grad()) {
        result.set_requires_grad(true);
        result.grad.setZero();
        
        // We need to store 'input' to compute mask during backward. Be careful with const reference.
        // Similar to matmul, we cast const away to store logic pointer.
        Tensor* inp = (Tensor*)&input;
        result.children.push_back(inp);

        result.backward_fn = [inp](Tensor& self) {
            if (inp->get_requires_grad()) {
                Eigen::MatrixXf mask = (inp->get_data().array() > 0.0f).cast<float>();
                inp->grad.array() += mask.array() * self.grad.array();
            }
        };
    }
    return result;
}

Tensor tanh(const Tensor& input) {
    Tensor result(input.get_data().rows(), input.get_data().cols(), false);
    result.set_data(input.get_data().array().tanh());

    if (input.get_requires_grad()) {
        result.set_requires_grad(true);
        result.grad.setZero();
        
        Tensor* inp = (Tensor*)&input;
        result.children.push_back(inp);

        // Store result data for efficient backward (1 - y^2) if we wanted optimize, 
        // but here we might recompute tanh or simpler: use output?
        // Using output 'self.data' is fine.
        
        result.backward_fn = [](Tensor& self) {
            // Wait, we need to push to child. We need child pointer.
            // Oh right, backward_fn captures child pointer 'inp'.
            // Can we use 'self.data' (which is tanh(x)) to compute deriv?
            // d(tanh(x))/dx = 1 - tanh^2(x) = 1 - y^2.
            // Yes!
            // But we need to write to inp->grad.
            // We need 'inp' in capture.
        };
        
        // Proper closure:
        result.backward_fn = [inp](Tensor& self) {
             if (inp->get_requires_grad()) {
                 // dy/dx = 1 - y^2
                 // y is self.data
                 Eigen::MatrixXf deriv = 1.0f - self.get_data().array().square();
                 inp->grad.array() += deriv.array() * self.grad.array();
             }
        };
    }
    return result;
}
