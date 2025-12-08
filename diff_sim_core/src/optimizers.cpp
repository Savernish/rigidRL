#include "optimizers.h"
#include <iostream>

Optimizer::Optimizer(std::vector<Tensor*> params, float lr)
    : parameters(params), learning_rate(lr) {}

Optimizer::~Optimizer() {}

void Optimizer::zero_grad() {
    for (Tensor* p : parameters) {
        p->zero_grad();
    }
}

SGD::SGD(std::vector<Tensor*> params, float lr)
    : Optimizer(params, lr) {}

void SGD::step() {
    for (Tensor* p : parameters) {
        if (!p->get_requires_grad()) continue;
        
        // Ensure gradient exists
        if (p->grad.size() == 0) continue;

        // Basic SGD: p = p - lr * grad
        p->data -= learning_rate * p->grad;
    }
}

// ---------------- Adam ----------------

Adam::Adam(std::vector<Tensor*> params, float lr, float beta1, float beta2, float epsilon)
    : Optimizer(params, lr), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0) {
    
    // Initialize m and v with zeros matching param shapes
    for (Tensor* p : params) {
        m.push_back(Eigen::MatrixXf::Zero(p->rows(), p->cols()));
        v.push_back(Eigen::MatrixXf::Zero(p->rows(), p->cols()));
    }
}

void Adam::step() {
    t++;
    for (size_t i = 0; i < parameters.size(); ++i) {
        Tensor* p = parameters[i];
        if (!p->get_requires_grad() || p->grad.size() == 0) continue;

        // Current Gradient
        const Eigen::MatrixXf& g = p->grad;

        // Update biased first moment estimate
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;

        // Update biased second raw moment estimate
        v[i] = beta2 * v[i] + (1.0f - beta2) * g.array().square().matrix();

        // Compute bias-corrected first moment estimate
        Eigen::MatrixXf m_hat = m[i] / (1.0f - std::pow(beta1, t));

        // Compute bias-corrected second raw moment estimate
        Eigen::MatrixXf v_hat = v[i] / (1.0f - std::pow(beta2, t));

        // Update parameters
        // theta = theta - lr * m_hat / (sqrt(v_hat) + epsilon)
        p->data.array() -= learning_rate * m_hat.array() / (v_hat.array().sqrt() + epsilon);
    }
}


// ---------------- AdamW ----------------


AdamW::AdamW(std::vector<Tensor*> params, float lr, float beta1, float beta2, float epsilon, float weight_decay)
    : Optimizer(params, lr), beta1(beta1), beta2(beta2), epsilon(epsilon), weight_decay(weight_decay), t(0) {
    
    for (Tensor* p : params) {
        m.push_back(Eigen::MatrixXf::Zero(p->rows(), p->cols()));
        v.push_back(Eigen::MatrixXf::Zero(p->rows(), p->cols()));
    }
}
void AdamW::step() {
    t++;
    for (size_t i = 0; i < parameters.size(); ++i) {
        Tensor* p = parameters[i];
        if (!p->get_requires_grad() || p->grad.size() == 0) continue;
        // AdamW Decoupled Weight Decay
        if (weight_decay > 0) {
            p->data -= learning_rate * weight_decay * p->data;
        }
        const Eigen::MatrixXf& g = p->grad;
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g.array().square().matrix();
        Eigen::MatrixXf m_hat = m[i] / (1.0f - std::pow(beta1, t));
        Eigen::MatrixXf v_hat = v[i] / (1.0f - std::pow(beta2, t));
        p->data.array() -= learning_rate * m_hat.array() / (v_hat.array().sqrt() + epsilon);
    }
}

// ---------------- RMSProp ----------------

