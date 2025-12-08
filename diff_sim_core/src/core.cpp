#include "core.h"
#include <iostream>
#include <unordered_set>

Tensor::Tensor(int rows, int cols, bool req_grad) {
    data.resize(rows, cols);
    data.setZero();
    set_requires_grad(req_grad);
}

Tensor::Tensor(int size, bool req_grad) {
    data.resize(size, 1);
    data.setZero();
    set_requires_grad(req_grad);
}

Tensor::Tensor(std::vector<float> data_list, bool req_grad) {
    data.resize(data_list.size(), 1);
    for (size_t i = 0; i < data_list.size(); ++i) {
        data(i, 0) = data_list[i];
    }
    set_requires_grad(req_grad);
}

void Tensor::set(int r, int c, float value) {
    if (r >= 0 && r < data.rows() && c >= 0 && c < data.cols()) {
        data(r, c) = value;
    }
}

float Tensor::get(int r, int c) const {
    if (r >= 0 && r < data.rows() && c >= 0 && c < data.cols()) {
        return data(r, c);
    }
    return 0.0f;
}

void Tensor::set_requires_grad(bool req) {
    this->requires_grad = req;
    if (req && grad.size() == 0) {
        grad.resizeLike(data);
        grad.setZero();
    }
}

void Tensor::zero_grad() {
    if (grad.size() > 0) {
        grad.setZero();
    }
}

void Tensor::backward() {
    if (!requires_grad) {
        std::cerr << "Warning: called backward() on a Tensor that does not require grad." << std::endl;
        return;
    }

    if (grad.size() == 0) {
        grad.resizeLike(data);
    }
    grad.setOnes();

    // Iterative Topological Sort to avoid Stack Overflow
    std::vector<Tensor*> topo;
    std::unordered_set<Tensor*> visited;
    std::vector<Tensor*> stack;
    
    std::unordered_set<Tensor*> expanded;
    stack.push_back(this);
    
    while (!stack.empty()) {
        Tensor* node = stack.back();
        
        if (visited.count(node)) {
            stack.pop_back();
            continue;
        }
        
        if (expanded.count(node)) {
            visited.insert(node);
            topo.push_back(node);
            stack.pop_back();
        } else {
            expanded.insert(node);
            for (Tensor* child : node->children) {
                if (visited.find(child) == visited.end()) {
                    stack.push_back(child);
                }
            }
        }
    }
    

    // Backward Pass
    int count = 0;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        Tensor* node = *it;
        if (node->backward_fn) {
             // count++;
            node->backward_fn(*node);
        }
    }
} // Backward Pass


int Tensor::rows() const { return data.rows(); }
int Tensor::cols() const { return data.cols(); }
float* Tensor::data_ptr() { return data.data(); }

// Reductions

// Sum (Scalar) - Already exists
Tensor Tensor::sum() {
    Tensor result(1, 1, false); // Scalar result
    result.data(0, 0) = this->data.sum();
    if (this->requires_grad) {
        result.set_requires_grad(true);
        result.grad.resizeLike(result.data);
        result.grad.setZero();
        
        result.children.push_back(this);
        
        result.backward_fn = [this](Tensor& self) {
            if (this->requires_grad) {
                // Gradient of sum is 1 for all elements
                float grad_val = self.grad(0, 0); 
                this->grad.array() += grad_val;
            }
        };
    }
    return result;
}

// Mean (Scalar) - Already exists


// Min (Scalar)
Tensor Tensor::min() {
    Tensor result(1, 1, false);
    Eigen::Index r, c;
    float val = this->data.minCoeff(&r, &c);
    result.data(0,0) = val;

    if (this->requires_grad) {
        result.set_requires_grad(true);
        result.grad.setZero();
        result.children.push_back(this);
        // Only the min element gets gradient 1
        result.backward_fn = [this, r, c](Tensor& self) {
             if (this->requires_grad) {
                 this->grad(r, c) += self.grad(0,0);
             }
        };
    }
    return result;
}

// Max (Scalar)
Tensor Tensor::max() {
    Tensor result(1, 1, false);
    Eigen::Index r, c;
    float val = this->data.maxCoeff(&r, &c);
    result.data(0,0) = val;

    if (this->requires_grad) {
        result.set_requires_grad(true);
        result.grad.setZero();
        result.children.push_back(this);
        result.backward_fn = [this, r, c](Tensor& self) {
             if (this->requires_grad) {
                 this->grad(r, c) += self.grad(0,0);
             }
        };
    }
    return result;
}

// Sum (Axis)
Tensor Tensor::sum(int axis) {
    if (axis != 0 && axis != 1) throw std::runtime_error("Axis must be 0 or 1");

    Tensor result(0,0);
    if (axis == 0) {
        result = Tensor(1, data.cols(), false);
        result.data = data.colwise().sum();
    } else {
        result = Tensor(data.rows(), 1, false);
        result.data = data.rowwise().sum();
    }

    if (this->requires_grad) {
        result.set_requires_grad(true);
        result.grad.setZero();
        result.children.push_back(this);
        result.backward_fn = [this, axis](Tensor& self) {
            if (this->requires_grad) {
                // Broadcast gradient back using replicate
                if (axis == 0) {
                    // self.grad is (1, cols). Replicate to (rows, cols)
                    this->grad.array() += self.grad.array().row(0).replicate(this->rows(), 1);
                } else {
                    // self.grad is (rows, 1). Replicate to (rows, cols)
                    this->grad.array() += self.grad.array().col(0).replicate(1, this->cols());
                }
            }
        };
    }
    return result;
}

// Mean (Axis)
Tensor Tensor::mean(int axis) {
     if (axis != 0 && axis != 1) throw std::runtime_error("Axis must be 0 or 1");
    Tensor result(0,0);
    if (axis == 0) {
        result = Tensor(1, data.cols(), false);
        result.data = data.colwise().mean();
    } else {
        result = Tensor(data.rows(), 1, false);
        result.data = data.rowwise().mean();
    }

    if (this->requires_grad) {
        result.set_requires_grad(true);
        result.grad.setZero();
        result.children.push_back(this);
        result.backward_fn = [this, axis](Tensor& self) {
            if (this->requires_grad) {
                float n = (axis == 0) ? (float)this->rows() : (float)this->cols();
                if (axis == 0) {
                     this->grad.array() += (self.grad.array().row(0) / n).replicate(this->rows(), 1);
                } else {
                     this->grad.array() += (self.grad.array().col(0) / n).replicate(1, this->cols());
                }
            }
        };
    }
    return result;
}



Tensor Tensor::sin() {
    Tensor result(data.rows(), data.cols(), false);
    result.data = data.array().sin().matrix();

    if (this->requires_grad) {
        result.set_requires_grad(true);
        result.grad.resizeLike(result.data);
        result.grad.setZero();
        
        result.children.push_back(this);

        result.backward_fn = [this](Tensor& self) {
            if (this->requires_grad) {
                // dL/dx += dL/dy * cos(x)
                this->grad.array() += self.grad.array() * this->data.array().cos();
            }
        };
    }
    return result;
}

Tensor Tensor::cos() {
    Tensor result(data.rows(), data.cols(), false);
    result.data = data.array().cos().matrix();

    if (this->requires_grad) {
        result.set_requires_grad(true);
        result.grad.resizeLike(result.data);
        result.grad.setZero();
        
        result.children.push_back(this);

        result.backward_fn = [this](Tensor& self) {
            if (this->requires_grad) {
                // dL/dx += dL/dy * -sin(x)
                this->grad.array() -= self.grad.array() * this->data.array().sin();
            }
        };
    }
    return result;
}

Tensor Tensor::mean() {
    Tensor result(1, 1, false);
    result.data(0, 0) = this->data.mean();
    if (this->requires_grad) {
        result.set_requires_grad(true);
        result.grad.resizeLike(result.data);
        result.grad.setZero();
        
        result.children.push_back(this);
        
        result.backward_fn = [this](Tensor& self) {
            if (this->requires_grad) {
                this->grad.array() += self.grad.array() / this->data.rows();
            }
        };
    }
    return result;
}

Tensor Tensor::select(int idx) {
    Tensor result(1, 1, false);
    // Boundary check? For performance we skip, but for safety:
    if (idx < 0 || idx >= data.size()) {
        std::cerr << "Error: Index " << idx << " out of bounds for Tensor size " << data.size() << std::endl;
        return result; 
    }
    result.data(0,0) = this->data(idx);
    
    if (this->requires_grad) {
        result.set_requires_grad(true);
        result.grad.setZero();
        result.children.push_back(this);
        
        // Capture 'idx' and 'this'
        result.backward_fn = [this, idx](Tensor& self) {
            if (this->requires_grad) {
                // Accumulate gradient into the specific index
                this->grad(idx) += self.grad(0,0);
            }
        };
    }
    return result;
}

Tensor Tensor::stack(const std::vector<Tensor*>& tensors) {
    int n = tensors.size();
    if (n == 0) return Tensor(0, 1);
    
    Tensor result(n, 1, false); 
    bool any_grad = false;
    for(auto* t : tensors) {
        if(t->get_requires_grad()) any_grad = true;
    }
    
    if (any_grad) {
        result.set_requires_grad(true);
        result.grad.setZero();
        
        for(int i=0; i<n; ++i) {
            result.data(i, 0) = tensors[i]->data(0,0); // Assume scalars
            if(tensors[i]->get_requires_grad()) {
                result.children.push_back(tensors[i]);
            }
        }
        
        // We need to know WHICH child corresponds to WHICH index.
        // But 'children' vector order matches 'tensors' order ONLY if all require grad.
        // Better approach: Capture the vector of pointers in the closure.
        // We trust the pointers remain valid (Python keeps them alive).
        
        std::vector<Tensor*> inputs = tensors; // Copy vector of pointers
        
        result.backward_fn = [inputs](Tensor& self) {
            for(size_t i=0; i<inputs.size(); ++i) {
                Tensor* inp = inputs[i];
                if(inp->get_requires_grad()) {
                    inp->grad(0,0) += self.grad(i,0);
                }
            }
        };
    } else {
         for(int i=0; i<n; ++i) {
            result.data(i, 0) = tensors[i]->data(0,0);
        }
    }
    return result;
}


// ---------------- Operators ----------------

Tensor Tensor::operator+(const Tensor& other) {
    Tensor result(data.rows(), data.cols(), false);
    result.data = this->data + other.data;

    if (this->requires_grad || other.requires_grad) {
        result.set_requires_grad(true);
        result.grad.resizeLike(result.data);
        result.grad.setZero();
        
        result.children.push_back(const_cast<Tensor*>(this));
        result.children.push_back(const_cast<Tensor*>(&other));

        result.backward_fn = [this, &other](Tensor& self) {
            if (this->requires_grad) {
                const_cast<Tensor*>(this)->grad += self.grad;
            }
            if (other.requires_grad) {
                const_cast<Tensor*>(&other)->grad += self.grad;
            }
        };
    }
    return result;
}

Tensor Tensor::operator-(const Tensor& other) {
    Tensor result(data.rows(), data.cols(), false);
    result.data = this->data - other.data;

    if (this->requires_grad || other.requires_grad) {
        result.set_requires_grad(true);
        result.grad.resizeLike(result.data);
        result.grad.setZero();
        
        result.children.push_back(const_cast<Tensor*>(this));
        result.children.push_back(const_cast<Tensor*>(&other));

        result.backward_fn = [this, &other](Tensor& self) {
            if (this->requires_grad) {
                const_cast<Tensor*>(this)->grad += self.grad;
            }
            if (other.requires_grad) {
                const_cast<Tensor*>(&other)->grad -= self.grad;
            }
        };
    }
    return result;
}

Tensor Tensor::operator*(const Tensor& other) {
    Tensor result(data.rows(), data.cols(), false);
    result.data = (this->data.array() * other.data.array()).matrix();

    if (this->requires_grad || other.requires_grad) {
        result.set_requires_grad(true);
        result.grad.resizeLike(result.data);
        result.grad.setZero();

        result.children.push_back(const_cast<Tensor*>(this));
        result.children.push_back(const_cast<Tensor*>(&other));

        result.backward_fn = [this, &other](Tensor& self) {
            if (this->requires_grad) {
                const_cast<Tensor*>(this)->grad.array() += self.grad.array() * other.data.array();
            }
            if (other.requires_grad) {
                const_cast<Tensor*>(&other)->grad.array() += self.grad.array() * this->data.array();
            }
        };
    }
    return result;
}

Tensor Tensor::operator/(const Tensor& other) {
    Tensor result(data.rows(), data.cols(), false);
    result.data = (this->data.array() / other.data.array()).matrix();

    if (this->requires_grad || other.requires_grad) {
        result.set_requires_grad(true);
        result.grad.resizeLike(result.data);
        result.grad.setZero();

        result.children.push_back(const_cast<Tensor*>(this));
        result.children.push_back(const_cast<Tensor*>(&other));

        result.backward_fn = [this, &other](Tensor& self) {
            if (this->requires_grad) {
                const_cast<Tensor*>(this)->grad.array() += self.grad.array() / other.data.array();
            }
            if (other.requires_grad) {
                const_cast<Tensor*>(&other)->grad.array() -= self.grad.array() * this->data.array() / (other.data.array().square());
            }
        };
    }
    return result;
}

Tensor Tensor::operator*(float scalar) {
    Tensor result(data.rows(), data.cols(), false);
    result.data = this->data * scalar;

    if (this->requires_grad) {
        result.set_requires_grad(true);
        result.grad.resizeLike(result.data);
        result.grad.setZero();
        
        result.children.push_back(const_cast<Tensor*>(this));

        result.backward_fn = [this, scalar](Tensor& self) {
            if (this->requires_grad) {
                // dL/dx += dL/dz * scalar
                const_cast<Tensor*>(this)->grad.array() += self.grad.array() * scalar;
            }
        };
    }
    return result;
}

// Mathematical Functions


// Transpose
Tensor Tensor::transpose() {
    Tensor result(this->data.cols(), this->data.rows(), false); // Swapped dims
    result.data = this->data.transpose();
    if (this->requires_grad) {
        result.set_requires_grad(true);
        result.grad.setZero();
        result.children.push_back(this);
        result.backward_fn = [this](Tensor& self) {
            if (this->requires_grad) {
                this->grad += self.grad.transpose();
            }
        };
    }
    return result;
}

// Power
Tensor Tensor::pow(float exponent) {
    Tensor result(this->data.rows(), this->data.cols(), false);
    result.data = this->data.array().pow(exponent);
    
    if (this->requires_grad) {
        result.set_requires_grad(true);
        result.grad.setZero();
        result.children.push_back(this);
        
        result.backward_fn = [this, exponent](Tensor& self) {
            if (this->requires_grad) {
                // dy/dx = p * x^(p-1) * grad_out
                this->grad.array() += exponent * this->data.array().pow(exponent - 1.0f) * self.grad.array();
            }
        };
    }
    return result;
}

// Exp
Tensor Tensor::exp() {
    Tensor result(this->data.rows(), this->data.cols(), false);
    result.data = this->data.array().exp();
    if (this->requires_grad) {
        result.set_requires_grad(true);
        result.grad.setZero();
        result.children.push_back(this);
        result.backward_fn = [this](Tensor& self) {
            if (this->requires_grad) {
                // dy/dx = exp(x) * grad = y * grad
                this->grad.array() += self.data.array() * self.grad.array(); 
            }
        };
    }
    return result;
}

// Log
Tensor Tensor::log() {
    Tensor result(this->data.rows(), this->data.cols(), false);
    // Safety: we assume x > 0.
    result.data = this->data.array().log();
    if (this->requires_grad) {
        result.set_requires_grad(true);
        result.grad.setZero();
        result.children.push_back(this);
        result.backward_fn = [this](Tensor& self) {
            if (this->requires_grad) {
                // dy/dx = 1/x * grad
                this->grad.array() += self.grad.array() / this->data.array();
            }
        };
    }
    return result;
}

// Sqrt
Tensor Tensor::sqrt() {
    Tensor result(this->data.rows(), this->data.cols(), false);
    result.data = this->data.array().sqrt();
    if (this->requires_grad) {
        result.set_requires_grad(true);
        result.grad.setZero();
        result.children.push_back(this);
        result.backward_fn = [this](Tensor& self) {
            if (this->requires_grad) {
                // dy/dx = 1/(2*sqrt(x)) * grad = 1/(2y) * grad
                this->grad.array() += 0.5f * self.grad.array() / self.data.array();
            }
        };
    }
    return result;
}

// Abs
Tensor Tensor::abs() {
    Tensor result(this->data.rows(), this->data.cols(), false);
    result.data = this->data.array().abs();
    if (this->requires_grad) {
        result.set_requires_grad(true);
        result.grad.setZero();
        result.children.push_back(this);
        result.backward_fn = [this](Tensor& self) {
            if (this->requires_grad) {
                // dy/dx = sign(x) * grad
                this->grad.array() += self.grad.array() * this->data.array().sign();
            }
        };
    }
    return result;
}

// Clamp
Tensor Tensor::clamp(float min_val, float max_val) {
    Tensor result(this->data.rows(), this->data.cols(), false);
    result.data = this->data.cwiseMax(min_val).cwiseMin(max_val);
    if (this->requires_grad) {
        result.set_requires_grad(true);
        result.grad.setZero();
        result.children.push_back(this);
        result.backward_fn = [this, min_val, max_val](Tensor& self) {
            if (this->requires_grad) {
                // dy/dx = 1 if min < x < max else 0
                // Use logic mask on 'this->data'
                Eigen::ArrayXXf x = this->data.array();
                // (x >= min && x <= max) -> 1
                // Eigen doesn't verify range simply, so:
                this->grad.array() += self.grad.array() * (x >= min_val && x <= max_val).cast<float>();
            }
        };
    }
    return result;
}


// Reshape
Tensor Tensor::reshape(int r, int c) {
    if (r * c != this->data.size()) {
         std::cerr << "Error: Reshape size mismatch. Total " << this->data.size() << " requested " << r << "x" << c << std::endl;
         return Tensor(1,1); // Error
    }
    
    Tensor result(r, c, false);
    result.data = Eigen::Map<Eigen::MatrixXf>(this->data.data(), r, c); // Copy with reshape
    
    if (this->requires_grad) {
        result.set_requires_grad(true);
        result.grad.setZero();
        result.children.push_back(this);
        
        result.backward_fn = [this](Tensor& self) {
            if (this->requires_grad) {
                // Flatten and add gradients
                Eigen::Map<Eigen::VectorXf> flat_grad(this->grad.data(), this->grad.size());
                Eigen::Map<Eigen::VectorXf> flat_self(self.grad.data(), self.grad.size());
                flat_grad += flat_self;
            }
        };
    }
    return result;
}

// Concatenate
Tensor Tensor::cat(const std::vector<Tensor*>& tensors, int dim) {
    if (tensors.empty()) return Tensor(0, 0);

    int rows = tensors[0]->data.rows();
    int cols = tensors[0]->data.cols();
    
    // Calculate total size and validate
    int total_rows = 0;
    int total_cols = 0;
    
    if (dim == 0) {
        total_cols = cols;
        for (const auto* t : tensors) {
            if (t->data.cols() != cols) throw std::runtime_error("Dimension mismatch in cat(dim=0)");
            total_rows += t->data.rows();
        }
    } else if (dim == 1) {
        total_rows = rows;
        for (const auto* t : tensors) {
            if (t->data.rows() != rows) throw std::runtime_error("Dimension mismatch in cat(dim=1)");
            total_cols += t->data.cols();
        }
    } else {
        throw std::runtime_error("Invalid dimension for cat (only 0 or 1 supported)");
    }

    Tensor result(total_rows, total_cols, false);
    
    // Forward Copy
    int current_offset = 0;
    for (const auto* t : tensors) {
        if (dim == 0) {
            result.data.block(current_offset, 0, t->data.rows(), cols) = t->data;
            current_offset += t->data.rows();
        } else {
            result.data.block(0, current_offset, rows, t->data.cols()) = t->data;
            current_offset += t->data.cols();
        }
        
        if (t->get_requires_grad()) {
            result.set_requires_grad(true);
            result.children.push_back(const_cast<Tensor*>(t));
        }
    }
    
    if (result.get_requires_grad()) {
        result.grad.setZero();
        
        // Capture inputs to know sizes for backward slicing
        std::vector<Tensor*> inputs = tensors;
        
        result.backward_fn = [inputs, dim](Tensor& self) {
            int offset = 0;
            for (Tensor* t : inputs) {
                int r = t->data.rows();
                int c = t->data.cols();
                
                if (t->get_requires_grad()) {
                    if (dim == 0) {
                        t->grad += self.grad.block(offset, 0, r, c);
                    } else {
                        t->grad += self.grad.block(0, offset, r, c);
                    }
                }
                
                if (dim == 0) offset += r;
                else offset += c;
            }
        };
    }
    
    return result;
}


// Matrix Multiplication
Tensor Tensor::matmul(const Tensor& other) {
    if (this->data.cols() != other.data.rows()) {
        throw std::runtime_error("Shape mismatch for matmul");
    }
    Tensor result(this->data.rows(), other.data.cols(), false);
    result.data = this->data * other.data;
    if (this->requires_grad || other.requires_grad) {
        result.set_requires_grad(true);
        result.grad.setZero();
        if (this->requires_grad) result.children.push_back((Tensor*)this);
        if (other.requires_grad) result.children.push_back((Tensor*)&other);
        Tensor* A = (Tensor*)this;
        Tensor* B = (Tensor*)&other;
        result.backward_fn = [A, B](Tensor& self) {
            if (A->get_requires_grad()) {
                A->grad += self.grad * B->data.transpose();
            }
            if (B->get_requires_grad()) {
                B->grad += A->data.transpose() * self.grad;
            }
        };
    }
    return result;
}


// Accessors
Eigen::MatrixXf Tensor::get_data() const { return data; }
void Tensor::set_data(const Eigen::MatrixXf& d) { data = d; }

Eigen::MatrixXf Tensor::get_grad() const { return grad; }
void Tensor::set_grad(const Eigen::MatrixXf& g) { grad = g; }

bool Tensor::get_requires_grad() const { return requires_grad; }
