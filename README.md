# forgeNN++

**High-Performance C++ Autograd Engine for Differentiable Physics**

## Overview

forgeNN++ is a C++17 port of the forgeNN deep learning framework. It replaces the NumPy-based autograd engine with a compiled, graph-based C++ core backed by Eigen. It exposes a Python interface via Pybind11, offering significant performance improvements for physics-heavy optimization loops and differential simulations.

## Key Features

- **C++17 Core**: Zero-overhead abstraction over Eigen3 matrix operations.
- **Reverse-Mode Autograd**: Dynamic computation graph with automatic differentiation.
- **Python Bindings**: Seamless integration with Python ecosystem (Matplotlib, NumPy).
- **Optimizers**: SGD, Adam, AdamW (with decoupled weight decay).
- **Differentiable Physics**: Production-hardened on 6-DOF Drone, Cartpole, and Projectile simulations.

## Performance

Benchmark comparison running a 6-DOF Drone Simulation (200 epochs of optimization):

| Implementation | Execution Time | Speedup |
| :--- | :--- | :--- |
| forgeNN (Python) | 15.25s | 1.0x |
| **forgeNN++ (C++)** | **1.29s** | **11.7x** |

## Installation

### Requirements

- CMake 3.10+
- C++17 Compiler (GCC/Clang)
- Python 3.x development headers
- Eigen3 (System or Submodule)
- Pybind11 (System or Submodule)

### Build

### Build
Run `compile.bat` on Windows or:

```bash
pip install -e .
```

This compiles the C++ core and installs the `forgeNN_cpp` Python module in development mode.
You can verify the build with `python examples/test_engine_loop.py`.

This compiles the C++ core and generates the `forgeNN_cpp` Python module in the current directory.

## Quick Start

```python
import forgeNN_cpp as rigid

# 1. Create Tensors
x = rigid.Tensor([2.0], requires_grad=True)
w = rigid.Tensor([3.0], requires_grad=True)

# 2. Forward Pass
y = x * w + x.sin()

# 3. Backward Pass
y.backward()

# 4. Access Gradients
# dy/dx = w + cos(x) = 3 + cos(2)
print(f"dy/dx: {x.grad.data[0]}") 
# dy/dw = x = 2
print(f"dy/dw: {w.grad.data[0]}") 
```

## Differentiable Simulation Examples

The `examples/` directory contains fully differentiable physics simulations powered by forgeNN++.

### 1. Drone Landing (6-DOF Quadrotor)
Optimizes motor thrusts to land a drone safely using a custom loss function with ground barriers.

![Drone Simulation](examples/drone.gif)

### 2. Cartpole Stabilization
Classic non-linear control problem solved via gradient descent through time.

![Cartpole Simulation](examples/cartpole.gif)

### 3. Projectile Optimization
Basic trajectory plotting and targeting.

### 3. Projectile Optimization
Basic trajectory plotting and targeting.

### 4. Engine Loop Demo
Demonstrates the new C++ Engine Loop class with automatic physics and rendering.

```bash
python examples/test_engine_loop.py
```

```bash
python examples/drone.py
```

## Author

**Savern** - [https://savern.me](https://savern.me)
