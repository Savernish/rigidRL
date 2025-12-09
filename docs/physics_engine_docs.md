# Physics Engine Core: Implementation Status

## Overview
We have successfully transitioned from a tensor-only library to a **Differentiable Physics Engine** foundation. The first component, the **Rigid Body System**, is now active.

## 1. The Rigid Body Class (`Body`)
Located in:
- `include/body.h` (Definition)
- `src/body.cpp` (Implementation)

The `Body` class represents a physical object in the simulation. Unlike traditional engines (Box2D), every property is a **Differentiable Tensor**.

### Structure
| Member | Type | Dimensions | Gradient Support |
| :--- | :--- | :--- | :--- |
| `pos` | `Tensor` | (2, 1) | **Yes** (Flows back to `step`) |
| `vel` | `Tensor` | (2, 1) | **Yes** |
| `mass` | `Tensor` | (1, 1) | No (Static, but adaptable) |
| `inertia` | `Tensor` | (1, 1) | No |
| `shapes` | `std::vector<Shape>`| N/A | For Collision/Rendering |

### Integration Step (`Body::step`)
We use a **Semi-Implicit Euler** integrator implemented purely with differentiable operators:
1.  **Inverse Properties**: `inv_mass = 1.0 / mass`
2.  **Linear Acceleration**: `acc = forces * inv_mass` (Broadcasting support added)
3.  **Velocity Update**: `vel = vel + acc * dt`
4.  **Position Update**: `pos = pos + vel * dt`
This ensures that if we differentiate the final `pos` w.r.t `forces`, we get valid gradients.

## 2. Core Autograd Upgrades
To support the physics logic, we upgraded the core `forgeNN` engine:

### A. Scalar Broadcasting (`src/core.cpp`)
Implemented automatic broadcasting for `operator*` and `operator/`.
- **Before**: `Tensor(2,1) * Tensor(1,1)` would crash.
- **After**: Automatic broadcasting allows `Force (2,1) / Mass (1,1)` to work naturally.
- **Gradients**: The backward pass correctly sums gradients for the scalar term.

### B. Const-Correctness
- Refactored all operators (`+`, `-`, `*`, `/`) to be `const`-correct, allowing them to be used safely inside class methods without mutable hacks.

### C. Constructor Strictness
- Eliminated implicit C++ initializer lists that caused narrowing/ambiguity errors. All tensors inside `Body` are initialized via explicit `std::vector<float>` constructors for maximum stability.

## 3. Verification
We created `examples/falling_box.py` to validate the physics.
- **Scenario**: Box (m=1.0) dropped from y=10.0 under gravity (-9.81).
- **Result**: Position updates correctly match Euler integration math.
- **Stability**: No segmentation faults or memory leaks observed.

## Next Steps
With the physics core operational and SDL2 installed (v2.30.0 detected), we are ready to implement:
1.  **The Rendering System (`Renderer`)**: A C++ class wrapping SDL2 to draw `Body` shapes.
2.  **The Engine Loop**: A C++ main loop integrating `Body::step` and `Renderer::draw`.
