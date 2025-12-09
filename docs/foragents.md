# Agent Handover Documentation: Differentiable Physics Engine

**Target Audience:** AI Agents (and humans) picking up this project.

**Current Date:** December 2025

**Core Objective:** Build a high-performance, Differentiable Physics Engine in C++ with Python bindings, specifically designed for Reinforcement Learning (RL) robotics.

---

## 1. Project Architecture

The project is a hybrid C++/Python system.

- **C++ Core (`diff_sim_core/`)**: Handles heavy computation (Physics, Autograd, Rendering).

- **Python Interface (`forgeNN_cpp` module)**: Users define networks and simulation loops here.

### Directory Structure

```
diff_sim/
├── diff_sim_core/
│   ├── include/
│   │   ├── engine/          # Core Physics & Math (Tensor, Body, Activations)
│   │   ├── renderer/        # Rendering Interface & Implementations
│   │   └── ...
│   ├── src/
│   │   ├── engine/          # Implementation of Physics/Math
│   │   │   ├── tensor.cpp   # THE AUTOGRAD ENGINE (formerly core.cpp)
│   │   │   ├── body.cpp     # Rigid Body Physics
│   │   │   └── ...
│   │   ├── renderer/        # Rendering Implementation
│   │   │   └── sdl_renderer.cpp # SDL2 backend
│   │   └── bindings.cpp     # Pybind11 definitions (The Bridge)
│   └── CMakeLists.txt       # Cross-platform Build System (Linux/Windows)
└── examples/                # Python scripts to test/verify features
```

---

## 2. Core Technical Components

### A. The Autograd Engine (`src/engine/tensor.cpp`)

*   **Custom Tensor Library**: Built on `Eigen::MatrixXf`.

*   **Graph Construction**: Dynamic computation graph. Every operation on a tensor with `requires_grad=true` creates a node.

*   **CRITICAL Implementation Details**:

    *   **Iterative Backward Pass**: We replaced recursive DFS with an **Iterative Topological Sort** to prevent stack overflows in long simulations (1000+ steps). Do NOT revert to recursion.

    *   **Broadcasting**: `operator*` and `operator/` support scalar broadcasting (Vector * Scalar). The backward pass manually handles gradient summation for the scalar term.

    *   **Const-Correctness**: All operators are `const`-qualified to work with reference logic.

### B. Rigid Body Physics (`src/engine/body.cpp`)

*   **Differentiable State**: Position (`pos`), Velocity (`vel`), Rotation (`rot`) are Tensors.

*   **Integration**: Uses **Semi-Implicit Euler**.

    *   `acc = force / mass`
    *   `vel += acc * dt`
    *   `pos += vel * dt`

*   **Why implementation matters**: We use `inv_mass = 1.0 / mass` explicitly to generate a graph compatible with our scalar broadcasting.

### C. Rendering (`src/renderer/`)

*   **Decoupled Architecture**: `Renderer` is an abstract base class (`include/renderer/renderer.h`).

*   **Implementation**: `SDLRenderer` uses SDL2.

*   **Bindings**: Exposed to Python so the user can control the loop: `renderer.process_events()`, `renderer.draw_box()`.

### D. Python Bindings (`src/bindings.cpp`)

*   **Memory Management**: Uses `py::keep_alive<0, 1>()` extensively. **DO NOT REMOVE**. This keeps C++ objects alive as long as their Python wrappers or dependent tensors exist. Without this, the graph segfaults during backward passes.

---

## 3. Build System & Platform Support

The project supports both **Linux** and **Windows**.

*   **Linux**: Uses `pkg-config` for SDL2.

*   **Windows**: Uses `vcpkg` (in Config mode) for SDL2, Eigen3, and Pybind11. Handles MSVC flags (`/O2`). We provide `compile.bat` for easy building.

*   **CMake**: The `CMakeLists.txt` detects the platform (`if(WIN32)`) and adjusts linking strategy accordingly.

---

## 4. The Roadmap (Status: Phase 2 - Rendering)

### ✅ Completed

1.  **Tensor Autograd**: Functioning, optimized, tested.

2.  **Rigid Body Core**: `Body` class with differentiable physics step.

3.  **Visuals (Foundation)**: SDL2 Renderer implementation and Python bindings.

4.  **Refactoring**: Engine/Renderer separation.

5.  **Rendering Loop Integration**: `Engine` class implemented with automated body/renderer management and python-friendly loop.

### 🚧 Immediate Next Steps (You are here)

1.  **Differentiable Collision**: This is the "Holy Grail". We need a way to calculate collision forces/penalties that allows gradients to flow.

    *   *Idea*: Soft-body or Penalty-based methods (Spring-Damper forces upon intersection).

### 🔮 Future Goals

1.  **Joints & Constraints**: Making `Body` parts connect (Legs attached to Body). To do this differentiably, we likely need a constraint solver or just stiff springs.

2.  **3D Rendering**: Switching the `Renderer` implementation to use OpenGL/Vulkan (while keeping the Interface).

3.  **URDF Loading**: Parsing robot descriptions.

---

## 5. How to Continue

1.  **Check `task.md`**: For granular tasks.

2.  **Check `examples/`**:

    *   `falling_box.py`: Verifies physics math (headless).
    *   `test_engine_loop.py`: Demonstrates the new Engine Loop.

3.  **Maintain Hygiene**:

    *   Always update `CMakeLists.txt` when adding files.
    *   Always verify builds on Windows (mentally check MSVC compatibility).
    *   Keep `bindings.cpp` updated with new classes.
    *   Use `compile.bat` or `pip install -e .` to rebuild.

**Good Luck, Agent.**
