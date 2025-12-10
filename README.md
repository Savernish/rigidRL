# rigidRL

**High-Performance Differentiable Rigid Body Physics for Reinforcement Learning**

## Overview

rigidRL is a differentiable 2D physics engine built for RL robotics research. It features a C++17 core with automatic differentiation, real-time SDL2 rendering, and Python bindings for seamless integration with ML workflows.

## Features

- **Rigid Body Physics**: Multi-point collision detection, impulse-based response, friction
- **Differentiable Core**: Reverse-mode autograd for gradients through physics
- **Real-time Rendering**: SDL2 visualization with frame rate control
- **Python Interface**: Easy integration via `import rigidRL as rigid`
- **Optimizers**: SGD, Adam, AdamW built-in

## Installation

**Requirements:** CMake 3.10+, C++17 compiler, Python 3.x, Eigen3, SDL2, Pybind11

```bash
# Windows
.\compile.bat

# Linux/Mac
pip install -e .
```

## Quick Start

```python
import rigidRL as rigid

# Create engine (width, height, scale, dt, substeps)
engine = rigid.Engine(800, 600, 50, 0.016, 30)
engine.set_gravity(0, -9.81)

# Add static colliders (floor, walls, slopes)
engine.add_collider(0, -1, 20, 1, 0)  # Flat floor

# Add dynamic bodies (x, y, mass, width, height)
box = rigid.Body(0, 5, 1.0, 1, 1)
engine.add_body(box)

# Run simulation loop
while engine.step():
    pass  # Rendering & frame rate handled automatically
```

## Examples

| Example | Description |
|---------|-------------|
| `stress_test.py` | 28 boxes, pyramids, slopes, multi-body stacking |
| `rotation_test.py` | Box settling from initial rotation |
| `slope_test.py` | Sliding on angled surfaces |
| `impulse_test.py` | Collision response testing |
| `falling_box_visual.py` | Visual demo with varied box sizes |

### RL Examples (Differentiable)

| Example | Description |
|---------|-------------|
| `drone.py` | 6-DOF quadrotor landing optimization |
| `cartpole.py` | Classic control via gradient descent |
| `projectile.py` | Trajectory optimization |

![Drone Simulation](examples/drone.gif)
![Cartpole Simulation](examples/cartpole.gif)

## Physics API

```python
# Body properties
box.get_x(), box.get_y()      # Position
box.get_rotation()            # Angle (radians)
box.set_rotation(angle)       # Set initial rotation
box.friction                  # Friction coefficient (0-1)
box.restitution              # Bounciness (0-1)
box.is_static                # Static body flag

# Engine methods
engine.add_collider(x, y, w, h, rotation)  # Static geometry
engine.add_body(body)                       # Dynamic body
engine.set_gravity(x, y)                    # Gravity vector
engine.step()                               # Update + render
```

## Author

**Savern** - [https://savern.me](https://savern.me)
