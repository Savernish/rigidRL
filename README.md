# rigidRL

A 2D physics engine with Python bindings for reinforcement learning research.

## Overview

rigidRL provides a C++ physics simulation core with Python bindings, designed for training RL agents on control tasks. The engine supports rigid body dynamics, collision detection, and motor-based actuation.

Key features:
- 2D rigid body physics with box colliders
- Impulse-based collision response with friction
- Motor attachments for thrust-based control
- Real-time SDL2 visualization (optional headless mode)
- Gymnasium-compatible environment interface
- Works with standard RL libraries (stable-baselines3, etc.)

## Requirements

- Python 3.10+
- C++ compiler with C++17 support
- CMake 3.14+
- Eigen3 (linear algebra library)
- SDL2 (for visualization)

## Installation

### Windows

**Prerequisites:**
1. Install Python 3.10+ from https://python.org
2. Install Visual Studio Build Tools with "Desktop development with C++" workload
3. Eigen3 and SDL2 are included in `diff_sim_core/extern/`

**Build:**
```batch
git clone https://github.com/Savernish/forgeNN.git
cd forgeNN
compile.bat
```

### Ubuntu/Debian

**Prerequisites:**
```bash
sudo apt update
sudo apt install python3 python3-pip build-essential cmake libsdl2-dev
```

Note: Eigen is downloaded automatically during build.

**Build:**
```bash
git clone https://github.com/Savernish/forgeNN.git
cd forgeNN
chmod +x compile.sh
./compile.sh
```

### Manual Installation (Any Platform)

If the build scripts fail, install manually:
```bash
pip install -e .
```

For verbose output to debug issues:
```bash
pip install -e . -v
```

## Verification

After installation, verify the build:
```bash
python -c "import rigidRL; print('OK')"
python examples/train_drone_sb3.py --test
```

## Usage

### Direct Engine API

```python
import rigidRL as rigid

# Create engine: width, height, scale, dt, substeps
engine = rigid.Engine(800, 600, 50.0, 0.016, 20)
engine.set_gravity(0, -9.81)

# Add ground collider
engine.Collider(0, -1, 20, 1, 0)

# Create body with motors
drone = rigid.Body(0, 2, 1.0, 1.0, 0.2)
motor_left = rigid.Motor(-0.5, 0, 0.15, 0.1, 0.1, 10.0)
motor_right = rigid.Motor(0.5, 0, 0.15, 0.1, 0.1, 10.0)
drone.add_motor(motor_left)
drone.add_motor(motor_right)
engine.add_body(drone)

# Simulation loop
while engine.step():
    motor_left.thrust = 5.0
    motor_right.thrust = 5.0
```

### Gymnasium Environment

```python
from rigidrl_py.envs import DroneEnv

env = DroneEnv(headless=True)
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Training with Stable-Baselines3

```bash
pip install stable-baselines3
python examples/train_drone_sb3.py --train --timesteps 50000
python examples/train_drone_sb3.py --demo  # visualize trained policy
```

## API Reference

### Engine

| Method | Description |
|--------|-------------|
| `Engine(w, h, scale, dt, substeps, headless=False)` | Create engine |
| `add_body(body)` | Add dynamic body |
| `Collider(x, y, w, h, rot, friction=0.5)` | Add static collider |
| `set_gravity(x, y)` | Set gravity vector |
| `step()` | Run one frame (physics + render) |
| `update()` | Run physics only |
| `clear_bodies()` | Remove all dynamic bodies |
| `is_headless()` | Check if running without visualization |

### Body

| Method | Description |
|--------|-------------|
| `Body(x, y, mass, w, h)` | Create body |
| `add_motor(motor)` | Attach motor |
| `get_x()`, `get_y()`, `get_rotation()` | Get position/angle |
| `vel`, `ang_vel` | Velocity tensors |

### Motor

| Property | Description |
|----------|-------------|
| `Motor(local_x, local_y, w, h, mass, max_thrust)` | Create motor |
| `thrust` | Current thrust (0 to max_thrust) |
| `angle` | Thrust direction in radians |

## Troubleshooting

### Windows: "cl.exe not found"
Install Visual Studio Build Tools with C++ workload.

### Linux: "SDL.h: No such file"
```bash
sudo apt install libsdl2-dev
```

### Import fails after build
Ensure you run from the project root, not inside `diff_sim_core/`.

## Author

Enbiya Çabuk - https://savern.me
