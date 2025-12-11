# rigidRL

**2-in-1 Physics + RL Framework for Robotics Research**

rigidRL is a differentiable 2D physics engine with built-in reinforcement learning, designed for training control policies on simulated robots. The project combines a C++17 physics core with automatic differentiation, real-time SDL2 rendering, and Python bindings for seamless integration with machine learning workflows.

The engine supports rigid body dynamics with multi-point collision detection, impulse-based response, and friction. Bodies can have motors attached for thrust-based control, making it suitable for simulating drones, rockets, and other actuated systems. The autograd system tracks computation graphs through physics steps, enabling gradient-based optimization and differentiable planning.

The built-in RL module implements policy gradient algorithms using the engine's own Tensor and optimizer classes. This means you can train control policies without any external dependencies like PyTorch or TensorFlow. The physics, autograd, and RL all run in one unified system.

## Quick Start

```bash
# Install (Windows)
.\compile.bat

# Install (Linux/Mac)
pip install -e .
```

```python
import rigidRL as rigid

# Create physics engine
engine = rigid.Engine(800, 600, 50, 0.016, 20)
engine.set_gravity(0, -9.81)
engine.add_collider(0, -1, 20, 1, 0)  # Ground

# Create drone with motors
drone = rigid.Body(0, 2, 1.0, 1.0, 0.2)
motor_left = rigid.Motor(-0.5, 0, 0.15, 0.1, 0.1, 10.0)
motor_right = rigid.Motor(0.5, 0, 0.15, 0.1, 0.1, 10.0)
drone.add_motor(motor_left)
drone.add_motor(motor_right)
engine.add_body(drone)

# Simulation loop
while engine.step():
    motor_left.thrust = 5.0  # Control motors
    motor_right.thrust = 5.0
```

## Examples

### Physics Demos
- `falling_box_visual.py` - Boxes falling and stacking
- `impulse_test.py` - Collision response on slopes
- `drone_liftoff.py` - Drone hover with altitude control
- `drone_unstable.py` - Asymmetric payload tipping

### RL Training
- `train_drone.py` - Train drone to reach target with REINFORCE
- `projectile.py` - Trajectory optimization via autograd
- `drone.py` / `cartpole.py` - Classic control examples

## API Reference

### Engine
```python
engine = rigid.Engine(width, height, scale, dt, substeps)
engine.set_gravity(x, y)
engine.add_body(body)
engine.add_collider(x, y, w, h, rotation)
engine.step()        # Physics + render + events
engine.update()      # Physics only
engine.clear_bodies()  # Reset for new episode
```

### Body
```python
body = rigid.Body(x, y, mass, width, height)
body.get_x(), body.get_y(), body.get_rotation()
body.vel, body.ang_vel  # Tensor properties
body.add_motor(motor)
```

### Motor
```python
motor = rigid.Motor(local_x, local_y, mass, width, height, max_thrust)
motor.thrust = 5.0  # Set current thrust
```

### Built-in RL
```python
from rigidrl_py.rl import REINFORCE

agent = REINFORCE(state_dim=6, action_dim=2, hidden_sizes=[64, 64], lr=0.003)
action = agent.select_action(state)
agent.store_reward(reward)
agent.update()  # Policy gradient update
```

## Author

**Savern** - [https://savern.me](https://savern.me)
