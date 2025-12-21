"""
DroneEnv - 2D drone hovering environment

A Gymnasium-compatible environment for training a drone to hover at a target position.
The drone has configurable motors that apply upward thrust.
"""

import numpy as np
import os
from .base_env import RigidEnv
from .spaces import Box
from ..configs import EnvConfig, DroneConfig, MotorConfig

# Import rigidRL
import sys
core_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'diff_sim_core')
if os.path.exists(core_dir):
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(core_dir)
    sys.path.insert(0, core_dir)

try:
    import rigidRL as rigid
except ImportError:
    rigid = None


class DroneEnv(RigidEnv):
    """
    Drone hovering environment.
    
    Task: Control a 2D drone with motors to reach and maintain a target position.
    
    Observation Space (6D):
        [dx, dy, vx, vy, rotation, angular_velocity]
        - dx, dy: Relative position to target (meters)
        - vx, vy: Velocity (m/s)
        - rotation: Body angle (radians)
        - angular_velocity: Angular velocity (rad/s)
    
    Action Space (n motors):
        [motor_0_thrust, motor_1_thrust, ...]
        - Each in range [0, max_thrust]
    """
    
    def __init__(self, config: EnvConfig, render_mode=None):
        """
        Initialize drone environment.
        
        Args:
            config: EnvConfig object with drone, target, spawn settings
            render_mode: "human" for window, None for headless
        """
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=config.max_steps,
            width=config.window_width,
            height=config.window_height,
            scale=config.scale
        )
        
        self.config = config
        self.target = np.array(config.target, dtype=np.float32)
        self.spawn_points = config.spawn_points
        
        # Get max thrust from first motor (assume all same for action space)
        max_thrust = config.drone.motors[0].max_thrust if config.drone.motors else 10.0
        num_motors = len(config.drone.motors)
        
        # Observation space: [dx, dy, vx, vy, rotation, angular_velocity]
        self.observation_space = Box(
            low=np.array([-10, -10, -10, -10, -np.pi, -10], dtype=np.float32),
            high=np.array([10, 10, 10, 10, np.pi, 10], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: one thrust per motor
        self.action_space = Box(
            low=np.zeros(num_motors, dtype=np.float32),
            high=np.full(num_motors, max_thrust, dtype=np.float32),
            dtype=np.float32
        )
        
        # Drone components (set in _setup_scene)
        self.drone = None
        self.motors = []
        
    @classmethod
    def from_yaml(cls, path: str, render_mode=None) -> "DroneEnv":
        """Create environment from YAML config file."""
        config = EnvConfig.from_yaml(path)
        return cls(config=config, render_mode=render_mode)
    
    @classmethod
    def default(cls, render_mode=None) -> "DroneEnv":
        """Create environment with default config."""
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            "configs", "defaults", "drone.yaml"
        )
        return cls.from_yaml(default_path, render_mode=render_mode)
        
    def _setup_scene(self):
        """Create the drone simulation scene."""
        self.engine.set_gravity(0, -9.81)
        
        # Ground plane
        self.engine.Collider(0, -1, 20, 1, 0)
        
        # Randomly select spawn point
        spawn_idx = np.random.randint(0, len(self.spawn_points))
        spawn_x, spawn_y = self.spawn_points[spawn_idx]
        
        # Create drone body from config
        drone_cfg = self.config.drone
        self.drone = rigid.Body(spawn_x, spawn_y, drone_cfg.mass, drone_cfg.width, drone_cfg.height)
        
        # Add motors from config
        self.motors = []
        for motor_cfg in drone_cfg.motors:
            motor = rigid.Motor(
                motor_cfg.x, motor_cfg.y,
                motor_cfg.width, motor_cfg.height,
                motor_cfg.mass, motor_cfg.max_thrust
            )
            self.drone.add_motor(motor)
            self.motors.append(motor)
        
        self.engine.add_body(self.drone)
        
    def _get_obs(self) -> np.ndarray:
        """Get current observation with relative target position."""
        dx = self.target[0] - self.drone.get_x()
        dy = self.target[1] - self.drone.get_y()
        
        return np.array([
            dx,
            dy,
            self.drone.vel.get(0, 0),
            self.drone.vel.get(1, 0),
            self.drone.get_rotation(),
            self.drone.ang_vel.get(0, 0),
        ], dtype=np.float32)
        
    def _apply_action(self, action: np.ndarray):
        """Apply motor thrusts."""
        for i, motor in enumerate(self.motors):
            thrust = float(np.clip(action[i], 0, motor.max_thrust))
            motor.thrust = thrust
            
    def _compute_reward(self) -> float:
        """Compute reward based on distance to target."""
        dx = self.target[0] - self.drone.get_x()
        dy = self.target[1] - self.drone.get_y()
        dist = np.sqrt(dx*dx + dy*dy)
        
        # Exponential distance reward (stronger gradient as we get close)
        reward = np.exp(-dist)
        
        # Bonus for being very close
        if dist < 0.5:
            reward += 2.0
        if dist < 0.2:
            reward += 5.0
            
        # Crash penalty
        if self.drone.get_y() < 0.1:
            reward -= 10.0
            
        return float(reward)
        
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        # Crashed into ground
        if self.drone.get_y() < 0.1:
            return True
        # Flipped over
        if abs(self.drone.get_rotation()) > np.pi / 2:
            return True
        return False

