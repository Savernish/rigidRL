"""
DroneEnv - 2D drone hovering environment

A Gymnasium-compatible environment for training a drone to hover at a target position.
The drone has two motors (left/right) that can apply upward thrust.
"""

import numpy as np
from .base_env import RigidEnv
from .spaces import Box

# Import rigidRL
import sys
import os
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
    
    Task: Control a 2D drone with two motors to reach and maintain a target position.
    
    Observation Space (6D):
        [x, y, vx, vy, rotation, angular_velocity]
        - x, y: Position (meters)
        - vx, vy: Velocity (m/s)
        - rotation: Body angle (radians)
        - angular_velocity: Angular velocity (rad/s)
    
    Action Space (2D continuous):
        [left_thrust, right_thrust]
        - Each in range [0, max_thrust]
    
    Reward:
        - Negative distance to target
        - Velocity penalty
        - Angle penalty
        - Bonus for being close to target
        
    Episode ends when:
        - Terminated: Drone crashes (y < 0.1) or flips (|angle| > π/2)
        - Truncated: Max steps reached (default 500)
    """
    
    def __init__(
        self,
        render_mode=None,
        max_thrust: float = 10.0,
        target_x: float = 0.0,
        target_y: float = 3.0,
        max_episode_steps: int = 500,
        **kwargs
    ):
        """
        Initialize drone environment.
        
        Args:
            render_mode: "human" for window, None for headless
            max_thrust: Maximum thrust per motor (N)
            target_x: Target X position (meters)
            target_y: Target Y position (meters)
            max_episode_steps: Steps before truncation
        """
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            **kwargs
        )
        
        self.max_thrust = max_thrust
        self.target = np.array([target_x, target_y], dtype=np.float32)
        
        # Observation space: [x, y, vx, vy, rotation, angular_velocity]
        self.observation_space = Box(
            low=np.array([-10, -10, -10, -10, -np.pi, -10], dtype=np.float32),
            high=np.array([10, 10, 10, 10, np.pi, 10], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: [left_thrust, right_thrust]
        self.action_space = Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([max_thrust, max_thrust], dtype=np.float32),
            dtype=np.float32
        )
        
        # Drone components (set in _setup_scene)
        self.drone = None
        self.motor_left = None
        self.motor_right = None
        
    def _setup_scene(self):
        """Create the drone simulation scene."""
        # Set gravity
        self.engine.set_gravity(0, -9.81)
        
        # Ground plane
        self.engine.add_collider(0, -1, 20, 1, 0)
        
        # Create drone body
        # x=0, y=2, mass=1kg, width=1m, height=0.2m
        self.drone = rigid.Body(0, 2, 1.0, 1.0, 0.2)
        
        # Add motors
        # Motor(local_x, local_y, width, height, mass, max_thrust)
        self.motor_left = rigid.Motor(-0.4, 0, 0.1, 0.1, 0.05, self.max_thrust)
        self.motor_right = rigid.Motor(0.4, 0, 0.1, 0.1, 0.05, self.max_thrust)
        
        self.drone.add_motor(self.motor_left)
        self.drone.add_motor(self.motor_right)
        
        self.engine.add_body(self.drone)
        
    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        return np.array([
            self.drone.get_x(),
            self.drone.get_y(),
            self.drone.vel.get(0, 0),
            self.drone.vel.get(1, 0),
            self.drone.get_rotation(),
            self.drone.ang_vel.get(0, 0),
        ], dtype=np.float32)
        
    def _apply_action(self, action: np.ndarray):
        """Apply motor thrusts."""
        # Clip actions to valid range
        left_thrust = float(np.clip(action[0], 0, self.max_thrust))
        right_thrust = float(np.clip(action[1], 0, self.max_thrust))
        
        self.motor_left.thrust = left_thrust
        self.motor_right.thrust = right_thrust
        
    def _compute_reward(self) -> float:
        """Compute reward based on current state."""
        obs = self._get_obs()
        pos = obs[:2]
        vel = obs[2:4]
        angle = obs[4]
        
        # Distance to target (main objective)
        dist = np.linalg.norm(pos - self.target)
        
        # Survival bonus - crucial for learning to stay aloft
        reward = 0.5
        
        # Distance reward (exponential for better gradient)
        reward -= dist * 0.5
        
        # Height reward - encourage going up
        reward += pos[1] * 0.2
        
        # Velocity penalty (smaller)
        reward -= 0.05 * np.linalg.norm(vel)
        
        # Angle penalty (smaller, encourage level flight)
        reward -= 0.2 * abs(angle)
        
        # Bonus for being close to target
        if dist < 0.5:
            reward += 2.0
        if dist < 0.2:
            reward += 5.0
            
        return float(reward)
        
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        y = self.drone.get_y()
        angle = self.drone.get_rotation()
        
        # Crashed into ground
        if y < 0.1:
            return True
            
        # Flipped over
        if abs(angle) > np.pi / 2:
            return True
            
        return False


# Simple test
if __name__ == "__main__":
    print("Testing DroneEnv...")
    
    # Test headless mode
    env = DroneEnv(render_mode=None)
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Initial obs: {obs}")
    
    # Take a few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, pos=({obs[0]:.2f}, {obs[1]:.2f})")
        if term or trunc:
            break
            
    env.close()
    print("Test passed!")
