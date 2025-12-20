"""
DroneEnv RL Training Demo - Train with Stable-Baselines3

This example demonstrates:
1. Training with multiple spawn points for generalization
2. Testing on an UNSEEN 6th spawn point
3. Visualizing the trained policy

Requirements:
    pip install stable-baselines3

Usage:
    python examples/demo_rl_training.py
"""

import sys
import os

# Add project to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

import numpy as np
from rigidrl_py.envs import DroneEnv
from rigidrl_py.configs import EnvConfig, DroneConfig, MotorConfig

# Check for stable-baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("ERROR: stable-baselines3 not installed")
    print("Run: pip install stable-baselines3")
    exit(1)


# === Configuration ===
TRAIN_TIMESTEPS = 400_000
NUM_ENVS = 4

# 5 spawn points for TRAINING
TRAIN_SPAWN_POINTS = [
    (0.0, 1.5),    # Center
    (-2.0, 1.5),   # Left
    (2.0, 1.5),    # Right
    (0.0, 2.5),    # Higher
    (1.0, 1.0),    # Right-low
]

# 6th spawn point for TESTING (never seen during training!)
TEST_SPAWN_POINT = (-1.5, 2.0)

# Shared drone config
DRONE = DroneConfig(
    mass=1.0,
    width=1.0,
    height=0.2,
    motors=[
        MotorConfig(x=-0.4, max_thrust=10.0),
        MotorConfig(x=0.4, max_thrust=10.0)
    ]
)
TARGET = (0.0, 4.0)

# Training config (5 spawn points)
TRAIN_CONFIG = EnvConfig(
    drone=DRONE,
    target=TARGET,
    spawn_points=TRAIN_SPAWN_POINTS,
    max_steps=500,
    window_width=800,
    window_height=600,
    scale=60.0
)

# Test config (6th unseen spawn point)
TEST_CONFIG = EnvConfig(
    drone=DRONE,
    target=TARGET,
    spawn_points=[TEST_SPAWN_POINT],
    max_steps=500,
    window_width=1000,
    window_height=800,
    scale=80.0
)


def train():
    """Train PPO agent on drone environment with multiple spawn points."""
    print("=" * 50)
    print("Training PPO on DroneEnv")
    print("=" * 50)
    print(f"Timesteps: {TRAIN_TIMESTEPS:,}")
    print(f"Parallel envs: {NUM_ENVS}")
    print(f"Training spawn points: {len(TRAIN_SPAWN_POINTS)}")
    for i, sp in enumerate(TRAIN_SPAWN_POINTS):
        print(f"  {i+1}. {sp}")
    print()
    
    # Create vectorized training environment
    train_env = make_vec_env(
        lambda: DroneEnv(config=TRAIN_CONFIG, render_mode=None),
        n_envs=NUM_ENVS
    )
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01
    )
    
    # Train
    print("Training...")
    model.learn(total_timesteps=TRAIN_TIMESTEPS)
    
    train_env.close()
    return model


def visualize(model):
    """Visualize trained policy on UNSEEN spawn point."""
    print()
    print("=" * 50)
    print("Testing Generalization on UNSEEN Spawn Point")
    print("=" * 50)
    print(f"Test spawn point: {TEST_SPAWN_POINT}")
    print("(This point was NEVER seen during training!)")
    print("Close window to exit")
    
    # Create visual environment with TEST config
    env = DroneEnv(config=TEST_CONFIG, render_mode="human")
    
    # Run episodes
    num_episodes = 3
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        
        obs, info = env.reset()
        total_reward = 0
        step = 0
        
        while True:
            # Get action from trained policy
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            if info.get("window_closed"):
                print("Window closed")
                env.close()
                return
            
            if terminated or truncated:
                status = "Crashed" if terminated else "Timeout"
                print(f"  {status} at step {step}, Reward: {total_reward:.1f}")
                break
    
    env.close()


def main():
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║     DroneEnv Generalization Demo                 ║") # looks cool tbh
    print("╚══════════════════════════════════════════════════╝")
    print()
    print("This demo trains on 5 spawn points, then tests on")
    print("a 6th UNSEEN spawn point to demonstrate generalization!")
    print()
    
    # Train
    model = train()
    
    # Visualize
    print()
    input("Training complete! Press Enter to test on UNSEEN spawn point...")
    visualize(model)
    
    print()
    print("Demo complete!")


if __name__ == "__main__":
    main()
