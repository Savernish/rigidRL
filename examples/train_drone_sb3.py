"""
DroneEnv Training Script with Stable Baselines3

This script demonstrates training a drone to hover at a target position
using PPO (Proximal Policy Optimization).

Requirements:
    pip install stable-baselines3

Usage:
    python examples/train_drone_sb3.py
"""

import os
import sys

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

from rigidrl_py.envs import DroneEnv, make_drone_vec_env

# Check for stable-baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.callbacks import EvalCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("stable-baselines3 not installed. Run: pip install stable-baselines3")


def test_environment():
    """Validate environment with Gymnasium env checker."""
    print("=" * 50)
    print("Testing DroneEnv...")
    print("=" * 50)
    
    # Create environment with default config
    env = DroneEnv.default()
    
    # Run env checker
    if HAS_SB3:
        try:
            check_env(env, warn=True)
            print("✓ Environment passed Gymnasium validation!")
        except Exception as e:
            print(f"✗ Environment check failed: {e}")
            return False
    
    # Manual test
    obs, info = env.reset()
    print(f"  Observation shape: {obs.shape}")
    print(f"  Action space: {env.action_space}")
    
    # Take some random steps
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        if term or trunc:
            break
    
    print(f"  Ran {i+1} steps, total reward: {total_reward:.2f}")
    env.close()
    print()
    return True


def test_vectorized():
    """Test vectorized environment."""
    print("=" * 50)
    print("Testing Vectorized Environment...")
    print("=" * 50)
    
    vec_env = make_drone_vec_env(num_envs=4)
    obs, _ = vec_env.reset()
    print(f"  Observation shape: {obs.shape} (should be (4, 6))")
    
    # Take a step
    actions = vec_env.action_space.sample()
    print(f"  Action shape: {actions.shape} (should be (4, 2))")
    
    obs, rewards, terms, truncs, infos = vec_env.step(actions)
    print(f"  Rewards shape: {rewards.shape}")
    print(f"✓ Vectorized environment works!")
    
    vec_env.close()
    print()


def train(total_timesteps=100_000, save_path="drone_ppo"):
    """Train a PPO agent on DroneEnv."""
    if not HAS_SB3:
        print("Cannot train without stable-baselines3. Install with:")
        print("  pip install stable-baselines3")
        return None
    
    print("=" * 50)
    print(f"Training PPO for {total_timesteps} timesteps...")
    print("=" * 50)
    
    # Use SB3's make_vec_env for proper wrapping
    from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env
    
    train_env = sb3_make_vec_env(
        lambda: DroneEnv.default(),
        n_envs=8
    )
    
    # Create model with better hyperparameters
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,  # Stable learning rate
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,  # Standard entropy
        tensorboard_log="./logs/"
    )
    
    # Train
    model.learn(total_timesteps=total_timesteps)
    
    # Save
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")
    
    train_env.close()
    return model


def demo(model_path="drone_ppo", episodes=3):
    """Run trained model with visualization."""
    if not HAS_SB3:
        print("Cannot run demo without stable-baselines3.")
        return
    
    print("=" * 50)
    print("Running Demo with Visualization...")
    print("=" * 50)
    
    # Load model
    if os.path.exists(f"{model_path}.zip"):
        model = PPO.load(model_path)
        print(f" Loaded model from {model_path}.zip")
    else:
        print(f" Model not found at {model_path}.zip")
        print("  Run training first!")
        return
    
    # Create environment with rendering
    env = DroneEnv(render_mode="human")
    
    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        step = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            step += 1
            
            if term or trunc:
                print(f"  Episode {ep+1}: {step} steps, reward={total_reward:.2f}")
                break
    
    env.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DroneEnv Training")
    parser.add_argument("--test", action="store_true", help="Test environment only")
    parser.add_argument("--train", action="store_true", help="Train PPO agent")
    parser.add_argument("--demo", action="store_true", help="Run trained agent with visualization")
    parser.add_argument("--steps", type=int, default=100_000, help="Training timesteps")
    args = parser.parse_args()
    
    if args.test or (not args.train and not args.demo):
        test_environment()
        test_vectorized()
    
    if args.train:
        train(total_timesteps=args.steps)
    
    if args.demo:
        demo()
