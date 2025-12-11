"""
Drone Training with Exploration - Better Learning
"""
import sys
import os
import math

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
core_dir = os.path.join(root_dir, 'diff_sim_core')
os.add_dll_directory(core_dir)
sys.path.insert(0, core_dir)
sys.path.insert(0, root_dir)

import rigidRL as rigid
from rigidrl_py.rl import REINFORCE
import random


def run():
    print("=== Drone RL Training with Exploration ===\n")
    
    target_x, target_y = 0, 2
    num_episodes = 300
    max_steps = 200
    
    # Agent with exploration
    agent = REINFORCE(
        state_dim=6, 
        action_dim=2,
        hidden_sizes=[64, 64],
        lr=0.005,
        gamma=0.99, 
        action_scale=5.0,
        exploration_start=1.5,   # Start with high exploration
        exploration_end=0.1,
        exploration_decay=0.99
    )
    print("Policy: 6 -> [64, 64] -> 2")
    print(f"Episodes: {num_episodes}, Exploration: 1.5 -> 0.1\n")
    
    best_reward = -float('inf')
    reward_history = []
    
    for episode in range(num_episodes):
        # Random starting position for variety
        start_x = random.uniform(-3, -1)
        start_y = random.uniform(0, 2)
        
        engine = rigid.Engine(800, 600, 50, 0.016, 15)
        engine.set_gravity(0, -9.81)
        engine.add_collider(0, -5.5, 25, 0.5, 0)
        
        target = rigid.Body(target_x, target_y, 0.01, 0.4, 0.4)
        target.is_static = True
        engine.add_body(target)
        
        drone = rigid.Body(start_x, start_y, 1.0, 1.0, 0.2)
        motor_left = rigid.Motor(-0.5, 0, 0.15, 0.1, 0.1, 20.0)
        motor_right = rigid.Motor(0.5, 0, 0.15, 0.1, 0.1, 20.0)
        drone.add_motor(motor_left)
        drone.add_motor(motor_right)
        engine.add_body(drone)
        
        hover = 5.89
        total_reward = 0
        
        render = (episode % 30 == 0) or (episode == num_episodes - 1)
        
        for step in range(max_steps):
            x, y = drone.get_x(), drone.get_y()
            vx, vy = drone.vel.get(0, 0), drone.vel.get(1, 0)
            theta, omega = drone.get_rotation(), drone.ang_vel.get(0, 0)
            
            state = [x / 5.0, (y - target_y) / 5.0, vx / 10.0, vy / 10.0, theta, omega / 5.0]
            
            action = agent.select_action(state, training=True)
            
            motor_left.thrust = max(0, min(15, hover + action.data[0, 0]))
            motor_right.thrust = max(0, min(15, hover + action.data[1, 0]))
            
            if render:
                if not engine.step():
                    return
            else:
                engine.update()
            
            # Reward
            dist = math.sqrt((x - target_x)**2 + (y - target_y)**2)
            reward = -dist * 0.5 - abs(theta) * 2.0 - abs(omega) * 0.1
            
            if dist < 1.0:
                reward += 3.0
            if dist < 0.5:
                reward += 10.0
            if y < -5:
                reward -= 50
                break
            
            agent.store_reward(reward)
            total_reward += reward
        
        agent.update()
        reward_history.append(total_reward)
        
        if total_reward > best_reward:
            best_reward = total_reward
        
        if episode % 20 == 0:
            avg_reward = sum(reward_history[-20:]) / min(20, len(reward_history))
            exp = agent.get_exploration()
            print(f"Ep {episode:3d}: reward={total_reward:7.1f}, avg={avg_reward:7.1f}, best={best_reward:7.1f}, exp={exp:.2f}")
    
    print(f"\n=== Training Complete ===")
    print(f"Best reward: {best_reward:.1f}")
    
    # Final demo
    print("\nFinal policy demo...")
    engine = rigid.Engine(800, 600, 50, 0.016, 15)
    engine.set_gravity(0, -9.81)
    engine.add_collider(0, -5.5, 25, 0.5, 0)
    target = rigid.Body(target_x, target_y, 0.01, 0.4, 0.4)
    target.is_static = True
    engine.add_body(target)
    drone = rigid.Body(-2, 1, 1.0, 1.0, 0.2)
    motor_left = rigid.Motor(-0.5, 0, 0.15, 0.1, 0.1, 20.0)
    motor_right = rigid.Motor(0.5, 0, 0.15, 0.1, 0.1, 20.0)
    drone.add_motor(motor_left)
    drone.add_motor(motor_right)
    engine.add_body(drone)
    
    for step in range(300):
        x, y = drone.get_x(), drone.get_y()
        vx, vy = drone.vel.get(0, 0), drone.vel.get(1, 0)
        theta, omega = drone.get_rotation(), drone.ang_vel.get(0, 0)
        state = [x / 5.0, (y - target_y) / 5.0, vx / 10.0, vy / 10.0, theta, omega / 5.0]
        
        action = agent.select_action(state, training=False)  # No exploration
        motor_left.thrust = max(0, min(15, 5.89 + action.data[0, 0]))
        motor_right.thrust = max(0, min(15, 5.89 + action.data[1, 0]))
        
        if not engine.step():
            break
    
    print(f"Final position: ({drone.get_x():.2f}, {drone.get_y():.2f})")


if __name__ == "__main__":
    run()
