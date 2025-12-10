import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../diff_sim_core'))

import rigidRL as rigid

# 1. Setup Simulation
target = rigid.Tensor([100.0, 0.0])
start_pos = rigid.Tensor([0.0, 0.0])
gravity = rigid.Tensor([0.0, -9.81])

# We optimize the initial velocity vector [vx, vy] directly
velocity = rigid.Tensor([10.0, 10.0], requires_grad=True)

optimizer = rigid.SGD([velocity], lr=0.001)

dt = 0.1
steps = 200

print("Starting Sim...")
for epoch in range(50):
    optimizer.zero_grad()
    
    # Reset state
    pos = start_pos
    curr_vel = velocity # This allows gradients to flow back to 'velocity'
    
    # Physics Loop (Euler Integration)
    for i in range(steps):
        pos = pos + curr_vel * dt
        curr_vel = curr_vel + gravity * dt
    
    # Loss: MSE distance to target
    diff = pos - target
    # loss = sum((pos - target)^2)
    loss = (diff * diff).sum()
    
    loss.backward()
    optimizer.step()
    
    if epoch % 5 == 0:
        # loss.data is a 1x1 Matrix, access [0,0]
        l = loss.data[0,0]
        vx = velocity.data[0,0]
        vy = velocity.data[1,0]
        print(f"Epoch {epoch}: Loss={l:.4f} | Vel=[{vx:.2f}, {vy:.2f}]")

# --- Visualization ---
import matplotlib.pyplot as plt
import numpy as np

print("\nGenerating visualization...")

# Run one final simulation with recording
pos = start_pos
curr_vel = velocity # Optimized velocity
dt = 0.1
traj_x = [pos.data[0,0]]
traj_y = [pos.data[1,0]]

for i in range(steps):
    pos = pos + curr_vel * dt
    curr_vel = curr_vel + gravity * dt
    traj_x.append(pos.data[0,0])
    traj_y.append(pos.data[1,0])

plt.figure(figsize=(10, 6))
plt.plot(traj_x, traj_y, 'b-', linewidth=2, label='Optimized Trajectory')
plt.plot(0, 0, 'go', label='Start')
plt.plot(100, 0, 'r*', markersize=15, label='Target')
plt.axhline(0, color='k', linestyle='--')
plt.title(f"Optimized Projectile Path\nFinal Velocity: [{velocity.data[0,0]:.2f}, {velocity.data[1,0]:.2f}]")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.legend()
plt.grid()
plt.savefig("projectile_result.png")
print("Saved plot to 'projectile_result.png'")
plt.show()
        