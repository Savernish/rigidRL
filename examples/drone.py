import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../diff_sim_core'))

import rigidRL as rigid
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

# --- 1. Simulation Constants ---
g = 9.81
m = 1.0   # Mass (kg)
I = 0.5   # Moment of Inertia
L = 0.25  # Arm length
dt = 0.04
steps = 100 # 4 seconds

# --- 2. Setup ---
# Initial State: [x, vx, y, vy, theta, omega]
state_init = rigid.Tensor([-5.0, 0.0, 5.0, 0.0, 0.0, 0.0], requires_grad=False)

# Controls: Two tensors of shape (steps, 1) effectively
# We initialize them as flat vector (steps, 1) so we can select(i)
# Init at hover thrust (mg/2 = 4.9)
initial_thrust = [4.9 for _ in range(steps)]
thrust_left = rigid.Tensor(initial_thrust, requires_grad=True)
thrust_right = rigid.Tensor(initial_thrust, requires_grad=True)

# Optimizer
optimizer = rigid.AdamW([thrust_left, thrust_right], lr=0.1, weight_decay=0.0)

print("Starting Drone Optimization (Vectorized)...")

def simulate_step(state, tl, tr):
    # state: [x, vx, y, vy, theta, omega]
    # Extract components
    x = state[0]
    vx = state[1]
    y = state[2]
    vy = state[3]
    theta = state[4]
    omega = state[5] # dtheta

    # Physics
    F = tl + tr
    tau = (tr - tl) * rigid.Tensor([L])
    
    sin_th = theta.sin()
    cos_th = theta.cos()
    
    # Accelerations
    # ax = -F * sin_th / m
    ax = (rigid.Tensor([-1.0]) * F * sin_th) / rigid.Tensor([m])
    # ay = F * cos_th / m - g
    ay = (F * cos_th) / rigid.Tensor([m]) - rigid.Tensor([g])
    # alpha = tau / I
    alpha = tau / rigid.Tensor([I])
    
    # Integration (Euler)
    new_x = x + vx * dt
    new_vx = vx + ax * dt
    new_y = y + vy * dt
    new_vy = vy + ay * dt
    new_theta = theta + omega * dt
    new_omega = omega + alpha * dt
    
    # Reassemble state using Stack or Cat
    # Stack creates (6, 1) from scalars
    next_state = rigid.Tensor.stack([new_x, new_vx, new_y, new_vy, new_theta, new_omega])
    
    return next_state

# --- 3. Optimization Loop ---
start_time = time.time()

for epoch in range(200):
    optimizer.zero_grad()
    
    curr_state = state_init
    total_loss = rigid.Tensor([0.0])
    
    # Tracking
    min_y = 100.0
    
    for i in range(steps):
        # Select actions for this timestep
        tl = thrust_left[i]
        tr = thrust_right[i]
        
        curr_state = simulate_step(curr_state, tl, tr)
        
        # Extract for loss
        x = curr_state[0]
        y = curr_state[2]
        vx = curr_state[1]
        vy = curr_state[3]
        theta = curr_state[4]
        
        # Track min_y
        y_val = y.data[0,0]
        if y_val < min_y: min_y = y_val
            
        # Loss Terms
        # Target: x=0, y=0.5
        # Use pow(2) instead of x*x
        dist_sq = x.pow(2.0) + (y - rigid.Tensor([0.5])).pow(2.0)
        vel_sq = vx.pow(2.0) + vy.pow(2.0)
        angle_sq = theta.pow(2.0)
        
        # Floor Barrier: 1.0 / (y + 5.0) -> Use clamp to avoid div by zero if very low? 
        # Or standard barrier.
        # Let's use relu to ensure y+5 is positive? No, barrier should push back.
        # Just standard.
        floor_dist = y + rigid.Tensor([5.0])
        floor_penalty = rigid.Tensor([1.0]) / floor_dist
        
        # Effort
        hover = rigid.Tensor([4.9])
        effort = (tl - hover).pow(2.0) + (tr - hover).pow(2.0)
        
        step_loss = dist_sq * 10.0 + vel_sq * 1.0 + angle_sq * 10.0 + effort * 0.001 + floor_penalty * 20.0
        total_loss = total_loss + step_loss

    total_loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        pos_x = curr_state[0].data[0,0]
        pos_y = curr_state[2].data[0,0]
        print(f"Epoch {epoch}: Loss={total_loss.data[0,0]:.2f} | MinY={min_y:.2f} | FinalPos=({pos_x:.2f}, {pos_y:.2f})")

opt_time = time.time() - start_time
print(f"Optimization Complete (Time: {opt_time:.4f}s)")

# --- 4. Animation ---
print("\nGenerating Animation...")
history_x = []
history_y = []
history_theta = []

curr_state = state_init
for i in range(steps):
    tl = thrust_left[i]
    tr = thrust_right[i]
    curr_state = simulate_step(curr_state, tl, tr)
    
    history_x.append(curr_state[0].data[0,0])
    history_y.append(curr_state[2].data[0,0])
    history_theta.append(curr_state[4].data[0,0])

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-6, 6)
ax.set_ylim(-1, 8)
ax.grid()

line, = ax.plot([], [], 'k-', linewidth=3)
l_motor, = ax.plot([], [], 'ro', markersize=8)
r_motor, = ax.plot([], [], 'bo', markersize=8)

def animate(i):
    cx = history_x[i]
    cy = history_y[i]
    th = history_theta[i]
    
    lx = cx - L * np.cos(th)
    ly = cy - L * np.sin(th)
    rx = cx + L * np.cos(th)
    ry = cy + L * np.sin(th)
    
    line.set_data([lx, rx], [ly, ry])
    l_motor.set_data([lx], [ly])
    r_motor.set_data([rx], [ry])
    return line, l_motor, r_motor

ani = animation.FuncAnimation(fig, animate, frames=steps, interval=40, blit=True)
ani.save('drone_optimized.gif', writer='pillow', fps=25)
print("Saved drone_optimized.gif")