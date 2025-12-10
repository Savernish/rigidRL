import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../diff_sim_core'))

import rigidRL as rigid
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- 1. Simulation Constants ---
g = 9.81
mp = 0.1   # Mass of the pole (kg)
mc = 1.0   # Mass of the cart (kg)
L = 0.5    # Half-length of the pole (m)
dt = 0.05
steps = 100 # 5 seconds horizon

# --- Setup ---
# Initialize separate scalar tensors for state so we can do math on them
x = rigid.Tensor([0.0])
dx = rigid.Tensor([0.0])
theta = rigid.Tensor([0.1]) # Start with small angle (approx 6 deg)
dtheta = rigid.Tensor([0.0])

# Control: A sequence of forces to optimize
forces = []
for i in range(steps):
    t = rigid.Tensor([0.0], requires_grad=True)
    forces.append(t)

# Optimizer takes the list of force tensors
optimizer = rigid.Adam(forces, lr=0.05)

# --- 2. Simulation Loop ---

for epoch in range(200):
    optimizer.zero_grad()

    curr_x = x
    curr_dx = dx
    curr_theta = theta
    curr_dtheta = dtheta

    total_loss = rigid.Tensor([0.0])

    for i in range(steps):
        r = forces[i]

        sin_th = curr_theta.sin()
        cos_th = curr_theta.cos()

        dth_sq = curr_dtheta * curr_dtheta
        partial = r + sin_th * dth_sq * (mp * L)

        total_m = mc + mp

       # theta_acc calculation
        # numer = g * sin_th - cos_th * partial / total_m
        numer = sin_th * g - cos_th * (partial / rigid.Tensor([total_m])) # div needs tensor? scalar div works? 
        # Actually our C++ supports Tensor/Tensor and Tensor/float. 
        # (partial / total_m) works if total_m is float.
        
        # denom = L * (4/3 - mp * cos_th^2 / total_m)
        denom_term = (cos_th * cos_th) * (mp / total_m)
        denom = (rigid.Tensor([4.0/3.0]) - denom_term) * L
        
        theta_acc = numer / denom
        
        # x_acc calculation
        # x_acc = (partial - mp * L * theta_acc * cos_th) / total_m
        x_acc = (partial - cos_th * theta_acc * (mp * L)) / rigid.Tensor([total_m])
        
        # 4. Euler Integration
        curr_x = curr_x + curr_dx * dt
        curr_dx = curr_dx + x_acc * dt
        curr_theta = curr_theta + curr_dtheta * dt
        curr_dtheta = curr_dtheta + theta_acc * dt
        
        # 5. Accumulate Loss
        # Target: x=0, theta=0 (UP)
        # Loss = x^2 + 10 * theta^2
        # (We want it to stay upright at x=0)
        # Penalty: Angle + Position + Velocity + Effort
        curr_loss = (curr_theta * curr_theta) * 20.0 + \
                    (curr_x * curr_x) * 10.0 + \
                    (curr_dx * curr_dx) * 1.0 + \
                    (r * r) * 0.01
        total_loss = total_loss + curr_loss
    # Optimization Step
    total_loss = total_loss.sum() # Ensure scalar
    total_loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {total_loss.data[0,0]}")

# --- Animation ---
print("Generating Animation...")
# 1. Rerun simulation with optimized forces to capture trajectory
history_x = []
history_theta = []
curr_x = x
curr_dx = dx
curr_theta = theta
curr_dtheta = dtheta
for i in range(steps):
    r = forces[i]
    sin_th = curr_theta.sin()
    cos_th = curr_theta.cos()
    dth_sq = curr_dtheta * curr_dtheta
    partial = r + sin_th * dth_sq * (mp * L)
    total_m = mc + mp
    numer = sin_th * g - cos_th * (partial / rigid.Tensor([total_m]))
    denom_term = (cos_th * cos_th) * (mp / total_m)
    denom = (rigid.Tensor([4.0/3.0]) - denom_term) * L
    theta_acc = numer / denom
    x_acc = (partial - cos_th * theta_acc * (mp * L)) / rigid.Tensor([total_m])
    
    curr_x = curr_x + curr_dx * dt
    curr_dx = curr_dx + x_acc * dt
    curr_theta = curr_theta + curr_dtheta * dt
    curr_dtheta = curr_dtheta + theta_acc * dt
    
    history_x.append(curr_x.data[0,0])
    history_theta.append(curr_theta.data[0,0])
# 2. Animate
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(-5, 5)
ax.set_ylim(-2, 2)
ax.grid()
cart_w, cart_h = 1.0, 0.5
cart = plt.Rectangle((-cart_w/2, -cart_h/2), cart_w, cart_h, color='black')
pole, = ax.plot([], [], 'r-', linewidth=3)
ax.add_patch(cart)
def animate(i):
    cx = history_x[i]
    th = history_theta[i]
    
    # Cart Pos
    cart.set_xy((cx - cart_w/2, -cart_h/2))
    
    # Pole Pos
    px = cx + 2 * L * np.sin(th)
    py = 2 * L * np.cos(th)
    pole.set_data([cx, px], [0, py])
    
    return cart, pole
ani = animation.FuncAnimation(fig, animate, frames=len(history_x), interval=50, blit=True)
ani.save('cartpole.gif', writer='pillow', fps=20)
print("Saved cartpole.gif")
plt.close()