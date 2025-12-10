"""
Slope Test - Box sliding down a slope
Tests friction and collision on angled surfaces
"""
import sys
import os
import math

script_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(os.path.dirname(script_dir), 'diff_sim_core')
os.add_dll_directory(core_dir)
sys.path.insert(0, core_dir)

import rigidRL as rigid
import time

def run():
    print("=== Slope Test ===")
    engine = rigid.Engine(800, 600, 50, 0.016, 30)
    renderer = engine.get_renderer()
    engine.set_gravity(0, -9.81)
    
    # Box at top of slope
    box = rigid.Body(-3, 4, 1.0, 1.0, 1.0)
    engine.add_body(box)
    
    # Slope: 30 degree angle, using add_collider with rotation
    slope_angle = 30  # degrees
    slope_length = 8
    slope_cx = -2.5  # center x
    slope_cy = 1.5   # center y
    engine.add_collider(slope_cx, slope_cy, slope_length, 0.5, math.radians(-slope_angle))
    
    # Flat floor
    engine.add_collider(2, -2, 10, 1, 0)
    
    print("Box should slide down slope...")
    
    frame = 0
    while frame < 300:
        if not engine.step():
            break
            
        if frame % 30 == 0:
            x = box.get_x()
            y = box.get_y()
            rot = math.degrees(box.get_rotation())
            print(f"Frame {frame}: pos=({x:.2f}, {y:.2f}) rot={rot:.1f}°")
        
        frame += 1

if __name__ == "__main__":
    run()
