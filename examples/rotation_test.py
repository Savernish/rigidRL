"""
Simple rotation test - drop box on edge to see if it rotates
"""
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(os.path.dirname(script_dir), 'diff_sim_core')
os.add_dll_directory(core_dir)
sys.path.insert(0, core_dir)

import forgeNN_cpp as fnn
import time
import math

def run():
    print("=== Rotation Test ===")
    engine = fnn.Engine(800, 600, 80, 0.016, 30)  # 30 substeps for stability
    engine.set_gravity(0, -9.81)
    
    # Just a floor - platform was causing overlap issues
    engine.add_collider(0, -0.5, 30, 1, 0)  # Floor at y=-0.5, top at y=0
    
    # Dynamic box with initial tilt - should rotate when landing on corner
    box = fnn.Body(0, 5, 1.0, 1, 1)
    # Set initial rotation (15 degrees) to land on corner
    initial_rot = math.radians(15)
    box.rotation.set(0, 0, initial_rot)
    engine.add_body(box)
    
    print(f"Initial rotation: {math.degrees(box.get_rotation()):.1f}°")
    print("Box should rotate when corner hits platform...")
    
    frame = 0
    while frame < 200:
        if not engine.step():
            break
        
        if frame % 20 == 0:
            r = math.degrees(box.get_rotation())
            print(f"Frame {frame}: pos=({box.get_x():.2f}, {box.get_y():.2f}) rot={r:.1f}°")
        
        frame += 1
        time.sleep(0.016)

if __name__ == "__main__":
    run()
