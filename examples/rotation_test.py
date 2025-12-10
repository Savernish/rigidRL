"""
Rotation Test - Box lands on corner and rotates to settle flat
"""
import sys
import os
import math

script_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(os.path.dirname(script_dir), 'diff_sim_core')
os.add_dll_directory(core_dir)
sys.path.insert(0, core_dir)

import rigidRL as rigid

def run():
    print("=== Rotation Test ===")
    engine = rigid.Engine(800, 600, 80, 0.016, 30)
    engine.set_gravity(0, -9.81)
    
    # Floor
    engine.add_collider(0, -0.5, 30, 1, 0)
    
    # Dynamic box with initial tilt
    box = rigid.Body(0, 5, 1.0, 1, 1)
    box.set_rotation(math.radians(15))  # Start tilted
    engine.add_body(box)
    
    print(f"Initial rotation: 15.0°")
    print("Box should rotate when corner hits platform...")
    
    frame = 0
    while frame < 200:
        if not engine.step():
            break
        
        if frame % 20 == 0:
            x = box.get_x()
            y = box.get_y()
            rot = math.degrees(box.get_rotation())
            print(f"Frame {frame}: pos=({x:.2f}, {y:.2f}) rot={rot:.1f}°")
        
        frame += 1

if __name__ == "__main__":
    run()
