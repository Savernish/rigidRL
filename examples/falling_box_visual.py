"""
Simple Physics Test - Multiple boxes falling on terrain
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
    print("=== Simple Physics Test ===")
    engine = rigid.Engine(1200, 800, 80, 0.016, 30)
    engine.set_gravity(0, -9.81)
    
    # Create boxes
    box1 = rigid.Body(0, 6, 4, 1.0, 1.0)
    box2 = rigid.Body(-2, 8, 1.5, 0.2, 0.8)
    box3 = rigid.Body(5, 10, 4, 1.2, 1.2)
    box4 = rigid.Body(6, 12, 0.5, 1.2, 1.2)
    box5 = rigid.Body(5, 15, 0.1, 0.6, 1.2)
    engine.add_body(box1)
    engine.add_body(box2)
    engine.add_body(box3)
    engine.add_body(box4)
    engine.add_body(box5)
    
    # Ground - flat floor
    engine.add_collider(0, -1, 25, 1, 0)
    
    # V-shaped valley (two angled platforms)
    engine.add_collider(-4, 2.5, 5, 0.5, math.radians(-30))  # Left slope
    engine.add_collider(0, 2.5, 5, 0.5, math.radians(30))    # Right slope
    
    # Peak (two angled platforms)
    engine.add_collider(5, 1.5, 4, 0.5, math.radians(40))    # Left side
    engine.add_collider(7, 1.5, 4, 0.5, math.radians(-40))   # Right side
    
    print("Press ESC or close window to exit")
    
    while engine.step():
        pass  # Frame rate is handled by engine.step()

if __name__ == "__main__":
    run()
