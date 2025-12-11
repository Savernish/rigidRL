"""
Impulse Physics Test - Tests collision response with floor and slope
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
    print("=== Impulse Physics Test ===")
    engine = rigid.Engine(800, 600, 80, 0.016, 30)
    engine.set_gravity(0, -9.81)
    
    # Floor
    engine.add_collider(0, -0.5, 30, 1, 0)
    
    # Slope (30 degree angle)
    engine.add_collider(0, 3, 6, 0.5, math.radians(40))
    
    # Dynamic boxes
    box1 = rigid.Body(0, 5, 600000, 1, 1)      # Above floor
    box2 = rigid.Body(-4, 0.00011, 100.0, 0.8, 0.8)  # Above slope
    
    engine.add_body(box1)
    engine.add_body(box2)
    
    print("Press ESC or close window to exit")
    print("Boxes should land and settle on floor and slope...")
    
    frame = 0
    while frame < 300:
        if not engine.step():
            break
        
        if frame % 30 == 0:
            r1 = math.degrees(box1.get_rotation())
            r2 = math.degrees(box2.get_rotation())
            print(f"Frame {frame}: box1=({box1.get_x():.2f}, {box1.get_y():.2f}, rot={r1:.1f}°) box2=({box2.get_x():.2f}, {box2.get_y():.2f}, rot={r2:.1f}°)")
        
        frame += 1

if __name__ == "__main__":
    run()
