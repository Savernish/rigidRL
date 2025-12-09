"""
Simple Impulse Physics Test - Tests new impulse-based collision
"""
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(os.path.dirname(script_dir), 'diff_sim_core')
os.add_dll_directory(core_dir)
sys.path.insert(0, core_dir)

import forgeNN_cpp as fnn
import time

def run():
    print("=== Impulse Physics Test ===")
    engine = fnn.Engine(800, 600, 80, 0.016, 30)  # 30 substeps for stability
    renderer = engine.get_renderer()
    engine.set_gravity(0, -9.81)
    
    # Add ground as a wide box collider
    # add_collider(x, y, width, height, rotation)
    engine.add_collider(0, -0.5, 30, 1, 0)  # Floor: center at (0, -0.5), 30 wide, 1 tall
    
    # Add a slope (rotated collider)
    import math
    engine.add_collider(0, 2, 6, 0.5, math.radians(30))  # 30 degree slope
    
    # Add dynamic boxes
    box1 = fnn.Body(0, 5, 1.0, 1, 1)  # Box above floor
    box2 = fnn.Body(-5, 6, 1.0, 0.8, 0.8)  # Box above slope
    
    engine.add_body(box1)
    engine.add_body(box2)
    
    print("Press ESC or close window to exit")
    print("Boxes should land and settle on floor and slope...")
    
    frame = 0
    while frame < 300:
        if not engine.step():
            break
        
        if frame % 30 == 0:
            import math
            r1 = math.degrees(box1.get_rotation())
            r2 = math.degrees(box2.get_rotation())
            print(f"Frame {frame}: box1=({box1.get_x():.2f}, {box1.get_y():.2f}, rot={r1:.1f}°) box2=({box2.get_x():.2f}, {box2.get_y():.2f}, rot={r2:.1f}°)")
        
        frame += 1
        time.sleep(0.016)

if __name__ == "__main__":
    run()
