"""
Slope Test - One box on a slope
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
    print("=== Slope Test ===")
    engine = fnn.Engine(800, 600, 100, 0.016, 10)
    renderer = engine.get_renderer()
    engine.set_gravity(0, -9.81)
    
    # Box above the slope
    box = fnn.Body(-3, 4, 1.0, 1.0, 1.0)
    engine.add_body(box)
    
    # Single slope going down-right
    # From (-5, 3) to (0, 0)
    engine.add_ground_segment(-5, 3, 0, 0, 0.5)
    
    # Flat floor to catch it
    engine.add_ground_segment(-10, -1, 10, -1, 0.5)
    
    print("Box should slide down slope...")
    
    frame = 0
    start = time.time()
    while time.time() - start < 5:
        if not renderer.process_events():
            break
            
        engine.update()
        
        if frame % 30 == 0:
            x = box.get_x()
            y = box.get_y()
            print(f"Frame {frame}: pos=({x:.3f}, {y:.3f})")
            if abs(x) > 100 or abs(y) > 100:
                print("EXPLOSION!")
                break
        
        renderer.clear()
        renderer.draw_line(-5, 3, 0, 0, 1.0, 1.0, 0.0)  # Slope
        renderer.draw_line(-10, -1, 10, -1, 0.0, 1.0, 0.0)  # Floor
        engine.render_bodies()
        renderer.present()
        
        frame += 1
        time.sleep(0.016)

if __name__ == "__main__":
    run()
