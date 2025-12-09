"""
Minimal Debug Test - One box, flat floor
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
    print("=== Minimal Debug Test ===")
    engine = fnn.Engine(800, 600, 100, 0.016, 10)
    renderer = engine.get_renderer()
    engine.set_gravity(0, -9.81)
    
    # Just ONE box, higher up
    box = fnn.Body(0, 3, 1.0, 1.0, 1.0)
    engine.add_body(box)
    
    # Just flat floor
    engine.add_ground_segment(-10, 0, 10, 0, 0.5)
    
    print("Dropping one box onto flat floor")
    print("Watching position for 2 seconds...")
    
    frame = 0
    start = time.time()
    while time.time() - start < 5:
        if not renderer.process_events():
            break
            
        engine.update()
        
        # Print position every 30 frames
        if frame % 30 == 0:
            x = box.get_x()
            y = box.get_y()
            rot = box.get_rotation()
            print(f"Frame {frame}: pos=({x:.3f}, {y:.3f}) rot={rot:.3f}")
            
            # Check for explosion
            if abs(x) > 100 or abs(y) > 100:
                print("EXPLOSION DETECTED!")
                break
        
        renderer.clear()
        renderer.draw_line(-10, 0, 10, 0, 0.0, 1.0, 0.0)
        engine.render_bodies()
        renderer.present()
        
        frame += 1
        time.sleep(0.016)
    
    print(f"Final pos: ({box.get_x():.3f}, {box.get_y():.3f})")

if __name__ == "__main__":
    run()
