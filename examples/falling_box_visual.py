"""
Simple Physics Test
"""
import sys
import os

# Add diff_sim_core to path for module and DLL loading
script_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(os.path.dirname(script_dir), 'diff_sim_core')
os.add_dll_directory(core_dir)  # For Windows DLL loading
sys.path.insert(0, core_dir)

import forgeNN_cpp as fnn
import time

def run():
    print("=== Simple Physics Test ===")
    engine = fnn.Engine(1200, 800, 80, 0.016, 10)
    renderer = engine.get_renderer()
    engine.set_gravity(0, -9.81)
    
    # Create boxes (avoid landing on segment endpoints)
    box1 = fnn.Body(0, 6, 1.0, 1.0, 1.0)
    box2 = fnn.Body(-2, 8, 1.5, 0.8, 0.8)
    box3 = fnn.Body(5, 10, 0.5, 1.2, 1.2)  # At x=5, lands between peak segments
    engine.add_body(box1)
    engine.add_body(box2)
    engine.add_body(box3)
    
    # Ground
    engine.add_ground_segment(-10, 0, 10, 0, 0.8)
    
    # V-shaped valley
    engine.add_ground_segment(-6, 4, -2, 1, 0.5)
    engine.add_ground_segment(-2, 1, 2, 4, 0.5)
    
    # Peak
    engine.add_ground_segment(4, 0, 6, 3, 0.5)
    engine.add_ground_segment(6, 3, 8, 0, 0.5)
    
    print("Press ESC or close window to exit")
    
    while True:
        if not renderer.process_events():
            break
            
        engine.update()
        renderer.clear()
        
        # Draw terrain
        renderer.draw_line(-10, 0, 10, 0, 0.0, 1.0, 0.0)  # Floor
        renderer.draw_line(-6, 4, -2, 1, 0.3, 0.3, 1.0)   # V left
        renderer.draw_line(-2, 1, 2, 4, 0.3, 0.3, 1.0)    # V right
        renderer.draw_line(4, 0, 6, 3, 1.0, 1.0, 0.0)     # Peak left
        renderer.draw_line(6, 3, 8, 0, 1.0, 1.0, 0.0)     # Peak right
        
        engine.render_bodies()
        renderer.present()
        time.sleep(0.016)

if __name__ == "__main__":
    run()
