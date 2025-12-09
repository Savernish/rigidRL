"""
V-Shape Test - The terrain from falling_box_visual.py
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
    print("=== V-Shape Test ===")
    engine = fnn.Engine(1200, 800, 80, 0.016, 10)  # Same as falling_box_visual
    renderer = engine.get_renderer()
    engine.set_gravity(0, -9.81)
    
    # Same boxes as falling_box_visual.py (moved box3 to avoid apex)
    box1 = fnn.Body(1, 6, 1.0, 1.0, 1.0)
    box2 = fnn.Body(-2, 8, 1.5, 0.8, 0.8)
    box3 = fnn.Body(3, 10, 0.5, 1.2, 1.2)  # Moved from x=2 to x=3 to avoid apex
    engine.add_body(box1)
    engine.add_body(box2)
    engine.add_body(box3)
    
    # Terrain with wide flat-bottomed V (needs to be wide enough for boxes)
    engine.add_ground_segment(-10, 0, 10, 0, 0.8)  # Ground
    engine.add_ground_segment(-6, 4, -3.5, 1, 0.5)   # V left (ends earlier)
    engine.add_ground_segment(-3.5, 1, -0.5, 1, 0.5) # V bottom (WIDER: 3 units)
    engine.add_ground_segment(-0.5, 1, 2, 4, 0.5)    # V right (starts later)
    engine.add_ground_segment(4, 0, 6, 3, 0.5)     # Peak left
    engine.add_ground_segment(6, 3, 8, 0, 0.5)     # Peak right
    
    print("Testing with flat-bottomed V-valley...")
    
    frame = 0
    while frame < 300:
        if not renderer.process_events():
            break
            
        engine.update()
        
        if frame % 30 == 0:
            print(f"Frame {frame}: box1=({box1.get_x():.2f},{box1.get_y():.2f}) box2=({box2.get_x():.2f},{box2.get_y():.2f}) box3=({box3.get_x():.2f},{box3.get_y():.2f})")
            if any(abs(b.get_x()) > 100 or abs(b.get_y()) > 100 for b in [box1, box2, box3]):
                print("EXPLOSION!")
                break
        
        renderer.clear()
        renderer.draw_line(-10, 0, 10, 0, 0.0, 1.0, 0.0)
        renderer.draw_line(-6, 4, -3.5, 1, 0.3, 0.3, 1.0)    # V left
        renderer.draw_line(-3.5, 1, -0.5, 1, 1.0, 0.5, 0.0)  # V bottom (orange, wide)
        renderer.draw_line(-0.5, 1, 2, 4, 0.3, 0.3, 1.0)     # V right
        renderer.draw_line(4, 0, 6, 3, 1.0, 1.0, 0.0)
        renderer.draw_line(6, 3, 8, 0, 1.0, 1.0, 0.0)
        engine.render_bodies()
        renderer.present()
        
        frame += 1
        time.sleep(0.016)

if __name__ == "__main__":
    run()
