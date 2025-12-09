"""
Box Stacking Test - Tests body-to-body collision
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
    print("=== Box Stacking Test ===")
    engine = fnn.Engine(800, 600, 80, 0.016, 10)
    renderer = engine.get_renderer()
    engine.set_gravity(0, -9.81)
    
    # Bottom box (heavy, on floor)
    box_bottom = fnn.Body(0, 1, 5.0, 2.0, 1.0)
    
    # Top box (falls onto bottom)
    box_top = fnn.Body(0, 5, 1.0, 1.0, 1.0)
    
    engine.add_body(box_bottom)
    engine.add_body(box_top)
    
    # Floor
    engine.add_ground_segment(-10, 0, 10, 0, 0.8)
    
    print("Top box should land on bottom box and stack...")
    
    frame = 0
    while frame < 300:
        if not renderer.process_events():
            break
            
        engine.update()
        
        if frame % 30 == 0:
            y_top = box_top.get_y()
            y_bot = box_bottom.get_y()
            print(f"Frame {frame}: bottom_y={y_bot:.2f}, top_y={y_top:.2f}")
        
        renderer.clear()
        renderer.draw_line(-10, 0, 10, 0, 0.0, 1.0, 0.0)
        engine.render_bodies()
        renderer.present()
        
        frame += 1
        time.sleep(0.016)

if __name__ == "__main__":
    run()
