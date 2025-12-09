"""
Multi-box Test - 3 boxes on flat floor
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
    print("=== Multi-box Test ===")
    engine = fnn.Engine(800, 600, 100, 0.016, 10)
    renderer = engine.get_renderer()
    engine.set_gravity(0, -9.81)
    
    # 3 boxes spread apart
    box1 = fnn.Body(-2, 6, 1.0, 1.0, 1.0)
    box2 = fnn.Body(0, 8, 1.5, 0.8, 0.8)
    box3 = fnn.Body(2, 10, 0.5, 1.2, 1.2)
    engine.add_body(box1)
    engine.add_body(box2)
    engine.add_body(box3)
    
    # Just flat floor
    engine.add_ground_segment(-10, 0, 10, 0, 0.5)
    
    print("3 boxes dropping onto flat floor...")
    
    frame = 0
    while frame < 300:
        if not renderer.process_events():
            break
            
        engine.update()
        
        if frame % 30 == 0:
            print(f"Frame {frame}: box1=({box1.get_x():.2f},{box1.get_y():.2f}) box2=({box2.get_x():.2f},{box2.get_y():.2f}) box3=({box3.get_x():.2f},{box3.get_y():.2f})")
        
        renderer.clear()
        renderer.draw_line(-10, 0, 10, 0, 0.0, 1.0, 0.0)
        engine.render_bodies()
        renderer.present()
        
        frame += 1
        time.sleep(0.016)

if __name__ == "__main__":
    run()
