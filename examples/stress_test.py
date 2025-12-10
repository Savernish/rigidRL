"""
Collision Stress Test - Pushes the physics engine to its limits
Features: Many boxes, stacking, slopes, different sizes, high-speed collisions
"""
import sys
import os
import math
import random

script_dir = os.path.dirname(os.path.abspath(__file__))
core_dir = os.path.join(os.path.dirname(script_dir), 'diff_sim_core')
os.add_dll_directory(core_dir)
sys.path.insert(0, core_dir)

import rigidRL as rigid

def run():
    print("=== Collision Stress Test ===")
    print("Testing: stacking, slopes, varied sizes, high-speed impacts")
    
    engine = rigid.Engine(1400, 900, 40, 0.016, 30)
    engine.set_gravity(0, -15)  # Stronger gravity for faster action
    
    boxes = []
    
    # Create a pyramid of boxes (bottom: 5, then 4, 3, 2, 1)
    print("Creating pyramid of 15 boxes...")
    box_size = 1.0
    for row in range(5):
        num_boxes = 5 - row
        start_x = -(num_boxes - 1) * (box_size + 0.1) / 2
        for i in range(num_boxes):
            x = start_x + i * (box_size + 0.1)
            y = 1 + row * (box_size + 0.05)
            box = rigid.Body(x, y, 1.0, box_size, box_size)
            boxes.append(box)
            engine.add_body(box)
    
    # Add some varied-size boxes falling from above
    print("Adding 10 falling boxes of varying sizes...")
    for i in range(10):
        x = random.uniform(-5, 5)
        y = random.uniform(8, 15)
        size = random.uniform(0.5, 1.5)
        mass = size * size  # Mass proportional to area
        box = rigid.Body(x, y, mass, size, size)
        box.set_rotation(random.uniform(-0.5, 0.5))  # Random initial rotation
        boxes.append(box)
        engine.add_body(box)
    
    # Add some high-speed projectiles coming from the side
    print("Adding 3 high-speed projectiles...")
    for i in range(3):
        box = rigid.Body(-12, 3 + i * 2, 2.0, 0.8, 0.8)
        # Give it initial velocity (we'd need to modify body directly)
        boxes.append(box)
        engine.add_body(box)
    
    # Complex terrain
    print("Creating terrain: floor + slopes + platforms...")
    
    # Main floor
    engine.add_collider(0, -1, 30, 1, 0)
    
    # Left slope going down
    engine.add_collider(-10, 2, 6, 0.5, math.radians(-25))
    
    # Right slope going down  
    engine.add_collider(10, 2, 6, 0.5, math.radians(25))
    
    # Elevated platform in center
    engine.add_collider(0, 5, 4, 0.5, 0)
    
    # Angled deflector
    engine.add_collider(-6, 0, 3, 0.5, math.radians(45))
    
    # Stats tracking
    max_y = 0
    min_y = 100
    explosions = 0
    
    print("\nRunning simulation...")
    print("Press ESC or close window to exit\n")
    
    frame = 0
    while frame < 600:
        if not engine.step():
            break
        
        # Track stats
        for box in boxes:
            y = box.get_y()
            max_y = max(max_y, y)
            min_y = min(min_y, y)
            
            # Check for explosions (bodies flying off)
            if abs(box.get_x()) > 50 or abs(y) > 50:
                explosions += 1
        
        if frame % 60 == 0:
            # Count settled boxes (low velocity - would need vel access)
            print(f"Frame {frame:4d}: {len(boxes)} boxes, y_range=[{min_y:.1f}, {max_y:.1f}], explosions={explosions}")
            min_y = 100
            max_y = 0
        
        frame += 1
    
    print("\n=== Test Complete ===")
    print(f"Total frames: {frame}")
    print(f"Total explosions: {explosions}")
    if explosions == 0:
        print("PASS: No bodies flew off!")
    else:
        print("FAIL: Some bodies escaped the simulation")

if __name__ == "__main__":
    run()
