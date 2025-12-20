"""
Circle Collision Demo - Rolling balls on ramps
"""
import rigidRL as rigid
import math

# === Configuration ===
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 950
SCALE = 100.0
RAMP_ANGLE = math.radians(20)

# === Setup ===
engine = rigid.Engine(
    width=WINDOW_WIDTH, height=WINDOW_HEIGHT, 
    scale=SCALE, dt=0.016, substeps=10, headless=False
)
engine.set_gravity(0, -9.81)

# Colliders
engine.add_collider(x=0, y=-1, width=20, height=1, rotation=0)              # Ground
engine.add_collider(x=-4, y=4, width=6, height=0.3, rotation=-RAMP_ANGLE)   # Left ramp
engine.add_collider(x=4, y=2, width=6, height=0.3, rotation=RAMP_ANGLE)     # Right ramp
engine.add_collider(x=0, y=0.5, width=8, height=0.3, rotation=0)            # Catcher

# Bodies
engine.add_body(rigid.Body.Circle(x=-6, y=5.5, mass=1.0, radius=0.4, friction=0.2, restitution=1))
engine.add_body(rigid.Body.Circle(x=-5, y=6.0, mass=0.8, radius=0.3, friction=0.2, restitution=0.4))
engine.add_body(rigid.Body.Circle(x=-4, y=6.5, mass=0.5, radius=0.2, friction=0.3, restitution=0.5))
engine.add_body(rigid.Body.Rect(x=5, y=4, mass=1.0, width=0.6, height=0.6, friction=0.2, restitution=0.4))

# === Run ===
while engine.step():
    pass
