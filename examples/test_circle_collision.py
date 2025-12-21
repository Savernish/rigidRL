"""
Circle Collision Demo - Rolling balls on ramps
"""
import rigidRL as rigid
import math

# === Configuration ===
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 950
SCALE = 100.0
RAMP_ANGLE = math.radians(30)

# === Setup ===
engine = rigid.Engine(
    width=WINDOW_WIDTH, height=WINDOW_HEIGHT, 
    scale=SCALE, dt=0.016, substeps=10, headless=False
)
engine.set_gravity(0, -9.81)

# Colliders
engine.Collider(x=0, y=-1, width=20, height=1, rotation=0)              # Ground
engine.Collider(x=-4, y=4, width=6, height=0.3, rotation=-RAMP_ANGLE)   # Left ramp
engine.Collider(x=4, y=2, width=6, height=0.3, rotation=RAMP_ANGLE)     # Right ramp
engine.Collider(x=0, y=0.5, width=8, height=0.3, rotation=0)            # Catcher

# Bodies
engine.add_body(rigid.Body.Circle(x=-6, y=7.5, mass=2.0, radius=0.4, friction=0.5, restitution=0.1))
engine.add_body(rigid.Body.Circle(x=-4, y=6.0, mass=1.5, radius=0.3, friction=0.3, restitution=0.4))
engine.add_body(rigid.Body.Circle(x=-5, y=9.5, mass=0.5, radius=0.2, friction=0.5, restitution=1.7)) # bouncy boy :O
engine.add_body(rigid.Body.Rect(x=5, y=4, mass=1.0, width=0.6, height=0.6, friction=0.5, restitution=0.4))

# === Run ===
while engine.step():
    pass
