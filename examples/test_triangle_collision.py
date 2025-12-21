"""
Triangle Collision Playground - Multiple shapes interacting
"""
import rigidRL as rigid
import math

# === Setup ===
engine = rigid.Engine(
    width=1200, height=900,
    scale=80.0, dt=0.016, substeps=10, headless=False
)
engine.set_gravity(0, -9.81)

# === Static Colliders ===
engine.Collider(x=0, y=0, width=14, height=0.5, rotation=0)         # Ground
engine.Collider(x=-5, y=2, width=4, height=0.3, rotation=-0.6)       # Left ramp
engine.Collider(x=5, y=1, width=4, height=0.3, rotation=0.3)         # Right ramp
engine.Collider(x=-6.5, y=0, width=1, height=3, rotation=0)          # Left wall
engine.Collider(x=6.5, y=0, width=1, height=3, rotation=0)           # Right wall
engine.Collider(x=0, y=3, width=3, height=0.2, rotation=0)           # Platform

# === Triangles ===
engine.add_body(rigid.Body.Triangle(
    x=-4, y=5, mass=1.0,
    x1=-0.3, y1=-0.3, x2=0.3, y2=-0.3, x3=0.0, y3=0.4,
    friction=0.5, restitution=0.2
))

engine.add_body(rigid.Body.Triangle(
    x=0, y=6, mass=0.8,
    x1=-0.4, y1=-0.2, x2=0.4, y2=-0.2, x3=0.0, y3=0.3,
    friction=0.5, restitution=0.3
))

engine.add_body(rigid.Body.Triangle(
    x=3, y=5.5, mass=1.2,
    x1=-0.25, y1=-0.25, x2=0.25, y2=-0.25, x3=0.0, y3=0.35,
    friction=0.4, restitution=0.4
))

# === Circles ===
engine.add_body(rigid.Body.Circle(x=-2, y=7, mass=0.6, radius=0.25, friction=0.3, restitution=0.6))
engine.add_body(rigid.Body.Circle(x=2, y=6.5, mass=0.5, radius=0.2, friction=0.3, restitution=0.7))
engine.add_body(rigid.Body.Circle(x=0.2, y=8, mass=0.4, radius=0.15, friction=0.2, restitution=0.8))

# === Boxes ===
engine.add_body(rigid.Body.Rect(x=-1, y=5, mass=0.7, width=0.5, height=0.5, friction=0.5, restitution=0.3))
engine.add_body(rigid.Body.Rect(x=4, y=6, mass=0.9, width=0.4, height=0.6, friction=0.4, restitution=0.4))

# === Run ===
while engine.step():
    pass
