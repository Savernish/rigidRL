"""
Configuration classes for RL environments
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import yaml


@dataclass
class MotorConfig:
    """Motor configuration"""
    x: float                    # Local X position on body
    y: float = 0.0              # Local Y position on body
    max_thrust: float = 10.0
    mass: float = 0.05
    width: float = 0.1
    height: float = 0.1


@dataclass  
class DroneConfig:
    """Drone body configuration"""
    mass: float = 1.0
    width: float = 1.0
    height: float = 0.2
    motors: List[MotorConfig] = field(default_factory=lambda: [
        MotorConfig(x=-0.4),
        MotorConfig(x=0.4)
    ])
    
    @classmethod
    def from_dict(cls, data: dict) -> "DroneConfig":
        motors = [MotorConfig(**m) for m in data.get("motors", [])]
        return cls(
            mass=data.get("mass", 1.0),
            width=data.get("width", 1.0),
            height=data.get("height", 0.2),
            motors=motors if motors else None
        )


@dataclass
class EnvConfig:
    """Environment configuration"""
    drone: DroneConfig = field(default_factory=DroneConfig)
    target: Tuple[float, float] = (0.0, 4.0)
    spawn_points: List[Tuple[float, float]] = field(default_factory=lambda: [(0.0, 1.5)])
    max_steps: int = 500
    window_width: int = 800
    window_height: int = 600
    scale: float = 50.0
    
    @classmethod
    def from_dict(cls, data: dict) -> "EnvConfig":
        drone_data = data.get("drone", {})
        drone = DroneConfig.from_dict(drone_data)
        
        target = tuple(data.get("target", [0.0, 4.0]))
        spawn_points = [tuple(p) for p in data.get("spawn_points", [[0.0, 1.5]])]
        
        return cls(
            drone=drone,
            target=target,
            spawn_points=spawn_points,
            max_steps=data.get("max_steps", 500),
            window_width=data.get("window_width", 800),
            window_height=data.get("window_height", 600),
            scale=data.get("scale", 50.0)
        )
    
    @classmethod
    def from_yaml(cls, path: str) -> "EnvConfig":
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)


# Convenience exports
__all__ = ["MotorConfig", "DroneConfig", "EnvConfig"]
