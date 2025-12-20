"""
Vectorized environment wrappers for rigidRL.

Uses Gymnasium's built-in vectorization for parallel environment execution.
"""

from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from typing import Optional, Callable, Union
from ..configs import EnvConfig


def make_vec_env(
    env_fn: Callable,
    num_envs: int = 4,
    async_envs: bool = False,
) -> SyncVectorEnv:
    """
    Create a vectorized environment for parallel training.
    
    Args:
        env_fn: Factory function that creates a single environment instance
        num_envs: Number of parallel environments
        async_envs: If True, use AsyncVectorEnv (multiprocessing)
                   
    Returns:
        Vectorized environment
        
    Example:
        >>> from rigidrl_py.envs import DroneEnv
        >>> vec_env = make_vec_env(lambda: DroneEnv.default(), num_envs=8)
    """
    env_fns = [env_fn for _ in range(num_envs)]
    
    if async_envs:
        return AsyncVectorEnv(env_fns)
    else:
        return SyncVectorEnv(env_fns)


def make_drone_vec_env(
    config: Union[EnvConfig, str, None] = None,
    num_envs: int = 4,
    async_envs: bool = False,
) -> SyncVectorEnv:
    """
    Create vectorized DroneEnv instances.
    
    Args:
        config: EnvConfig object, path to YAML file, or None for default
        num_envs: Number of parallel environments
        async_envs: Use multiprocessing if True
        
    Returns:
        Vectorized DroneEnv
        
    Example:
        >>> from rigidrl_py.envs import make_drone_vec_env
        >>> vec_env = make_drone_vec_env(num_envs=8)
        >>> vec_env = make_drone_vec_env("configs/my_drone.yaml", num_envs=4)
    """
    from .drone_env import DroneEnv
    
    def make_env():
        if config is None:
            return DroneEnv.default()
        elif isinstance(config, str):
            return DroneEnv.from_yaml(config)
        else:
            return DroneEnv(config=config)
    
    return make_vec_env(make_env, num_envs=num_envs, async_envs=async_envs)
