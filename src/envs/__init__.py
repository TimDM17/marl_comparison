"""
Environment wrappers for MARL training
"""

from src.envs.mamujoco_wrapper import MaMuJoCoWrapper
from src.envs.vectorized_mamujoco_wrapper import VectorizedMaMuJoCoWrapper

__all__ = ['MaMuJoCoWrapper', 'VectorizedMaMuJoCoWrapper']