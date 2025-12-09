"""
MARL Comparison Framework

Main package exports for easy access to core components.
"""

from src.agents import NQMIX, FACMAC
from src.envs import MaMuJoCoWrapper, VectorizedMaMuJoCoWrapper
from src.training import Trainer, Evaluator
from src.utils import Logger, load_config, save_config

__all__ = [
    # Agents
    'FACMAC',
    'NQMIX',
    # Environments
    'MaMuJoCoWrapper',
    'VectorizedMaMuJoCoWrapper',
    # Training
    'Trainer',
    'Evaluator',
    # Utils
    'Logger',
    'load_config',
    'save_config',
]

__version__ = '0.1.0'
