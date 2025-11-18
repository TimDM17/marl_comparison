"""
MARL Comparison Framework

Main package exports for easy access to core components.
"""

from src.agents import BaseAgent, NQMIX, FACMAC
from src.envs import MaMuJoCoWrapper
from src.training import Trainer, Evaluator
from src.utils import Logger, load_config, save_config

__all__ = [
    # Agents
    'BaseAgent',
    'NQMIX',
    'FACMAC',
    # Environments
    'MaMuJoCoWrapper',
    # Training
    'Trainer',
    'Evaluator',
    # Utils
    'Logger',
    'load_config',
    'save_config',
]

__version__ = '0.1.0'
