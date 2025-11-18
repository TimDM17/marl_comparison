"""
MARL agents module.

Available algorithms:
- NQMIX: Non-monotonic Q-value mixing for continuous actions
- FACMAC: (To be implemented)
"""

from src.agents.base_agent import BaseAgent
from src.agents.nqmix import NQMIX
# from src.agents.facmac import FACMAC

__all__ = ['BaseAgent', 'NQMIX']