"""Abstract base class for all MARL agents"""

from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
import torch


class BaseAgent(ABC):
    """
    Abstract base for multi-agent RL algorithms.
    
    All algorithms (NQMIX, QMIX, FACMAC) must implement this interface.
    """
    
    @abstractmethod
    def select_actions(
        self, 
        observations: List[np.ndarray],
        last_actions: List[np.ndarray],
        hiddens: List[torch.Tensor],
        explore: bool = True,
        noise_scale: float = 0.1
    ) -> Tuple[List[np.ndarray], List[torch.Tensor]]:
        """Select actions for all agents"""
        pass
    
    @abstractmethod
    def train_step(self, batch_size: int) -> float:
        """Single training step, returns loss"""
        pass
    
    @abstractmethod
    def init_hidden_states(self) -> List[torch.Tensor]:
        """Initialize recurrent hidden states"""
        pass
    
    @abstractmethod
    def store_episode(self, episode_data: dict):
        """Store episode in replay buffer"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save model checkpoint"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load model checkpoint"""
        pass
    
    @property
    @abstractmethod
    def replay_buffer(self):
        """Access to replay buffer"""
        pass