"""
Abstract base class for all MARL agents

All algorithms must implement this interface to work with the training system
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
import numpy as np
import torch


class BaseAgent(ABC):
    """
    Abstract base for multi-agent RL algorithms.
    
    Design principles:
    - Algorithm-agnostic interface
    - Works with any MARL algorithm
    - Enables fair comparison between algorithms
    
    Required methods:
    - select_actions: Get actions for all agents
    - train_step: Single training update
    - init_hidden_states: Initialize recurrent states
    - store_episode: Add episode to buffer
    - save/load: Model checkpointing
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
        """
        Select actions for all agents
        
        Args:
            observations: List of observations per agent
            last_actions: List of previous actions per agent
            hiddens: List of recurrent hidden states (if applicable)
            explore: Whether to add exploration noise
            noise_scale: Scale of exploration noise

        Returns:
            actions: List of selected actions per agent
            new_hiddens: Updated hidden states
        """
        pass
    
    @abstractmethod
    def train_step(self, batch_size: int) -> float:
        """
        Perform single training step

        Args:
            batch_size: Number of episodes/transitions to sample
        
        Returns:
            loss: Training loss (for monitoring)
        """
        pass
    
    @abstractmethod
    def init_hidden_states(self) -> List[torch.Tensor]:
        """
        Initialize recurrent hidden states for all agents

        Returns:
            List of initialized hidden states
        """
        pass
    
    @abstractmethod
    def store_episode(self, episode_data: Dict):
        """
        Store episode in replay buffer

        Args:
            episode_data: Dictionary containing episode information
        """
        self.replay_buffer.push(episode_data)
    
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


print("✓ BaseAgent abstract class defined")