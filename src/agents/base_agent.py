"""
Abstract base class for all MARL agents.

All algorithms must implement this interface to work with the training system.
This enables fair comparison between different MARL algorithms.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional
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
    - save/load: Model checkpointing
    
    Required properties:
    - replay_buffer: Access to experience storage
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
        Select actions for all agents.
        
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
    def train_step(self, batch_size: int) -> Optional[float]:
        """
        Perform single training step.
        
        Args:
            batch_size: Number of episodes/transitions to sample
        
        Returns:
            loss: Training loss (for monitoring), None if not enough data
        """
        pass
    
    @abstractmethod
    def init_hidden_states(self) -> List[torch.Tensor]:
        """
        Initialize recurrent hidden states for all agents.
        
        Returns:
            List of initialized hidden states (one per agent)
        """
        pass
    
    def store_episode(self, episode_data: Dict):
        """
        Store episode in replay buffer.
        
        Default implementation uses self.replay_buffer.push().
        Override if custom storage logic is needed.
        
        Args:
            episode_data: Dictionary containing episode information
                {
                    'observations': [[obs_t0, obs_t1, ...], ...],  # Per agent
                    'actions': [[act_t0, act_t1, ...], ...],       # Per agent
                    'last_actions': [[last_act_t0, ...], ...],     # Per agent
                    'states': [state_t0, state_t1, ...],           # Global
                    'rewards': [reward_t0, reward_t1, ...]         # Shared
                }
        """
        self.replay_buffer.push(episode_data)
    
    @abstractmethod
    def save(self, path: str):
        """
        Save model checkpoint.
        
        Args:
            path: File path to save checkpoint
        """
        pass
    
    @abstractmethod
    def load(self, path: str):
        """
        Load model checkpoint.
        
        Args:
            path: File path to load checkpoint from
        """
        pass
    
    @property
    @abstractmethod
    def replay_buffer(self):
        """
        Access to replay buffer.
        
        Returns:
            ReplayBuffer instance or None if agent doesn't use replay buffer
            
        Note:
            On-policy algorithms (like MAPPO) may return None.
            Off-policy algorithms (like QMIX, MADDPG) should return a buffer.
        """
        pass


