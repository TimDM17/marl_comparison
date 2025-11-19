"""
MaMuJoCo environment wrapper for clean interface.

Provides consistent API for different MaMuJoCo environments
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


class MaMuJoCoWrapper:
    """
    Wrapper for MaMuJoCo parallel environments

    Provides:
    - Clean observation/state extraction
    - Consistent action interface
    - Episode termination handling
    - Dimension information
    """

    def __init__(self, env_name: str = "Humanoid", partitioning: str = "9|8"):
        """
        Initialize MaMuJoCo environment.

        Args:
            env_name: Environment name (e.g., "Humanoid", "Ant")
            partitioning: Action partitioning (e.g., "9|8" for Humanoid)
        """
        # Import here to allow framework to work even without gymnasium_robotics
        from gymnasium_robotics import mamujoco_v1

        self.env = mamujoco_v1.parallel_env(env_name, partitioning)
        self.env_name = env_name
        self.partitioning = partitioning

        # Store agent information
        self.possible_agents = self.env.possible_agents
        self.n_agents = len(self.possible_agents)

        # Extract dimensions
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces

        self.obs_dims = [
            self.observation_spaces[agent].shape[0] 
            for agent in self.possible_agents
        ]
        self.action_dims = [
            self.action_spaces[agent].shape[0] 
            for agent in self.possible_agents
        ]
        self.state_dim = sum(self.obs_dims)

    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        """
        Reset environment.
        
        Args:
            seed: Random seed for episode
        
        Returns:
            observations: Dict of observations per agent
            info: Additional information
        """
        return self.env.reset(seed=seed)
    

    def step(self, actions: List[np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Step environment with actions.
        
        Args:
            actions: List of actions per agent
        
        Returns:
            observations: Next observations per agent
            rewards: Rewards per agent
            terminated: Termination flags per agent
            truncated: Truncation flags per agent
            info: Additional information
        """
        # Convert list to dict for PettingZoo API
        action_dict = {
            self.possible_agents[i]: actions[i] 
            for i in range(self.n_agents)
        }
        return self.env.step(action_dict)
    

    def close(self) -> None:
        """Close environment"""
        self.env.close()

    def render(self) -> Optional[np.ndarray]:
        """Render environment (returns frame if render_mode='rgb_array')."""
        return self.env.render()


