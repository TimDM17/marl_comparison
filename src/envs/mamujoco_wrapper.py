"""
MaMuJoCo environment wrapper fro MARL training

Purpose:
    Wraps the MaMuJoCo (Multi-Agent MuJoCo) environment from gymnasium_robotics
    to provide a clean, consistent interface for our MARL algorithms

Key Concepts:
    - MaMuJoCo: Multi-agent version of MuJoCo physics simulator
    - Partitioning: How robot joints are split between agents (e.g., "9|8" means
    agent 0 controls 9 joints, agent 1 controls 8 joints)
    - PettingZoo: Multi-agent environment API that use dict-based observations/actions

Connections:
    - Used by: scripts/train.py, scripts/evaluate.py
    - Used by: src/training/trainer.py, src/training/evaluator.py
"""

import numpy as np
from typing import Tuple, List, Dict, Optional


class MaMuJoCoWrapper:
    
    def __init__(self, env_name: str = "Humanoid", partitioning: str = "9|8", render_mode: Optional[str] = None):
        """
        Initialize MaMuJoCo environment.

        Args:
            env_name: Environment name (e.g., "Humanoid", "Ant")
            partitioning: Action partitioning (e.g., "9|8" for Humanoid)
            render_mode: Rendering mode ('rgb_array' for video, None for no rendering)
        """
        # Import here to allow framework to work even without gymnasium_robotics
        from gymnasium_robotics import mamujoco_v1

        # Create parallel environment (all agents act simultaneously)
        # Unlike turn-based environments, all agents observe and act at the same time
        self.env = mamujoco_v1.parallel_env(env_name, partitioning, render_mode=render_mode)
        self.env_name = env_name
        self.partitioning = partitioning

        # Store agent information
        self.possible_agents = self.env.possible_agents
        self.n_agents = len(self.possible_agents)

        # Extract observation and action spaces for each agent
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces

        # Extract dimensions as simple integers for network construction
        self.obs_dims = [
            self.observation_spaces[agent].shape[0] 
            for agent in self.possible_agents
        ]
        self.action_dims = [
            self.action_spaces[agent].shape[0] 
            for agent in self.possible_agents
        ]
        # Global state dimension = sum of all observations
        # Used by mixing network for centralized training
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
        Execute one environment step with given actions
        
        Args:
            actions: List of actions per agent
        
        Returns:
            observations: Dict mapping agent_id -> observation array
            rewards: Dict mapping agent_id -> reward (same value for cooperative task)
            terminated: Dict mapping agent_id -> bool (episode ended naturally)
            truncated: Dict mapping agent_id -> bool (episode ended by time limit)
            info: Dict with additional environment information
        """
        # Convert our list format to PettingZoo's dict format
        # PettingZoo expects: {'agent_0': action_0, 'agent_1': action_1}
        # We use: [action_0, action_1] for simpler indexing
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


