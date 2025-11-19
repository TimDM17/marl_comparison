import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List



class MixingNetwork(nn.Module):
    """
    Non-monotonic mixing network for NQMIX

    Architecture:
        - Input: [Q_1, Q_2, ..., Q_n, state]
        - Hidden layer: 32 * (n_agents + 4) - n_agents units with ReLU
        - Output layer: 1 unit (Q_tot)

    Key difference from QMIX:
        - QMIX: Uses hypernetworks with non-negative weights (monotonic)
        - NQMIX: Uses simple MLP with no constraints (non_monotonic)

    This allows ∂Q_tot/∂Q_a to be positive OR negative!
    """
    def __init__(self, n_agents: int, state_dim: int):
        """
        Initialize mixing network following NQMIX paper specification

        Args:
            n_agents: Number of Agents (2 for MaMuJoCo Humanoid 9"|8")
            state_dim: Dimension of gloab state
                        (For Humanoid: sum of both agent's observations)
        
        Paper formula for hidden layer size:
            hidden_dim = 32 x (n_agents + 4) - n_agents
        
        Example for 2 agents:
            hidden_dim = 32 x (2 + 4) - 2 = 192 - 2 = 190
        """
        super(MixingNetwork, self).__init__()

        # Input: concatenation of all Q-values and global state
        #[Q_1, Q_2, ..., Q_n, s] where each Q_i is scalar
        input_dim = n_agents + state_dim

        # Hidden layer size from paper 
        # Formula designed to match parameter count with QMIX's hypernetwork
        # For 2 agents: 32 × (2+4) - 2 = 190 units
        hidden_dim = 32 * (n_agents + 4) - n_agents

        # ==============================================
        # Layer 1: Input -> Hidden (with ReLU activation)
        # ==============================================
        # Takes all individual Q-values plus global state
        # No weight constraint (unlike QMIX which enforces non-negative)
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # ==============================================
        # Layer 2: Hidden -> Output (single Q-tot value)
        # ==============================================
        # Outputs scalar: Q_tot = f(Q_1, ..., Q_n, s)
        # Can be non-monotonic: ∂Q_tot/∂Q_i can be negative!
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, q_values: List[torch.Tensor], state: torch.Tensor) -> torch.Tensor:
        """
        Mix individual Q-values into joint Q_tot

        This is the centralized critic in Centralized Training 
        Decentralized Execution framework

        Args:
            q_values : List of individual Q-values, one per agent
                       [Q_1, Q_2, ..., Q_n] where Q_i has shape [batch, 1]
            state: Global state [batch, state_dim]
                   Contains full information (both agent's observations)

        Returns:
            q_tot: Joint action_value Q_tot(τ, u) [batch, 1]
                   Represents value of joint action given full state
        
        Mathematical representation:
            Q_tot = f([Q_1(τ_1, u_1), Q_2(τ_2, u_2), ..., Q_n(τ_n, u_n)], s)
        
        Key property (enables non-monotonic coordination):
            ∂Q_tot/∂Q_i can be positive (agent helps) OR negative (agent should reduce)
        """
        # ==============================================
        # Step 1: Concatenate all Q-values
        # ==============================================
        # q_values is a list: [Q_1, Q_2] for 2 agents
        # Each Q_i has shape [batch, 1]
        # After concat: [batch, n_agents]
        q_cat = torch.cat(q_values, dim=-1) # [batch, 2] for 2 agents

        # ==============================================
        # Step 2: Concatenate Q-values with global state
        # ==============================================
        # Combine individual values with global context
        # This allows mixer to reason about coordination given full information
        # Shape: [batch, n_agents + state_dim]
        x = torch.cat([q_cat, state], dim=-1)

        # ==============================================
        # Step 3: Pass through MLP
        # ==============================================
        # First layer with ReLU non-linearity
        x = F.relu(self.fc1(x)) # [batch, hidden_dim]

        # Output layer (no activation)
        q_tot = self.fc2(x)    # [batch, 1]

        return q_tot
    
