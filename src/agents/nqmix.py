"""
NQMIX: Non-Monotonic Q-Value Mixing for Continuous Action Spaces

Implementation of Algorithm 2 from:
"QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning"
Extended to continuous actions with non-monotonic mixing as per NQMIX paper.

Key innovations:
- Non-monotonic mixing network (allows negative gradients ∂Q_tot/∂Q_a)
- Sign-based actor gradient for proper credit assignment
- Works with continuous action spaces
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional

from src.networks import AgentNetwork, MixingNetwork
from src.memory import ReplayBuffer
from src.agents.base_agent import BaseAgent


class NQMIX(BaseAgent):
    """
    NQMIX agent for continuous action space (Algorithm 2).

    Implements CTDE (Centralized Training, Decentralized Execution):
    - Training: Uses global state and mixer network (centralized critic)
    - Execution: Each agent uses only local observation (decentralized actor)

    Key components:
    - Agent networks: One per agent (actor + critic)
    - Mixing network: Combines individual Q-values into Q_tot
    - Target networks: Stabilize learning (DDPG-style)
    - Seperate optimizers: One for critic, one per agent's actor
    """
    def __init__(
            self,
            n_agents: int,
            obs_dims: List[int],
            action_dims: List[int],
            state_dim: int,
            hidden_dim: int = 64,
            lr_actor: float = 5e-4,
            lr_critic: float = 5e-4,
            gamma: float = 0.99,
            tau: float = 0.001,
            buffer_capacity: int = 2000,
            action_low: float = -0.4,
            action_high: float = 0.4
    ):
        """
        Initialize NQMIX agent following paper specifications

        Args:
            n_agents: Number of agents (2 for Humanoid "9|8")
            obs_dims: List of observation dimension per agent
            action_dims: List of action dimensions per agent
            state_dim: Global state dimension (usually sum of obs_dims)
            hidden_dim: GRU hidden state size (paper uses 64)
            lr_actor: Actor learning rate (paper: 5e-4)
            lr_critic: Critic learning rate (paper: 5e-4)
            gamma: Discount factor (paper: 0.99)
            tau: Soft target update rate (paper: 0.001, very slow!)
            buffer_capacity: Replay buffer size (paper uses 5000 for experiments)
        """
        # ======================================================================
        # Store hyperparameters
        # ======================================================================
        self.n_agents = n_agents
        self.obs_dims = obs_dims  # Different per agent (heteregeneous)
        self.action_dims = action_dims # Different per agent
        self.state_dim = state_dim
        self.gamma = gamma # For TD target: R + γ*Q'
        self.tau = tau # For soft update: θ' ← τ*θ + (1-τ)*θ'
        self.action_low = action_low
        self.action_high = action_high

        # ======================================================================
        # Device setup (GPU if available, else CPU)
        # ======================================================================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # All networks will be moved to this device

        # ======================================================================
        # Initialize EVALUATION agent networks (used for current policy/critic)
        # ======================================================================
        # Create one AgentNetwork per agent
        # ModuleList allows PyTorch to track parameters properly
        # Each agent can have different obs_dim and action_dim (heterogeneous!)
        self.agent_eval = nn.ModuleList([
            AgentNetwork(
                obs_dim=obs_dims[i],       # Agent-specific observation size
                action_dim=action_dims[i], # Agent-specific action size
                hidden_dim=hidden_dim,      # Shared hidden size(64)
                action_low=action_low,
                action_high=action_high
            ).to(self.device)
            for i in range(n_agents)
        ])

        # Example for Humanoid "9|8":
        # agent_eval[0]: AgentNetwork(obs_dim=242, action_dim=9, hidden_dim=64)
        # agent_eval[1]: AgentNetwork(obs_dim=170, action_dim=8, hidden_dim=64)

        # ======================================================================
        # Initialize TARGET agent networks (used for stable TD targets)
        # ======================================================================
        # Target networks are copies of eval networks
        # Updated slowly (tau=0.001) for training stability
        # Paper: "We used tau = 0.001 for the soft target updates"
        self.agent_target = nn.ModuleList([
            AgentNetwork(
                obs_dim=obs_dims[i],
                action_dim=action_dims[i],
                hidden_dim=hidden_dim,
                action_low=action_low,
                action_high=action_high
            ).to(self.device)
            for i in range(n_agents)
        ])

        # ======================================================================
        # Initialize EVALUATION mixing network (current Q_tot)
        # ======================================================================
        # Combines individual Q-values: Q_tot = f(Q_1, ..., Q_n, state)
        # Non-monotonic: ∂Q_tot/∂Q_i can be positive OR negative
        self.mixer_eval = MixingNetwork(n_agents, state_dim).to(self.device)

        # ======================================================================
        # Initialize TARGET mixing network (stable Q_tot for TD target)
        # ======================================================================
        self.mixer_target = MixingNetwork(n_agents, state_dim).to(self.device)

        # ======================================================================
        # Initialize target networks with same weights as eval networks
        # ======================================================================
        # Hard update: Copy weights exactly (θ_target ← θ_eval)
        # This ensures target and eval start with identical parameters
        self._hard_update(self.agent_target, self.agent_eval)
        self._hard_update([self.mixer_target], [self.mixer_eval])

        # After this:
        # agent_target[i].parameters() == agent_eval[i].parameters() ✓
        # mixer_target.parameters() == mixer_eval.parameters() ✓

        # ======================================================================
        # CRITIC OPTIMIZER (single optimizer for all critic components)
        # ======================================================================
        # Paper: "We use RMSprop for learning critic parameters with lr=5e-4"

        # Critic includes:
        # 1. All agent network's critic heads (Q_a estimation)
        # 2. Mixing network (Q_tot estimation)

        # Why single optimizer?
        # - All these components work together to estimate Q_tot
        # - Gradient Flows: Q_tot -> mixer -> individual Q_values -> agent critics
        # - Joint optmization ensures consistency
        
        # Build list of critic parameters from each agent network
        critic_params = []
        for i in range(n_agents):
            critic_params.extend(self.agent_eval[i].fc1.parameters()) # shared encoder
            critic_params.extend(self.agent_eval[i].gru.parameters()) # shared GRU
            critic_params.extend(self.agent_eval[i].critic_fc.parameters()) # critic head
            critic_params.extend(self.agent_eval[i].critic_out.parameters()) # critic output

        self.critic_optimizer = torch.optim.RMSprop(
            # Combine all critic-related parameters
            critic_params +   # All agent network critic params
            list(self.mixer_eval.parameters()),    # Mixer params
            lr=lr_critic
        )

        
        # ======================================================================
        # ACTOR OPTIMIZERS (separate optimizer per agent's actor)
        # ======================================================================
        # Paper: "We use RMSprop for learning each actor policy parameters 
        #         with lr=5e-4"

        # Why separate optimizers?
        # - Each agent's actor updated independently with sign-based gradient
        # - sign(∂Q_tot/∂Q_i) different for each agent
        # - Agent i's actor only updates its own policy parameters

        # Only optimize actor-specific layers (not entire network)
        self.actor_optimizers = []
        for i in range(n_agents):
            # Only optimize actor-specific layers, NOT shared encoder/GRU
            actor_params = list(self.agent_eval[i].actor_fc.parameters()) + \
                           list(self.agent_eval[i].actor_out.parameters()) 
            self.actor_optimizers.append(
                torch.optim.RMSprop(actor_params, lr=lr_actor)
            )
            

        # Example for 2 agents:
        # actor_optimizers[0]: Optimizes agent_eval[0]'s actor (9 actions)
        # actor_optimizers[1]: Optimizes agent_eval[1]'s actor (8 actions)

        # ======================================================================
        # REPLAY BUFFER (stores complete episodes for off-policy learning)
        # ======================================================================
        self._replay_buffer = ReplayBuffer(buffer_capacity)

        # Paper: Uses ~5000 capacity for experiments
        # Stores episodes: {observations, actions, rewards, states}
        # Enables off-policy learning: learn from old policy's experience

    @property
    def replay_buffer(self):
        """Access to replay buffer (required by BaseAgent)."""
        return self._replay_buffer
    
    def _hard_update(self, target_nets: nn.ModuleList, eval_nets: nn.ModuleList) -> None:
        """
        Hard update: Copy weights exactly from eval to target networks.

        Used at initialization to ensure target and eval start identical.
        Formula: θ_target ← θ_eval (complete copy, no mixing)

        Args:
            target_nets: List of target networks to update
            eval_nets: List of eval networks to copy from
        """
        for target_net, eval_net in zip(target_nets, eval_nets):
            # load_state_dict: Copy all parameters exactly
            target_net.load_state_dict(eval_net.state_dict())

        # After this, target and eval have identical weights
        # Example: agent_target[0] becomes exact copy of agent_eval[0]

    def _soft_update(self) -> None:
        """
        Soft update: Slowly move target networks toward eval networks.
        
        Called after each training step (Algorithm 2, Line 15-16).
        Formula: θ' ← τ*θ + (1-τ)*θ'
        
        Paper: "We used τ = 0.001 for the soft target updates"
        
        Why soft update?
        - Hard update (θ' ← θ) causes instability (target jumps around)
        - Soft update (τ=0.001) means target changes by only 0.1% per step
        - Provides stable TD targets for learning
        
        Example with τ=0.001:
            If eval weight = 1.0 and target weight = 0.5:
            new_target = 0.001 * 1.0 + 0.999 * 0.5
                      = 0.001 + 0.4995
                      = 0.5005 (tiny change!)
            
            After 1000 updates: target ≈ 0.632 (slowly approaches 1.0)
        """
        # ======================================================================
        # Update all agent target networks
        # ======================================================================
        for target_agent, eval_agent in zip(self.agent_target, self.agent_eval):
            # Update every parameter in the agent network
            for target_param, eval_param in zip(
                target_agent.parameters(),
                eval_agent.parameters()
            ):
                # Polyak averaging: weighted average of eval and target
                target_param.data.copy_(
                    self.tau * eval_param.data +           # Small part from eval
                    (1 - self.tau) * target_param.data     # Large part stays same
                )
        
        # ======================================================================
        # Update mixing target network
        # ======================================================================
        for target_param, eval_param in zip(
            self.mixer_target.parameters(),
            self.mixer_eval.parameters()
        ):
            target_param.data.copy_(
                self.tau * eval_param.data +
                (1 - self.tau) * target_param.data
            )

        # Result: Target networks slowly track eval networks
        # Provides stability: TD target doesn't change drastically


    def select_actions(
            self,
            observations: List[np.ndarray],
            last_actions: List[np.ndarray],
            hiddens: List[torch.Tensor],
            explore: bool = True,
            noise_scale: float = 0.1
    ) -> Tuple[List[np.ndarray], List[torch.Tensor]]:
        """
        Select actions using deterministic policy with optional exploration noise.
        Corresponds to Algorithm 2, Line 2: u_t^a = μ_a(τ_t^a|θ) + noise

        Args:
            observation: List of observations for each agent [obs_0, obs_1]
            last_action: List of previous actions [last_act_0, last_act_1]
            hiddens: List of GRU hidden states [hidden_0, hidden_1]
            explore: Whether to add exploration noise (True for training, False for eval)
            noise_scale: Standard deviation of Gaussian noise (scales exploration)

        Returns:
            actions: List of selected actions [action_0, action_1]
                     Each action is numpy array in [-0.4, 0.4] for Humanoid
            new_hiddens: List of updated GRU hidden states for next timestep

        Note: Uses torch.no_grad() because this is action selection, not training.
              No gradient computation needed -> saves memory and computation
        """
        actions = []
        new_hiddens = []

        # Disable gradient computation (this is inference, not training)
        with torch.no_grad():
            for i in range(self.n_agents):
                # Convert numpy arrays to PyTorch tensors
                # unsqueeze(0) adds batch dimension: (obs_dim,) -> (1, obs_dim)
                obs = torch.FloatTensor(observations[i]).unsqueeze(0).to(self.device)
                last_act = torch.FloatTensor(last_actions[i]).unsqueeze(0).to(self.device)
                hidden = hiddens[i].to(self.device)

                # Get action from policy (no current_action -> critic return None)
                # action = μ_a(τ_t^a|θ) from actor network
                _, action, new_hidden = self.agent_eval[i](obs, last_act, hidden)
                # action shape: (1, action_dim), already scaled to [-0.4, 0.4]

                if explore:
                    # Add Gaussian noise for exploration: N(0, noise_scale²)
                    noise = torch.randn_like(action) * noise_scale
                    action = action + noise

                    # Clamp to correct action space bounds!
                    action = torch.clamp(
                        action,
                        self.agent_eval[i].action_low,  
                        self.agent_eval[i].action_high  
                    )
                
                # Convert back to numpy and remove batch dimension
                # (1, action_dim) -> (action_dim)
                actions.append(action.cpu().numpy()[0])
                new_hiddens.append(new_hidden)
        
        return actions, new_hiddens
    

    def init_hidden_states(self) -> List[torch.Tensor]:
        """
        Initialize GRU hidden states for all agents at episode start.
        
        Returns:
            List of zero-initialized hidden states [hidden_0, hidden_1]
            Each has shape (1, hidden_dim) for batch_size=1
        
        Usage:
            hiddens = nqmix.init_hidden_states()  # Start of episode
            actions, hiddens = nqmix.select_actions(obs, last_act, hiddens)  # Use
        """
        return [agent.init_hidden().to(self.device) for agent in self.agent_eval]
    
    
    def train_step(self, batch_size: int = 32) -> Optional[Dict[str, float]]:
        """
        Training step implementing Algorithm 2 from NQMIX paper.

        Algorithm 2 structure:
        1. Sample mini-batch of episodes (Line 4)
        2. For each episode:
           a. Initialize discount accumulator I = 1 (Line 5)
           b. For each timestep t (Line 6):
              - Critic update: Minimize TD error (Lines 7-12)
              - Actor update: Sign-based gradient (Line 13)
              - Decay I ← γ*I (Line 14)
        3. Soft update target networks (Lines 15-16)

        Returns:
            Average critic loss across batch (for monitoring)
        """
        # Wait until buffer has enough episodes
        if len(self.replay_buffer) < batch_size:
            return None
        
        # Line 4: Sample mini-batch from replay buffer (OFF-POLICY!)
        episodes = self.replay_buffer.sample(batch_size)
        total_critic_loss = 0.0
        total_actor_loss = 0.0

        # Process each episode in the batch
        for episode in episodes:
            # Validate episode data structure
            T = len(episode['rewards'])
            for i in range(self.n_agents):
                if len(episode['observations'][i]) != T:
                    raise ValueError(f"Agent {i} observations length {len(episode['observations'][i])} != rewards length {T}")
                if len(episode['actions'][i]) != T:
                    raise ValueError(f"Agent {i} actions length {len(episode['actions'][i])} != rewards length {T}")
                if len(episode['last_actions'][i]) != T:
                    raise ValueError(f"Agent {i} last_actions length {len(episode['last_actions'][i])} != rewards length {T}")
            if len(episode['states']) != T:
                raise ValueError(f"States length {len(episode['states'])} != rewards length {T}")

            # Initialize hidden states for this episode
            # Each episode starts with empty history (zero hidden states)
            hiddens_eval = [agent.init_hidden(1).to(self.device)
                           for agent in self.agent_eval]
            hiddens_target = [agent.init_hidden(1).to(self.device)
                             for agent in self.agent_target]

            episode_critic_loss = 0.0
            episode_actor_loss = 0.0

            # Line 5: I ← 1 (Discount accumulator for multi-step updates)
            # Purpose: Weight gradients by temporal discount γ^t
            I = 1.0

            # Line 6: For each timestep t in episode
            for t in range(len(episode['rewards'])):
                # ========================================================
                # PREPARE DATA FOR TIMESTEP t
                # ========================================================
                # Convert all data to PyTorch tensors and move to device
                obs_t = [torch.FloatTensor(episode['observations'][i][t]).unsqueeze(0).to(self.device)
                        for i in range(self.n_agents)]
                actions_t = [torch.FloatTensor(episode['actions'][i][t]).unsqueeze(0).to(self.device)
                            for i in range(self.n_agents)]
                last_actions_t = [torch.FloatTensor(episode['last_actions'][i][t]).unsqueeze(0).to(self.device)
                                 for i in range(self.n_agents)]
                state_t = torch.FloatTensor(episode['states'][t]).unsqueeze(0).to(self.device)
                # Reward tensor shape [1, 1] to match mixer output shape
                reward = torch.FloatTensor([[episode['rewards'][t]]]).to(self.device)

                # ========================================================
                # CRITIC UPDATE (Algorithm 2, Lines 7-12)
                # ========================================================

                # Line 7: Q_a(τ_t^a, u_t^a) ← Agent_eval(o_t^a, u_{t-1}^a, u_t^a) ∀a
                # Evaluate Q-values for actions that were ACTUALLY TAKEN (from buffer)
                q_values_eval = []
                new_hiddens_eval = []
                for i in range(self.n_agents):
                    q_val, _, new_hidden = self.agent_eval[i](
                        obs_t[i],           # Current observation
                        last_actions_t[i],  # Previous action
                        hiddens_eval[i],    # Current hidden state (history τ)
                        actions_t[i]        # Action to evaluate (from buffer)
                    )
                    q_values_eval.append(q_val)
                    new_hiddens_eval.append(new_hidden)
                
                # Line 9: Q_tot ← Mixing_eval[Q_1, Q_2, ..., Q_n, s_t]
                # Combine individual Q-values into joint Q-value
                q_tot = self.mixer_eval(q_values_eval, state_t)

                # ========================================================
                # COMPUTE TD TARGET (Lines 8, 10-11)
                # ========================================================
                if t < len(episode['rewards']) - 1:  # Not terminal state
                    # Get next state data
                    obs_next = [torch.FloatTensor(episode['observations'][i][t+1]).unsqueeze(0).to(self.device)
                                for i in range(self.n_agents)]
                    state_next = torch.FloatTensor(episode['states'][t+1]).unsqueeze(0).to(self.device)

                    # Compute target Q-value (no gradients needed)
                    with torch.no_grad():
                        q_values_target = []
                        new_hiddens_target = []
                        
                        for i in range(self.n_agents):
                            # Line 8: Get action from target policy
                            # μ_a(τ_{t+1}^a|θ') - what would target policy do?
                            _, target_action, new_hidden_target = self.agent_target[i](
                                obs_next[i],
                                actions_t[i],  # Last action at t (for GRU input)
                                hiddens_target[i]
                            )
                            
                            # Evaluate Q-value for target action
                            # Q'_a(τ_{t+1}^a, μ_a(τ_{t+1}^a|θ'))
                            q_val, _, _ = self.agent_target[i](
                                obs_next[i],
                                actions_t[i],
                                hiddens_target[i],
                                target_action  # Evaluate target policy action
                            )
                            q_values_target.append(q_val)
                            new_hiddens_target.append(new_hidden_target)
                        
                        # Line 10: Mix target Q-values
                        # Q'_tot ← Mixing_target[Q'_1, ..., Q'_n, s_{t+1}]
                        q_tot_target = self.mixer_target(q_values_target, state_next)
                        
                        # Line 11: TD target = R + γ * Q'_tot
                        td_target = reward + self.gamma * q_tot_target
                else:
                    # Terminal state: no future value
                    td_target = reward

                # Line 11: Compute TD error
                # δ = target - current = [R + γ*Q'_tot] - Q_tot
                td_error = td_target.detach() - q_tot
                
                # Line 12: Critic loss = δ²
                # Minimize squared TD error (standard Q-learning)
                critic_loss = td_error.pow(2).mean()

                # Update critic parameters
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                
                # Gradient clipping prevents exploding gradients
                torch.nn.utils.clip_grad_norm_(
                    list(self.agent_eval.parameters()) + list(self.mixer_eval.parameters()),
                    max_norm=10.0
                )
                self.critic_optimizer.step()

                episode_critic_loss += critic_loss.item()

                # ========================================================
                # DETACH HIDDEN STATES (Memory Management)
                # ========================================================
                # Prevents backpropagation through entire episode history
                # (Truncated BPTT - necessary for long episodes)
                hiddens_eval = [h.detach() for h in new_hiddens_eval]
                if t < len(episode['rewards']) - 1:
                    hiddens_target = [h.detach() for h in new_hiddens_target]

                # ========================================================
                # ACTOR UPDATE (Algorithm 2, Line 13 - KEY INNOVATION!)
                # ========================================================
                # This is what makes NQMIX different from QMIX:
                # Sign-based gradient allows non-monotonic coordination
                
                # Step 1: Get current policy actions (not from buffer!)
                current_actions = []
                current_hiddens_for_actor = [h.detach() for h in hiddens_eval]
                
                for i in range(self.n_agents):
                    # Get action from CURRENT policy μ_a(τ_t^a|θ)
                    _, act, _ = self.agent_eval[i](
                        obs_t[i].detach(),
                        last_actions_t[i].detach(),
                        current_hiddens_for_actor[i]
                    )
                    current_actions.append(act)

                # Step 2: Evaluate Q-values for current policy actions
                q_values_for_actor = []
                for i in range(self.n_agents):
                    q_val, _, _ = self.agent_eval[i](
                        obs_t[i].detach(),
                        last_actions_t[i].detach(),
                        current_hiddens_for_actor[i],
                        current_actions[i]  # Evaluate current policy action
                    )
                    q_values_for_actor.append(q_val)

                # Step 3: Compute joint Q-value for gradient computation
                q_tot_for_actor = self.mixer_eval(q_values_for_actor, state_t.detach())

                # Step 4: Update each agent's actor with sign-based gradient
                # Line 13: θ ← θ + α·I·sign(∂Q_tot/∂Q_a)·∇_θμ_a·∇_uQ_a
                for i in range(self.n_agents):
                    # Compute ∂Q_tot/∂Q_a (how Q_tot changes with Q_a)
                    grad_q_tot_wrt_qa = torch.autograd.grad(
                        q_tot_for_actor,
                        q_values_for_actor[i],
                        retain_graph=True,  # Keep graph for other agents
                        create_graph=False  # Don't need second-order gradients
                    )[0]

                    # Check for NaN/Inf gradients
                    if torch.isnan(grad_q_tot_wrt_qa).any() or torch.isinf(grad_q_tot_wrt_qa).any():
                        # Skip this agent's update if gradient is invalid
                        sign_grad = torch.zeros_like(grad_q_tot_wrt_qa)
                    else:
                        # KEY INNOVATION: Extract sign of gradient
                        # sign > 0: Agent helps team → gradient ASCENT (maximize Q_a)
                        # sign < 0: Agent hurts team → gradient DESCENT (minimize Q_a)
                        sign_grad = torch.sign(grad_q_tot_wrt_qa).detach()
                    
                    # Actor loss with sign weighting and temporal discount I
                    # Negative sign for gradient ascent when sign_grad > 0
                    actor_loss = -(sign_grad * q_values_for_actor[i] * I).mean()
                    
                    # Update this agent's actor parameters
                    self.actor_optimizers[i].zero_grad()
                    actor_loss.backward(retain_graph=(i < self.n_agents - 1))
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(
                        list(self.agent_eval[i].actor_fc.parameters()) +
                        list(self.agent_eval[i].actor_out.parameters()),
                        max_norm=10.0
                    )
                    self.actor_optimizers[i].step()
                    
                    episode_actor_loss += actor_loss.item()

                # Line 14: I ← γ·I (Decay discount accumulator)
                # After T steps: I = γ^T (weights later timesteps less)
                I *= self.gamma

            # Accumulate losses across episodes
            total_critic_loss += episode_critic_loss
            total_actor_loss += episode_actor_loss

        # Line 15-16: Soft update target networks
        # θ' ← τ·θ + (1-τ)·θ' (slow tracking for stability)
        self._soft_update()

        # Return average losses for monitoring
        avg_critic_loss = total_critic_loss / len(episodes)
        avg_actor_loss = total_actor_loss / len(episodes)

        return avg_critic_loss
    
    
    def save(self, path: str) -> None:
        """
        Save model checkpoint to disk.

        Saves:
        - All agent eval networks (current policy/critic)#
        - All agent target networks (stable targets)
        - Mixer eval network
        - Mixer target network

        Note: Does NOT save optimizer states. If you need to resume training
        exactly, also save optmizer.state_dict()

        Args:
            path: File path to save checkpoint (e.g., "model.pth")
        """
        torch.save({
            'agent_eval': [a.state_dict() for a in self.agent_eval],
            'agent_target': [a.state_dict() for a in self.agent_target],
            'mixer_eval': self.mixer_eval.state_dict(),
            'mixer_target': self.mixer_target.state_dict(),
        }, path)

    
    def load(self, path: str) -> None:
        """
        Load model checkpoint from disk.

        Load all network parameters and sets model to evaluation mode.
        Use this for:
        - Continuing training from checkpoint
        - Evaluating a trainded model
        - Deploying the policy

        Args:
            path: File path to load checkpoint from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load all agent networks
        for i, agent in enumerate(self.agent_eval):
            agent.load_state_dict(checkpoint['agent_eval'][i])
        for i, agent in enumerate(self.agent_target):
            agent.load_state_dict(checkpoint['agent_target'][i])
        
        # Load mixer networks
        self.mixer_eval.load_state_dict(checkpoint['mixer_eval'])
        self.mixer_target.load_state_dict(checkpoint['mixer_target'])



    