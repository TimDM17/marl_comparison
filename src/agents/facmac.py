"""
FACMAC: Factored Multi-Agent Centralised Policy Gradients

This implementation uses non-monotonic mixing (FACMAC-nonmonotonic variant)
Paper shows this often outperforms monotonic version

Only 2 changes from NQMIX:
1. Single joint actor optimizer (vs per-agent optimizers)
2. Direct policy gradient (vs sign-based gradient)

Paper quote:

"One key advantage of adopting value factorisation in an actor-critic framework
is that ... One can employ any type of factorisation, including nonmonotonic
factorisations that value-based methods cannot directly use."
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional

from src.networks import AgentNetwork, MixingNetwork # Reuse NQMIX's non-monotonic mixer!
from src.memory import ReplayBuffer
from src.agents.base_agent import BaseAgent

class FACMAC(BaseAgent):
    """
    FACMAC agent with non-monotonic mixing (FACMAC-nonmonotonic variants)

    This is 99% identical to NQMIX with only 2 changes:

    Similarities (same as NQMIX):
    - Agent networks (actor + critic per agent)
    - Non-monotonic mixing network (MixingNetwork)
    - Replay buffer and episode storage
    - Critic update (TD error minimization)
    - Target networks with soft update
    - Action selection with exploration

    Differences from NQMIX (only 2!):
    1. Single joint actor optimizer (vs per-agent optimizers)
    2. Direct policy gradient (vs sign-based gradient)
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
        buffer_capacity: int = 5000,
        action_low: float = -0.4,
        action_high: float = 0.4
    ):
        """Initialize FACMAC agent (same parameters as NQMIX)"""
        # Store hyperparameters (identical to NQMIX)
        self.n_agents = n_agents
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.action_low = action_low
        self.action_high = action_high
        
        # Device setup (identical to NQMIX)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize agent networks (identical to NQMIX)
        self.agent_eval = nn.ModuleList([
            AgentNetwork(
                obs_dim=obs_dims[i],
                action_dim=action_dims[i],
                hidden_dim=hidden_dim,
                action_low=action_low,
                action_high=action_high
            ).to(self.device)
            for i in range(n_agents)
        ])
        
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
        # NON-MONOTONIC mixing network (SAME as NQMIX!)
        # ======================================================================
        # Paper (Section 3.1): FACMAC can use "nonmonotonic factorisations"
        # Figure 9 shows FACMAC-nonmonotonic works well (sometimes better than monotonic)
        self.mixer_eval = MixingNetwork(n_agents, state_dim).to(self.device)
        self.mixer_target = MixingNetwork(n_agents, state_dim).to(self.device)
        
        # Initialize target networks (identical to NQMIX)
        self._hard_update(self.agent_target, self.agent_eval)
        self._hard_update([self.mixer_target], [self.mixer_eval])
        
        # Critic optimizer - ONLY critic parameters, NOT actor!
        # Reference: facmac_learner.py separates critic from actor completely
        critic_params = []
        for agent in self.agent_eval:
            # Only include critic layers (NOT actor_fc/actor_out)
            critic_params += list(agent.fc1.parameters())
            critic_params += list(agent.gru.parameters())
            critic_params += list(agent.critic_fc.parameters())
            critic_params += list(agent.critic_out.parameters())
        critic_params += list(self.mixer_eval.parameters())
        self.critic_params = critic_params  # Store for gradient clipping in train_step

        # Reference: weight_decay: True, weight_decay_factor: 0.0001
        # Critical for preventing Q-value explosion!
        self.critic_optimizer = torch.optim.Adam(
            critic_params,
            lr=lr_critic,
            eps=0.01,  # Reference: optimizer_epsilon: 0.01
            weight_decay=1e-4  # Reference: weight_decay_factor: 0.0001
        )
        
        # ======================================================================
        # CHANGE 1: Single JOINT actor optimizer (ONLY difference in __init__)
        # ======================================================================
        # NQMIX has: self.actor_optimizers = [RMSprop(...) for each agent]
        # FACMAC has: self.actor_optimizer = RMSprop(all actor params)
        #
        # This enables: ∇θ J(µ) where µ = {µ1, ..., µn} (Equation 7, pg 5)
        actor_params = []
        for agent in self.agent_eval:
            actor_params += list(agent.actor_fc.parameters())
            actor_params += list(agent.actor_out.parameters())
        self.actor_params = actor_params  # Store for gradient clipping in train_step

        # Use Adam optimizer as per reference (optimizer: adam)
        self.actor_optimizer = torch.optim.Adam(
            actor_params,
            lr=lr_actor,
            eps=0.01,  # Reference: optimizer_epsilon: 0.01
            weight_decay=1e-4  # Reference: weight_decay_factor: 0.0001
        )
        
        # Replay buffer (identical to NQMIX)
        self._replay_buffer = ReplayBuffer(buffer_capacity)
    
    @property
    def replay_buffer(self):
        """Access to replay buffer (identical to NQMIX)"""
        return self._replay_buffer
    
    def _hard_update(self, target_nets: nn.ModuleList, eval_nets: nn.ModuleList) -> None:
        """Hard update target networks (identical to NQMIX)"""
        for target_net, eval_net in zip(target_nets, eval_nets):
            target_net.load_state_dict(eval_net.state_dict())
    
    def _soft_update(self) -> None:
        """Soft update target networks (identical to NQMIX)"""
        for target_agent, eval_agent in zip(self.agent_target, self.agent_eval):
            for target_param, eval_param in zip(
                target_agent.parameters(),
                eval_agent.parameters()
            ):
                target_param.data.copy_(
                    self.tau * eval_param.data + (1 - self.tau) * target_param.data
                )
        
        for target_param, eval_param in zip(
            self.mixer_target.parameters(),
            self.mixer_eval.parameters()
        ):
            target_param.data.copy_(
                self.tau * eval_param.data + (1 - self.tau) * target_param.data
            )
    
    def select_actions(
        self,
        observations: List[np.ndarray],
        last_actions: List[np.ndarray],
        hiddens: List[torch.Tensor],
        explore: bool = True,
        noise_scale: float = 0.1
    ) -> Tuple[List[np.ndarray], List[torch.Tensor]]:
        """Select actions (100% identical to NQMIX)"""
        actions = []
        new_hiddens = []

        with torch.no_grad():
            for i in range(self.n_agents):
                obs = torch.FloatTensor(observations[i]).unsqueeze(0).to(self.device)
                last_act = torch.FloatTensor(last_actions[i]).unsqueeze(0).to(self.device)
                hidden = hiddens[i].to(self.device)

                _, action, new_hidden = self.agent_eval[i](obs, last_act, hidden)

                if explore:
                    noise = torch.randn_like(action) * noise_scale
                    action = action + noise
                    action = torch.clamp(
                        action,
                        self.agent_eval[i].action_low,
                        self.agent_eval[i].action_high
                    )

                actions.append(action.cpu().numpy()[0])
                new_hiddens.append(new_hidden)

        return actions, new_hiddens

    def select_actions_batched(
        self,
        observations_batch: List[List[np.ndarray]],
        last_actions_batch: List[List[np.ndarray]],
        hiddens_batch: List[List[torch.Tensor]],
        explore: bool = True,
        noise_scale: float = 0.1
    ) -> Tuple[List[List[np.ndarray]], List[List[torch.Tensor]]]:
        """
        GPU-batched action selection for all environments in parallel.

        Key optimization: Instead of processing n_envs sequentially,
        batch all environments together for each agent network.

        Args:
            observations_batch: [n_envs][n_agents] observations
            last_actions_batch: [n_envs][n_agents] last actions
            hiddens_batch: [n_envs][n_agents] hidden states

        Returns:
            actions_batch: [n_envs][n_agents] actions
            new_hiddens_batch: [n_envs][n_agents] new hidden states
        """
        n_envs = len(observations_batch)

        # Reorganize: [n_envs][n_agents] -> [n_agents][n_envs]
        # This allows batching all envs for each agent network
        obs_per_agent = [
            [observations_batch[env][agent] for env in range(n_envs)]
            for agent in range(self.n_agents)
        ]
        last_act_per_agent = [
            [last_actions_batch[env][agent] for env in range(n_envs)]
            for agent in range(self.n_agents)
        ]
        hidden_per_agent = [
            [hiddens_batch[env][agent] for env in range(n_envs)]
            for agent in range(self.n_agents)
        ]

        # Process each agent with batched envs
        actions_per_agent = []
        new_hiddens_per_agent = []

        with torch.no_grad():
            for i in range(self.n_agents):
                # Stack all envs for this agent: [n_envs, feature_dim]
                obs_stacked = torch.FloatTensor(np.stack(obs_per_agent[i])).to(self.device)
                last_act_stacked = torch.FloatTensor(np.stack(last_act_per_agent[i])).to(self.device)
                hidden_stacked = torch.cat(hidden_per_agent[i], dim=0).to(self.device)

                # Forward pass with batch_size = n_envs
                _, action_batch, new_hidden_batch = self.agent_eval[i](
                    obs_stacked, last_act_stacked, hidden_stacked
                )

                if explore:
                    noise = torch.randn_like(action_batch) * noise_scale
                    action_batch = action_batch + noise
                    action_batch = torch.clamp(
                        action_batch,
                        self.agent_eval[i].action_low,
                        self.agent_eval[i].action_high
                    )

                # Split back to list of [n_envs] actions
                actions_np = action_batch.cpu().numpy()
                actions_per_agent.append([actions_np[env] for env in range(n_envs)])

                # Split hidden states: [n_envs, hidden_dim] -> list of [1, hidden_dim]
                new_hiddens_per_agent.append([
                    new_hidden_batch[env:env+1] for env in range(n_envs)
                ])

        # Reorganize back: [n_agents][n_envs] -> [n_envs][n_agents]
        actions_batch_out = [
            [actions_per_agent[agent][env] for agent in range(self.n_agents)]
            for env in range(n_envs)
        ]
        new_hiddens_batch_out = [
            [new_hiddens_per_agent[agent][env] for agent in range(self.n_agents)]
            for env in range(n_envs)
        ]

        return actions_batch_out, new_hiddens_batch_out
    
    def init_hidden_states(self) -> List[torch.Tensor]:
        """Initialize hidden states (100% identical to NQMIX)"""
        return [agent.init_hidden().to(self.device) for agent in self.agent_eval]
    
    def train_step(self, batch_size: int = 32) -> Optional[Dict[str, float]]:
        """
        FACMAC training step with BATCHED tensor operations (research standard).

        Key optimization: Process entire batch in parallel instead of sequential loops.
        Reference: facmac-main/src/learners/facmac_learner.py

        Structure:
        1. Sample batch as pre-padded tensors
        2. Loop only over TIME (not episodes) - O(T) instead of O(batch*T)
        3. Process all episodes in parallel at each timestep
        4. Use masking for variable-length episodes
        """
        if len(self.replay_buffer) < batch_size:
            return None

        # Get batched tensors (research standard: pre-pad and stack)
        batch = self.replay_buffer.sample_batch(batch_size, self.device)

        observations = batch['observations']      # List of [B, T, obs_dim]
        actions = batch['actions']                # List of [B, T, action_dim]
        last_actions = batch['last_actions']      # List of [B, T, action_dim]
        states = batch['states']                  # [B, T, state_dim]
        rewards = batch['rewards']                # [B, T, 1]
        mask = batch['mask']                      # [B, T, 1]
        terminated = batch['terminated']          # [B, T, 1] - 1 if terminal, 0 otherwise
        max_seq_length = batch['max_seq_length']
        B = batch['batch_size']

        # Initialize hidden states for entire batch
        hiddens_eval = [agent.init_hidden(B).to(self.device) for agent in self.agent_eval]
        hiddens_target = [agent.init_hidden(B).to(self.device) for agent in self.agent_target]

        # Collect Q-values over time (for critic)
        q_taken_list = []

        # Collect target Q-values over time
        target_q_list = []

        # Collect actor outputs over time
        actor_actions_list = []
        actor_q_list = []

        # ================================================================
        # FORWARD PASS: Loop only over TIME (batch processed in parallel)
        # ================================================================
        for t in range(max_seq_length):
            # Get data at timestep t for ALL episodes: [B, feature_dim]
            obs_t = [observations[i][:, t, :] for i in range(self.n_agents)]
            actions_t = [actions[i][:, t, :] for i in range(self.n_agents)]
            last_actions_t = [last_actions[i][:, t, :] for i in range(self.n_agents)]
            state_t = states[:, t, :]  # [B, state_dim]

            # ----- CRITIC: Evaluate Q for actions taken -----
            q_values_eval = []
            new_hiddens_eval = []
            for i in range(self.n_agents):
                q_val, _, new_hidden = self.agent_eval[i](
                    obs_t[i], last_actions_t[i], hiddens_eval[i], actions_t[i]
                )
                q_values_eval.append(q_val)  # [B, 1]
                new_hiddens_eval.append(new_hidden)

            q_tot = self.mixer_eval(q_values_eval, state_t)  # [B, 1]
            q_taken_list.append(q_tot)

            # ----- TARGET: Compute target Q-values -----
            with torch.no_grad():
                q_values_target = []
                new_hiddens_target = []
                for i in range(self.n_agents):
                    # Get target action
                    _, target_action, new_hidden_target = self.agent_target[i](
                        obs_t[i], last_actions_t[i], hiddens_target[i]
                    )
                    # Evaluate Q for target action (with updated hidden)
                    q_val, _, _ = self.agent_target[i](
                        obs_t[i], last_actions_t[i], new_hidden_target, target_action
                    )
                    q_values_target.append(q_val)
                    new_hiddens_target.append(new_hidden_target)

                q_tot_target = self.mixer_target(q_values_target, state_t)
                target_q_list.append(q_tot_target)

            # ----- ACTOR: Get current policy actions and Q-values -----
            current_actions = []
            q_values_for_actor = []
            actor_hiddens = [h.detach() for h in hiddens_eval]

            for i in range(self.n_agents):
                _, act, _ = self.agent_eval[i](
                    obs_t[i].detach(), last_actions_t[i].detach(), actor_hiddens[i]
                )
                current_actions.append(act)

            for i in range(self.n_agents):
                q_val, _, _ = self.agent_eval[i](
                    obs_t[i].detach(), last_actions_t[i].detach(),
                    actor_hiddens[i], current_actions[i]
                )
                q_values_for_actor.append(q_val)

            q_tot_actor = self.mixer_eval(q_values_for_actor, state_t.detach())
            actor_q_list.append(q_tot_actor)
            actor_actions_list.append(torch.cat(current_actions, dim=-1))

            # Update hidden states (detach to prevent BPTT through entire episode)
            hiddens_eval = [h.detach() for h in new_hiddens_eval]
            hiddens_target = [h.detach() for h in new_hiddens_target]

        # ================================================================
        # COMPUTE LOSSES (vectorized over entire batch)
        # ================================================================
        # Stack over time: [B, T, 1]
        q_taken = torch.stack(q_taken_list, dim=1)
        target_q = torch.stack(target_q_list, dim=1)
        actor_q = torch.stack(actor_q_list, dim=1)
        actor_actions = torch.stack(actor_actions_list, dim=1)  # [B, T, total_action_dim]

        # Compute TD targets: r_t + gamma * (1 - terminated) * Q'(s_{t+1}, a'_{t+1})
        # Shift target_q by 1 timestep (target at t uses Q from t+1)
        target_q_shifted = torch.zeros_like(target_q)
        target_q_shifted[:, :-1, :] = target_q[:, 1:, :]  # Q'(s_{t+1})

        # Create mask for valid timesteps (exclude padding)
        # We need next timestep to be valid for TD target
        mask_shifted = torch.zeros_like(mask)
        mask_shifted[:, :-1, :] = mask[:, 1:, :]  # Shift mask by 1
        mask_for_critic = mask * mask_shifted  # Both current AND next must be valid

        # TD target with proper terminal handling (reference: facmac_learner.py line 105)
        # For terminal states: target = reward (no bootstrap, gamma * 0 * Q' = 0)
        # For non-terminal: target = reward + gamma * Q'
        td_targets = rewards + self.gamma * (1.0 - terminated) * target_q_shifted

        # TD error
        td_error = (td_targets.detach() - q_taken)

        # Masked TD error (only include valid timesteps)
        masked_td_error = td_error * mask_for_critic
        critic_loss = (masked_td_error ** 2).sum() / mask_for_critic.sum().clamp(min=1)

        # Actor loss: maximize Q, with action regularization
        # Reference: pg_loss = -chosen_action_qvals.mean() + (pi**2).mean() * 1e-3
        mask_for_actor = mask[:, :-1, :]  # Exclude last timestep
        actor_q_masked = actor_q[:, :-1, :] * mask_for_actor
        action_reg = (actor_actions[:, :-1, :] ** 2 * mask_for_actor).sum() / mask_for_actor.sum().clamp(min=1)
        actor_loss = -actor_q_masked.sum() / mask_for_actor.sum().clamp(min=1) + action_reg * 1e-3

        # ================================================================
        # GRADIENT UPDATE
        # ================================================================
        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(self.critic_params, max_norm=0.5)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_params, max_norm=0.5)

        self.critic_optimizer.step()
        self.actor_optimizer.step()

        # Soft update targets
        self._soft_update()

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }
    
    def save(self, path: str) -> None:
        """Save model (100% identical to NQMIX)"""
        torch.save({
            'agent_eval': [a.state_dict() for a in self.agent_eval],
            'agent_target': [a.state_dict() for a in self.agent_target],
            'mixer_eval': self.mixer_eval.state_dict(),
            'mixer_target': self.mixer_target.state_dict(),
        }, path)
    
    def load(self, path: str) -> None:
        """Load model (100% identical to NQMIX)"""
        checkpoint = torch.load(path, map_location=self.device)
        
        for i, agent in enumerate(self.agent_eval):
            agent.load_state_dict(checkpoint['agent_eval'][i])
        for i, agent in enumerate(self.agent_target):
            agent.load_state_dict(checkpoint['agent_target'][i])
        
        self.mixer_eval.load_state_dict(checkpoint['mixer_eval'])
        self.mixer_target.load_state_dict(checkpoint['mixer_target'])