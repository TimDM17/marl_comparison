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

        self.critic_optimizer = torch.optim.Adam(
            critic_params,
            lr=lr_critic,
            eps=0.01  # Reference uses optimizer_epsilon: 0.01
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
            eps=0.01  # Reference: optimizer_epsilon: 0.01
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
    
    def init_hidden_states(self) -> List[torch.Tensor]:
        """Initialize hidden states (100% identical to NQMIX)"""
        return [agent.init_hidden().to(self.device) for agent in self.agent_eval]
    
    def train_step(self, batch_size: int = 32) -> Optional[Dict[str, float]]:
        """
        FACMAC training step with centralised policy gradient (CORRECTED: Batched updates)

        FIXED: Now accumulates gradients across entire batch, then updates once.
        Research-standard implementation: 1 gradient update per train_step (not per timestep!)
        """
        if len(self.replay_buffer) < batch_size:
            return None

        episodes = self.replay_buffer.sample(batch_size)

        # Collect losses across all episodes and timesteps (avoid in-place operations)
        critic_losses = []
        actor_losses = []

        for episode in episodes:
            # Validate episode
            T = len(episode['rewards'])
            for i in range(self.n_agents):
                assert len(episode['observations'][i]) == T
                assert len(episode['actions'][i]) == T
                assert len(episode['last_actions'][i]) == T
            assert len(episode['states']) == T

            # Initialize hiddens
            hiddens_eval = [agent.init_hidden(1).to(self.device) for agent in self.agent_eval]
            hiddens_target = [agent.init_hidden(1).to(self.device) for agent in self.agent_target]

            for t in range(len(episode['rewards'])):
                # Prepare data
                obs_t = [torch.FloatTensor(episode['observations'][i][t]).unsqueeze(0).to(self.device)
                        for i in range(self.n_agents)]
                actions_t = [torch.FloatTensor(episode['actions'][i][t]).unsqueeze(0).to(self.device)
                            for i in range(self.n_agents)]
                last_actions_t = [torch.FloatTensor(episode['last_actions'][i][t]).unsqueeze(0).to(self.device)
                                 for i in range(self.n_agents)]
                state_t = torch.FloatTensor(episode['states'][t]).unsqueeze(0).to(self.device)
                reward = torch.FloatTensor([[episode['rewards'][t]]]).to(self.device)

                # ============================================================
                # CRITIC LOSS ACCUMULATION (No backward/step yet!)
                # ============================================================
                q_values_eval = []
                new_hiddens_eval = []
                for i in range(self.n_agents):
                    q_val, _, new_hidden = self.agent_eval[i](
                        obs_t[i],
                        last_actions_t[i],
                        hiddens_eval[i],
                        actions_t[i]
                    )
                    q_values_eval.append(q_val)
                    new_hiddens_eval.append(new_hidden)

                q_tot = self.mixer_eval(q_values_eval, state_t)

                # Compute TD target
                if t < len(episode['rewards']) - 1:
                    obs_next = [torch.FloatTensor(episode['observations'][i][t+1]).unsqueeze(0).to(self.device)
                                for i in range(self.n_agents)]
                    state_next = torch.FloatTensor(episode['states'][t+1]).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        q_values_target = []
                        new_hiddens_target = []

                        for i in range(self.n_agents):
                            # First get the action and updated hidden state
                            _, target_action, new_hidden_target = self.agent_target[i](
                                obs_next[i],
                                actions_t[i],  # Last action at time t
                                hiddens_target[i]
                            )

                            # FIXED: Use new_hidden_target (not hiddens_target[i]) for Q evaluation
                            # This ensures Q is evaluated with the correct history
                            q_val, _, _ = self.agent_target[i](
                                obs_next[i],
                                actions_t[i],
                                new_hidden_target,  # FIX: Use updated hidden state
                                target_action
                            )
                            q_values_target.append(q_val)
                            new_hiddens_target.append(new_hidden_target)

                        q_tot_target = self.mixer_target(q_values_target, state_next)
                        td_target = reward + self.gamma * q_tot_target
                else:
                    td_target = reward

                td_error = td_target.detach() - q_tot
                critic_loss = td_error.pow(2).mean()

                # Accumulate loss (don't update yet!)
                critic_losses.append(critic_loss)

                # Detach hiddens
                hiddens_eval = [h.detach() for h in new_hiddens_eval]
                if t < len(episode['rewards']) - 1:
                    hiddens_target = [h.detach() for h in new_hiddens_target]

                # ============================================================
                # ACTOR LOSS ACCUMULATION (No backward/step yet!)
                # ============================================================
                current_actions = []
                current_hiddens = [h.detach() for h in hiddens_eval]

                for i in range(self.n_agents):
                    _, act, _ = self.agent_eval[i](
                        obs_t[i].detach(),
                        last_actions_t[i].detach(),
                        current_hiddens[i]
                    )
                    current_actions.append(act)

                # Evaluate Q-values for current policy actions
                q_values_for_actor = []
                for i in range(self.n_agents):
                    q_val, _, _ = self.agent_eval[i](
                        obs_t[i].detach(),
                        last_actions_t[i].detach(),
                        current_hiddens[i],
                        current_actions[i]
                    )
                    q_values_for_actor.append(q_val)

                # Mix Q-values
                q_tot_for_actor = self.mixer_eval(q_values_for_actor, state_t.detach())

                # FACMAC: Centralized policy gradient with action regularization
                # Reference (facmac_learner.py:137): pg_loss = -chosen_action_qvals.mean() + (pi**2).mean() * 1e-3
                actions_tensor = torch.cat(current_actions, dim=-1)  # Combine all agent actions
                action_reg = (actions_tensor ** 2).mean() * 1e-3  # Prevents extreme actions
                actor_loss = -q_tot_for_actor.mean() + action_reg

                # Accumulate loss (don't update yet!)
                actor_losses.append(actor_loss)

        # ============================================================
        # SINGLE GRADIENT UPDATE FOR ENTIRE BATCH
        # ============================================================
        # Average loss over all transitions in batch
        avg_critic_loss = torch.stack(critic_losses).mean()
        avg_actor_loss = torch.stack(actor_losses).mean()

        # CRITICAL: Compute BOTH gradients BEFORE any optimizer.step()
        # (optimizer.step() modifies parameters in-place, breaking the graph)

        # Compute critic gradients
        self.critic_optimizer.zero_grad()
        avg_critic_loss.backward(retain_graph=True)  # Keep graph for actor backward
        # Reference: grad_norm_clip: 0.5 (NOT 10.0!)
        torch.nn.utils.clip_grad_norm_(self.critic_params, max_norm=0.5)

        # Compute actor gradients (before critic.step() modifies parameters!)
        self.actor_optimizer.zero_grad()
        avg_actor_loss.backward()  # Can release graph now
        # Reference: grad_norm_clip: 0.5 (NOT 10.0!)
        torch.nn.utils.clip_grad_norm_(self.actor_params, max_norm=0.5)

        # Now apply both gradient updates
        self.critic_optimizer.step()
        self.actor_optimizer.step()

        # Soft update targets
        self._soft_update()

        return {
            'critic_loss': avg_critic_loss.item(),
            'actor_loss': avg_actor_loss.item()
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