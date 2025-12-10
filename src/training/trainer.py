"""
Training loop orchestrator for MARL algorithms.

Purpose:
    Manages the complete training process: collecting episodes, calling agent
    training updates, decaying exploration noise, logging metrics and triggering
    periodic evaluations

Key Concepts:
    - Episode Collection: Run policy in environment, store experience
    - Off-Policy Training: Learn from past episodes stored in replay buffer
    - Exploration Decay: Start with high noise, gradually reduce for exploitation
    - Observation Normalization: Running mean/std for stable training
    - Start Steps: Random exploration before policy learning

Connections:
    - Uses: BaseAgent (for training), MaMuJoCo (environment),
            Logger (metrics), Evaluator (periodic testing)
    - Called by: scripts/train.py
"""

import time
import numpy as np
from typing import TYPE_CHECKING, Optional, Tuple, List

from src.utils.normalization import ObservationNormalizer

if TYPE_CHECKING:
    from src.agents.base_agent import BaseAgent
    from src.envs.mamujoco_wrapper import MaMuJoCoWrapper
    from src.utils.logger import Logger
    from src.training.evaluator import Evaluator


class Trainer:
    """
    Orchestrates the training loop for MARL algorithms.

    Handles episode collection, training updates, and periodic evaluation.
    """

    def __init__(
        self,
        agent: "BaseAgent",
        env,
        logger: "Logger",
        evaluator: Optional["Evaluator"] = None,
        # Training parameters
        total_episodes: int = 10000,
        batch_size: int = 32,
        train_every: int = 1,
        train_steps: int = 1,
        # Exploration parameters
        noise_scale: float = 0.1,
        noise_decay: float = 1.0,
        min_noise: float = 0.01,
        start_steps: int = 0,
        # Evaluation parameters
        eval_every: int = 100,
        # Buffer parameters
        min_buffer_size: int = 100,
        # Logging parameters
        log_every: int = 10,
        # Normalization parameters
        normalize_observations: bool = True,
        obs_clip_range: float = 10.0
    ):
        """
        Initialize the trainer.

        Args:
            agent: The MARL agent to train
            env: The training environment
            logger: Logger for training metrics
            evaluator: Optional evaluator for periodic evaluation
            total_episodes: Total number of training episodes
            batch_size: Batch size for training updates
            train_every: Train after every N episodes
            train_steps: Number of training steps per training call
            noise_scale: Initial exploration noise scale
            noise_decay: Noise decay factor per episode
            min_noise: Minimum noise scale
            start_steps: Number of initial timesteps with random actions
            eval_every: Evaluate every N episodes
            min_buffer_size: Minimum buffer size before training starts
            log_every: Log training metrics every N episodes
            normalize_observations: Whether to normalize observations
            obs_clip_range: Clip range for normalized observations
        """
        self.agent = agent
        self.env = env
        self.logger = logger
        self.evaluator = evaluator

        # Detect if we're using vectorized environments
        self.is_vectorized = hasattr(env, 'n_envs')
        self.n_envs = env.n_envs if self.is_vectorized else 1

        if self.is_vectorized:
            self.logger.info(f"Using vectorized training with {self.n_envs} parallel environments")

        # Training parameters
        self.total_episodes = total_episodes
        self.batch_size = batch_size
        self.train_every = train_every
        self.train_steps = train_steps

        # Exploration parameters
        self.noise_scale = noise_scale
        self.noise_decay = noise_decay
        self.min_noise = min_noise
        self.current_noise = noise_scale
        self.start_steps = start_steps

        # Evaluation parameters
        self.eval_every = eval_every

        # Buffer parameters
        self.min_buffer_size = min_buffer_size

        # Logging parameters
        self.log_every = log_every

        # Observation normalization
        self.normalize_observations = normalize_observations
        if normalize_observations:
            self.obs_normalizer = ObservationNormalizer(
                obs_dims=env.obs_dims,
                clip_range=obs_clip_range
            )
            self.logger.info("Using observation normalization (running mean/std)")
        else:
            self.obs_normalizer = None

        # Training state
        self.total_timesteps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.start_time = None

    def train(self) -> dict:
        """
        Run the full training loop.

        Returns:
            Dictionary with training summary
        """
        self.logger.info(f"Starting training for {self.total_episodes} episodes")
        self.start_time = time.time()

        episodes_collected = 0

        while episodes_collected < self.total_episodes:
            if self.is_vectorized:
                # Collect from vectorized environments
                rewards, lengths = self._collect_episode_vectorized()
                self.episode_rewards.extend(rewards)
                self.episode_lengths.extend(lengths)
                episodes_collected += self.n_envs
            else:
                # Collect single episode (backward compatibility)
                reward, length = self._collect_episode()
                self.episode_rewards.append(reward)
                self.episode_lengths.append(length)
                episodes_collected += 1

            # Current episode number for logging
            episode = episodes_collected

            # Train if buffer is ready
            if self._can_train() and episode % self.train_every == 0:
                for _ in range(self.train_steps):
                    loss = self.agent.train_step(self.batch_size)
                    if loss is not None:
                        self.losses.append(loss)

            # Decay exploration noise
            self.current_noise = max(
                self.min_noise,
                self.current_noise * self.noise_decay
            )

            # Log training metrics
            if episode % self.log_every == 0:
                self._log_training(episode)

            # Evaluate (check if we've reached eval milestone)
            if self.evaluator and episode // self.eval_every > (episode - self.n_envs) // self.eval_every:
                self.evaluator.evaluate(episode)

        self.logger.info("Training complete!")

        return {
            'total_episodes': self.total_episodes,
            'total_timesteps': self.total_timesteps,
            'final_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'best_reward': self.evaluator.best_reward if self.evaluator else None
        }

    def _collect_episode(self) -> tuple[float, int]:
        """
        Collect a single training episode.

        Features:
            - Observation normalization (running mean/std)
            - Start steps random exploration
            - Proper hidden state management

        Returns:
            Tuple of (episode_reward, episode_length)
        """
        obs_dict, _ = self.env.reset()
        hiddens = self.agent.init_hidden_states()

        # Initialize last actions
        last_actions = [
            np.zeros(self.env.action_dims[i])
            for i in range(self.env.n_agents)
        ]

        # Convert observations to list
        observations = [obs_dict[agent_id] for agent_id in self.env.possible_agents]

        # Normalize observations if enabled
        if self.obs_normalizer is not None:
            observations = self.obs_normalizer.normalize(observations, update=True)

        # Episode storage
        episode_data = {
            'observations': [[] for _ in range(self.env.n_agents)],
            'actions': [[] for _ in range(self.env.n_agents)],
            'last_actions': [[] for _ in range(self.env.n_agents)],
            'states': [],
            'rewards': [],
            'terminated': []  # Track terminal states for proper TD targets
        }

        total_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            # Store current observations and last actions
            for i in range(self.env.n_agents):
                episode_data['observations'][i].append(observations[i])
                episode_data['last_actions'][i].append(last_actions[i])

            # Store global state (concatenation of all observations)
            state = np.concatenate(observations)
            episode_data['states'].append(state)

            # Select actions: random during start_steps, then policy
            if self.total_timesteps < self.start_steps:
                # Random exploration during start_steps
                actions = [
                    np.random.uniform(
                        self.agent.action_low,
                        self.agent.action_high,
                        size=self.env.action_dims[i]
                    )
                    for i in range(self.env.n_agents)
                ]
            else:
                # Policy-based action selection
                actions, hiddens = self.agent.select_actions(
                    observations=observations,
                    last_actions=last_actions,
                    hiddens=hiddens,
                    explore=True,
                    noise_scale=self.current_noise
                )
                # Detach hidden states to prevent gradient flow through episode
                hiddens = [h.detach() if hasattr(h, 'detach') else h for h in hiddens]

            # Store actions
            for i in range(self.env.n_agents):
                episode_data['actions'][i].append(actions[i])

            # Step environment
            obs_dict, rewards_dict, terminated_dict, truncated_dict, _ = self.env.step(actions)

            # Store reward (shared reward - verify all agents have same reward)
            rewards = list(rewards_dict.values())
            if len(set(rewards)) != 1:
                raise ValueError(f"Expected shared reward, got different rewards: {rewards}")
            reward = rewards[0]
            episode_data['rewards'].append(reward)

            # Store terminated signal (True if episode ended naturally, not truncated)
            # This is critical for TD targets: terminal states should NOT bootstrap
            is_terminated = any(terminated_dict.values())
            episode_data['terminated'].append(is_terminated)

            # Update for next step
            observations = [obs_dict[agent_id] for agent_id in self.env.possible_agents]

            # Normalize new observations
            if self.obs_normalizer is not None:
                observations = self.obs_normalizer.normalize(observations, update=True)

            last_actions = actions

            total_reward += reward
            episode_length += 1
            self.total_timesteps += 1

            # Check termination
            done = (
                any(terminated_dict.values()) or
                any(truncated_dict.values())
            )

        # Store episode in replay buffer
        self.agent.store_episode(episode_data)

        return total_reward, episode_length
    
    def _collect_episode_vectorized(self) -> Tuple[List[float], List[int]]:
        """
        Collect episodes from vectorized environments in parallel.

        Features:
            - GPU-batched action selection (all envs in single forward pass)
            - Observation normalization (running mean/std)
            - Start steps random exploration

        Returns:
            episode_rewards: List of rewards from each environment
            episode_lengths: List of lengths from each environment
        """
        # Reset all environments
        observations_list = self.env.reset()  # [n_envs][n_agents]

        # Normalize observations if enabled
        if self.obs_normalizer is not None:
            observations_list = self.obs_normalizer.normalize_batch(
                observations_list, update=True
            )

        # Initialize hidden states for each environment: [n_envs][n_agents]
        hiddens_list = [
            self.agent.init_hidden_states()
            for _ in range(self.n_envs)
        ]

        # Initialize last actions for each environment: [n_envs][n_agents]
        last_actions_list = [
            [np.zeros(self.env.action_dims[i]) for i in range(self.env.n_agents)]
            for _ in range(self.n_envs)
        ]

        # Episode storage for each environment
        episode_data_list = [
            {
                'observations': [[] for _ in range(self.env.n_agents)],
                'actions': [[] for _ in range(self.env.n_agents)],
                'last_actions': [[] for _ in range(self.env.n_agents)],
                'states': [],
                'rewards': [],
                'terminated': []  # Track terminal states for proper TD targets
            }
            for _ in range(self.n_envs)
        ]

        # Track which environments are still running
        dones = [False] * self.n_envs
        episode_rewards = [0.0] * self.n_envs
        episode_lengths = [0] * self.n_envs

        while not all(dones):
            # Get indices of active (not done) environments
            active_indices = [i for i, done in enumerate(dones) if not done]
            if not active_indices:
                break

            # ================================================================
            # ACTION SELECTION (random during start_steps, then policy)
            # ================================================================
            if self.total_timesteps < self.start_steps:
                # Random exploration during start_steps
                active_actions = [
                    [
                        np.random.uniform(
                            self.agent.action_low,
                            self.agent.action_high,
                            size=self.env.action_dims[agent_idx]
                        )
                        for agent_idx in range(self.env.n_agents)
                    ]
                    for _ in active_indices
                ]
                # Keep hidden states unchanged during random exploration
                active_new_hiddens = [hiddens_list[i] for i in active_indices]
            else:
                # GPU-BATCHED ACTION SELECTION (key optimization)
                active_obs = [observations_list[i] for i in active_indices]
                active_last_actions = [last_actions_list[i] for i in active_indices]
                active_hiddens = [hiddens_list[i] for i in active_indices]

                # Single batched call to agent (GPU parallel processing)
                active_actions, active_new_hiddens = self.agent.select_actions_batched(
                    observations_batch=active_obs,
                    last_actions_batch=active_last_actions,
                    hiddens_batch=active_hiddens,
                    explore=True,
                    noise_scale=self.current_noise
                )

            # Detach hidden states and map back to full list
            actions_list = [None] * self.n_envs
            for idx, env_idx in enumerate(active_indices):
                actions_list[env_idx] = active_actions[idx]
                # Detach hidden states to prevent gradient flow through episode
                hiddens_list[env_idx] = [
                    h.detach() if hasattr(h, 'detach') else h
                    for h in active_new_hiddens[idx]
                ]

            # Store transitions for active environments
            for idx, env_idx in enumerate(active_indices):
                for agent_idx in range(self.env.n_agents):
                    episode_data_list[env_idx]['observations'][agent_idx].append(
                        observations_list[env_idx][agent_idx]
                    )
                    episode_data_list[env_idx]['last_actions'][agent_idx].append(
                        last_actions_list[env_idx][agent_idx]
                    )
                    episode_data_list[env_idx]['actions'][agent_idx].append(
                        active_actions[idx][agent_idx]
                    )

                # Store global state
                state = np.concatenate(observations_list[env_idx])
                episode_data_list[env_idx]['states'].append(state)

            # ================================================================
            # ENVIRONMENT STEP
            # ================================================================
            # Create full actions list (with dummy actions for done envs)
            full_actions_list = []
            for env_idx in range(self.n_envs):
                if dones[env_idx]:
                    full_actions_list.append(
                        [np.zeros(dim) for dim in self.env.action_dims]
                    )
                else:
                    full_actions_list.append(actions_list[env_idx])

            # Step all environments
            observations_list, rewards_list, dones_list, infos_list = self.env.step(full_actions_list)

            # Normalize new observations
            if self.obs_normalizer is not None:
                observations_list = self.obs_normalizer.normalize_batch(
                    observations_list, update=True
                )

            # Process results for active environments
            for env_idx in active_indices:
                reward = rewards_list[env_idx]
                episode_data_list[env_idx]['rewards'].append(reward)
                episode_rewards[env_idx] += reward
                episode_lengths[env_idx] += 1
                self.total_timesteps += 1

                # Store terminated signal
                # Check infos for truncation info, otherwise use done signal
                # Terminal = episode ended naturally (should NOT bootstrap)
                # Truncated = hit time limit (SHOULD bootstrap, but we approximate as non-terminal)
                info = infos_list[env_idx] if env_idx < len(infos_list) else {}
                is_truncated = info.get('TimeLimit.truncated', False) or info.get('truncated', False)
                is_terminated = dones_list[env_idx] and not is_truncated
                episode_data_list[env_idx]['terminated'].append(is_terminated)

                # Update last actions
                last_actions_list[env_idx] = actions_list[env_idx]

                # Check if this environment is done
                if dones_list[env_idx]:
                    dones[env_idx] = True
                    # Store complete episode
                    self.agent.store_episode(episode_data_list[env_idx])

        return episode_rewards, episode_lengths

    def _can_train(self) -> bool:
        """Check if training can start based on buffer size."""
        if self.agent.replay_buffer is None:
            return False
        return len(self.agent.replay_buffer) >= self.min_buffer_size

    def _log_training(self, episode: int) -> None:
        """Log training metrics."""
        # Compute recent metrics
        recent_rewards = self.episode_rewards[-10:]
        avg_reward_10 = np.mean(recent_rewards) if recent_rewards else 0

        # Current episode metrics
        current_reward = self.episode_rewards[-1] if self.episode_rewards else 0
        current_length = self.episode_lengths[-1] if self.episode_lengths else 0

        # Loss
        loss = 0.0
        if self.losses:
            recent_losses = self.losses[-self.log_every * self.train_steps:]
            # Handle both dict losses (FACMAC) and float losses (NQMIX)
            if recent_losses and isinstance(recent_losses[0], dict):
                # FACMAC: average critic_loss only for main metric
                loss = np.mean([l['critic_loss'] for l in recent_losses])
            else:
                # NQMIX: average directly
                loss = np.mean(recent_losses)

        # Buffer size
        buffer_size = 0
        if self.agent.replay_buffer is not None:
            buffer_size = len(self.agent.replay_buffer)

        # Time elapsed
        time_min = (time.time() - self.start_time) / 60 if self.start_time else 0

        metrics = {
            'episode': episode,
            'reward': current_reward,
            'avg_reward_10': avg_reward_10,
            'length': int(current_length),
            'loss': loss,
            'buffer_size': buffer_size,
            'time_min': time_min
        }

        self.logger.log_train(metrics)
