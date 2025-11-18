"""
Trainer for orchestrating the MARL training loop.
"""

import time
import numpy as np
from typing import TYPE_CHECKING, Optional

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
        env: "MaMuJoCoWrapper",
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
        # Evaluation parameters
        eval_every: int = 100,
        # Buffer parameters
        min_buffer_size: int = 100,
        # Logging parameters
        log_every: int = 10
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
            eval_every: Evaluate every N episodes
            min_buffer_size: Minimum buffer size before training starts
            log_every: Log training metrics every N episodes
        """
        self.agent = agent
        self.env = env
        self.logger = logger
        self.evaluator = evaluator

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

        # Evaluation parameters
        self.eval_every = eval_every

        # Buffer parameters
        self.min_buffer_size = min_buffer_size

        # Logging parameters
        self.log_every = log_every

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

        for episode in range(1, self.total_episodes + 1):
            # Collect episode
            episode_reward, episode_length = self._collect_episode()
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

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

            # Evaluate
            if self.evaluator and episode % self.eval_every == 0:
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

        # Episode storage
        episode_data = {
            'observations': [[] for _ in range(self.env.n_agents)],
            'actions': [[] for _ in range(self.env.n_agents)],
            'last_actions': [[] for _ in range(self.env.n_agents)],
            'states': [],
            'rewards': []
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

            # Select actions with exploration
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

            # Update for next step
            observations = [obs_dict[agent_id] for agent_id in self.env.possible_agents]
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
