"""
Evaluator for running deterministic evaluation episodes.
"""

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.base_agent import BaseAgent
    from src.envs.mamujoco_wrapper import MaMuJoCoWrapper
    from src.utils.logger import Logger


class Evaluator:
    """
    Runs deterministic evaluation episodes without exploration noise.

    Used to assess the agent's true performance during training.
    """

    def __init__(
        self,
        agent: "BaseAgent",
        env: "MaMuJoCoWrapper",
        logger: "Logger",
        n_eval_episodes: int = 10,
        save_best: bool = True,
        save_path: str = "checkpoints/best_model"
    ):
        """
        Initialize the evaluator.

        Args:
            agent: The agent to evaluate
            env: The evaluation environment
            logger: Logger for evaluation metrics
            n_eval_episodes: Number of episodes to run per evaluation
            save_best: Whether to save the best model
            save_path: Path to save the best model
        """
        self.agent = agent
        self.env = env
        self.logger = logger
        self.n_eval_episodes = n_eval_episodes
        self.save_best = save_best
        self.save_path = save_path

        self.best_reward = float('-inf')

    def evaluate(self, episode: int) -> dict:
        """
        Run evaluation episodes and compute metrics.

        Args:
            episode: Current training episode number (for logging)

        Returns:
            Dictionary with evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []

        for _ in range(self.n_eval_episodes):
            reward, length = self._run_episode()
            episode_rewards.append(reward)
            episode_lengths.append(length)

        # Compute statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)

        # Check for best model
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            if self.save_best:
                self.agent.save(self.save_path)

        # Log evaluation results
        self.logger.log_eval(
            episode=episode,
            eval_reward=mean_reward,
            eval_length=mean_length,
            best_reward=self.best_reward
        )

        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_length': mean_length,
            'best_reward': self.best_reward
        }

    def _run_episode(self) -> tuple[float, int]:
        """
        Run a single deterministic evaluation episode.

        Returns:
            Tuple of (total_reward, episode_length)
        """
        obs_dict, _ = self.env.reset()
        hiddens = self.agent.init_hidden_states()

        # Initialize last actions to zeros
        last_actions = [
            np.zeros(self.env.action_dims[i])
            for i in range(self.env.n_agents)
        ]

        # Convert observations dict to list
        observations = [obs_dict[agent_id] for agent_id in self.env.possible_agents]

        total_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            # Select actions deterministically (no exploration)
            actions, hiddens = self.agent.select_actions(
                observations=observations,
                last_actions=last_actions,
                hiddens=hiddens,
                explore=False
            )

            # Detach hidden states to prevent gradient accumulation
            hiddens = [h.detach() if hasattr(h, 'detach') else h for h in hiddens]

            # Step environment
            obs_dict, rewards_dict, terminated_dict, truncated_dict, _ = self.env.step(actions)

            # Update observations
            observations = [obs_dict[agent_id] for agent_id in self.env.possible_agents]
            last_actions = actions

            # Accumulate reward (shared reward for all agents)
            rewards = list(rewards_dict.values())
            reward = rewards[0]
            total_reward += reward
            episode_length += 1

            # Check termination
            done = (
                any(terminated_dict.values()) or
                any(truncated_dict.values())
            )

        return total_reward, episode_length
