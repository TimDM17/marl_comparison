"""
Evaluation script for trained MARL models.

Usage:
    python scripts/evaluate.py --checkpoint results/nqmix_humanoid/best_model --config configs/nqmix_humanoid.yaml
    python scripts/evaluate.py --checkpoint results/nqmix_humanoid/best_model --config configs/nqmix_humanoid.yaml --episodes 50
"""

import argparse
import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents import NQMIX
from src.envs import MaMuJoCoWrapper
from src.training import Evaluator
from src.utils import load_config, Logger


def create_agent(config: dict, env: MaMuJoCoWrapper):
    """Create agent based on config."""
    algorithm = config.get('algorithm', 'nqmix').lower()
    agent_params = config.get('agent_params', {})

    if algorithm == 'nqmix':
        agent = NQMIX(
            n_agents=env.n_agents,
            obs_dims=env.obs_dims,
            action_dims=env.action_dims,
            state_dim=env.state_dim,
            hidden_dim=agent_params.get('hidden_dim', 64),
            lr_actor=agent_params.get('lr_actor', 5e-4),
            lr_critic=agent_params.get('lr_critic', 5e-4),
            gamma=agent_params.get('gamma', 0.99),
            tau=agent_params.get('tau', 0.001),
            buffer_capacity=agent_params.get('buffer_capacity', 5000)
        )
    elif algorithm == 'facmac':
        raise NotImplementedError("FACMAC not yet implemented")
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return agent


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained MARL models')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render environment (if supported)')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Initialize logger
    logger = Logger(verbose=True)

    logger.info(f"Evaluating checkpoint: {args.checkpoint}")
    logger.info(f"Config: {args.config}")

    # Create environment
    env_name = config.get('env_name', 'Humanoid')
    partitioning = config.get('partitioning', '9|8')
    env = MaMuJoCoWrapper(env_name=env_name, partitioning=partitioning)

    logger.info(f"Environment: {env_name} ({partitioning})")

    # Create and load agent
    agent = create_agent(config, env)
    agent.load(args.checkpoint)
    logger.info(f"Model loaded from {args.checkpoint}")

    # Create evaluator
    evaluator = Evaluator(
        agent=agent,
        env=env,
        logger=logger,
        n_eval_episodes=args.episodes,
        save_best=False
    )

    # Run evaluation
    logger.info(f"\nRunning {args.episodes} evaluation episodes...\n")

    episode_rewards = []
    episode_lengths = []

    for i in range(args.episodes):
        reward, length = evaluator._run_episode()
        episode_rewards.append(reward)
        episode_lengths.append(length)

        if (i + 1) % 10 == 0:
            logger.info(f"Completed {i + 1}/{args.episodes} episodes")

    # Compute statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    min_reward = np.min(episode_rewards)
    max_reward = np.max(episode_rewards)
    mean_length = np.mean(episode_lengths)

    # Print results
    logger.info(f"\n{'='*50}")
    logger.info("EVALUATION RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"Episodes:     {args.episodes}")
    logger.info(f"Mean Reward:  {mean_reward:.2f} +/- {std_reward:.2f}")
    logger.info(f"Min Reward:   {min_reward:.2f}")
    logger.info(f"Max Reward:   {max_reward:.2f}")
    logger.info(f"Mean Length:  {mean_length:.1f}")
    logger.info(f"{'='*50}\n")

    env.close()

    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'min_reward': min_reward,
        'max_reward': max_reward,
        'mean_length': mean_length
    }


if __name__ == '__main__':
    main()
