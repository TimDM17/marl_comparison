"""
Main training script for MARL algorithms.

Purpose:
    Entry point for training. Loads configuration, creates all components,
    runs training loop, and saves results.

Usage:
    python scripts/train.py --config configs/nqmix_humanoid.yaml
    python scripts/train.py --config configs/nqmix_humanoid.yaml --seed 123

Flow:
    1. Parse arguments and load config
    2. Set random seeds for reproducibility
    3. Create environment wrapper
    4. Create agent (NQMIX)
    5. Create evaluator for periodic testing
    6. Create trainer to orchestrate training
    7. Run training loop
    8. Save final model and log summary
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import random
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents import NQMIX
from src.envs import MaMuJoCoWrapper
from src.training import Trainer, Evaluator
from src.utils import load_config, Logger


def set_seed(seed: int):
    """
    Set random seeds for reproducibility across all libraries.

    Why we need this:
        - Python's random: Used in replay buffer sampling
        - NumPy: Used in action noise, environment resets
        - PyTorch CPU: Network initialization, dropout
        - PyTorch CUDA: GPU operations
        - cuDNN: GPU-accelerated operations

    Note: Setting deterministic=True and benchmark=False ensures
    reproducibility but may slightly reduce performance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        torch.backends.cudnn.deterministic = True  # Reproducible
        torch.backends.cudnn.benchmark = False     # Disable auto-tuning


def create_agent(config: dict, env: MaMuJoCoWrapper):
    """
    Factory function to create agent based on config.

    This pattern allows easy switching between algorithms by changing
    the 'algorithm' field in the config file.

    Args:
        config: Configuration dictionary
        env: Environment wrapper (provides dimensions)

    Returns:
        Initialized agent ready for training
    """
    algorithm = config.get('algorithm', 'nqmix').lower()
    agent_params = config.get('agent_params', {})

    if algorithm == 'nqmix':
        # Create NQMIX agent with parameters from config
        # Environment provides: n_agents, obs_dims, action_dims, state_dim
        # Config provides: hidden_dim, learning rates, gamma, tau, etc.
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
            buffer_capacity=agent_params.get('buffer_capacity', 5000),
            action_low=agent_params.get('action_low', -0.4),
            action_high=agent_params.get('action_high', 0.4)
        )
    elif algorithm == 'facmac':
        # Future: Implement FACMAC here
        raise NotImplementedError("FACMAC not yet implemented")
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return agent


def main():
    # ================================================================
    # ARGUMENT PARSING AND CONFIG LOADING
    # ================================================================
    parser = argparse.ArgumentParser(description='Train MARL algorithms')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    args = parser.parse_args()

    config = load_config(args.config)

    # Command-line seed overrides config file seed
    if args.seed is not None:
        config['seed'] = args.seed

    seed = config.get('seed', 42)
    set_seed(seed)

    # ================================================================
    # SETUP DIRECTORIES AND LOGGING
    # ================================================================
    save_dir = Path(config.get('save_dir', './results/default'))
    save_dir.mkdir(parents=True, exist_ok=True)

    log_file = save_dir / 'training.log'
    tensorboard_dir = save_dir / 'tensorboard'
    logger = Logger(log_file=str(log_file), verbose=True, tensorboard_dir=str(tensorboard_dir))

    logger.info(f"Config: {args.config}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Save dir: {save_dir}")

    # ================================================================
    # CREATE ENVIRONMENT
    # ================================================================
    env_name = config.get('env_name', 'Humanoid')
    partitioning = config.get('partitioning', '9|8')
    env = MaMuJoCoWrapper(env_name=env_name, partitioning=partitioning)

    logger.info(f"Environment: {env_name} ({partitioning})")
    logger.info(f"Agents: {env.n_agents}, Obs dims: {env.obs_dims}, Action dims: {env.action_dims}")

    # ================================================================
    # CREATE AGENT
    # ================================================================
    agent = create_agent(config, env)
    logger.info(f"Algorithm: {config.get('algorithm', 'nqmix').upper()}")

    # ================================================================
    # CREATE EVALUATOR
    # ================================================================
    # Separate environment for evaluation to avoid interference
    eval_env = MaMuJoCoWrapper(env_name=env_name, partitioning=partitioning)
    evaluator = Evaluator(
        agent=agent,
        env=eval_env,
        logger=logger,
        n_eval_episodes=config.get('n_eval_episodes', 10),
        save_best=True,
        save_path=str(save_dir / 'best_model.pth')
    )

    # ================================================================
    # CALCULATE EXPLORATION NOISE DECAY
    # ================================================================
    # Noise decays exponentially from start to end over decay_episodes
    # Formula: noise(t) = noise_start * (decay_rate)^t
    # Solving: noise_end = noise_start * decay_rate^decay_episodes
    # Therefore: decay_rate = (noise_end / noise_start)^(1/decay_episodes)
    noise_start = config.get('noise_scale_start', 0.1)
    noise_end = config.get('noise_scale_end', 0.01)
    noise_decay_episodes = config.get('noise_decay_episodes', 1000)
    noise_decay = (noise_end / noise_start) ** (1.0 / noise_decay_episodes)

    # ================================================================
    # CREATE TRAINER
    # ================================================================
    trainer = Trainer(
        agent=agent,
        env=env,
        logger=logger,
        evaluator=evaluator,
        total_episodes=config.get('n_episodes', 3000),
        batch_size=config.get('batch_size', 32),
        train_every=1,          # Train after every episode
        train_steps=1,          # One gradient update per episode
        noise_scale=noise_start,
        noise_decay=noise_decay,
        min_noise=noise_end,
        eval_every=config.get('eval_freq', 100),
        min_buffer_size=config.get('batch_size', 32) * 2,  # Need 2x batch for stable training
        log_every=config.get('log_freq', 10)
    )

    # ================================================================
    # RUN TRAINING
    # ================================================================
    logger.info(f"\nStarting training for {config.get('n_episodes', 3000)} episodes...\n")
    start_time = time.time()

    try:
        results = trainer.train()
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        results = {'interrupted': True}
    finally:
        # Always close environments to free resources
        env.close()
        eval_env.close()

    # ================================================================
    # SAVE RESULTS AND CLEANUP
    # ================================================================
    total_time = (time.time() - start_time) / 60
    summary = {
        'total_time_min': total_time,
        'final_avg_reward': results.get('final_reward', 0),
        'best_eval_reward': evaluator.best_reward,
        'total_episodes': results.get('total_episodes', 0)
    }
    logger.log_summary(summary)
    logger.closeTensorBoard()

    # Save final model (might be different from best model)
    agent.save(str(save_dir / 'final_model.pth'))
    logger.info(f"Models saved to {save_dir}")

    return results


if __name__ == '__main__':
    main()
