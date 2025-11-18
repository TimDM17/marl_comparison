"""
Plot training results from log files.

Usage:
    python scripts/plot_results.py --log results/nqmix_humanoid/training.log
    python scripts/plot_results.py --log results/nqmix_humanoid/training.log --output results/nqmix_humanoid/plots
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_log_file(log_path: str) -> dict:
    """
    Parse training log file and extract metrics.

    Args:
        log_path: Path to the training log file

    Returns:
        Dictionary with lists of metrics
    """
    metrics = {
        'episodes': [],
        'rewards': [],
        'avg_rewards': [],
        'lengths': [],
        'losses': [],
        'buffer_sizes': [],
        'times': [],
        'eval_episodes': [],
        'eval_rewards': [],
        'best_rewards': []
    }

    # Regex patterns for parsing
    train_pattern = re.compile(
        r'Ep\s+(\d+)\s+\|\s+'
        r'R:\s+([-\d.]+)\s+\|\s+'
        r'R10:\s+([-\d.]+)\s+\|\s+'
        r'Len:\s+(\d+)\s+\|\s+'
        r'Loss:\s+([-\d.]+)\s+\|\s+'
        r'Buf:\s+(\d+)\s+\|\s+'
        r'T:\s+([\d.]+)m'
    )

    eval_pattern = re.compile(
        r'EVAL @ Ep\s+(\d+)\s+\|\s+'
        r'R:\s+([-\d.]+)\s+\|\s+'
        r'Len:\s+([\d.]+)\s+\|\s+'
        r'Best:\s+([-\d.]+)'
    )

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Try to match training log
            train_match = train_pattern.search(line)
            if train_match:
                metrics['episodes'].append(int(train_match.group(1)))
                metrics['rewards'].append(float(train_match.group(2)))
                metrics['avg_rewards'].append(float(train_match.group(3)))
                metrics['lengths'].append(int(train_match.group(4)))
                metrics['losses'].append(float(train_match.group(5)))
                metrics['buffer_sizes'].append(int(train_match.group(6)))
                metrics['times'].append(float(train_match.group(7)))
                continue

            # Try to match evaluation log
            eval_match = eval_pattern.search(line)
            if eval_match:
                metrics['eval_episodes'].append(int(eval_match.group(1)))
                metrics['eval_rewards'].append(float(eval_match.group(2)))
                metrics['best_rewards'].append(float(eval_match.group(4)))

    return metrics


def smooth(data: list, window: int = 10) -> np.ndarray:
    """Apply moving average smoothing."""
    if len(data) < window:
        return np.array(data)
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_training_curves(metrics: dict, output_dir: Path, show: bool = True):
    """
    Generate training plots.

    Args:
        metrics: Dictionary with training metrics
        output_dir: Directory to save plots
        show: Whether to display plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_size = (10, 6)

    # 1. Reward plot
    fig, ax = plt.subplots(figsize=fig_size)
    episodes = metrics['episodes']
    rewards = metrics['rewards']

    ax.plot(episodes, rewards, alpha=0.3, label='Episode Reward')
    if len(rewards) >= 10:
        smoothed = smooth(rewards, 10)
        ax.plot(episodes[9:], smoothed, label='Smoothed (10 ep)')
    ax.plot(episodes, metrics['avg_rewards'], label='Running Avg (10 ep)', linewidth=2)

    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Reward')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'reward.png', dpi=150)
    if show:
        plt.show()
    plt.close()

    # 2. Evaluation reward plot
    if metrics['eval_episodes']:
        fig, ax = plt.subplots(figsize=fig_size)
        ax.plot(metrics['eval_episodes'], metrics['eval_rewards'],
                marker='o', label='Eval Reward')
        ax.plot(metrics['eval_episodes'], metrics['best_rewards'],
                linestyle='--', label='Best Reward')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Evaluation Reward')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / 'eval_reward.png', dpi=150)
        if show:
            plt.show()
        plt.close()

    # 3. Loss plot
    fig, ax = plt.subplots(figsize=fig_size)
    losses = metrics['losses']
    ax.plot(episodes, losses, alpha=0.5)
    if len(losses) >= 10:
        smoothed_loss = smooth(losses, 10)
        ax.plot(episodes[9:], smoothed_loss, label='Smoothed', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'loss.png', dpi=150)
    if show:
        plt.show()
    plt.close()

    # 4. Episode length plot
    fig, ax = plt.subplots(figsize=fig_size)
    lengths = metrics['lengths']
    ax.plot(episodes, lengths, alpha=0.5)
    if len(lengths) >= 10:
        smoothed_len = smooth(lengths, 10)
        ax.plot(episodes[9:], smoothed_len, label='Smoothed', linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Length')
    ax.set_title('Episode Length')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'length.png', dpi=150)
    if show:
        plt.show()
    plt.close()

    # 5. Combined summary plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Reward
    ax = axes[0, 0]
    ax.plot(episodes, metrics['avg_rewards'], linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Avg Reward')
    ax.set_title('Training Reward')

    # Eval reward
    ax = axes[0, 1]
    if metrics['eval_episodes']:
        ax.plot(metrics['eval_episodes'], metrics['eval_rewards'],
                marker='o', markersize=4)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Eval Reward')
    ax.set_title('Evaluation Reward')

    # Loss
    ax = axes[1, 0]
    if len(losses) >= 10:
        smoothed_loss = smooth(losses, 10)
        ax.plot(episodes[9:], smoothed_loss, linewidth=2)
    else:
        ax.plot(episodes, losses, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')

    # Episode length
    ax = axes[1, 1]
    if len(lengths) >= 10:
        smoothed_len = smooth(lengths, 10)
        ax.plot(episodes[9:], smoothed_len, linewidth=2)
    else:
        ax.plot(episodes, lengths, linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Length')
    ax.set_title('Episode Length')

    plt.tight_layout()
    plt.savefig(output_dir / 'summary.png', dpi=150)
    if show:
        plt.show()
    plt.close()

    print(f"Plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Plot training results')
    parser.add_argument('--log', type=str, required=True,
                        help='Path to training log file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for plots (default: same as log)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots')
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)

    # Parse log file
    print(f"Parsing log file: {log_path}")
    metrics = parse_log_file(str(log_path))

    if not metrics['episodes']:
        print("Error: No training data found in log file")
        sys.exit(1)

    print(f"Found {len(metrics['episodes'])} training entries")
    print(f"Found {len(metrics['eval_episodes'])} evaluation entries")

    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = log_path.parent / 'plots'

    # Generate plots
    plot_training_curves(metrics, output_dir, show=not args.no_show)


if __name__ == '__main__':
    main()
