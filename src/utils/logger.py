"""
Clean logging system for MARL training.

Design: Minimal, one-line logs, essential metrics only.
"""


from typing import Dict
from pathlib import Path


class Logger:
    """
    Minimal logger for MARL training.
    
    Features:
    - One-line training logs
    - Clean evaluation logs
    - Summary statistics
    - Optional file logging
    """
    
    def __init__(self, log_file: str = None, verbose: bool = True):
        """
        Initialize logger.
        
        Args:
            log_file: Path to log file (optional)
            verbose: Whether to print to console
        """
        self.log_file = log_file
        self.verbose = verbose
        
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            # Clear existing log
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("")
    
    def info(self, message: str):
        """Print info message"""
        if self.verbose:
            print(message)
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
    
    def log_train(self, metrics: Dict):
        """
        Log training progress (one line).
        
        Args:
            metrics: Dictionary with training metrics
        """
        log_str = (
            f"Ep {metrics['episode']:4d} | "
            f"R: {metrics['reward']:7.1f} | "
            f"R̄10: {metrics['avg_reward_10']:7.1f} | "
            f"Len: {metrics['length']:4d} | "
            f"Loss: {metrics['loss']:7.4f} | "
            f"Buf: {metrics['buffer_size']:4d} | "
            f"T: {metrics['time_min']:6.1f}m"
        )
        self.info(log_str)
    
    def log_eval(self, episode: int, eval_reward: float, 
                 eval_length: float, best_reward: float):
        """
        Log evaluation results.
        
        Args:
            episode: Current episode
            eval_reward: Average evaluation reward
            eval_length: Average evaluation length
            best_reward: Best reward so far
        """
        self.info(f"\n{'='*70}")
        self.info(f"EVAL @ Ep {episode:4d} | "
                 f"R: {eval_reward:7.1f} | "
                 f"Len: {eval_length:6.1f} | "
                 f"Best: {best_reward:7.1f}")
        self.info(f"{'='*70}\n")
    
    def log_summary(self, summary: Dict):
        """
        Log final training summary.
        
        Args:
            summary: Dictionary with summary statistics
        """
        self.info(f"\n{'='*70}")
        self.info("TRAINING COMPLETE")
        self.info(f"{'='*70}")
        self.info(f"Time:         {summary['total_time_min']:7.1f} min")
        self.info(f"Final reward: {summary['final_avg_reward']:7.1f}")
        self.info(f"Best reward:  {summary['best_eval_reward']:7.1f}")
        self.info(f"Episodes:     {summary['total_episodes']:7d}")
        self.info(f"{'='*70}\n")


