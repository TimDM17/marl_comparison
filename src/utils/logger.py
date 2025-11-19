"""
Clean logging system for MARL training.

Design: Minimal, one-line logs, essential metrics only.
"""


from typing import Dict, Optional
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

class Logger:
    """
    Minimal logger for MARL training.
    
    Features:
    - One-line training logs
    - Clean evaluation logs
    - Summary statistics
    - Optional file logging
    """
    
    def __init__(self, log_file: Optional[str] = None, verbose: bool = True, tensorboard_dir: Optional[str] = None):
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
        
        self.writer = None
        if tensorboard_dir:
            self.writer = SummaryWriter(log_dir=tensorboard_dir)
    
    def info(self, message: str) -> None:
        """Print info message"""
        if self.verbose:
            print(message)
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
    
    def log_train(self, metrics: Dict) -> None:
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

        if self.writer:
            self.writer.add_scalar('train/episode_reward', metrics['reward'], metrics['episode'])
            self.writer.add_scalar('train/avg_reward_10', metrics['avg_reward_10'], metrics['episode'])
            self.writer.add_scalar('train/episode_length', metrics['length'], metrics['episode'])
            self.writer.add_scalar('train/loss', metrics['loss'], metrics['episode'])
            self.writer.add_scalar('train/buffer_size', metrics['buffer_size'], metrics['episode'])
    
    def log_eval(self, episode: int, eval_reward: float, 
                 eval_length: float, best_reward: float) -> None:
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

        if self.writer:
            self.writer.add_scalar('eval/mean_reward', eval_reward, episode)
            self.writer.add_scalar('eval/mean_length', eval_length, episode)
            self.writer.add_scalar('eval/best_reward', best_reward, episode)
    
    def log_summary(self, summary: Dict) -> None:
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


    def closeTensorBoard(self) -> None:

        if self.writer:
            self.writer.close()
