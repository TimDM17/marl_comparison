"""
Render saved model with video recording.

RESEARCH USE: Use this script AFTER training to generate proper evaluation videos
of your best model for thesis/papers.

Usage:
    # Render best model (PRIMARY RESEARCH RESULTS)
    python scripts/render_model.py --checkpoint results/nqmix_humanoid/best_model.pth --config configs/nqmix_humanoid.yaml

    # Render with more episodes for statistics
    python scripts/render_model.py --checkpoint results/nqmix_humanoid/best_model.pth --config configs/nqmix_humanoid.yaml --episodes 10

    # Render final model for comparison
    python scripts/render_model.py --checkpoint results/nqmix_humanoid/final_model.pth --config configs/nqmix_humanoid.yaml --episodes 10
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import imageio

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents import NQMIX
from src.envs import MaMuJoCoWrapper
from src.utils import load_config


def render_model(checkpoint_path: str, config_path: str, n_episodes: int = 3, output_path: str = None):
    """
    Load a saved model and render evaluation episodes with video recording.
    
    This is the RESEARCH-STANDARD way to evaluate and visualize your trained model.
    
    Args:
        checkpoint_path: Path to saved model checkpoint
        config_path: Path to config file
        n_episodes: Number of episodes to render
        output_path: Custom output video path (optional)
    """
    print("="*70)
    print("NQMIX Model Rendering (Research Evaluation)")
    print("="*70)
    
    # Load config
    config = load_config(config_path)
    env_name = config.get('env_name', 'Humanoid')
    partitioning = config.get('partitioning', '9|8')
    
    # Create environment WITH render mode
    print(f"\nCreating environment: {env_name} ({partitioning})")
    env = MaMuJoCoWrapper(env_name=env_name, partitioning=partitioning, render_mode='rgb_array')
    
    # Get dimensions
    obs_dims = env.obs_dims
    action_dims = env.action_dims
    state_dim = env.state_dim
    
    print(f"Observation dims: {obs_dims}")
    print(f"Action dims: {action_dims}")
    print(f"State dim: {state_dim}")
    
    # Create agent
    agent_params = config.get('agent_params', {})
    print(f"\nCreating NQMIX agent...")
    agent = NQMIX(
        n_agents=env.n_agents,
        obs_dims=obs_dims,
        action_dims=action_dims,
        state_dim=state_dim,
        hidden_dim=agent_params.get('hidden_dim', 64),
        lr_actor=agent_params.get('lr_actor', 5e-4),
        lr_critic=agent_params.get('lr_critic', 5e-4),
        gamma=agent_params.get('gamma', 0.99),
        tau=agent_params.get('tau', 0.001),
        buffer_capacity=agent_params.get('buffer_capacity', 5000),
        action_low=agent_params.get('action_low', -0.4),
        action_high=agent_params.get('action_high', 0.4)
    )
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    agent.load(checkpoint_path)
    print("✓ Checkpoint loaded successfully")
    
    # Setup video path
    if output_path is None:
        save_dir = Path(config.get('save_dir', './results/nqmix_humanoid'))
        video_dir = save_dir / 'videos'
        video_dir.mkdir(parents=True, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = video_dir / f"render_{timestamp}.mp4"
    else:
        video_path = Path(output_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nVideo will be saved to: {video_path}")
    print(f"\nRendering {n_episodes} episodes...")
    print("="*70)
    
    # Run episodes and collect frames
    episode_rewards = []
    episode_lengths = []
    all_frames = []
    
    for ep in range(n_episodes):
        obs, _ = env.env.reset()
        observations = [obs[agent_id] for agent_id in env.possible_agents]
        
        hiddens = agent.init_hidden_states()
        last_actions = [np.zeros(action_dims[i]) for i in range(env.n_agents)]
        
        episode_reward = 0
        episode_length = 0
        episode_frames = []
        done = False
        
        while not done:
            # Capture frame
            frame = env.render()
            if frame is not None:
                episode_frames.append(frame)
            
            # Select actions (no exploration - deterministic policy)
            actions, hiddens = agent.select_actions(
                observations=observations,
                last_actions=last_actions,
                hiddens=hiddens,
                explore=False
            )
            
            # Detach hidden states
            hiddens = [h.detach() if hasattr(h, 'detach') else h for h in hiddens]
            
            # Step environment
            action_dict = {env.possible_agents[i]: actions[i] for i in range(env.n_agents)}
            obs, rewards, terminated, truncated, _ = env.env.step(action_dict)
            
            # Update
            observations = [obs[agent_id] for agent_id in env.possible_agents]
            last_actions = actions
            
            # Accumulate reward
            reward = list(rewards.values())[0]
            episode_reward += reward
            episode_length += 1
            
            # Check done
            done = any(terminated.values()) or any(truncated.values())
        
        # Add episode frames to collection
        all_frames.extend(episode_frames)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {ep+1:2d} | Reward: {episode_reward:7.1f} | Length: {episode_length:4d} | Frames: {len(episode_frames)}")
    
    # Save video
    print(f"\nSaving video with {len(all_frames)} frames...")
    if len(all_frames) > 0:
        imageio.mimsave(str(video_path), all_frames, fps=30)
        print(f"✓ Video saved successfully")
    else:
        print("⚠ No frames captured - video not saved")
    
    # Close environment
    env.close()
    
    # Summary statistics (for research reporting)
    print("="*70)
    print("EVALUATION STATISTICS (Report these in your thesis!)")
    print("="*70)
    print(f"Average Reward: {np.mean(episode_rewards):7.1f} ± {np.std(episode_rewards):6.1f}")
    print(f"Min Reward:     {np.min(episode_rewards):7.1f}")
    print(f"Max Reward:     {np.max(episode_rewards):7.1f}")
    print(f"Average Length: {np.mean(episode_lengths):7.1f} ± {np.std(episode_lengths):6.1f}")
    print(f"\n✓ Results for checkpoint: {checkpoint_path}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Render NQMIX model with video recording (Research Standard)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (e.g., results/nqmix_humanoid/best_model.pth)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file (e.g., configs/nqmix_humanoid.yaml)')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of episodes to render (default: 3, recommend 10 for research)')
    parser.add_argument('--output', type=str, default=None,
                        help='Custom output video path (optional)')
    
    args = parser.parse_args()
    
    render_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        n_episodes=args.episodes,
        output_path=args.output
    )


if __name__ == '__main__':
    main()