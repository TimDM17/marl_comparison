import numpy as np
import random
import torch

# ============================================================================
# TRAINING CONFIGURATION FOR NQMIX ON MAMUJOCO HUMANOID
# ============================================================================

# Training hyperparameters for MaMuJoCo Humanoid "9|8" environment
# Adjusted based on NQMIX paper specifications and Humanoid complexity

"""
Paper specification
- hidden_dim: 64 (we use 128 for complex Humanoid)
- lr_actor: 5e-4
- lr_critic: 5e-4
- gamma: 0.99
- tau: 0.001 (we use 0.005 for faster convergence)
- buffer_capacity: ~5000
- optmizer: RMSprop

QUICK TEST CONFIG (for debugging):
Uncomment this for fast testing (~30-60 min on GPU)
"""
# CONFIG = {
#     'n_episodes': 500,
#     'max_steps': 500,
#     'batch_size': 8,
#     'eval_freq': 50,
#     'n_eval_episodes': 5,
#     'noise_scale_start': 0.12,  # Adjusted for [-0.4, 0.4] action space
#     'noise_scale_end': 0.02,    # Adjusted for [-0.4, 0.4] action space
#     'noise_decay_episodes': 300,
#     'hidden_dim': 128,
#     'lr_actor': 3e-4,
#     'lr_critic': 3e-4,
#     'gamma': 0.99,
#     'tau': 0.005,
#     'buffer_capacity': 1000,  # Reduced for Colab memory
#     'seed': 42,
# }

# ============================================================================
# FULL TRAINING CONFIG (recommended for good results)
# ============================================================================
CONFIG = {
    # ========================================================================
    # EPISODE AND TRAINING SETTINGS
    # ========================================================================
    'n_episodes': 3000,  # Total training episodes
    # Expected time: ~4-8 hours on GPU, ~16-24 hours on CPU
    # Humanoid typically needs 2000+ episodes to learn stable walking

    'max_steps': 1000,  # Maximum steps per episode
    # Humanoid episodes can be long (500-1000 steps)
    # Longer episodes -> better exploration but slower training

    'batch_size': 16,  # Number of episodes sampled per training step
    # Paper uses 8-16 for SMAC
    # Larger batch -> more stable gradients but more memory
    # Humanoid: 16 is good balance

    # ========================================================================
    # EVALUATION SETTINGS
    # ========================================================================
    'eval_freq': 100,  # Evaluate every N episodes
    # More frequent -> better monitoring but slower training
    # 100 episodes ≈ 5-10 minutes of training

    'n_eval_episodes': 10,  # Number of episodes for  evaluation
    # More episodes -> more reliable evaluation
    # 10 episodes provides good average without too much time

    # ========================================================================
    # EXPLORATION SETTINGS (CRITICAL FOR HUMANOID!)
    # ========================================================================
    'noise_scale_start': 0.12,  # Initial exploration noise
    # Adjusted for Humanoid Action Space [-0.4, 0.4]
    # Paper uses 0.3 for [-1, 1] action space
    # For [-0.4, 0.4]: scale proportionally -> 0.3 * 0.4 = 0.12
    # This is 30% of action range (strong exploration)

    'noise_scale_end': 0.02,   # Final exploration noise
    # Paper uses 0.05 for [-1, 1] action space
    # For [-0.4, 0.4]: 0.05 * 0.4 = 0.02
    # This is 5% of action range (fine-tuning)

    'noise_decay_episodes': 1500,  # Decay noise over these episodes
    # Linear decay: high exploration early → low exploration late
    # Half of total episodes (1500/3000) for smooth transition
    # After 1500 episodes: noise stays at minimum (exploitation)

    # ========================================================================
    # NETWORK ARCHITECTURE
    # ========================================================================
    'hidden_dim': 128,   # GRU hidden state dimension
    # Paper uses 64 for SMAC (simpler environment)
    # Humanoid is more complex → 128 provides more capacity

    # ========================================================================
    # LEARNING RATES (from paper)
    # ========================================================================
     'lr_actor': 3e-4,    # Actor learning rate (RMSprop)
    # Paper: 5e-4, we use 3e-4 for stability
    # Lower LR → slower learning but more stable
    # Humanoid is sensitive to actor updates
    
    'lr_critic': 3e-4,   # Critic learning rate (RMSprop)
    # Paper: 5e-4, we use 3e-4 for stability
    # Critic and actor should have similar learning rates

    # ========================================================================
    # RL HYPERPARAMETERS
    # ========================================================================
    'gamma': 0.99,       # Discount factor
    # Paper specification: 0.99
    # Standard for long-horizon tasks (Humanoid walking)
    # γ=0.99 means rewards 100 steps away worth 36.6% of current
    
    'tau': 0.005,        # Soft target update rate
    # Paper: 0.001 (very slow!)
    # We use 0.005 (5x faster) for faster convergence
    # Trade-off: faster learning vs stability
    # τ=0.005 means target updates 0.5% per step

    # ========================================================================
    # REPLAY BUFFER
    # ========================================================================
    'buffer_capacity': 5000,  # Maximum episodes in replay buffer
    # Paper uses ~5000 for experiments
    # Each episode: ~1000 steps × 2 agents × (obs + action + state)
    # Memory estimate: ~4-5 GB for 5000 episodes
    # For Colab (12-16 GB RAM): 5000 is safe
    # Reduce to 2000-3000 if memory issues
    
    # ========================================================================
    # REPRODUCIBILITY
    # ========================================================================
    'seed': 42,          # Random seed for reproducibility
    # Sets seeds for: numpy, torch, random
    # Same seed → same results (on same hardware)
    # Change seed for different runs: 42, 123, 456, etc.
}

# ============================================================================
# DISPLAY FINAL CONFIGURATION
# ============================================================================
print("\nFinal Training Configuration:")
print("="*70)
for key, value in CONFIG.items():
    print(f"  {key:25s}: {value}")
print("="*70)

# ============================================================================
# SET RANDOM SEEDS FOR REPRODUCIBILITY
# ============================================================================
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
random.seed(CONFIG['seed'])

# For CUDA reproducibility (if using GPU)
if torch.cuda.is_available():
    torch.cuda.manual_seed(CONFIG['seed'])
    torch.cuda.manual_seed_all(CONFIG['seed'])
    # Note: Full reproducibility on GPU is hard, these help but don't guarantee
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print("\n✓ Random seeds set for reproducibility")
print(f"  NumPy seed: {CONFIG['seed']}")
print(f"  PyTorch seed: {CONFIG['seed']}")
print(f"  Python random seed: {CONFIG['seed']}")
if torch.cuda.is_available():
    print(f"  CUDA seed: {CONFIG['seed']}")
    print("  ⚠️  Note: GPU results may still vary slightly due to hardware")