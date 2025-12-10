from collections import deque
from typing import Dict, List, Tuple
import random
import numpy as np
import torch

class ReplayBuffer:
    """
    Experience replay buffer for storing complete episodes.

    Key design for NQMIX:
    - Stores FULL EPISODES (not individual transitions like DQN)
    - Enables off-policy learning by reusing old experience
    - Uses deque with maxlen for automatic old data removal
    """
    def __init__(self, capacity: int = 1000):
        """
        Initialize replay buffer with fixed capacity

        Args: 
            capacity: Maximum number of episodes to store
        
        Implementation:
        - Uses deque (double-ended queue) from collections
        - maxlen=capacity means oldest episodes auto-removed when full
        - FIFO (First In, First Out) when capacity exceeded
        """
        # deque with maxlen automatically removes oldest when new items added
        # More efficient than list for this use case (O(1) append/pop)
        self.buffer = deque(maxlen=capacity)

    def push(self, episode: Dict) -> None:
        """
        Store a complete episode in the buffer

        Args:
            episode: Dictionary containing full episode data
            {
                    'observations': [[obs_agent0_t0, obs_t1, ...], 
                                    [obs_agent1_t0, obs_t1, ...]],  # Per agent
                    'actions': [[act_agent0_t0, act_t1, ...],
                               [act_agent1_t0, act_t1, ...]],       # Per agent
                    'last_actions': [[last_act_agent0_t0, ...],
                                    [last_act_agent1_t0, ...]],     # Per agent
                    'states': [state_t0, state_t1, ...],            # Global states
                    'rewards': [reward_t0, reward_t1, ...]          # Shared rewards
                }
        
        Episode structure:
        - Length: Variable (episode can end early due to termination)
        - All lists have same length (number of timesteps in episode)
        - Agent-specific data (obs, actions) are nestes lists
        - Shared data (states, rewards) are flat lists
        """
        # Append episode to buffer
        # If buffer is full (len == capacity), oldest episode auto-removed
        self.buffer.append(episode)

    def sample(self, batch_size: int) -> List[Dict]:
        """
        Randomly sample a batch of episodes for training

        Args:
            batch_size: Number of episodes to sample

        Returns: 
            List of episode dictionaries (randomly sampled)

        Sampling strategy:
        - Uniform random sampling (each episode has equal probability)
        - Sampling WITH replacement within a batch (can get duplicates)
        - If buffer has fewer episodes than batch_size, return all available

        Why random sampling?
        - Breaks temporal correlation between consecutive episodes
        - Provides diverse training data (episodes from different policies)
        - Improves training stability and convergence
        """
        # random.sample() samples without replacement
        # min() ensures we don't try to sample more than available
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        """
        Get current number of episodes in buffer

        Returns:
            Integer count of stored episodes

        Usage:
        - Check if buffer has enough data before training
        - Monitor buffer fill rate during training
        - Implement training start condition (e.g., wait for 100 episodes)
        """
        return len(self.buffer)

    def sample_batch(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of episodes and return as padded tensors (research standard).

        Optimized implementation:
        1. Pre-allocate numpy arrays (faster than torch on CPU)
        2. Fill numpy arrays with episode data
        3. Convert to tensors once and move to device

        Args:
            batch_size: Number of episodes to sample
            device: torch device to place tensors on

        Returns:
            Dictionary with batched tensors:
            {
                'observations': [n_agents] list of [batch, max_time, obs_dim] tensors
                'actions': [n_agents] list of [batch, max_time, action_dim] tensors
                'last_actions': [n_agents] list of [batch, max_time, action_dim] tensors
                'states': [batch, max_time, state_dim] tensor
                'rewards': [batch, max_time, 1] tensor
                'mask': [batch, max_time, 1] tensor (1 for valid, 0 for padded)
                'max_seq_length': int
                'batch_size': int
            }
        """
        episodes = self.sample(batch_size)

        # Find dimensions from first episode
        n_agents = len(episodes[0]['observations'])
        obs_dims = [len(episodes[0]['observations'][i][0]) for i in range(n_agents)]
        action_dims = [len(episodes[0]['actions'][i][0]) for i in range(n_agents)]
        state_dim = len(episodes[0]['states'][0])

        # Find max sequence length in this batch
        max_seq_length = max(len(ep['rewards']) for ep in episodes)
        actual_batch_size = len(episodes)

        # Pre-allocate NUMPY arrays (faster for CPU operations)
        # Using np.zeros is faster than torch.zeros for filling
        obs_np = [
            np.zeros((actual_batch_size, max_seq_length, obs_dims[i]), dtype=np.float32)
            for i in range(n_agents)
        ]
        act_np = [
            np.zeros((actual_batch_size, max_seq_length, action_dims[i]), dtype=np.float32)
            for i in range(n_agents)
        ]
        last_act_np = [
            np.zeros((actual_batch_size, max_seq_length, action_dims[i]), dtype=np.float32)
            for i in range(n_agents)
        ]
        states_np = np.zeros((actual_batch_size, max_seq_length, state_dim), dtype=np.float32)
        rewards_np = np.zeros((actual_batch_size, max_seq_length, 1), dtype=np.float32)
        mask_np = np.zeros((actual_batch_size, max_seq_length, 1), dtype=np.float32)
        terminated_np = np.zeros((actual_batch_size, max_seq_length, 1), dtype=np.float32)

        # Fill numpy arrays with episode data (CPU operations)
        for b, episode in enumerate(episodes):
            ep_len = len(episode['rewards'])

            # Fill observations, actions, last_actions for each agent
            for i in range(n_agents):
                # Stack lists into numpy arrays (more efficient than individual assignments)
                obs_np[i][b, :ep_len] = np.array(episode['observations'][i][:ep_len])
                act_np[i][b, :ep_len] = np.array(episode['actions'][i][:ep_len])
                last_act_np[i][b, :ep_len] = np.array(episode['last_actions'][i][:ep_len])

            # Fill states and rewards
            states_np[b, :ep_len] = np.array(episode['states'][:ep_len])
            rewards_np[b, :ep_len, 0] = np.array(episode['rewards'][:ep_len])

            # Mask: 1 for valid timesteps, 0 for padding
            mask_np[b, :ep_len, 0] = 1.0

            # Terminated: 1 if this timestep is a terminal state, 0 otherwise
            # This is critical for TD targets - terminal states should NOT bootstrap
            if 'terminated' in episode and len(episode['terminated']) > 0:
                terminated_np[b, :ep_len, 0] = np.array(episode['terminated'][:ep_len], dtype=np.float32)
            # If terminated not available (backward compatibility), assume all zeros
            # (will use the old mask-based approach)

        # Convert to tensors ONCE and move to device (single transfer)
        # torch.from_numpy shares memory, then .to(device) copies to GPU
        observations = [
            torch.from_numpy(obs_np[i]).to(device)
            for i in range(n_agents)
        ]
        actions = [
            torch.from_numpy(act_np[i]).to(device)
            for i in range(n_agents)
        ]
        last_actions = [
            torch.from_numpy(last_act_np[i]).to(device)
            for i in range(n_agents)
        ]
        states = torch.from_numpy(states_np).to(device)
        rewards = torch.from_numpy(rewards_np).to(device)
        mask = torch.from_numpy(mask_np).to(device)
        terminated = torch.from_numpy(terminated_np).to(device)

        return {
            'observations': observations,      # List of [batch, time, obs_dim]
            'actions': actions,                # List of [batch, time, action_dim]
            'last_actions': last_actions,      # List of [batch, time, action_dim]
            'states': states,                  # [batch, time, state_dim]
            'rewards': rewards,                # [batch, time, 1]
            'mask': mask,                      # [batch, time, 1]
            'terminated': terminated,          # [batch, time, 1] - 1 if terminal, 0 otherwise
            'max_seq_length': max_seq_length,
            'batch_size': actual_batch_size
        }
    


