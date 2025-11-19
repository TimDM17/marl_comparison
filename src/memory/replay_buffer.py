from collections import deque
from typing import Dict
import random
from typing import Dict, List

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
    


