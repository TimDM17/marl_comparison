"""Evaluation logic for trained agents"""

import numpy as np

class Evaluator:
    """Evaluate agent performance without exploration"""

    def __init__(self, agent, env, config):
        self.agent = agent
        self.env = env
        self.config = config

    def evaluate(self, n_episodes: int = 10, seed: int = 999):
        """
        Evaluate agent for n_episodes without exploration

        Returns:
            avg_reward: Average episode reward
            avg_length: Average episode length
        """
        total_rewards = []
        total_lengths = []

        for ep in range(n_episodes):
            episode_reward, episode_length = self._eval_episode(seed + ep)
            total_rewards.append(episode_reward)
            total_lengths.append(episode_length)

        return np.mean(total_rewards), np.mean(total_lengths)
    
    def _eval_episode(self, seed: int):
        """Single evaluation episode"""
        obs, _ = self.env.reset(seed=seed)
        observations = [obs[agent] for agent in self.env.possible_agents]
        hiddens = self.agent.init_hidden_states()
        last_actions = [np.zeros(self.env.action_dims[i]) 
                       for i in range(self.env.n_agents)]
        
        episode_reward = 0
        episode_length = 0
        
        for step in range(self.config['max_steps']):
            # No exploration
            actions, hiddens = self.agent.select_actions(
                observations, last_actions, hiddens, explore=False
            )
            
            # Step
            action_dict = {self.env.possible_agents[i]: actions[i] 
                          for i in range(self.env.n_agents)}
            next_obs, rewards, terminated, truncated, _ = self.env.step(action_dict)
            
            episode_reward += sum(rewards.values())
            episode_length += 1
            
            # Update
            observations = [next_obs[agent] for agent in self.env.possible_agents]
            last_actions = actions
            
            if any(terminated.values()) or any(truncated.values()):
                break
        
        return episode_reward, episode_length