"""
Generic trainer for mutli-agent RL algorithms.
Works with any agent that implements the BaseAgent interface
"""

import time
import numpy as np
from typing import Dict, Optional
from src.utils.logger import Logger
from src.training.evaluator import Evaluator


class Trainer:
    """
    Generic trainer for MARL algorithms

    Design principles:
    - Algorithm-agnostic (works with NQMIX, FACMAC)
    - Clean separation of concerns
    - Minimal logging
    - Easy to extend
    """

    def __init__(
            self,
            agent,
            env,
            config: Dict,
            logger: Logger,
            save_dir: str = "./results/checkpoints"
    ):
        self.agent = agent
        self.env = env
        self.config = config
        self.logger = logger
        self.save_dir = save_dir

        # Initialize evaluator
        self.evaluator = Evaluator(agent, env, config)

        # Metrics
        self.episode_rewards = []
        self.episode_lenghts = []
        self.best_eval_reward = -np.inf

    def train(self):
        """Main training loop - clean and minimal"""

        self.logger.info("Starting training...")
        start_time = time.time()

        for episode in range(self.config['n_episodes']):
            # Collect episode
            episode_data, episode_reward, episode_length = self._collect_episode(episode)

            # Store in buffer
            self.agent.store_episode(episode_data)
            self.episode_rewards.append(episode_reward)
            self.episode_lenghts.append(episode_length)

            # Train
            loss = self._train_step(episode)

            # Log progress (minimal)
            if episode % self.config['log_freq'] == 0:
                self._log_progress(episode, loss, time.time() - start_time)

            # Evaluate
            if episode > 0 and episode % self.config['eval_freq'] == 0:
                self._evaluate(episode)

        # Final save and summary
        self._finalize_training(time.time() - start_time)
    
    
    def _collect_episode(self, episode: int):
        """Collect one episode of experience"""

        # Decay noise
        noise_scale = self._get_noise_scale(episode)

        # Reset
        obs, info = self.env.reset(seed=self.config['seed'] + episode)
        observations, state = self._parse_obs(obs)
        hiddens = self.agent.init_hidden_states()
        last_actions = self._get_zero_actions()

        # Episode data
        episode_data = self._init_episode_data()
        episode_reward = 0
        episode_length = 0

        # Rollout
        for step in range(self.config['max_steps']):
            # Select actions
            actions, hiddens = self.agent.select_actions(
                observations, last_actions, hiddens,
                explore=True, noise_scale=noise_scale
            )

            # Step environment
            next_obs, rewards, terminated, truncated, info = self.env.step(actions)
            done = self._is_done(terminated, truncated)
            reward = self._get_total_reward(rewards)

            # Store transition
            self._store_transition(
                episode_data, observations, actions,
                last_actions, state, reward
            )

            episode_reward += reward
            episode_length += 1

            # Update
            observations, state = self._parse_obs(next_obs)
            last_actions = actions

            if done:
                break

        # Store termination flags
        episode_data['terminated'] = any(terminated.values()) if isinstance(terminated, dict) else terminated
        episode_data['truncated'] = any(truncated.values()) if isinstance(truncated, dict) else truncated

        return episode_data, episode_reward, episode_length
    
    
    def _train_step(self, episode: int):
        """Single training step"""
        if len(self.agent.replay_buffer) >= self.config['batch_size']:
            return self.agent.train_step(batch_size=self.config['batch_size'])
        return None
    

    def _evaluate(self, episode: int):
        """Evaluate current policy"""
        eval_reward, eval_length = self.evaluator.evaluate(
            n_episodes=self.config['n_eval_episodes']
        )
        
        # Log
        self.logger.log_eval(episode, eval_reward, eval_length, self.best_eval_reward)
        
        # Save best
        if eval_reward > self.best_eval_reward:
            self.best_eval_reward = eval_reward
            self.agent.save(f"{self.save_dir}/best.pth")
            self.logger.info("[OK] New best model saved")
    
    
    def _log_progress(self, episode: int, loss: Optional[float], elapsed: float):
        """Minimal progress logging"""
        metrics = {
            'episode': episode,
            'reward': self.episode_rewards[-1],
            'avg_reward_10': np.mean(self.episode_rewards[-10:]),
            'length': self.episode_lenghts[-1],
            'loss': loss if loss else 0.0,
            'buffer_size': len(self.agent.replay_buffer),
            'time_min': elapsed / 60
        }
        self.logger.log_train(metrics)

    
    def _finalize_training(self, total_time: float):
        """Save final model and summary"""
        self.agent.save(f"{self.save_dir}/final.pth")

        summary = {
            'total_time_min': total_time / 60,
            'final_avg_reward': np.mean(self.episode_rewards[-100:]),
            'best_eval_reward': self.best_eval_reward,
            'total_episodes': len(self.episode_rewards)
        }
        self.logger.log_summary(summary)
        self.env.close()
    

    # Helper methods
    def _get_noise_scale(self, episode: int) -> float:
        progress = min(episode / self.config['noise_decay_episodes'], 1.0)
        return self.config['noise_scale_start'] + \
               (self.config['noise_scale_end'] - self.config['noise_scale_start']) * progress
    
    def _parse_obs(self, obs):
        observations = [obs[agent] for agent in self.env.possible_agents]
        state = np.concatenate(observations)
        return observations, state
    
    def _get_zero_actions(self):
        return [np.zeros(self.env.action_dims[i]) for i in range(self.env.n_agents)]
    
    def _init_episode_data(self):
        return {
            'observations': [[] for _ in range(self.env.n_agents)],
            'actions': [[] for _ in range(self.env.n_agents)],
            'last_actions': [[] for _ in range(self.env.n_agents)],
            'states': [],
            'rewards': [],
            'terminated': False,
            'truncated': False
        }
    
    def _is_done(self, terminated, truncated):
        return any(terminated.values()) or any(truncated.values())
    
    def _get_total_reward(self, rewards):
        return sum(rewards.values())
    
    def _store_transition(self, episode_data, observations, actions, 
                         last_actions, state, reward):
        for i in range(self.env.n_agents):
            episode_data['observations'][i].append(observations[i])
            episode_data['actions'][i].append(actions[i])
            episode_data['last_actions'][i].append(last_actions[i])
        episode_data['states'].append(state)
        episode_data['rewards'].append(reward)







        