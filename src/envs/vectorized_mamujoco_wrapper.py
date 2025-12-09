"""
Vectorized MaMuJoCo environment wrapper for parallel episode collection.

Purpose:
    Run multiple MaMuJoCo environments in parallel to speed up training.
    Uses AsyncVectorEnv pattern to run each env in a separate process.

Key Concepts:
    - Asynchronous execution: Each env runs in its own process
    - Automatic reset: When an env finishes, it auto-resets
    - Batched observations: Returns stacked observations from all envs
    - Episode tracking: Tracks which envs are done and their rewards

Connections:
    - Used by: src/training/trainer.py (replaces MaMuJoCoWrapper for training)
    - Uses: MaMuJoCoWrapper (creates N instances)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from multiprocessing import Process, Pipe
from src.envs.mamujoco_wrapper import MaMuJoCoWrapper


def worker(remote, parent_remote, env_name, partitioning):
    """
    Worker process for running a single environment.

    This function runs in a separate process and communicates via pipes.

    Args:
        remote: Child end of the pipe
        parent_remote: Parent end (closed in child)
        env_name: Environment name ('Humanoid')
        partitioning: Action partitioning ('9|8')
    """
    parent_remote.close()  # Close parent's end in child process
    env = MaMuJoCoWrapper(env_name=env_name, partitioning=partitioning)

    while True:
        try:
            cmd, data = remote.recv()

            if cmd == 'step':
                # Execute step and return results
                obs_dict, rewards_dict, term_dict, trunc_dict, info = env.step(data)

                # Check if episode is done
                done = any(term_dict.values()) or any(trunc_dict.values())

                # Convert dicts to lists for easier processing
                observations = [obs_dict[agent_id] for agent_id in env.possible_agents]
                rewards = [rewards_dict[agent_id] for agent_id in env.possible_agents]

                # If done, auto-reset for next episode
                if done:
                    reset_obs_dict, reset_info = env.reset()
                    reset_observations = [reset_obs_dict[agent_id] for agent_id in env.possible_agents]
                    remote.send((observations, rewards, done, reset_observations, info))
                else:
                    remote.send((observations, rewards, done, None, info))

            elif cmd == 'reset':
                obs_dict, info = env.reset(seed=data)
                observations = [obs_dict[agent_id] for agent_id in env.possible_agents]
                remote.send((observations, info))

            elif cmd == 'close':
                env.close()
                remote.close()
                break

            elif cmd == 'get_spaces':
                remote.send((env.n_agents, env.obs_dims, env.action_dims, env.state_dim))

        except EOFError:
            break


class VectorizedMaMuJoCoWrapper:
    """
    Vectorized wrapper for running multiple MaMuJoCo environments in parallel.

    Creates n_envs separate environment processes that run asynchronously.
    """

    def __init__(
        self,
        env_name: str = "Humanoid",
        partitioning: str = "9|8",
        n_envs: int = 8
    ):
        """
        Initialize vectorized environments.

        Args:
            env_name: Environment name (e.g., "Humanoid")
            partitioning: Action partitioning (e.g., "9|8")
            n_envs: Number of parallel environments
        """
        self.env_name = env_name
        self.partitioning = partitioning
        self.n_envs = n_envs
        self.waiting = False
        self.closed = False

        # Create pipes for communication with workers
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])

        # Create worker processes
        self.processes = []
        for work_remote, remote in zip(self.work_remotes, self.remotes):
            args = (work_remote, remote, env_name, partitioning)
            process = Process(target=worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()  # Close child's end in parent

        # Get environment specs from first worker
        self.remotes[0].send(('get_spaces', None))
        self.n_agents, self.obs_dims, self.action_dims, self.state_dim = self.remotes[0].recv()

        # Episode tracking
        self.episode_rewards = [0.0] * n_envs
        self.episode_lengths = [0] * n_envs

    def reset(self, seed: Optional[int] = None) -> List[List[np.ndarray]]:
        """
        Reset all environments.

        Args:
            seed: Base random seed (each env gets seed + env_idx)

        Returns:
            observations: List of [n_envs, n_agents] observations
        """
        for idx, remote in enumerate(self.remotes):
            env_seed = seed + idx if seed is not None else None
            remote.send(('reset', env_seed))

        # Collect reset observations from all envs
        results = [remote.recv() for remote in self.remotes]
        observations_list = [obs for obs, info in results]

        # Reset episode tracking
        self.episode_rewards = [0.0] * self.n_envs
        self.episode_lengths = [0] * self.n_envs

        return observations_list

    def step(
        self,
        actions_list: List[List[np.ndarray]]
    ) -> Tuple[List[List[np.ndarray]], List[float], List[bool], List[Dict]]:
        """
        Execute one step in all environments.

        Args:
            actions_list: List of [n_envs] action lists, where each is [n_agents] actions

        Returns:
            observations_list: [n_envs] lists of [n_agents] observations
            rewards_list: [n_envs] rewards (summed over agents)
            dones_list: [n_envs] done flags
            infos_list: [n_envs] info dicts (with episode_reward and episode_length when done)
        """
        # Send actions to all workers
        for remote, actions in zip(self.remotes, actions_list):
            remote.send(('step', actions))

        # Collect results
        results = [remote.recv() for remote in self.remotes]

        observations_list = []
        rewards_list = []
        dones_list = []
        infos_list = []

        for env_idx, (obs, rewards, done, reset_obs, info) in enumerate(results):
            # Sum rewards across agents (cooperative task)
            total_reward = sum(rewards)
            self.episode_rewards[env_idx] += total_reward
            self.episode_lengths[env_idx] += 1

            # If episode is done, add episode stats to info and use reset obs
            if done:
                info['episode'] = {
                    'r': self.episode_rewards[env_idx],
                    'l': self.episode_lengths[env_idx]
                }
                # Reset tracking for this env
                self.episode_rewards[env_idx] = 0.0
                self.episode_lengths[env_idx] = 0
                # Use reset observations for next episode
                observations_list.append(reset_obs)
            else:
                observations_list.append(obs)

            rewards_list.append(total_reward)
            dones_list.append(done)
            infos_list.append(info)

        return observations_list, rewards_list, dones_list, infos_list

    def close(self):
        """Close all worker processes."""
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def __del__(self):
        """Ensure processes are closed on deletion."""
        if not self.closed:
            self.close()