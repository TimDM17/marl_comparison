# Multi-Agent Reinforcement Learning Algorithm Comparison

Comparing state-of-the-art MARL algorithms on the MaMuJoCo Humanoid environment.
More environments coming soon.

## Algorithms

- **NQMIX** - Non-monotonic value function factorization for continuous actions
- **FACMAC** - Continuous action actor-critic (coming soon)

## Environment

**MaMuJoCo Humanoid "9|8" (Gymnasium Robotics)**
- 2 agents: Upper body (9 actions) + Lower body (8 actions)
- Cooperative task: Bipedal locomotion
- Partial observability per agent
- Continuous action space: [-0.4, 0.4]

## Installation

### Install dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### Training

Train NQMIX on Humanoid:
```bash
python scripts/train.py --config configs/nqmix_humanoid.yaml
```

With custom seed:
```bash
python scripts/train.py --config configs/nqmix_humanoid.yaml --seed 123
```

Training will:
1. Create `results/nqmix_humanoid/` directory
2. Save logs to `training.log`
3. Save best model to `best_model/`
4. Save final model to `final_model/`

### Evaluation

Evaluate a trained model:
```bash
python scripts/evaluate.py \
    --checkpoint results/nqmix_humanoid/best_model \
    --config configs/nqmix_humanoid.yaml \
    --episodes 100
```

### Plotting Results

Generate training plots:
```bash
python scripts/plot_results.py --log results/nqmix_humanoid/training.log
```

Save plots without displaying:
```bash
python scripts/plot_results.py \
    --log results/nqmix_humanoid/training.log \
    --output results/nqmix_humanoid/plots \
    --no-show
```

## Configuration

Example config (`configs/nqmix_humanoid.yaml`):

```yaml
algorithm: nqmix
env_name: Humanoid
partitioning: "9|8"

# Training
n_episodes: 3000
batch_size: 16
seed: 42

# Evaluation
eval_freq: 100
n_eval_episodes: 10

# Exploration
noise_scale_start: 0.12
noise_scale_end: 0.02
noise_decay_episodes: 1500

# Agent parameters
agent_params:
  hidden_dim: 128
  lr_actor: 0.0003
  lr_critic: 0.0003
  gamma: 0.99
  tau: 0.005
  buffer_capacity: 5000

# Paths
save_dir: ./results/nqmix_humanoid
```

## Project Structure

```
marl_comparison/
├── configs/              # Algorithm configurations
│   └── nqmix_humanoid.yaml
├── src/
│   ├── agents/           # Algorithm implementations
│   │   ├── base_agent.py # Abstract interface
│   │   ├── nqmix.py      # NQMIX implementation
│   │   └── facmac.py     # FACMAC (placeholder)
│   ├── networks/         # Neural network modules
│   │   ├── agent_network.py  # Per-agent actor-critic
│   │   └── mixer_network.py  # Q-value mixer
│   ├── memory/           # Experience replay
│   │   └── replay_buffer.py
│   ├── envs/             # Environment wrappers
│   │   └── mamujoco_wrapper.py
│   ├── training/         # Training and evaluation
│   │   ├── trainer.py    # Training loop
│   │   └── evaluator.py  # Evaluation runner
│   └── utils/            # Utilities
│       ├── config.py     # Config loading
│       └── logger.py     # Training logger
├── scripts/              # Entry points
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│   └── plot_results.py   # Plotting script
└── results/              # Checkpoints and logs
```

## Training Output

During training, you'll see logs like:
```
Ep  100 | R:   -45.2 | R̄10:   -52.3 | Len:  142 | Loss:  0.0234 | Buf:  100 | T:   1.5m

======================================================================
EVAL @ Ep  100 | R:   -48.5 | Len:  138.2 | Best:   -48.5
======================================================================
```

- **R**: Episode reward
- **R̄10**: Running average over last 10 episodes
- **Len**: Episode length
- **Loss**: Training loss
- **Buf**: Replay buffer size
- **T**: Time elapsed

## Results

| Algorithm | Avg Reward | Training Time | Parameters |
|-----------|------------|---------------|------------|
| NQMIX     | TBD        | TBD           | TBD        |
| FACMAC    | TBD        | TBD           | TBD        |

## References

- **NQMIX**: Chen et al. (2020) - *Non-monotonic Value Function Factorization for Deep Multi-Agent Reinforcement Learning*
- **FACMAC**: Peng et al. (2021) - *FACMAC: Factored Multi-Agent Centralised Policy Gradients*
- **MaMuJoCo**: Gymnasium Robotics - *Multi-Agent MuJoCo*

## License

MIT
