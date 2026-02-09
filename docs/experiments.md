# Morality Gym Experiments

This directory contains experiments for the morality-gym-tabular framework. Morality Gym provides integration with multiple reinforcement learning frameworks to train and evaluate agents in trolley problem environments.

## Available Frameworks

Morality Gym currently supports running experiments with the following reinforcement learning frameworks:

### StableBaselines3

[StableBaselines3](https://github.com/DLR-RM/stable-baselines3) is a set of reliable implementations of reinforcement learning algorithms in PyTorch. The `experiments/baselines/` directory contains code for training and evaluating agents using StableBaselines3.

Key features:
- Simple interface for training standard RL algorithms (PPO, SAC, A2C, etc.)
- Support for MlpPolicy and CnnPolicy
- Easy experiment configuration via JSON files
- Integrated with Weights & Biases for experiment tracking

### OmniSafe

[OmniSafe](https://github.com/PKU-Alignment/omnisafe) is a comprehensive infrastructure framework for safe reinforcement learning research developed by PKU-Alignment. The `experiments/omnisafe/` directory contains code for training and evaluating agents with safety constraints.

Key features:
- Implementation of over 20 safe RL algorithms across different categories:
  - On-policy algorithms (PPO-Lagrangian, TRPO-Lagrangian, CPO, FOCOPS, etc.)
  - Off-policy algorithms (DDPG-Lagrangian, TD3-Lagrangian, SAC-Lagrangian, etc.)
  - Model-based algorithms (SafeLOOP, CCEPETS, etc.)
  - Offline algorithms (BCQL-Lagrangian, etc.)
- Support for cost constraints to enforce safety requirements
- Integration with morality metrics for evaluation
- Advanced logging and visualization tools
- Command-line interface for training and evaluation

## Running Experiments

### StableBaselines3 Experiments

To run an experiment using StableBaselines3:

```bash
python -m experiments.baselines.train \
  --config experiments/baselines/configs/SwitchStandard-v1.json \
  --seed 42
```

Configuration files in `experiments/baselines/configs/` define the environment, algorithm, and hyperparameters for different experimental setups.

### OmniSafe Experiments

To run an experiment using OmniSafe:

```bash
python -m experiments.omnisafe.train \
  --task "MoralityGym/Trolley-SwitchStandard-v0" \
  --morality-tree-id "Trolley-Common-StandardUtilitarian-v0" \
  --algo "ppo_lagrangian" \
  --epoch 100 \
  --seed 42
```

## Experiment Configuration

### BaselinesConfig

The configuration files for StableBaselines experiments are located in `experiments/baselines/configs/`. These JSON files include:

- Environment specification (environment ID, variant)
- Algorithm selection (PPO, A2C, etc.)
- Network architecture (MLP layers, activation functions)
- Training parameters (learning rate, batch size, etc.)
- Evaluation settings

Example configuration snippets:

```json
{
  "env": {
    "id": "MoralityGym/Trolley-SwitchStandard-v1",
    "morality_tree_id": "Trolley-Common-Utilitarian-UtilityHarm-v0"
  },
  "algorithm": "ppo",
  "hyperparameters": {
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "batch_size": 64
  }
}
```

### OmniSafe Configuration

OmniSafe experiments can be configured via command-line arguments or by modifying the configuration class in the training script:

```python
class SafeRLConfig:
    # Environment parameters
    task = "MoralityGym/Trolley-SwitchStandard-v0"
    morality_tree_id = "Trolley-Common-StandardUtilitarian-v0"
    cost_limit = 0.1
    
    # Algorithm parameters
    algo = "ppo_lagrangian"
    epochs = 100
    seed = 1
    
    # Training parameters
    steps_per_epoch = 1000
    update_iters = 10
```

OmniSafe also supports a convenient CLI interface:

```bash
# First install OmniSafe
pip install omnisafe

# Train a safe RL algorithm on our environment
omnisafe train --algo PPO-Lag --env-id MoralityGym/Trolley-SwitchStandard-v0 --total-steps 1000000
```

## Example Experiment Scenarios

Morality Gym provides configurations for various trolley problem scenarios:

- **Switch Variants**: SwitchStandard-v1, Switch3-v1, SwitchSelfSacrifice-v1
- **Push Variants**: PushStandard-v1, PushEasy-v1, PushSelfSacrifice-v1
- **Combination Variants**: PushOrSwitch-v1, PushOrSwitchSelfSacrifice-v1

Each scenario can be evaluated against different moral frameworks using various reinforcement learning algorithms to investigate agent behavior in moral dilemmas.

## Available OmniSafe Algorithms

When using OmniSafe, you can leverage a wide range of safe RL algorithms, including:

### On-Policy Algorithms
- Primal-Dual: TRPOLag, PPOLag, PDO, RCPO
- Convex Optimization: CPO, PCPO, FOCOPS, CUP
- Penalty Function: IPO, P3O
- Others: TRPOPID, CPPOPID, OnCRPO

### Off-Policy Algorithms
- Primal-Dual: DDPGLag, TD3Lag, SACLag, DDPGPID, TD3PID, SACPID

### Model-Based Algorithms
- Online Planning: SafeLOOP, CCEPETS, RCEPETS
- Pessimistic Estimate: CAPPETS

### Offline Algorithms
- Q-Learning Based: BCQLag, C-CRR
- DICE Based: COptDICE

## Evaluation

Both frameworks provide tools for evaluating trained agents:

```bash
# Evaluate a StableBaselines agent
python -m experiments.baselines.evaluate \
  --model_path path/to/saved/model \
  --episodes 100

# Evaluate an OmniSafe agent
python -m experiments.omnisafe.evaluate \
  --model_path path/to/saved/model \
  --morality-tree-id "Trolley-Common-Utilitarian-UtilityHarm-v0" \
  --episodes 100

# Or use OmniSafe's CLI
omnisafe eval path/to/saved/model --num-episode 100
```

## Notes

- The default logger uses Weights & Biases for experiment tracking
- The agent's policy is specially designed to handle the nested dictionary observations from morality environments
- For debugging, you can enable verbose logging with the `--verbose` flag
- Results are saved in the `runs/` directory by default 