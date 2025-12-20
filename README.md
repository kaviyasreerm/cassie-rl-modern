# Cassie Bipedal Locomotion with PPO

CS669 Project - Replication of "RL for Versatile, Dynamic, and Robust Bipedal Locomotion Control" (Li et al., 2024)

**Team**: 
- Kaviya Sree Ravikumar Meenakshi (kr549)
- Yash Chaudhary (ygc2)
- Gnana Midde (gam54)

---

## Overview

We replicated the core PPO training pipeline from the paper using modern tools (Python 3.9+, MuJoCo 3.0+, Stable-Baselines3). Main focus was on reward function engineering for bipedal balance.

**Key Results**:
- V1 @ 100K: 37 steps (policy degradation due to reward bug)
- V2 @ 500K: 183 steps (+395% improvement from fixing reward)
- V3 @ 10M: 107 steps (degraded - hyperparameters need tuning for extended training)

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Test environment
python scripts/test_env.py
```

Requirements: Python 3.9+, ~2GB disk, Windows 10+

### Training

```bash
# Train V2 model (recommended)
python scripts/train_longer.py --timesteps 500000

# Or baseline V1
python scripts/train_baseline.py --timesteps 100000
```

Training time: ~1 hour for V2 on modern laptop (CPU)

### Evaluation

```bash
# With visualization (if MuJoCo viewer works)
python scripts/demo_v2.py --episodes 3

# Without visualization (prints statistics only)
python scripts/demo_v2_headless.py --episodes 5
```

---

## Project Structure

```
cassie_rl_modern/
├── envs/
│   ├── cassie_env.py          # V1 (baseline)
│   ├── cassie_env_v2.py       # V2 (fixed reward)
│   └── cassie_env_full.py     # Full with randomization
├── scripts/
│   ├── train_baseline.py      # V1 training
│   ├── train_longer.py        # V2 training
│   ├── demo_v2.py             # Interactive demo
│   ├── demo_v2_headless.py    # Evaluation without visualization
│   └── test_env.py            # Environment test
├── checkpoints/v2/            # Best models here
├── assets/cassie.xml          # Robot model
└── requirements.txt
```

---

## What We Built

### V1 Baseline (Problem)
- Original reward function with high energy penalty
- Result: Policy degradation (76 steps → 37 steps with more training)
- Robot learned to freeze instead of balance

### V2 Fixed Reward (Solution)
- Added survival bonus (+1.0 per step)
- Reduced energy penalty 10x (0.00001 vs 0.0001)
- Changed height penalty → bonus
- Result: 4.9x improvement (37 → 183 steps)

### V3 Extended Training (Lesson Learned)
- Trained V2 to 10M steps (20x longer)
- Result: Performance degraded to 107 steps
- Lesson: Hyperparameters need tuning for extended training (LR scheduling needed)

---

## Key Findings

1. **Reward engineering > training time**: 10x change in one parameter → 4.9x performance gain
2. **Survival bonus is critical**: Without it, robot has no incentive to stay upright
3. **More training ≠ better**: V3 degraded due to constant LR/entropy (need scheduling)
4. **Visualization helps debugging**: Watching robot revealed V1's "freeze" strategy

---

CS669 - Reinforcement Learning, Fall 2025


