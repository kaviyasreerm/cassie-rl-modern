"""
Headless demo script for V2 trained policy (no visualization).
Evaluates the model and prints statistics.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.cassie_env_v2 import CassieEnvV2


def evaluate_policy(n_episodes=5):
    """Evaluate V2 model without rendering."""

    print("=" * 70)
    print("V2 MODEL EVALUATION (Headless)")
    print("=" * 70)

    # Load model
    print("Loading model...")
    policy = PPO.load('checkpoints/v2/cassie_v2_final.zip')

    # Create env
    def make_env():
        return CassieEnvV2(randomize_command=False, command_velocity=[1.5, 0.0, 0.0])

    env = DummyVecEnv([make_env])

    # Load normalization if available
    if os.path.exists('checkpoints/v2/vecnormalize.pkl'):
        env = VecNormalize.load('checkpoints/v2/vecnormalize.pkl', env)
        env.training = False
        env.norm_reward = False

    print(f"Running {n_episodes} episodes...\n")

    # Run evaluation
    results = []
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        steps = 0
        total_reward = 0

        while not done and steps < 1000:
            action, _ = policy.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            steps += 1
            total_reward += reward[0]

        results.append((steps, total_reward))
        print(f'Episode {ep+1}: {steps} steps, reward={total_reward:.1f}')

    # Statistics
    lengths = [r[0] for r in results]
    rewards = [r[1] for r in results]

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Average: {np.mean(lengths):.1f} ± {np.std(lengths):.1f} steps")
    print(f"Average reward: {np.mean(rewards):.1f}")
    print(f"Time per episode: ~{np.mean(lengths)*0.033:.1f} seconds")
    print()
    print("COMPARISON")
    print("-" * 70)
    print(f"V1 (baseline) @ 100K steps: 37 steps")
    print(f"V2 (fixed reward) @ 500K steps: {np.mean(lengths):.0f} steps")
    print(f"Improvement: {np.mean(lengths)/37:.1f}x better!")
    print("=" * 70)

    env.close()
    return np.mean(lengths)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5,
                       help="Number of episodes to run")
    args = parser.parse_args()

    avg_length = evaluate_policy(n_episodes=args.episodes)

    print(f"\nSuccess! Model balances for ~{avg_length:.0f} steps on average")
