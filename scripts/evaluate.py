"""
Evaluate a trained Cassie policy and optionally record videos.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from stable_baselines3 import PPO
from envs.cassie_env import CassieEnv
import time


def evaluate_policy(
    model_path,
    n_episodes=10,
    render=True,
    save_video=False,
    video_path="videos"
):
    """Evaluate a trained policy."""

    print("="*60)
    print("Cassie RL - Policy Evaluation")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Render: {render}")
    print("="*60)

    # Load model
    print("\nLoading model...")
    model = PPO.load(model_path)

    # Create environment
    render_mode = "human" if render else None
    env = CassieEnv(
        model_path="assets/cassie.xml",
        frame_skip=10,
        randomize_command=False,  # Fixed command for evaluation
        command_velocity=[1.5, 0.0, 0.0],  # Walk forward at 1.5 m/s
        render_mode=render_mode
    )

    # Evaluate
    episode_rewards = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False

        print(f"\n--- Episode {ep+1}/{n_episodes} ---")
        print(f"Command: {info['command']}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            done = terminated or truncated

            if render:
                time.sleep(0.01)  # Slow down for visualization

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)

        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Steps: {steps}")
        if terminated:
            print("  Status: Terminated (fall)")
        else:
            print("  Status: Truncated (time limit)")

    # Summary statistics
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    print(f"Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print("="*60)

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Render evaluation")
    parser.add_argument("--no-render", dest="render", action="store_false")
    parser.set_defaults(render=False)

    args = parser.parse_args()

    evaluate_policy(
        model_path=args.model,
        n_episodes=args.episodes,
        render=args.render,
    )
