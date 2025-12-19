"""
Baseline training script using Stable-Baselines3 PPO.
This trains a standard MLP policy (no dual-history yet).
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import torch

from envs.cassie_env import CassieEnv


def make_env():
    """Create and wrap a Cassie environment."""
    def _init():
        env = CassieEnv(
            model_path="assets/cassie.xml",
            frame_skip=10,
            randomize_command=True,
            history_length=50
        )
        env = Monitor(env)
        return env
    return _init


def train_baseline(
    total_timesteps=5_000_000,
    n_envs=1,
    save_dir="checkpoints",
    log_dir="logs",
    eval_freq=50000,
    save_freq=100000,
):
    """Train baseline PPO policy."""

    print("="*60)
    print("Cassie RL - Baseline Training")
    print("="*60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Number of environments: {n_envs}")
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("="*60)

    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create vectorized environments
    if n_envs > 1:
        env = SubprocVecEnv([make_env() for _ in range(n_envs)])
        eval_env = DummyVecEnv([make_env()])
    else:
        env = DummyVecEnv([make_env()])
        eval_env = DummyVecEnv([make_env()])

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=save_dir,
        name_prefix="cassie_ppo",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=eval_freq // n_envs,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # Create PPO model
    print("\nCreating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=4096,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256, 128], vf=[256, 256, 128])],
            activation_fn=torch.nn.ReLU,
        ),
        tensorboard_log=log_dir,
        verbose=1,
        device='auto',
    )

    # Train
    print("\nStarting training...\n")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )

        # Save final model
        final_path = os.path.join(save_dir, "cassie_ppo_final")
        model.save(final_path)
        print(f"\n✓ Training complete! Final model saved to: {final_path}")

    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted by user")
        interrupted_path = os.path.join(save_dir, "cassie_ppo_interrupted")
        model.save(interrupted_path)
        print(f"Model saved to: {interrupted_path}")

    # Cleanup
    env.close()
    eval_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=5_000_000, help="Total training timesteps")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save models")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory for tensorboard logs")
    parser.add_argument("--eval_freq", type=int, default=50000, help="Evaluation frequency")
    parser.add_argument("--save_freq", type=int, default=100000, help="Checkpoint save frequency")

    args = parser.parse_args()

    train_baseline(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
    )
