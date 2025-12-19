"""
Training script for V2 environment (fixed reward)
Run for 500K-1M+ steps
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import torch
import argparse
import time

from envs.cassie_env_v2 import CassieEnvV2


def make_env():
    def _init():
        env = CassieEnvV2(
            model_path="assets/cassie.xml",
            frame_skip=10,
            randomize_command=True,
            history_length=50
        )
        env = Monitor(env)
        return env
    return _init


def train_longer(
    total_timesteps=500_000,
    n_envs=1,
    save_dir="checkpoints/v2",
    log_dir="logs/v2",
    eval_freq=25_000,
    save_freq=50_000,
    resume_from=None,
):
    print("=" * 70)
    print("Training Cassie with V2 Reward Function")
    print("=" * 70)
    print(f"Timesteps: {total_timesteps:,}")
    print(f"Environments: {n_envs}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("=" * 70)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Setup environments
    print("\nCreating environments...")
    if n_envs > 1:
        env = SubprocVecEnv([make_env() for _ in range(n_envs)])
        eval_env = DummyVecEnv([make_env()])
    else:
        env = DummyVecEnv([make_env()])
        eval_env = DummyVecEnv([make_env()])

    # VecNormalize really helps training stability
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Callbacks for saving checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1),
        save_path=save_dir,
        name_prefix="cassie_v2",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # Create or resume model
    if resume_from and os.path.exists(resume_from):
        print(f"\nResuming from: {resume_from}")
        model = PPO.load(resume_from, env=env)
    else:
        print("\nCreating new PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=128,  # reduced for CPU training
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                activation_fn=torch.nn.ReLU,
            ),
            tensorboard_log=log_dir,
            verbose=1,
            device='cpu',
        )

    # Estimate time
    steps_per_second = 150  # rough estimate on CPU
    estimated_minutes = total_timesteps / steps_per_second / 60
    print(f"\nEstimated time: ~{estimated_minutes:.0f} minutes")
    print(f"Started: {time.strftime('%H:%M:%S')}")

    # Train!
    print("\nTraining...\n")
    start_time = time.time()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )

        # Save final
        final_path = os.path.join(save_dir, "cassie_v2_final")
        model.save(final_path)
        env.save(os.path.join(save_dir, "vecnormalize.pkl"))

        elapsed = time.time() - start_time
        print(f"\n{'=' * 70}")
        print(f"Done! Took {elapsed / 60:.1f} minutes")
        print(f"Saved to: {final_path}.zip")
        print(f"{'=' * 70}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user - saving...")
        interrupted_path = os.path.join(save_dir, "cassie_v2_interrupted")
        model.save(interrupted_path)
        env.save(os.path.join(save_dir, "vecnormalize_interrupted.pkl"))
        print(f"Saved to: {interrupted_path}.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000,
                       help="Total timesteps (default: 500K)")
    parser.add_argument("--n_envs", type=int, default=1,
                       help="Parallel environments")
    parser.add_argument("--save_dir", type=str, default="checkpoints/v2")
    parser.add_argument("--log_dir", type=str, default="logs/v2")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from checkpoint")

    args = parser.parse_args()

    train_longer(
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        resume_from=args.resume,
    )
