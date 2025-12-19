"""
Demo script for V2 trained policy (with improved reward function).
Shows robot balancing for ~183 steps (~6 seconds)!
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from envs.cassie_env_v2 import CassieEnvV2
import time


def demo_v2_policy(model_path="checkpoints/v2/cassie_v2_final.zip",
                   vecnorm_path="checkpoints/v2/vecnormalize.pkl",
                   n_episodes=3):
    """Demonstrate the V2 trained policy."""

    print("=" * 70)
    print("CASSIE RL - V2 POLICY DEMONSTRATION")
    print("=" * 70)
    print(f"Model: {model_path}")
    print("Expected: ~183 steps per episode (~6 seconds)")
    print("=" * 70)

    # Check files exist
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return

    # Load model
    print("\nLoading policy...")
    policy = PPO.load(model_path)
    print("Policy loaded!")

    # Create environment
    print("Creating environment...")

    def make_env():
        return CassieEnvV2(randomize_command=False, command_velocity=[1.5, 0.0, 0.0])

    env = DummyVecEnv([make_env])

    # Load VecNormalize if available
    if os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize from {vecnorm_path}...")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        print("VecNormalize loaded!")

    # Get MuJoCo model for visualization
    mj_model = mujoco.MjModel.from_xml_path("assets/cassie.xml")
    mj_data = mujoco.MjData(mj_model)

    # Set visualization extent for better default view
    mj_model.stat.extent = 80

    # Get actuated joint IDs
    actuated_joint_ids = []
    for i in range(mj_model.nu):
        joint_id = mj_model.actuator_trnid[i, 0]
        actuated_joint_ids.append(int(joint_id))

    # PD gains
    kp = np.array([400, 200, 200, 500, 20, 400, 200, 200, 500, 20])
    kd = np.array([4, 4, 10, 20, 4, 4, 4, 10, 20, 4])

    all_lengths = []
    all_rewards = []

    print("\n" + "=" * 70)
    print("CONTROLS: Drag to rotate, scroll to zoom, close window to exit")
    print("=" * 70 + "\n")

    # Launch viewer
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15
        viewer.cam.distance = 5.0
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

        for ep in range(n_episodes):
            if not viewer.is_running():
                break

            print(f"\n{'='*70}")
            print(f"Episode {ep + 1}/{n_episodes}")
            print(f"{'='*70}")

            # Reset
            obs = env.reset()
            mujoco.mj_resetData(mj_model, mj_data)
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            episode_reward = 0
            steps = 0
            done = False

            while not done and viewer.is_running():
                # Get action from policy
                action, _ = policy.predict(obs, deterministic=True)

                # Step environment (for reward calculation)
                obs, reward, done, info = env.step(action)
                episode_reward += reward[0]

                # Also step MuJoCo for visualization
                target_positions = action[0] * 0.5
                current_positions = mj_data.qpos[actuated_joint_ids]
                current_velocities = mj_data.qvel[actuated_joint_ids]

                torques = kp * (target_positions - current_positions) - kd * current_velocities
                torques = np.clip(torques, -100, 100)
                mj_data.ctrl[:] = torques

                for _ in range(10):
                    mujoco.mj_step(mj_model, mj_data)

                viewer.sync()
                steps += 1

                # Progress update
                if steps % 25 == 0:
                    height = mj_data.qpos[2]
                    print(f"  Step {steps}: height={height:.2f}m")

                # Check termination
                height = mj_data.qpos[2]
                quat = mj_data.qpos[3:7]
                if height < 0.4 or abs(quat[0]) < 0.5:
                    done = True

                if steps >= 1000:
                    done = True
                    print("  Reached max steps!")

            print(f"\n  Episode Summary:")
            print(f"  - Steps: {steps}")
            print(f"  - Reward: {episode_reward:.1f}")
            print(f"  - Time: {steps * 0.033:.1f} seconds")

            all_lengths.append(steps)
            all_rewards.append(episode_reward)

            if viewer.is_running() and ep < n_episodes - 1:
                print("\n  (Pausing 2 seconds...)")
                time.sleep(2)

        # Keep viewer open
        print("\n" + "="*70)
        print("Demo complete! Close window to exit.")
        print("="*70)

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.1)

    # Summary
    if all_lengths:
        print("\n" + "="*70)
        print("FINAL STATISTICS")
        print("="*70)
        print(f"Episodes: {len(all_lengths)}")
        print(f"Avg Length: {np.mean(all_lengths):.1f} +/- {np.std(all_lengths):.1f} steps")
        print(f"Avg Reward: {np.mean(all_rewards):.1f}")
        print(f"Best Episode: {max(all_lengths)} steps")
        print("="*70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/v2/cassie_v2_final.zip")
    parser.add_argument("--vecnorm", default="checkpoints/v2/vecnormalize.pkl")
    parser.add_argument("--episodes", type=int, default=3)

    args = parser.parse_args()

    demo_v2_policy(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        n_episodes=args.episodes
    )
