"""
Test script to verify Cassie environment is working correctly.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from envs.cassie_env import CassieEnv


def test_basic_functionality():
    """Test basic environment functionality."""
    print("="*60)
    print("Testing Cassie Environment")
    print("="*60)

    # Create environment
    print("\n1. Creating environment...")
    try:
        env = CassieEnv(render_mode=None)
        print("   [OK] Environment created successfully")
    except Exception as e:
        print(f"   [FAIL] Failed to create environment: {e}")
        return False

    # Test reset
    print("\n2. Testing reset...")
    try:
        obs, info = env.reset()
        print(f"   [OK] Reset successful")
        print(f"   - Observation shape: {obs.shape}")
        print(f"   - Observation space: {env.observation_space}")
        print(f"   - Action space: {env.action_space}")
        print(f"   - Command: {info['command']}")
    except Exception as e:
        print(f"   [FAIL] Reset failed: {e}")
        return False

    # Test random actions
    print("\n3. Testing random actions...")
    try:
        n_steps = 100
        total_reward = 0

        for step in range(n_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                print(f"   Episode ended at step {step}")
                break

        print(f"   [OK] Ran {step+1} steps successfully")
        print(f"   - Total reward: {total_reward:.2f}")
        print(f"   - Average reward: {total_reward/(step+1):.4f}")
        print(f"   - Final observation shape: {obs.shape}")

    except Exception as e:
        print(f"   [FAIL] Step execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test episode rollout
    print("\n4. Testing full episode rollout...")
    try:
        obs, info = env.reset()
        episode_reward = 0
        steps = 0

        while True:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

            if terminated:
                print(f"   Episode terminated (fall detected)")
                break
            if truncated:
                print(f"   Episode truncated (time limit)")
                break

        print(f"   [OK] Episode completed")
        print(f"   - Total steps: {steps}")
        print(f"   - Episode reward: {episode_reward:.2f}")

    except Exception as e:
        print(f"   [FAIL] Episode rollout failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Clean up
    env.close()

    print("\n" + "="*60)
    print("[OK] All tests passed!")
    print("="*60)
    return True


def test_observation_components():
    """Test that observation components are correct."""
    print("\n5. Testing observation components...")

    env = CassieEnv()
    obs, info = env.reset()

    print(f"   Observation breakdown:")

    # Based on the environment implementation
    n_joints = env.n_joints
    idx = 0

    print(f"   - Joint positions: {obs[idx:idx+n_joints]}")
    idx += n_joints

    print(f"   - Joint velocities: {obs[idx:idx+n_joints]}")
    idx += n_joints

    print(f"   - Torso quaternion: {obs[idx:idx+4]}")
    idx += 4

    print(f"   - Torso angular velocity: {obs[idx:idx+3]}")
    idx += 3

    print(f"   - Command velocity: {obs[idx:idx+3]}")
    idx += 3

    print(f"   - Previous action: {obs[idx:idx+n_joints]}")

    env.close()


if __name__ == "__main__":
    success = test_basic_functionality()

    if success:
        test_observation_components()
    else:
        print("\n[WARN] Basic tests failed. Please check the environment implementation.")
        sys.exit(1)
