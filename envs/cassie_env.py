"""
Cassie Bipedal Robot Environment for Gymnasium

This environment wraps the Cassie robot in MuJoCo for reinforcement learning.
Based on the paper: "Reinforcement Learning for Versatile, Dynamic, and Robust
Bipedal Locomotion Control" (Li et al., 2024)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from collections import deque
import os


class CassieEnv(gym.Env):
    """
    Cassie bipedal robot locomotion environment.

    Observation Space:
        - Joint positions (10): actuated joint angles
        - Joint velocities (10): actuated joint velocities
        - Torso orientation (4): quaternion
        - Torso angular velocity (3)
        - Command (3): [forward_vel, lateral_vel, yaw_rate]
        - Previous action (10)
        - (Optional) Long-term history for dual-history controller

    Action Space:
        - 10 continuous actions: target positions for PD controllers on each actuated joint

    Reward:
        - Forward velocity tracking
        - Lateral velocity tracking
        - Yaw rate tracking
        - Energy penalty (torque squared)
        - Orientation penalty (staying upright)
        - Height penalty (maintaining standing height)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        model_path="assets/cassie.xml",
        frame_skip=10,
        render_mode=None,
        command_velocity=[1.0, 0.0, 0.0],
        randomize_command=True,
        history_length=50,
    ):
        super().__init__()

        # Get absolute path
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(__file__), "..", model_path)

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Environment parameters
        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * frame_skip
        self.render_mode = render_mode

        # Command parameters
        self.command_velocity = np.array(command_velocity, dtype=np.float32)
        self.randomize_command = randomize_command

        # History tracking (for dual-history architecture)
        self.history_length = history_length
        self.obs_history = deque(maxlen=history_length)
        self.action_history = deque(maxlen=history_length)

        # Cassie-specific parameters
        self.actuated_joint_ids = self._get_actuated_joint_ids()
        self.n_joints = len(self.actuated_joint_ids)

        # PD controller gains (from paper defaults)
        self.kp = np.array([400, 200, 200, 500, 20, 400, 200, 200, 500, 20])
        self.kd = np.array([4, 4, 10, 20, 4, 4, 4, 10, 20, 4])

        # Action and observation spaces
        # Action: target joint positions
        self.action_space = spaces.Box(
            low=-np.ones(self.n_joints, dtype=np.float32),
            high=np.ones(self.n_joints, dtype=np.float32),
            dtype=np.float32
        )

        # Observation: joints + velocities + orientation + ang_vel + command + prev_action
        obs_dim = (
            self.n_joints  # joint positions
            + self.n_joints  # joint velocities
            + 4  # torso quaternion
            + 3  # torso angular velocity
            + 3  # command [vx, vy, yaw_rate]
            + self.n_joints  # previous action
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Rendering
        if self.render_mode == "human":
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None

        # Episode tracking
        self.steps = 0
        self.max_episode_steps = 1000
        self.prev_action = np.zeros(self.n_joints)

    def _get_actuated_joint_ids(self):
        """Get IDs of actuated joints controlled by actuators."""
        # Get joint IDs from actuator transmission
        # Each actuator controls one joint - read from model.actuator_trnid
        joint_ids = []
        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            joint_ids.append(int(joint_id))

        return joint_ids

    def _get_obs(self):
        """Get current observation."""
        # Joint positions and velocities
        qpos = self.data.qpos[self.actuated_joint_ids]
        qvel = self.data.qvel[self.actuated_joint_ids]

        # Torso orientation (quaternion) and angular velocity
        # Assuming floating base (first 7 qpos: position + quaternion)
        if self.model.nq >= 7:
            torso_quat = self.data.qpos[3:7]  # Quaternion
            torso_angvel = self.data.qvel[3:6]  # Angular velocity
        else:
            torso_quat = np.array([1, 0, 0, 0])  # Default orientation
            torso_angvel = np.zeros(3)

        # Concatenate observation
        obs = np.concatenate([
            qpos,
            qvel,
            torso_quat,
            torso_angvel,
            self.command_velocity,
            self.prev_action,
        ]).astype(np.float32)

        return obs

    def _get_reward(self):
        """Calculate reward based on paper."""
        # Get robot velocities
        if self.model.nq >= 7:
            base_vel = self.data.qvel[0:3]  # Linear velocity
            base_angvel = self.data.qvel[3:6]  # Angular velocity
        else:
            base_vel = np.zeros(3)
            base_angvel = np.zeros(3)

        # Velocity tracking reward
        vel_reward = 0.0
        vel_reward += 1.0 * np.exp(-2.0 * (base_vel[0] - self.command_velocity[0])**2)  # Forward
        vel_reward += 0.5 * np.exp(-2.0 * (base_vel[1] - self.command_velocity[1])**2)  # Lateral
        vel_reward += 0.3 * np.exp(-2.0 * (base_angvel[2] - self.command_velocity[2])**2)  # Yaw

        # Energy penalty (torque squared)
        torques = self.data.ctrl
        energy_penalty = 0.0001 * np.sum(torques**2)

        # Orientation penalty (keep upright)
        if self.model.nq >= 7:
            quat = self.data.qpos[3:7]
            # Quaternion for upright: [1, 0, 0, 0] or close
            # We want w (quat[0]) close to 1
            orientation_penalty = 1.0 * (1.0 - quat[0])**2
        else:
            orientation_penalty = 0.0

        # Height penalty (maintain standing height ~0.95m)
        if self.model.nq >= 3:
            height = self.data.qpos[2]
            target_height = 0.95
            height_penalty = 0.5 * (height - target_height)**2
        else:
            height_penalty = 0.0

        # Total reward
        reward = vel_reward - energy_penalty - orientation_penalty - height_penalty

        return reward

    def _is_terminated(self):
        """Check if episode should terminate."""
        # Terminate if robot falls (height too low or extreme orientation)
        if self.model.nq >= 7:
            height = self.data.qpos[2]
            quat = self.data.qpos[3:7]

            # Fall conditions
            if height < 0.4:  # Too low
                return True
            if abs(quat[0]) < 0.5:  # Extreme orientation
                return True

        return False

    def _is_truncated(self):
        """Check if episode should be truncated (time limit)."""
        return self.steps >= self.max_episode_steps

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset MuJoCo simulation - this sets the model's default stable pose
        mujoco.mj_resetData(self.model, self.data)

        # DON'T overwrite the default pose! The model's default qpos is already
        # a stable standing configuration. Previous code was replacing it with
        # unstable joint angles causing immediate tumbling.

        # Randomize command if enabled
        if self.randomize_command:
            self.command_velocity = np.array([
                np.random.uniform(-0.5, 3.0),  # Forward velocity
                np.random.uniform(-0.5, 0.5),  # Lateral velocity
                np.random.uniform(-1.0, 1.0),  # Yaw rate
            ], dtype=np.float32)

        # Reset history
        self.obs_history.clear()
        self.action_history.clear()
        self.prev_action = np.zeros(self.n_joints)
        self.steps = 0

        # Forward simulation to stabilize
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {"command": self.command_velocity.copy()}

        return obs, info

    def step(self, action):
        """Execute one environment step."""
        # Clip action
        action = np.clip(action, -1.0, 1.0)

        # Convert action to joint targets (scale from [-1, 1] to joint limits)
        # For now, we'll use a simple scaling; refine based on actual joint limits
        joint_targets = action * 0.5  # Scale factor

        # PD control to compute torques
        qpos = self.data.qpos[self.actuated_joint_ids]
        qvel = self.data.qvel[self.actuated_joint_ids]
        torques = self.kp * (joint_targets - qpos) - self.kd * qvel

        # Apply torques (clip to limits)
        torque_limits = np.array([80, 60, 80, 190, 45] * 2)  # From paper
        torques = np.clip(torques, -torque_limits, torque_limits)
        self.data.ctrl[:len(torques)] = torques

        # Step simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # Update tracking
        self.steps += 1
        self.prev_action = action

        # Get new observation, reward, termination
        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        info = {
            "command": self.command_velocity.copy(),
            "velocity": self.data.qvel[0:3].copy() if self.model.nq >= 3 else np.zeros(3),
        }

        # Render if needed
        if self.render_mode == "human" and self.viewer is not None:
            self.viewer.sync()

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            # Render to numpy array
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data)
            return renderer.render()
        elif self.render_mode == "human":
            # Already handled in step() with viewer
            pass

    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
