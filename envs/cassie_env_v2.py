"""
V2 environment - fixed reward function from v1

Main fixes:
- Added survival bonus (robot learns staying alive is good)
- Reduced energy penalty 10x
- Height bonus instead of penalty
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
from collections import deque
import os


class CassieEnvV2(gym.Env):
    """Cassie bipedal robot with improved reward"""

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

        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(__file__), "..", model_path)

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = frame_skip
        self.dt = self.model.opt.timestep * frame_skip
        self.render_mode = render_mode

        self.command_velocity = np.array(command_velocity, dtype=np.float32)
        self.randomize_command = randomize_command

        self.history_length = history_length
        self.obs_history = deque(maxlen=history_length)
        self.action_history = deque(maxlen=history_length)

        self.actuated_joint_ids = self._get_actuated_joint_ids()
        self.n_joints = len(self.actuated_joint_ids)

        # PD gains - tuned by trial and error
        self.kp = np.array([400, 200, 200, 500, 20, 400, 200, 200, 500, 20])
        self.kd = np.array([4, 4, 10, 20, 4, 4, 4, 10, 20, 4])

        # Actions are target joint positions in [-1, 1]
        self.action_space = spaces.Box(
            low=-np.ones(self.n_joints, dtype=np.float32),
            high=np.ones(self.n_joints, dtype=np.float32),
            dtype=np.float32
        )

        # Observation: joint pos, vel, torso quat, angvel, command, prev action
        obs_dim = self.n_joints + self.n_joints + 4 + 3 + 3 + self.n_joints

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        if self.render_mode == "human":
            from mujoco import viewer
            self.viewer = viewer.launch_passive(self.model, self.data)
        else:
            self.viewer = None

        self.steps = 0
        self.max_episode_steps = 1000
        self.prev_action = np.zeros(self.n_joints)

    def _get_actuated_joint_ids(self):
        joint_ids = []
        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            joint_ids.append(int(joint_id))
        return joint_ids

    def _get_obs(self):
        qpos = self.data.qpos[self.actuated_joint_ids]
        qvel = self.data.qvel[self.actuated_joint_ids]

        # Get torso state if available
        if self.model.nq >= 7:
            torso_quat = self.data.qpos[3:7]
            torso_angvel = self.data.qvel[3:6]
        else:
            torso_quat = np.array([1, 0, 0, 0])
            torso_angvel = np.zeros(3)

        obs = np.concatenate([
            qpos, qvel, torso_quat, torso_angvel,
            self.command_velocity, self.prev_action,
        ]).astype(np.float32)

        return obs

    def _get_reward(self):
        """
        Reward function v2 - this actually works!

        The v1 reward had energy penalty way too high, robot just learned to freeze.
        Fixed by adding survival bonus and making energy penalty 10x smaller.
        """
        if self.model.nq >= 7:
            base_vel = self.data.qvel[0:3]
            base_angvel = self.data.qvel[3:6]
            height = self.data.qpos[2]
            quat = self.data.qpos[3:7]
        else:
            base_vel = np.zeros(3)
            base_angvel = np.zeros(3)
            height = 1.0
            quat = np.array([1, 0, 0, 0])

        # Survival bonus - most important change!
        survival_bonus = 1.0

        # Velocity tracking (exponential)
        vel_x_reward = np.exp(-2.0 * (base_vel[0] - self.command_velocity[0])**2)
        vel_y_reward = np.exp(-2.0 * (base_vel[1] - self.command_velocity[1])**2)
        yaw_reward = np.exp(-2.0 * (base_angvel[2] - self.command_velocity[2])**2)
        vel_reward = vel_x_reward + 0.5 * vel_y_reward + 0.3 * yaw_reward

        # Forward bonus - encourages any forward motion
        forward_bonus = 0.5 * max(0, base_vel[0])

        # Height bonus (not penalty!) - reward tall posture
        height_bonus = 2.0 * max(0, (height - 0.5) / 0.5)

        # Small orientation penalty
        orientation_penalty = 0.5 * (1.0 - abs(quat[0]))**2

        # Energy penalty - REDUCED 10x from v1!
        torques = self.data.ctrl
        energy_penalty = 0.00001 * np.sum(torques**2)

        reward = (survival_bonus + vel_reward + forward_bonus + height_bonus
                  - orientation_penalty - energy_penalty)

        return reward

    def _is_terminated(self):
        """Check if robot fell"""
        if self.model.nq >= 7:
            height = self.data.qpos[2]
            quat = self.data.qpos[3:7]

            if height < 0.4:  # too low
                return True
            if abs(quat[0]) < 0.5:  # tilted too much
                return True

        return False

    def _is_truncated(self):
        return self.steps >= self.max_episode_steps

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Randomize command each episode
        if self.randomize_command:
            self.command_velocity = np.array([
                np.random.uniform(0.5, 2.0),  # forward
                np.random.uniform(-0.3, 0.3),  # lateral
                np.random.uniform(-0.5, 0.5),  # yaw
            ], dtype=np.float32)

        self.obs_history.clear()
        self.action_history.clear()
        self.prev_action = np.zeros(self.n_joints)
        self.steps = 0

        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {"command": self.command_velocity.copy()}

        return obs, info

    def step(self, action):
        # Clip and scale actions
        action = np.clip(action, -1.0, 1.0)
        joint_targets = action * 0.5

        # PD control to compute torques
        qpos = self.data.qpos[self.actuated_joint_ids]
        qvel = self.data.qvel[self.actuated_joint_ids]
        torques = self.kp * (joint_targets - qpos) - self.kd * qvel

        # Torque limits per joint
        torque_limits = np.array([80, 60, 80, 190, 45] * 2)
        torques = np.clip(torques, -torque_limits, torque_limits)
        self.data.ctrl[:len(torques)] = torques

        # Step physics
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.steps += 1
        self.prev_action = action

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_terminated()
        truncated = self._is_truncated()

        info = {
            "command": self.command_velocity.copy(),
            "velocity": self.data.qvel[0:3].copy() if self.model.nq >= 3 else np.zeros(3),
            "height": self.data.qpos[2] if self.model.nq >= 3 else 1.0,
        }

        if self.render_mode == "human" and self.viewer is not None:
            self.viewer.sync()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data)
            return renderer.render()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
