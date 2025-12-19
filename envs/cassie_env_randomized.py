"""
Cassie Environment with Domain Randomization for Milestone 4

This adds domain randomization on top of the base CassieEnv:
- Mass randomization (pelvis, thighs, shanks)
- Friction randomization (ground contact)
- Motor damping randomization (actuators)

Randomization range is configurable (±20% for baseline, ±50% for high randomization)
"""

import numpy as np
import mujoco
from envs.cassie_env import CassieEnv


class CassieEnvRandomized(CassieEnv):
    """
    Cassie environment with domain randomization.

    Additional Parameters:
        randomization_range (float): Randomization range as fraction (0.2 = ±20%, 0.5 = ±50%)
    """

    def __init__(
        self,
        model_path="assets/cassie.xml",
        frame_skip=10,
        render_mode=None,
        command_velocity=[1.0, 0.0, 0.0],
        randomize_command=True,
        history_length=50,
        randomization_range=0.2,  # Default ±20%
    ):
        # Store randomization parameters
        self.randomization_range = randomization_range

        # Call parent constructor
        super().__init__(
            model_path=model_path,
            frame_skip=frame_skip,
            render_mode=render_mode,
            command_velocity=command_velocity,
            randomize_command=randomize_command,
            history_length=history_length,
        )

        # Store original model parameters for resetting
        self.original_body_mass = self.model.body_mass.copy()
        self.original_geom_friction = self.model.geom_friction.copy()
        self.original_dof_damping = self.model.dof_damping.copy()

    def _randomize_dynamics(self):
        """
        Randomize physics parameters within specified range.
        Called at the start of each episode.
        """
        low = 1.0 - self.randomization_range
        high = 1.0 + self.randomization_range

        # 1. Mass randomization (for main body parts)
        # Randomize pelvis, thighs, and shanks
        # Body indices (adjust based on cassie.xml structure):
        # Typically: 0=world, 1=pelvis, 2-5=leg bodies
        for body_id in range(1, min(10, self.model.nbody)):  # Randomize first 10 bodies
            mass_multiplier = np.random.uniform(low, high)
            self.model.body_mass[body_id] = self.original_body_mass[body_id] * mass_multiplier

        # 2. Friction randomization (for ground contact)
        # Randomize friction coefficients for all geometries
        for geom_id in range(self.model.ngeom):
            friction_multiplier = np.random.uniform(low, high)
            # Friction has 3 components: sliding, torsional, rolling
            # We randomize sliding friction (first component)
            self.model.geom_friction[geom_id, 0] = (
                self.original_geom_friction[geom_id, 0] * friction_multiplier
            )

        # 3. Motor damping randomization (for actuators)
        # Randomize joint damping
        for dof_id in range(self.model.nv):
            damping_multiplier = np.random.uniform(low, high)
            self.model.dof_damping[dof_id] = (
                self.original_dof_damping[dof_id] * damping_multiplier
            )

    def reset(self, seed=None, options=None):
        """
        Reset environment and randomize dynamics.
        """
        # Call parent reset
        obs, info = super().reset(seed=seed, options=options)

        # Randomize dynamics for this episode
        self._randomize_dynamics()

        # Re-forward simulation after randomization
        mujoco.mj_forward(self.model, self.data)

        # Add randomization info
        info['randomization_range'] = self.randomization_range

        return obs, info
