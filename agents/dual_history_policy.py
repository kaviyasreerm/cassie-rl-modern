"""
Dual-History Policy Network
Replicates the architecture from the paper:
- Short-term history: 50 timesteps (1.1 sec)
- Long-term history: 200 timesteps (4.4 sec)
- Both processed by 1D CNNs
- Combined and fed to MLP for action output

Reference: "Reinforcement Learning for Versatile, Dynamic, and Robust Bipedal Locomotion Control"
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class DualHistoryCNN(BaseFeaturesExtractor):
    """
    Custom feature extractor with dual-history architecture.

    Architecture (from paper):
    - Short history (40 x 50):  Conv1D(16, k=8, s=4) -> Conv1D(32, k=4, s=2) -> Flatten
    - Long history (40 x 200):  Conv1D(16, k=8, s=4) -> Conv1D(32, k=4, s=2) -> Flatten
    - Concat features -> MLP
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        short_history_len: int = 50,
        long_history_len: int = 200,
    ):
        # Total observation dim = current_obs + short_history + long_history
        # current_obs: 40D (joint pos/vel, orientation, command, prev_action)
        # short_history: 40 x 50 = 2000
        # long_history: 40 x 200 = 8000
        # Total: 40 + 2000 + 8000 = 10040
        super().__init__(observation_space, features_dim)

        self.obs_dim = 40  # Base observation dimension
        self.short_history_len = short_history_len
        self.long_history_len = long_history_len

        # Short-term history CNN (processes last 50 timesteps)
        self.short_cnn = nn.Sequential(
            # Input: (batch, 40, 50)
            nn.Conv1d(in_channels=40, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            # Output: (batch, 16, 11)
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            # Output: (batch, 32, 4)
            nn.Flatten(),
            # Output: (batch, 128)
        )

        # Long-term history CNN (processes last 200 timesteps)
        self.long_cnn = nn.Sequential(
            # Input: (batch, 40, 200)
            nn.Conv1d(in_channels=40, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            # Output: (batch, 16, 49)
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            # Output: (batch, 32, 23)
            nn.Flatten(),
            # Output: (batch, 736)
        )

        # Calculate CNN output dimensions
        with torch.no_grad():
            short_dummy = torch.zeros(1, 40, short_history_len)
            long_dummy = torch.zeros(1, 40, long_history_len)
            short_features = self.short_cnn(short_dummy).shape[1]
            long_features = self.long_cnn(long_dummy).shape[1]

        # Combined feature processing
        combined_dim = self.obs_dim + short_features + long_features
        self.combine_mlp = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Process observations with dual-history architecture.

        Args:
            observations: Shape (batch, 10040)
                - [0:40]: Current observation
                - [40:2040]: Short history (40 x 50)
                - [2040:10040]: Long history (40 x 200)

        Returns:
            features: Shape (batch, features_dim)
        """
        batch_size = observations.shape[0]

        # Split observation into components
        current_obs = observations[:, :self.obs_dim]
        short_history = observations[:, self.obs_dim:self.obs_dim + self.obs_dim * self.short_history_len]
        long_history = observations[:, self.obs_dim + self.obs_dim * self.short_history_len:]

        # Reshape histories for CNN: (batch, obs_dim, time_steps)
        short_history = short_history.reshape(batch_size, self.obs_dim, self.short_history_len)
        long_history = long_history.reshape(batch_size, self.obs_dim, self.long_history_len)

        # Process through CNNs
        short_features = self.short_cnn(short_history)
        long_features = self.long_cnn(long_history)

        # Combine all features
        combined = torch.cat([current_obs, short_features, long_features], dim=1)

        # Final processing
        output = self.combine_mlp(combined)

        return output


class DualHistoryActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy with dual-history feature extractor.
    """

    def __init__(self, *args, **kwargs):
        # Force the use of our custom feature extractor
        kwargs["features_extractor_class"] = DualHistoryCNN
        kwargs["features_extractor_kwargs"] = {
            "features_dim": 256,
            "short_history_len": 50,
            "long_history_len": 200,
        }
        super().__init__(*args, **kwargs)


def create_dual_history_policy():
    """
    Factory function to create dual-history policy configuration.

    Returns:
        dict: Policy configuration for PPO
    """
    return {
        "policy_class": DualHistoryActorCriticPolicy,
        "policy_kwargs": {
            "features_extractor_class": DualHistoryCNN,
            "features_extractor_kwargs": {
                "features_dim": 256,
                "short_history_len": 50,
                "long_history_len": 200,
            },
            "net_arch": [dict(pi=[256, 128], vf=[256, 128])],
            "activation_fn": nn.ReLU,
        }
    }


if __name__ == "__main__":
    """Test the dual-history network"""
    print("Testing Dual-History Policy Network...")

    # Create dummy observation space
    obs_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(10040,),  # 40 + 40*50 + 40*200
        dtype=np.float32
    )

    # Create feature extractor
    extractor = DualHistoryCNN(obs_space, features_dim=256)

    # Test forward pass
    batch_size = 4
    dummy_obs = torch.randn(batch_size, 10040)

    print(f"\nInput shape: {dummy_obs.shape}")
    features = extractor(dummy_obs)
    print(f"Output shape: {features.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in extractor.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    print("\n[OK] Dual-history network test passed!")
    print("\nArchitecture summary:")
    print("- Current obs: 40D")
    print("- Short history: 40 x 50 timesteps (1.1 sec)")
    print("- Long history: 40 x 200 timesteps (4.4 sec)")
    print("- Output features: 256D")
