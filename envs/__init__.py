from gymnasium.envs.registration import register

from envs.cassie_env import CassieEnv

register(
    id='Cassie-v0',
    entry_point='envs.cassie_env:CassieEnv',
    max_episode_steps=1000,
)

__all__ = ['CassieEnv']
