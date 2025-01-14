from typing import Any

import gym
from gymnasium.envs.registration import make, pprint_registry, register, registry, spec

register(
    id="Kondo",
    entry_point=".final_:HumanoidStandupEnv",
    max_episode_steps=1000,
)

env=gym.make("Kondo")