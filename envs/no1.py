from os import path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.spaces import Space


DEFAULT_SIZE = 480

def expand_model_path(model_path: str) -> str:
    """Expands the `model_path` to a full path if it starts with '~' or '.' or '/'."""
    if model_path.startswith(".") or model_path.startswith("/"):
        fullpath = model_path
    elif model_path.startswith("~"):
        fullpath = path.expanduser(model_path)
    else:
        fullpath = path.join(path.dirname(__file__), "assets", model_path)
    if not path.exists(fullpath):
        raise OSError(f"File {fullpath} does not exist")

    return fullpath


full_path = expand_model_path("./kondo_scene_squat_stand.xml")
print (full_path)


class HumaoidEnv(gym.Env):
    def __init__(self, num_envs: int):
        # Create a single environment for rendering
        self.base_env = gym.make("HumanoidStandup-v5", xml_file=full_path,
                                 exclude_current_positions_from_observation=False, frame_skip=1, render_mode='human')

        self.num_envs = num_envs
        self.envs = [self.base_env for _ in range(num_envs)]

        action_min_value = -2
        action_max_value = 2
        action_low = np.full(self.base_env.action_space.shape[0], action_min_value)
        action_high = np.full(self.base_env.action_space.shape[0], action_max_value)
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_envs * 57,),
                                            dtype=np.float32)  # Adjusted shape
        self.state = np.zeros((num_envs, 57))  # Placeholder for observations

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[NDArray, Dict]:
        observations , info = zip(*[env.reset(seed=seed) for env in self.envs])
        return self.get_combined_observation(observations) , info

    def step(self, actions: NDArray) -> Tuple[NDArray, NDArray, NDArray, NDArray, Dict]:
        observations, rewards, dones,truncated, info = [], [], [], [],[]

        for env, action in zip(self.envs, actions):
            obs, reward, done,truncate, _info = env.step(action)
            observations.append(obs)
            truncated.append(truncate)
            rewards.append(reward)
            dones.append(done)

        combined_obs = self.get_combined_observation(observations)
        return combined_obs, rewards, dones, truncated, info

    def get_combined_observation(self, observations: NDArray) -> NDArray:
        # Combine the observations for all environments
        return np.concatenate(observations)

    def render(self, mode: str = 'human') -> None:
        # Render all robots in the same environment
        for env in self.envs:
            env.render(mode)

    def close(self) -> None:
        for env in self.envs:
            env.close()


if __name__ == "__main__":
    # Usage example
    num_envs = 3
    multi_env = HumaoidEnv(num_envs)

    obs = multi_env.reset()
    print (obs[0].shape)
    print (obs[1])
    multi_env.close()