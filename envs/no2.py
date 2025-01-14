import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class MultiHumanoidEnv(gym.Env):
    def __init__(self, num_envs):
        # Create a single environment for rendering
        self.base_env = gym.make("HumanoidStandup-v5", xml_file='./robot.xml',
                                 exclude_current_positions_from_observation=False, frame_skip=1, render_mode='human')

        self.num_envs = num_envs
        self.envs = [self.base_env for _ in range(num_envs)]

        action_min_value = -2
        action_max_value = 2
        action_low = np.full(self.base_env.action_space.shape[0], action_min_value)
        action_high = np.full(self.base_env.action_space.shape[0], action_max_value)
        self.action_space = Box(low=action_low, high=action_high, dtype=np.float32)

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(num_envs * 57,),
                                     dtype=np.float32)  # Adjusted shape
        self.state = np.zeros((num_envs, 57))  # Placeholder for observations

    def reset(self, seed=None, options= None):
        observations , info = [env.reset(seed=seed) for env in self.envs]
        return self.get_combined_observation(observations) , info

    def step(self, actions):
        observations, rewards, dones,truncated, info = [], [], [], [],[]

        for env, action in zip(self.envs, actions):
            obs, reward, done,truncate, _info = env.step(action)
            observations.append(obs)
            truncated.append(truncate)
            rewards.append(reward)
            dones.append(done)

        combined_obs = self.get_combined_observation(observations)
        return combined_obs, rewards, dones, truncated, info

    def get_combined_observation(self, observations):
        # Combine the observations for all environments
        return np.concatenate(observations)

    def render(self, mode='human'):
        # Render all robots in the same environment
        for env in self.envs:
            env.render(mode)

    def close(self):
        for env in self.envs:
            env.close()


# Usage example
num_envs = 3
multi_env = MultiHumanoidEnv(num_envs)

obs,info  = multi_env.reset()
while True:
    actions = np.random.rand(num_envs, multi_env.base_env.action_space.shape[0])  # Random actions for all robots
    new_obs, rewards, dones,truncated, info = multi_env.step(actions)
    multi_env.render()
