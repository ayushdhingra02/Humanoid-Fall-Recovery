import torch
import gymnasium as gym


class HistoryWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.env = env
        self.obs_history_length =env.observation_space.shape[0]

        self.num_obs_history = self.obs_history_length * self.num_obs
        self.obs_history = torch.zeros(self.env.num_envs, self.num_obs_history, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)
        self.num_privilleged_obs = self.num_privilleged_obs

    def step(self, action):
        obs, rew, done, trunc, info = self.env.step(action)
        privilleged_obs = None

        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], torch.tensor(obs, dtype=torch.float32).unsqueeze(0)), dim=-1)

        return obs, rew, done, trunc, info

    def get_observations(self):
        obs = self.env.get_observations()
        # privilleged_obs = self.env.get_privileeged_observations()

        return obs

    def reset_idx(self, env_ids):
        ret = super().reset_idx(env_ids)
        self.obs_history[env_ids:] = 0
        return ret

    def reset(self, seed=None, options=None):
        ret, info = super().reset(seed=seed, options=options)
        # privileged_obs = self.env.get_privileged_observations()
        self.obs_history[:, :] = 0
        return ret,info


if __name__ == "__main__":
    from tqdm import trange
    import matplotlib.pyplot as plt

    import ml_logger as logger

    from go1_gym_learn.ppo import Runner
    from go1_gym.envs.wrappers.history_wrapper import HistoryWrapper
    from go1_gym_learn.ppo.actor_critic import AC_Args

    from go1_gym.envs.base.legged_robot_config import Cfg
    from go1_gym.envs.mini_cheetah.mini_cheetah_config import config_mini_cheetah

    config_mini_cheetah(Cfg)

    test_env = gym.make("VelocityTrackingEasyEnv-v0", cfg=Cfg)
    env = HistoryWrapper(test_env)

    env.reset()
    action = torch.zeros(test_env.num_envs, 12)
    for i in trange(3):
        obs, rew, done, info = env.step(action)
        print(obs.keys())
        print(f"obs: {obs['obs']}")
        print(f"privileged obs: {obs['privileged_obs']}")
        print(f"obs_history: {obs['obs_history']}")

        img = env.render('rgb_array')
        plt.imshow(img)
        plt.show()