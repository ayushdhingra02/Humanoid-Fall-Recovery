import gymnasium as gym
from envs.gpt import CustomMujocoEnv
env = CustomMujocoEnv(render_mode='human')
obs, _ = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, done, _, _ = env.step(action)

    if done:
        env.reset()

env.close()