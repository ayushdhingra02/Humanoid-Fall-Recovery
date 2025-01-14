import gymnasium as gym
from stable_baselines3 import PPO
import torch as th
from stable_baselines3.common.env_util import make_vec_env

vec_env = make_vec_env("Humanoid-v4", n_envs=4)
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[32, 32], vf=[32, 32]))
# model = PPO("MlpPolicy", vec_env, verbose=1, policy_kwargs=policy_kwargs)
model = PPO.load("ppo_cartpole", env=vec_env)
model.learn(total_timesteps=2500000)

model.save("ppo_humanoid_stand_balanacing")

del model # remove to demonstrate saving and loading

model = PPO.load("ppo_humanoid_stand_balanacing")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")
