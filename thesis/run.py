from kondo_env import HumanoidEnv
import numpy as np
from gymnasium.wrappers import TimeLimit, OrderEnforcing, PassiveEnvChecker

# from ml_logger import logger
from ppo_f import Runner

if __name__ == '__main__':
    env = HumanoidEnv()
    env = TimeLimit(env, max_episode_steps=1000)
    env = OrderEnforcing(env)
    obs = env.reset()
    train_cfg = {
        "num_steps_per_env": 1000,  # Number of steps per environment
        "save_interval": 10,  # Interval for saving checkpoints
        "empirical_normalization": True  # Enable/disable empirical normalization
    }

    runner = Runner(env, train_cfg, device="cpu")
    # runner = Runner(env, device=f"cpu")
    runner.learn(num_learning_iterations=100000, init_at_random_ep_len=True, eval_freq=100)


#

# def make_env():
#     def _init():
#         env = HumanoidEnv()
#         env = TimeLimit(env, max_episode_steps=1000)
#         env = PassiveEnvChecker(env)
#         env = OrderEnforcing(env)
#         return env
#     return _init
# # while True:
# #     env.render()
# #     env.step(np.zeros(22))
# # qpos_squat = [
# #     0.00326883, 0, 0.215544,
# #     0.999472, 0, 0.0324943 ,0,
# #     0.00142608 ,0.0219053, - 1.31238 ,2.34488, - 1.09752 ,- 0.0219516,
# #                - 0.00142608,- 0.0219053, - 1.31742, 2.35333 ,- 1.10093, 0.0219516,
# #     0, 0,
# #                - 0.63612 ,- 0.04712, - 0.37696, - 1.67276,
# #                - 0.7068, - 0.04712, 0.98952, - 1.46072  # Right leg (hip, knee, ankle)
# #     ]
# # # envs.state=envs.unwrapped.state=qpos_squat
# # qvel_zero = np.zeros_like(env.data.qvel)
# # qpos_squat = np.array(qpos_squat)  # Convert to NumPy array
# # qvel_zero = np.array(qvel_zero)
# # env.unwrapped.set_state(qpos_squat, qvel_zero)
# # # env.data.qpos[:] = qpos_squat
# # # env.data.qvel[:] = qvel_zero
# # # env.render()
#
#
#
# from stable_baselines3 import PPO
# import torch as th
# from stable_baselines3.common.vec_env import DummyVecEnv
#
# env = DummyVecEnv([make_env()])
#
# # if __name__== "__main__":
# lr = 0.0003
# niters = 5000000
# model_no = 2
# model_name=f"ppo_f sit to stand scaled reward/1000{lr} {niters} {model_no}"
#
# policy_kwargs = dict(activation_fn=th.nn.Tanh,
#                          net_arch=dict(pi=[256, 256,256], vf=[256,256, 256]))
# model = PPO("MlpPolicy", env, verbose=1,
#             learning_rate=lr, ent_coef=0.0001,
#             policy_kwargs=policy_kwargs)
# model.learn(total_timesteps=niters)
# model.save(f"{model_name}")
# model = PPO.load(f"{model_name}")
# # model=PPO.load(f"ppo_humanoid_v5_custom_env_sb.zip")
# # env = HumanoidStandupEnv(render_mode='human')
# env=HumanoidEnv(render_mode='human')
# obs, _ = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     # print (action)
#     # action =np.random.rand(22)*10* np.random.choice([1, -1])
#     # print (action)
#     obs, rewards, dones, truncated, info = env.step(action)
#
