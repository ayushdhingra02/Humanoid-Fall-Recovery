from env import HumanoidEnv
import numpy as np
from gymnasium.wrappers import TimeLimit, OrderEnforcing, PassiveEnvChecker

# from ml_logger import logger
from ppo_f import Runner


env = HumanoidEnv(render_mode='human')
env = TimeLimit(env, max_episode_steps=1000)
env = PassiveEnvChecker(env)
env = OrderEnforcing(env)
obs, info=env.reset()
print(obs.shape)
i=1
while True:
    print(i)
    i+=1
    env.step(np.random.rand(22))
    obs=env.get_observations()
    print(obs[0])