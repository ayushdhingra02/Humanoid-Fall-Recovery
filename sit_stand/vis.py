import gymnasium as gym

from gymnasium.envs.mujoco.humanoidstandup_v5 import HumanoidStandupEnv
import numpy as np


class HumanoidEnv(HumanoidStandupEnv):
    def __init__(self,env):
        super().__init__()
        self.env = env
        self.render_mode='human'

    def reset(self,seed=None, options=None):
        # Reset the environment and return the observation
        observation ,info = self.env.reset(seed=seed, options=options)
        self.env.data.qpos[:]= self.env.model.key_qpos[1][:]
        return self.env.data, info

    def render(self, mode='human'):
        return self.env.render()

    def close(self):
        self.env.close()








env1 = gym.make("HumanoidStandup-v5", xml_file='./robot_google_squat.xml',
                    exclude_current_positions_from_observation=False, frame_skip=1)
env=HumanoidEnv(env1)
env.reset()
# env.unwrapped.model.key_qpos
# env1.render_mode='human'
env.render()
n=0
while True:
    n=n+1
    action = np.random.rand(env.action_space.shape[0])
    # env.step(action)
    # if n%100==0:
    #     env.reset()