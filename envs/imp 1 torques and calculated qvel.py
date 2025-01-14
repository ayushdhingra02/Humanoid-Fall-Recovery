import gymnasium as gym
import numpy as np
import torch
import time
from gymnasium.spaces import Box
from history_wrapper import HistoryWrapper
from scipy.spatial.transform import Rotation as R
from stable_baselines3 import PPO
import torch as th

class HumanoidEnv(gym.Env):
    def __init__(self,env):
        super().__init__()
        self.env = env
        self.num_envs=1
        self.device='cpu'
        self.num_action = self.env.unwrapped.model.nu
        action_min_value = -2
        action_max_value = 2
        action_low = np.full(self.num_action, action_min_value)
        action_high = np.full(self.num_action, action_max_value)

        self.num_qpos = self.env.unwrapped.model.nq
        self.num_qvel = self.env.unwrapped.model.nv
        self.num_obs = self.num_qpos+ self.num_qvel
        self.env.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.num_obs*2,), dtype=np.float32)

        self.env.action_space = Box(low=action_low, high=action_high, dtype=np.float32)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.num_privilleged_obs=self.num_obs

        self.action = np.full(self.num_action, 0)
        # print("hi")
        self.qpos_storage = []
        self.qvel_storage = []


    def reset(self,seed=None, options=None):
        # Reset the environment and return the observation
        observation ,info = self.env.reset(seed=seed, options=options)
        self._reset_episode_storage()
        if not isinstance(info, dict):
            info = {}
        return self.get_observation(observation) , info

    def step(self, action):
        # Take a step in the environment and return the new state
        self.action = self._compute_torques(action)
        observation, reward, done, truncated, info = self.env.step(self.action)
        return self.get_observation(observation), reward, done,truncated, info

    def _compute_torques(self,action):
        Kp = 2.0  # Proportional gain
        Kd = 1.0  # Derivative gain

        last_qpos = self.qpos_storage[-1]  # Last step qpos
        last_qvel = self.qvel_storage[-1]  # Last step qvel

        desired_pos=action

        position_error=desired_pos-last_qpos[7:29]
        torque = Kp * position_error - Kd * last_qvel[6:28]

        torque= np.clip(torque, -10, 10)
        return torque

    def quaternion_to_angular_velocity(self,current_orientation, previous_orientation, delta_t):
        # Convert the current and previous orientations (quaternions) to rotation objects
        r_current = R.from_quat(current_orientation)  # Current quaternion
        r_previous = R.from_quat(previous_orientation)  # Previous quaternion

        # Compute the relative rotation (change in orientation)
        r_relative = r_current * r_previous.inv()

        # Convert the relative rotation to a rotation vector (axis-angle representation)
        angular_velocity_vector = r_relative.as_rotvec() / delta_t  # Shape (3,)

        return angular_velocity_vector
    def get_observation(self, observation):
        # Customize the observation here
        # For example, modifying it before returning
        delta_t = self.env.unwrapped.model.opt.timestep

        # Slice current qpos and qvel from observation
        current_qpos = observation[:29]  # First 29 elements are qpos
        current_qvel = observation[29:57]  # Next 28 elements are qvel

        # Calculate qvel (velocity) using qpos difference and delta_t
        calculated_qvel = np.zeros(28)
        # for i in range(28):
        #     # Compute qvel (velocity) as the difference between current and previous qpos
        #     calculated_qvel[i] = (current_qpos[i] - self.qpos_storage[-1][i]) / delta_t
        calculated_qvel[6:]= (current_qpos[7:]- self.qpos_storage[-1][7:])/delta_t

        linear_velocity = (current_qpos[:3] - self.qpos_storage[-1][:3]) / delta_t  # Shape (3,)

        # Angular velocity (qvel[3:6])
        current_orientation = current_qpos[3:7]  # Quaternion (w, x, y, z)
        previous_orientation = self.qpos_storage[-1][3:7]  # Quaternion (w, x, y, z)
        angular_velocity = self.quaternion_to_angular_velocity(current_orientation, previous_orientation,
                                                          delta_t)  # Shape (3,)

        # Combine linear and angular velocities
        # qvel = np.concatenate((linear_velocity, angular_velocity))
        calculated_qvel[:3]=linear_velocity
        calculated_qvel[3:6]=angular_velocity
        # Store qpos and qvel for this step
        self._store_step_data(current_qpos, calculated_qvel)

        # If less than 2 steps of data, pad with zeros
        if len(self.qpos_storage) < 2:
            prev_qpos = np.zeros(29)
            prev_qvel = np.zeros(28)
        else:
            prev_qpos = self.qpos_storage[-2]
            prev_qvel = self.qvel_storage[-2]

        # Create an observation with the last two steps of qpos and qvel data
        custom_observation = np.concatenate((self.qpos_storage[-1], self.qvel_storage[-1], prev_qpos, prev_qvel))

        return custom_observation

    def _reset_episode_storage(self):
        """Reset storage for qpos and qvel at the beginning of a new episode."""
        # Initialize with reset model data
        self.qpos_storage = [self.env.unwrapped.data.qpos[:29].copy()]  # Assuming model provides the initial position
        self.qvel_storage = [np.zeros(28)]  # Initialize with zeros for velocity

    def _store_step_data(self, qpos, qvel):
        """Store qpos and qvel for each step during the episode."""
        self.qpos_storage.append(qpos.copy())  # Store a copy of qpos
        self.qvel_storage.append(qvel.copy())  # Store a copy of qvel

    def render(self, mode='human'):
        return self.env.render()

    def close(self):
        self.env.close()



# from stable_baselines3 import PPO
# import torch as th
# from stable_baselines3.common.vec_env import SubprocVecEnv

n=0
def make_env():
    global n
    n=n+1
    print(n)
    env1 = gym.make("HumanoidStandup-v5", xml_file='../using v-5 humaoid/robot.xml',
                    exclude_current_positions_from_observation=False, frame_skip=1)
    env = HumanoidEnv(env1)
    return env


def make_env1():
    sync_vector = gym.vector.SyncVectorEnv([make_env for _ in range(8)])
    return sync_vector


if __name__=="__main__":


    # env1 = gym.make("HumanoidStandup-v5", xml_file='../using v-5 humaoid/robot.xml',
    #                 exclude_current_positions_from_observation=False, frame_skip=1 , render_mode="human")
    # env = HumanoidEnv(env1)
    #
    # env.reset()
    # Create 8 parallel environments
    # envs = SubprocVecEnv([make_env for _ in range(4)])
    # policy_kwargs = dict(activation_fn=th.nn.ReLU,
    #                      net_arch=dict(pi=[256, 256,256], vf=[256,256, 256]))
    # model = PPO("MlpPolicy", envs, verbose=1, policy_kwargs=policy_kwargs)
    # model.learn(250000)
    # model.save("ppo_humanoid_v5_custom_env_sb")
    #
    # del model
    #
    # model=PPO.load("ppo_humanoid_v5_custom_env_sb")
    env1 = gym.make("HumanoidStandup-v5", xml_file='../using v-5 humaoid/robot.xml',
                    exclude_current_positions_from_observation=False, frame_skip=1,render_mode="human")
    env = HumanoidEnv(env1)
    obs ,info = env.reset()
    qpos_squat = [
        0, 0, 0.25,  # Lower body position (x, y, z)
        1, 0, 0, 0,  # Quaternion for orientation (standing upright)
        0, 0, -0.5236, 0.6981, -0.1745, 0,  # Torso and left arm (folded hands)
        0, 0, -0.5236, 0.6981, -0.1745, 0,  # Torso and right arm (folded hands)
        0, 0,  # Unchanged for torso twist or other body angles
        -0.7854, -0.2618, -0.6981, -1.7453,  # Left leg (hip, knee, ankle)
        -0.7854, -0.2618, 0.6981, -1.7453  # Right leg (hip, knee, ankle)
    ]
    # envs.state=envs.unwrapped.state=qpos_squat
    qvel_zero = np.zeros_like(env.env.unwrapped.data.qvel)
    env.env.unwrapped.data.qpos[:] = qpos_squat
    env.env.unwrapped.data.qvel[:] = qvel_zero
    env.render()
    while True:
        # action, _states = model.predict(obs)
        # obs, rewards, dones,truncated, info = env.step(action)
        obs, rewards, dones, truncated, info = env.step(np.zeros(22))



    # # wrapped_env = HistoryWrapper(env)
    # n=0
    # start_time=time.time()
    # while True:
    #     n+=1
    #     action=np.random.rand( env.action_space.shape[0])
    #     obs,rew,trunc,done,info=env.step(action)
    #     if n % 10000 == 0:
    #         # print(n)
    #         end_time = time.time()
    #
    #         # Calculate the time difference
    #         elapsed_time = end_time - start_time
    #         print (f"{n} steps in  {elapsed_time:.5f} seconds")
    #     if n==1000000:
    #         break
    # print(obs.shape)
# n = 0
#
# while True:
#
#     n += 1
#     action = np.random.rand(envs.num_envs, envs.envs[0].action_space.shape[0])
#     new_observation, reward, done, truncated, info = envs.step(action)
#     # print(new_observation.shape)
#     obs_tensor = torch.tensor(new_observation, dtype=torch.float32)
#     action_tensor = torch.tensor(action, dtype=torch.float32)
#
#     combined_tensor = torch.cat((obs_tensor, action_tensor), dim=1)  # Shape: (3, 79)
#
#     # Concatenate the combined tensor to obs_buff
#     obs_buff = torch.cat((obs_buff, combined_tensor), dim=0)  # Concatenate along the first dimension
#
#     curr_obs_buff = obs_buff[:envs.num_envs,:]
#
#     reshaped_obs_buff = obs_buff.reshape(3, n * 79)  # Reshape to (3, n*79)




