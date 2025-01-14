from typing import Dict, Tuple, Union

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

from scipy.spatial.transform import Rotation as R
from stable_baselines3 import PPO
import torch as th
from stable_baselines3.common.vec_env import SubprocVecEnv


DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 1,
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 0.8925)),
    "elevation": -20.0,
}
class HumanoidStandupEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
            self,
            xml_file: str = "./kondo_scene_sts.xml",
            frame_skip: int = 5,
            default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
            uph_cost_weight: float = 1,
            ctrl_cost_weight: float = 0.1,
            impact_cost_weight: float = 0.5e-6,
            impact_cost_range: Tuple[float, float] = (-np.inf, 10.0),
            reset_noise_scale: float = 1e-2,
            exclude_current_positions_from_observation: bool = True,
            include_cinert_in_observation: bool = True,
            include_cvel_in_observation: bool = True,
            include_qfrc_actuator_in_observation: bool = True,
            include_cfrc_ext_in_observation: bool = True,
            **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            uph_cost_weight,
            ctrl_cost_weight,
            impact_cost_weight,
            impact_cost_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            include_cinert_in_observation,
            include_cvel_in_observation,
            include_qfrc_actuator_in_observation,
            include_cfrc_ext_in_observation,
            **kwargs,
        )

        self._uph_cost_weight = uph_cost_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._impact_cost_weight = impact_cost_weight
        self._impact_cost_range = impact_cost_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        self._include_cinert_in_observation = include_cinert_in_observation
        self._include_cvel_in_observation = include_cvel_in_observation
        self._include_qfrc_actuator_in_observation = (
            include_qfrc_actuator_in_observation
        )
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation

        self.steps=0

        obs_size = 57
        # obs_size -= 2 * exclude_current_positions_from_observation
        # obs_size += 130 * include_cinert_in_observation
        # obs_size += 78 * include_cvel_in_observation
        # obs_size += 17 * include_qfrc_actuator_in_observation
        # obs_size += 78 * include_cfrc_ext_in_observation

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size -= 2 * exclude_current_positions_from_observation
        obs_size += self.data.cinert[1:].size * include_cinert_in_observation
        obs_size += self.data.cvel[1:].size * include_cvel_in_observation
        obs_size += (self.data.qvel.size - 6) * include_qfrc_actuator_in_observation
        obs_size += self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
                    - 2 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
            "cinert": self.data.cinert[1:].size * include_cinert_in_observation,
            "cvel": self.data.cvel[1:].size * include_cvel_in_observation,
            "qfrc_actuator": (self.data.qvel.size - 6)
                             * include_qfrc_actuator_in_observation,
            "cfrc_ext": self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation,
            "ten_length": 0,
            "ten_velocity": 0,
        }

        sim=self.model
        self.right_foot_geom_id = sim.body("RightFoot").id
        self.left_foot_geom_id = sim.body("LeftFoot").id

        # Get the positions of the feet in the world coordinate frame
        self.init_right_foot_pos = self.data.geom_xpos[self.right_foot_geom_id]
        self.init_left_foot_pos = self.data.geom_xpos[self.left_foot_geom_id]

        self.action_space = Box(low=-2.356 , high=2.356, shape=(self.action_space.shape[0],), dtype=np.float32)

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._include_cinert_in_observation is True:
            com_inertia = self.data.cinert[1:].flatten()
        else:
            com_inertia = np.array([])
        if self._include_cvel_in_observation is True:
            com_velocity = self.data.cvel[1:].flatten()
        else:
            com_velocity = np.array([])

        if self._include_qfrc_actuator_in_observation is True:
            actuator_forces = self.data.qfrc_actuator[6:].flatten()
        else:
            actuator_forces = np.array([])
        if self._include_cfrc_ext_in_observation is True:
            external_contact_forces = self.data.cfrc_ext[1:].flatten()
        else:
            external_contact_forces = np.array([])

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate(
            (
                position,
                velocity,
                com_inertia,
                com_velocity,
                actuator_forces,
                external_contact_forces,
            )
        )

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        pos_after = self.data.qpos[2]

        self.steps = self.steps+1
        trunc= False

        reward, reward_info = self._get_rew(pos_after, action)
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "z_distance_from_origin": self.data.qpos[2] - self.init_qpos[2],
            "tendon_length": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
            **reward_info,
        }
        if self.steps>100:
            trunc=True
        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return self._get_obs(), reward, False, trunc, info

    def _get_rew(self, pos_after: float, action):
        uph_cost = (pos_after - 0) / self.model.opt.timestep

        quad_ctrl_cost = self._ctrl_cost_weight * np.square(self.data.ctrl).sum()

        quad_impact_cost = (
                self._impact_cost_weight * np.square(self.data.cfrc_ext).sum()
        )
        min_impact_cost, max_impact_cost = self._impact_cost_range
        quad_impact_cost = np.clip(quad_impact_cost, min_impact_cost, max_impact_cost)

        curr_right_foot_pos = self.data.geom_xpos[self.right_foot_geom_id]
        curr_left_foot_pos = self.data.geom_xpos[self.left_foot_geom_id]

        foot_pos_cost= - (curr_left_foot_pos-self.init_left_foot_pos)**2  - (curr_right_foot_pos-self.init_right_foot_pos)**2
        foot_pos_cost_s=np.sum(foot_pos_cost)
        reward = uph_cost - quad_ctrl_cost - quad_impact_cost + 1 + 4* foot_pos_cost_s

        reward_info = {
            "reward_linup": uph_cost,
            "reward_quadctrl": -quad_ctrl_cost,
            "reward_impact": -quad_impact_cost,
        }




        # return reward, reward_info
        return reward / 1000, reward_info

    def reset_model(self):

        self.steps=0
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "z_distance_from_origin": self.data.qpos[2] - self.init_qpos[2],
            "tendon_length": self.data.ten_length,
            "tendon_velocity": self.data.ten_velocity,
        }


from stable_baselines3.common.callbacks import BaseCallback


class ActionLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ActionLoggerCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log the actions being sampled
        obs = self.locals['obs']
        with th.no_grad():
            actions, _ = self.model.policy(obs)
        print(f"Actions at step {self.num_timesteps}: {actions}")
        return True

# Initialize callback
action_logger = ActionLoggerCallback()



def make_env():
    env1 = HumanoidStandupEnv()
    env1.reset()
    return env1

if __name__== "__main__":


    lr=0.0003
    niters = 250000
    model_no=1
    model_name=f"ppo sit to stand scaled reward/1000{lr} {niters} {model_no}"

    # env = HumanoidStandupEnv(env = HumanoidStandupEnv())
    env = HumanoidStandupEnv()
    # action_space = env.action_space
    # print(action_space)
    # action_logger = ActionLoggerCallback()

    # Train with callback

    # env = SubprocVecEnv([make_env for _ in range(4)])
    policy_kwargs = dict(activation_fn=th.nn.Tanh,
                         net_arch=dict(pi=[256, 256,256], vf=[256,256, 256]))

    model = PPO("MlpPolicy", env, verbose=1,
                learning_rate=lr,ent_coef=0.0001,
                policy_kwargs=policy_kwargs)
    # model = PPO.load(f"{model_name}",env)
    # obs, _ = env.reset()
    # action, _ = model.policy.predict(obs)
    # print("Sampled action from policy:", action)


    model.learn(total_timesteps=niters)
    model.save(f"{model_name}")
    #
    del model
    # ppo_humanoid_v5_custom_env_sb
    model=PPO.load(f"{model_name}")
    # model=PPO.load(f"ppo_humanoid_v5_custom_env_sb.zip")
    env = HumanoidStandupEnv(render_mode='human')
    obs,_=env.reset()
    while True:
        action, _states = model.predict(obs)
        # print (action)
        # action =np.random.rand(22)*10* np.random.choice([1, -1])
        # print (action)
        obs, rewards, dones,truncated, info = env.step(action)
        # i=i+1
        # print(i)
        # action = np.random.rand(22)
        # obs,rew,d,t,inf=env.step(action)
