import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import os
import glfw

class CustomMujocoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, xml_file="kondo_scene_squat_stand.xml", render_mode=None):
        self.model = mujoco.MjModel.from_xml_path(xml_file)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode

        obs_size = self.model.nq + self.model.nv
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)

        self.viewer = None
        self.window = None

    def step(self, action):
        mujoco.mj_step(self.model, self.data)
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        reward = 0
        done = False
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        print("[DEBUG] Stepping simulation before rendering")
        for _ in range(10):  # Step the simulation to initialize rendering
            mujoco.mj_step(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)

        if self.render_mode == "human":
            self.render()

        obs = np.concatenate([self.data.qpos, self.data.qvel])
        return obs, {}

    def render(self):
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")

        if self.window is None:
            self.window = glfw.create_window(1000, 700, "MuJoCo Simulation", None, None)
            glfw.make_context_current(self.window)

        if glfw.window_should_close(self.window):
            glfw.terminate()
            return

        # 🔥 Explicitly create and set a camera
        renderer = mujoco.Renderer(self.model)

        DEFAULT_CAMERA_CONFIG = {
            "trackbodyid": 1,
            "distance": 4.0,
            "lookat": np.array((0.0, 0.0, .50)),
            "elevation": -20.0,
        }
        camera_config = DEFAULT_CAMERA_CONFIG

        camera = mujoco.MjvCamera()

        # Set camera position and target (lookat)
        camera.distance = camera_config["distance"]
        camera.lookat = camera_config["lookat"]
        camera.elevation = camera_config["elevation"]

        # Assuming `model` and `data` are your MuJoCo model and simulation data
        # Initialize the visualization options
        # mjv_default = mujoco.MjvOption()  # Create the visualization options object
        #
        # # Set the camera in the visualization options
        # mjv_default.camera = camera

        # 🔥 Render with the camera
        renderer.update_scene(self.data, camera=camera)
        renderer.render()

        glfw.swap_buffers(self.window)
        glfw.poll_events()

    def close(self):
        if self.window:
            glfw.destroy_window(self.window)
            glfw.terminate()
