from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym

# Import your custom HumanoidEnv class
from env import HumanoidEnv

# Function to create an instance of the custom environment
def make_env():
    def _init():
        return HumanoidEnv()
    return _init

# Number of environments to create
num_envs = 400

# Create a list of environment callables
env_list = [make_env() for _ in range(num_envs)]

# Create the SubprocVecEnv
vec_env = SubprocVecEnv(env_list)

# Test the environment
obs, info = vec_env.reset()
print(f"Observation shape: {obs.shape}")