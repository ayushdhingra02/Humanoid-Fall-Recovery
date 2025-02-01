import mujoco_py
import numpy as np
import math

# Load MuJoCo model
model = mujoco_py.load_model_from_path("humanoid.xml")
sim = mujoco_py.MjSim(model)

# Time step for integration
dt = sim.model.opt.timestep

# Extract data
data = sim.data
gravity = model.opt.gravity[2]  # Assuming gravity is along the z-axis

########## zmp calculation

def calculate_zmp(sim):
    m_total = 0
    num_links = model.nbody
    zmp_x, zmp_y, zmp_numerator_x, zmp_numerator_y = 0, 0, 0, 0

    for i in range(num_links):
        mass = model.body_mass[i]
        pos = data.xpos[i]
        acc = (data.cacc[i] + gravity)  # acceleration including gravity

        zmp_numerator_x += mass * (pos[0] * acc[2] - pos[2] * acc[0])
        zmp_numerator_y += mass * (pos[1] * acc[2] - pos[2] * acc[1])
        m_total += mass * acc[2]

    zmp_x = zmp_numerator_x / m_total
    zmp_y = zmp_numerator_y / m_total
    return zmp_x, zmp_y





################# rewards

def compute_reward(sim, target_height, zmp_target, target_orientation):
    data = sim.data
    model = sim.model

    # Standing success reward
    torso_height = data.body_xpos[model.body_name2id("torso")][2]
    r_stand = 10 * max(0, 1 - abs(torso_height - target_height))

    # Balance reward
    zmp_robot = calculate_zmp(sim)  # Use your ZMP calculation function
    squared_distance = sum((a - b) ** 2 for a, b in zip(zmp_robot, zmp_target))
    r_balance = 5 * math.exp(-squared_distance)

    # Smooth motion reward
    joint_accelerations = data.qacc
    r_smooth = -1 * np.sum(np.square(joint_accelerations))

    # Energy efficiency reward
    joint_torques = data.actuator_force
    r_energy = -1 * np.sum(np.square(joint_torques))

    # Posture reward
    torso_orientation = data.body_xquat[model.body_name2id("torso")]
    r_posture = 3 * np.exp(-np.linalg.norm(torso_orientation - target_orientation) ** 2)

    # Time penalty
    r_time = -0.1  # Constant penalty per step

    # Fall penalty
    if has_fallen(sim):  # Implement this function to check for falls
        r_fall = -100
    else:
        r_fall = 0

    # Total reward
    total_reward = r_stand + r_balance + r_smooth + r_energy + r_posture + r_time + r_fall
    return total_reward



left_foot_initial = sim.data.body_xpos[sim.model.body_name2id("LeftFoot")][:2]
right_foot_initial = sim.data.body_xpos[sim.model.body_name2id("RightFoot")][:2]
initial_foot_position = (left_foot_initial + right_foot_initial) / 2

# Assuming "left_foot" and "right_foot" are the names of the feet in the model
left_foot_position = sim.data.body_xpos[sim.model.body_name2id("left_foot")][:2]
right_foot_position = sim.data.body_xpos[sim.model.body_name2id("right_foot")][:2]

# Compute the average foot position
current_foot_position = (left_foot_position + right_foot_position) / 2

r_position = 5 * np.exp(-np.linalg.norm(current_foot_position - initial_foot_position)**2)

import numpy as np


def has_fallen(sim, height_threshold=0.5, tilt_threshold=45):
    """
    Detect if the humanoid robot has fallen.

    Parameters:
        sim: Mujoco simulation object.
        height_threshold (float): Minimum torso height to consider the robot standing.
        tilt_threshold (float): Maximum allowed tilt in degrees from upright orientation.

    Returns:
        bool: True if the robot has fallen, False otherwise.
    """
    data = sim.data
    model = sim.model

    # Get torso height
    torso_height = data.body_xpos[model.body_name2id("torso")][2]

    # Get torso orientation (quaternion)
    torso_orientation = data.body_xquat[model.body_name2id("torso")]

    # Calculate tilt angle from upright position
    # Assuming the upright orientation quaternion is approximately [1, 0, 0, 0]
    upright_orientation = np.array([1, 0, 0, 0])
    tilt_angle = np.arccos(np.clip(np.dot(torso_orientation, upright_orientation), -1.0, 1.0)) * (180 / np.pi)

    # Check conditions
    has_fallen_due_to_height = torso_height < height_threshold
    has_fallen_due_to_tilt = tilt_angle > tilt_threshold

    return has_fallen_due_to_height or has_fallen_due_to_tilt


if has_fallen(sim):
    print("The robot has fallen!")
    reward = -100  # Apply a penalty or terminate the episode
    done = True  # End the episode

left_foot_contact = sim.data.contact[sim.model.body_name2id("left_foot")]
right_foot_contact = sim.data.contact[sim.model.body_name2id("right_foot")]
has_lost_contact = not (left_foot_contact or right_foot_contact)