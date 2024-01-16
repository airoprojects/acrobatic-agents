import mujoco_py
import os
import numpy as np
import time

# Load the model from the XML file
model = mujoco_py.load_model_from_path("humanoid.xml")
sim = mujoco_py.MjSim(model)

# Create a viewer to render the simulation
viewer = mujoco_py.MjViewer(sim)

# Main simulation loop
while True:
    # Step the simulation
    sim.step()
    viewer.render()

    # Apply control to the joints
    for i in range(sim.model.njnt):
        joint_name = sim.model.joint_id2name(i)
        joint_pos = sim.data.get_joint_qpos(joint_name)
        joint_vel = sim.data.get_joint_qvel(joint_name)

        # Simple control logic (modify as needed)
        # For example, applying a sinusoidal signal based on the simulation time
        control_signal = np.sin(sim.data.time)

        # Apply the control signal to the joint
        sim.data.ctrl[i] = control_signal

    time.sleep(0.01)  # Adjust as needed for your simulation speed

    # if viewer.is_alive() is False:
    #     break
