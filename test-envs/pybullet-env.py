import pybullet as p
import time

# Start PyBullet in GUI mode
print("Before")
p.connect(p.GUI)

# Load the humanoid model from DeepMimic
humanoid = p.loadMJCF("humanoid.xml")[0]
print("After")


# Set up the simulation
p.setGravity(0, 10, -9.8)
p.setRealTimeSimulation(1)

# Run the simulation
while True:
    p.stepSimulation()
    time.sleep(1./240.)


