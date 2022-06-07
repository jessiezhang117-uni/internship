import mujoco_py as mp
import os
model = mp.load_model_from_path("./model/braccio_arm.xml")
sim = mp.MjSim(model)
viewer = mp.MjViewer(sim)

for i in range(3000):
    #sim.data.ctrl[:6]=1
    sim.step()
    viewer.render()

