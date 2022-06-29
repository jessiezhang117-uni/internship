from tkinter import FALSE
from gym import utils
from braccio_arm_gym import barobotGymEnv


class barobotGympickEnv(barobotGymEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        barobotGymEnv.__init__(
            self, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,isDiscrete=False,renders=False,
            reward_type=reward_type)
        utils.EzPickle.__init__(self)
